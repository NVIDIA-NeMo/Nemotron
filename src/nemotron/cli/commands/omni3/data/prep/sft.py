# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Omni3 SFT data preparation command."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import typer

from nemo_runspec import parse as parse_runspec
from nemo_runspec.config import (
    build_job_config,
    extract_train_config,
    generate_job_dir,
    parse_config,
    save_configs,
)
from nemo_runspec.display import display_job_config, display_job_submission
from nemo_runspec.env import parse_env
from nemo_runspec.execution import (
    build_env_vars,
    clone_git_repos_via_tunnel,
    execute_local,
    get_startup_commands,
    prepend_startup_to_cmd,
)
from nemo_runspec.recipe_config import RecipeConfig, parse_recipe_config
from nemo_runspec.recipe_typer import RecipeMeta
from nemo_runspec.squash import ensure_squashed_image

SCRIPT_PATH = "src/nemotron/recipes/omni3/stage0_sft/data_prep.py"
SPEC = parse_runspec(SCRIPT_PATH)

SETUP_COMMANDS = [
    "find . -type d -name __pycache__ -delete 2>/dev/null || true",
    "uv sync --reinstall-package nemotron",
]

META = RecipeMeta(
    name=SPEC.name,
    script_path=SCRIPT_PATH,
    config_dir=str(SPEC.config_dir),
    default_config=SPEC.config.default,
    input_artifacts={"data": "Prepared Valor32k Energon dataset path"},
    output_artifacts={"data": "Validated or staged Energon dataset metadata"},
)


def _execute_data_prep_sft(cfg: RecipeConfig):
    """Execute omni3 SFT data prep with Ray via nemo-run."""
    train_config = parse_config(cfg.ctx, SPEC.config_dir, SPEC.config.default)
    env = parse_env(cfg.ctx)

    job_config = build_job_config(
        train_config,
        cfg.ctx,
        SPEC.name,
        SCRIPT_PATH,
        cfg.argv,
        env_profile=env,
    )

    display_job_config(job_config, for_remote=False)

    if cfg.dry_run:
        return

    job_dir = generate_job_dir(SPEC.name)
    train_config_for_script = extract_train_config(job_config, for_remote=False)
    job_path, train_path = save_configs(job_config, train_config_for_script, job_dir)

    env_for_executor = job_config.run.env if hasattr(job_config.run, "env") else None
    env_vars = build_env_vars(job_config, env_for_executor)

    display_job_submission(job_path, train_path, env_vars, cfg.mode, artifacts=job_config.get("artifacts"))
    startup_commands = get_startup_commands(env_for_executor)

    if cfg.mode == "local":
        execute_local(
            SCRIPT_PATH,
            train_path,
            cfg.passthrough,
            torchrun=False,
            env_vars=env_vars,
            startup_commands=startup_commands,
        )
    else:
        _execute_ray_code_packager(
            train_path=train_path,
            env=env_for_executor,
            passthrough=cfg.passthrough,
            attached=cfg.attached,
            env_vars=env_vars,
            startup_commands=startup_commands,
            force_squash=cfg.force_squash,
        )


def _execute_ray_code_packager(
    train_path: Path,
    env,
    passthrough: list[str],
    attached: bool,
    env_vars: dict[str, str],
    startup_commands: list[str] | None,
    force_squash: bool,
):
    """Execute via Ray with code packager."""
    try:
        import nemo_run as run
        from nemo_run.run.ray.job import RayJob
    except ImportError:
        typer.echo("Error: nemo-run is required for --run/--batch execution", err=True)
        typer.echo("Install with: pip install nemo-run", err=True)
        raise typer.Exit(1)

    from nemo_runspec.packaging import CodePackager
    from nemo_runspec.run import (
        patch_nemo_run_ray_template_for_cpu,
        patch_nemo_run_rsync_accept_new_host_keys,
    )

    patch_nemo_run_rsync_accept_new_host_keys()
    patch_nemo_run_ray_template_for_cpu()

    def _get(key: str, default=None):
        if env is None:
            return default
        return env.get(key, default) if hasattr(env, "get") else getattr(env, key, default)

    tunnel = None
    remote_job_dir = _get("remote_job_dir")
    if _get("tunnel") == "ssh":
        tunnel = run.SSHTunnel(
            host=_get("host", "localhost"),
            user=_get("user"),
            job_dir=remote_job_dir,
        )

    packager = CodePackager(
        script_path=SCRIPT_PATH,
        train_path=str(train_path),
        exclude_dirs=("usage-cookbook", "use-case-examples"),
    )

    container_image = _get("container_image") or _get("container") or SPEC.image

    if container_image and tunnel and remote_job_dir:
        tunnel.connect()
        container_image = ensure_squashed_image(
            tunnel, container_image, remote_job_dir, env, force=force_squash
        )

    git_mounts = []
    if tunnel and remote_job_dir:
        tunnel.connect()
        git_mounts = clone_git_repos_via_tunnel(tunnel, remote_job_dir)

    partition = _get("run_partition") or _get("partition") if attached else _get("batch_partition") or _get("partition")

    raw_mounts = list(_get("mounts") or [])
    mounts = [mount for mount in raw_mounts if not mount.startswith("__auto_mount__:")]
    mounts.extend(git_mounts)
    mounts.append("/lustre:/lustre")

    if remote_job_dir:
        ray_temp_path = f"{remote_job_dir}/ray_temp"
        mounts.append(f"{ray_temp_path}:/ray-cluster")
        if tunnel:
            tunnel.run(f"mkdir -p {ray_temp_path}", hide=True)

    executor = run.SlurmExecutor(
        account=_get("account"),
        partition=partition,
        nodes=_get("nodes", 1),
        ntasks_per_node=_get("ntasks_per_node", 1),
        gpus_per_node=_get("gpus_per_node"),
        cpus_per_task=_get("cpus_per_task"),
        time=_get("time", "04:00:00"),
        container_image=container_image,
        container_mounts=mounts,
        tunnel=tunnel,
        packager=packager,
        mem=_get("mem"),
        env_vars=env_vars,
        launcher=None,
    )

    recipe_name = SPEC.name.replace("/", "-")
    job_name = f"{recipe_name}_{int(time.time())}"
    ray_job = RayJob(name=job_name, executor=executor)

    # Stage the resolved config under a job-scoped filename so we don't clobber
    # whatever the user happens to have at $CWD/config.yaml. The remote path
    # inside the Ray workdir stays `config.yaml` so the `cmd` template below
    # doesn't need to change; we only rename it on the local side.
    repo_config = Path.cwd() / f".nemotron-data-prep-{job_name}.yaml"
    shutil.copy2(train_path, repo_config)

    cmd = (_get("run_command", SPEC.run.cmd) or "uv run python {script} --config {config}").format(
        script=SCRIPT_PATH,
        config="config.yaml",
    )
    if passthrough:
        cmd += " " + " ".join(passthrough)
    if startup_commands:
        cmd = prepend_startup_to_cmd(startup_commands, cmd)

    runtime_env: dict = {"env_vars": dict(env_vars)}

    import tempfile
    import yaml as pyyaml

    runtime_env_yaml = None
    if runtime_env["env_vars"]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
            pyyaml.dump(runtime_env, handle)
            runtime_env_yaml = handle.name

    try:
        ray_job.start(
            command=cmd,
            workdir=str(Path.cwd()) + "/",
            pre_ray_start_commands=list(SETUP_COMMANDS),
            runtime_env_yaml=runtime_env_yaml,
        )

        remote_code_dir = f"{executor.tunnel.job_dir}/{job_name}/code"
        executor.tunnel.put(str(repo_config), f"{remote_code_dir}/config.yaml")
    finally:
        # Clean up local scratch files so we don't leak one per invocation.
        try:
            repo_config.unlink()
        except OSError:
            pass
        if runtime_env_yaml:
            try:
                Path(runtime_env_yaml).unlink()
            except OSError:
                pass

    if ray_job.backend.job_id is None:
        try:
            status = ray_job.backend.status(display=False)
            if status and status.get("job_id"):
                ray_job.backend.job_id = status["job_id"]
        except Exception:
            pass

    if attached:
        try:
            ray_job.logs(follow=True, timeout=600)
        except KeyboardInterrupt:
            typer.echo(f"\n[info] Detaching. Job {ray_job.backend.job_id} continues running.")
            raise typer.Exit(0)


def sft(ctx: typer.Context) -> None:
    """Validate or stage Valor32k Energon data for omni3 SFT."""
    cfg = parse_recipe_config(ctx)
    _execute_data_prep_sft(cfg)
