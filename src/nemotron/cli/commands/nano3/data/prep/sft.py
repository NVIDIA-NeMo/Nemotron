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

"""SFT data preparation command.

Execution backends:
- local: subprocess (default)
- slurm: CodePackager + RayJob via SSH tunnel
- lepton / dgxcloud: empty Packager + run.Experiment (pip install on remote)

Design: LLM-Native Recipe Architecture
- Execution logic visible and modifiable
- Fork this file to change how data prep jobs are submitted
"""

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
    create_slurm_executor,
    execute_cloud,
    execute_local,
    get_executor_type,
    get_startup_commands,
    prepend_startup_to_cmd,
)
from nemo_runspec.recipe_config import RecipeConfig, parse_recipe_config
from nemo_runspec.recipe_typer import RecipeMeta

# =============================================================================
# Recipe Metadata (read from [tool.runspec] in script)
# =============================================================================

SCRIPT_PATH = "src/nemotron/recipes/nano3/stage1_sft/data_prep.py"
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
    input_artifacts={"data": "Conversation data (JSONL/Parquet)"},
    output_artifacts={"data": "Packed SFT data (.npy format)"},
)


# =============================================================================
# Execution Logic
# =============================================================================


def _execute_data_prep_sft(cfg: RecipeConfig):
    """Execute SFT data prep with Ray via nemo-run."""
    train_config = parse_config(cfg.ctx, SPEC.config_dir, SPEC.config.default)
    env = parse_env(cfg.ctx)

    job_config = build_job_config(
        train_config, cfg.ctx, SPEC.name, SCRIPT_PATH, cfg.argv, env_profile=env,
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
            SCRIPT_PATH, train_path, cfg.passthrough,
            torchrun=False, env_vars=env_vars, startup_commands=startup_commands,
        )
    else:
        if get_executor_type(env_for_executor) in ("dgxcloud", "lepton"):
            execute_cloud(
                SCRIPT_PATH, train_path, env=env_for_executor,
                env_vars=env_vars, passthrough=cfg.passthrough,
                attached=cfg.attached, default_image=SPEC.image,
                script_resources=SPEC.resources,
                startup_commands=startup_commands,
            )
        else:
            _execute_slurm(
                train_path=train_path, env=env_for_executor,
                passthrough=cfg.passthrough, attached=cfg.attached,
                env_vars=env_vars, startup_commands=startup_commands,
                force_squash=cfg.force_squash,
            )


# =============================================================================
# Cloud Execution (Lepton / DGX Cloud)
# =============================================================================


def _execute_slurm(
    train_path: Path,
    env,
    passthrough: list[str],
    attached: bool,
    env_vars: dict[str, str],
    startup_commands: list[str] | None,
    force_squash: bool,
):
    """Execute data prep on Slurm via RayJob with CodePackager."""
    from nemo_run.run.ray.job import RayJob

    from nemo_runspec.packaging import CodePackager
    from nemo_runspec.run import (
        patch_nemo_run_ray_template_for_cpu,
        patch_nemo_run_rsync_accept_new_host_keys,
    )

    patch_nemo_run_rsync_accept_new_host_keys()
    patch_nemo_run_ray_template_for_cpu()

    _get = _make_env_getter(env)

    packager = CodePackager(
        script_path=SCRIPT_PATH,
        train_path=str(train_path),
        exclude_dirs=("usage-cookbook", "use-case-examples"),
    )
    executor = create_slurm_executor(
        env, env_vars, packager,
        default_image=SPEC.image,
        attached=attached,
        force_squash=force_squash,
        launcher=None,
    )

    recipe_name = SPEC.name.replace("/", "-")
    job_name = f"{recipe_name}_{int(time.time())}"
    ray_job = RayJob(name=job_name, executor=executor)

    repo_config = Path.cwd() / "config.yaml"
    shutil.copy2(train_path, repo_config)

    effective_run_command = _get("run_command", SPEC.run.cmd)
    cmd = effective_run_command.format(script=SCRIPT_PATH, config="config.yaml")
    if passthrough:
        cmd += " " + " ".join(passthrough)
    if startup_commands:
        cmd = prepend_startup_to_cmd(startup_commands, cmd)

    runtime_env: dict = {"env_vars": dict(env_vars)}

    import tempfile

    import yaml as pyyaml

    runtime_env_yaml = None
    if runtime_env["env_vars"]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            pyyaml.dump(runtime_env, f)
            runtime_env_yaml = f.name

    ray_job.start(
        command=cmd,
        workdir=str(Path.cwd()) + "/",
        pre_ray_start_commands=list(SETUP_COMMANDS),
        runtime_env_yaml=runtime_env_yaml,
    )

    if hasattr(executor, "tunnel") and executor.tunnel:
        remote_code_dir = f"{executor.tunnel.job_dir}/{job_name}/code"
        executor.tunnel.put(str(repo_config), f"{remote_code_dir}/config.yaml")

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


# =============================================================================
# Helpers
# =============================================================================


def _make_env_getter(env):
    """Return a helper to read values from env config (OmegaConf or dict)."""
    def _get(key: str, default=None):
        if env is None:
            return default
        return env.get(key, default) if hasattr(env, "get") else getattr(env, key, default)
    return _get


# =============================================================================
# CLI Entry Point
# =============================================================================


def sft(ctx: typer.Context) -> None:
    """Prepare data for SFT (packed Parquet format).

    This command prepares conversation data for supervised fine-tuning.
    Uses Ray for distributed processing with the xenna data pipeline.
    """
    cfg = parse_recipe_config(ctx)
    _execute_data_prep_sft(cfg)
