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

"""RL command implementation.

This module defines the `rl` command for the super3 recipe with
**visible execution logic**. The RL stage uses Ray instead of standard
Slurm execution.

Key differences from pretrain/sft:
- Uses RayJob instead of Slurm Experiment
- Has workdir (/opt/nemo-rl) and pre_ray_start_commands
- Uses custom run_command for uv run

To change the execution backend, modify _execute_rl() in this file.

Design: LLM-Native Recipe Architecture
- Execution logic visible and modifiable
- Fork this file to change how Ray jobs are submitted
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
    clone_git_repos_via_tunnel,
    execute_local,
    get_startup_commands,
    prepend_startup_to_cmd,
)
from nemo_runspec.packaging import REMOTE_CONFIG, REMOTE_SCRIPT
from nemo_runspec.squash import ensure_squashed_image
from nemo_runspec.recipe_config import RecipeConfig, parse_recipe_config
from nemo_runspec.recipe_typer import RecipeMeta

# =============================================================================
# Recipe Metadata (read from [tool.runspec] in script)
# =============================================================================

SCRIPT_PATH = "src/nemotron/recipes/super3/stage2_rl/train.py"
SPEC = parse_runspec(SCRIPT_PATH)

# Ray-specific execution settings
# workdir comes from SPEC, but these commands are implementation details
PRE_RAY_START_COMMANDS = [
    f"cp {REMOTE_SCRIPT} {SPEC.run.workdir}/",
    f"cp {REMOTE_CONFIG} {SPEC.run.workdir}/",
]

# For help panels
META = RecipeMeta(
    name=SPEC.name,
    script_path=SCRIPT_PATH,
    config_dir=str(SPEC.config_dir),
    default_config=SPEC.config.default,
    input_artifacts={
        "model": "SFT model checkpoint (from sft stage)",
        "data": "RL data artifact (JSONL prompts)",
    },
    output_artifacts={"model": "RL-trained model checkpoint"},
)


# =============================================================================
# Execution Logic
# =============================================================================


def _execute_rl(cfg: RecipeConfig):
    """Execute RL with Ray via nemo-run.

    This function contains the VISIBLE execution logic. RL uses Ray
    instead of standard Slurm execution.

    Args:
        cfg: Parsed recipe configuration
    """
    # =========================================================================
    # 1. Parse configuration
    # =========================================================================
    train_config = parse_config(cfg.ctx, SPEC.config_dir, SPEC.config.default)
    env = parse_env(cfg.ctx)

    # Build full job config with provenance
    job_config = build_job_config(
        train_config,
        cfg.ctx,
        SPEC.name,
        SCRIPT_PATH,
        cfg.argv,
        env_profile=env,
    )

    # Display compiled configuration
    for_remote = cfg.mode in ("run", "batch")
    display_job_config(job_config, for_remote=for_remote)

    # Handle dry-run mode
    if cfg.dry_run:
        return

    # =========================================================================
    # 2. Save configs and prepare execution
    # =========================================================================
    job_dir = generate_job_dir(SPEC.name)
    train_config_for_script = extract_train_config(job_config, for_remote=for_remote)
    job_path, train_path = save_configs(job_config, train_config_for_script, job_dir)

    # Get env config from job_config.run.env (merged YAML + env.toml)
    env_for_executor = job_config.run.env if hasattr(job_config.run, "env") else None

    env_vars = build_env_vars(job_config, env_for_executor)

    # Display job submission summary
    display_job_submission(job_path, train_path, env_vars, cfg.mode)

    # Get startup commands from env config
    startup_commands = get_startup_commands(env_for_executor)

    # =========================================================================
    # 3. Execute based on mode
    # =========================================================================
    if cfg.mode == "local":
        execute_local(
            SCRIPT_PATH,
            train_path,
            cfg.passthrough,
            torchrun=False,  # Ray handles distribution
            env_vars=env_vars,
            startup_commands=startup_commands,
        )
    else:
        # Remote execution via Ray
        _execute_ray(
            train_path=train_path,
            env=env_for_executor,
            passthrough=cfg.passthrough,
            attached=cfg.attached,
            env_vars=env_vars,
            startup_commands=startup_commands,
            force_squash=cfg.force_squash,
        )


def _execute_ray(
    train_path: Path,
    env,
    passthrough: list[str],
    attached: bool,
    env_vars: dict[str, str],
    startup_commands: list[str] | None,
    force_squash: bool,
):
    """Execute via Ray (RayJob).

    This is the VISIBLE Ray execution logic. The key differences from Slurm:
    - Uses RayJob instead of Experiment
    - Has workdir, pre_ray_start_commands, run_command
    - Config is rsynced and copied to workdir

    FORK POINT: Replace this function for different Ray submission logic.
    """
    try:
        import nemo_run as run
        from nemo_run.run.ray.job import RayJob
    except ImportError:
        typer.echo("Error: nemo-run is required for --run/--batch execution", err=True)
        typer.echo("Install with: pip install nemo-run", err=True)
        raise typer.Exit(1)

    from nemo_runspec.packaging import SelfContainedPackager
    from nemo_runspec.run import (
        patch_nemo_run_ray_template_for_cpu,
        patch_nemo_run_rsync_accept_new_host_keys,
    )

    # Apply nemo-run patches
    patch_nemo_run_rsync_accept_new_host_keys()
    patch_nemo_run_ray_template_for_cpu()

    # Helper for accessing env config (OmegaConf or dict)
    def _get(key: str, default=None):
        if env is None:
            return default
        return env.get(key, default) if hasattr(env, "get") else getattr(env, key, default)

    # Build Executor for Ray
    tunnel = None
    remote_job_dir = _get("remote_job_dir")
    if _get("tunnel") == "ssh":
        tunnel = run.SSHTunnel(
            host=_get("host", "localhost"),
            user=_get("user"),
            job_dir=remote_job_dir,
        )

    # Build packager - explicit choice of how code is bundled
    packager = SelfContainedPackager(
        script_path=SCRIPT_PATH,
        train_path=str(train_path),
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

    if attached:
        partition = _get("run_partition") or _get("partition")
    else:
        partition = _get("batch_partition") or _get("partition")

    raw_mounts = list(_get("mounts") or [])
    mounts = [m for m in raw_mounts if not m.startswith("__auto_mount__:")]
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
        launcher=None,  # Ray handles distribution
    )

    # Ray-specific setup
    recipe_name = SPEC.name.replace("/", "-")
    job_name = f"{recipe_name}_{int(time.time())}"
    ray_job = RayJob(name=job_name, executor=executor)

    # Copy train.yaml to repo root so it gets rsynced
    repo_config = Path.cwd() / REMOTE_CONFIG
    shutil.copy2(train_path, repo_config)

    # For self_contained packager, create inlined main.py at repo root
    from nemo_runspec.packaging.self_contained_packager import inline_imports

    script_file = Path(SCRIPT_PATH)
    if not script_file.is_absolute():
        script_file = Path.cwd() / SCRIPT_PATH
    inlined = inline_imports(script_file, repo_root=Path.cwd(), package_prefix="nemotron")
    repo_main = Path.cwd() / REMOTE_SCRIPT
    repo_main.write_text(inlined, encoding="utf-8")

    # Check for YAML overrides for workdir, pre_ray_start_commands, run_command
    effective_workdir = _get("workdir", SPEC.run.workdir)
    effective_pre_ray_start_commands = _get("pre_ray_start_commands", PRE_RAY_START_COMMANDS)
    effective_run_command = _get("run_command", SPEC.run.cmd)

    # Build setup commands
    if effective_pre_ray_start_commands is not None:
        setup_commands = list(effective_pre_ray_start_commands)
    else:
        setup_commands = [
            "find . -type d -name __pycache__ -delete 2>/dev/null || true",
        ]
        if effective_workdir:
            setup_commands.extend(
                [
                    f"cp {REMOTE_SCRIPT} {effective_workdir}/",
                    f"cp {REMOTE_CONFIG} {effective_workdir}/",
                ]
            )

    # Build the command to run
    if effective_run_command:
        cmd = effective_run_command.format(script=REMOTE_SCRIPT, config=REMOTE_CONFIG)
        if effective_workdir:
            cmd = f"cd {effective_workdir} && {cmd}"
    elif effective_workdir:
        cmd = f"cd {effective_workdir} && python {REMOTE_SCRIPT} --config {REMOTE_CONFIG}"
    else:
        cmd = f"uv run python {REMOTE_SCRIPT} --config {REMOTE_CONFIG}"

    if passthrough:
        cmd += " " + " ".join(passthrough)

    if startup_commands:
        cmd = prepend_startup_to_cmd(startup_commands, cmd)

    # Build runtime_env with environment variables for Ray workers
    runtime_env: dict = {"env_vars": dict(env_vars)}

    import tempfile
    import yaml as pyyaml

    runtime_env_yaml = None
    if runtime_env["env_vars"]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            pyyaml.dump(runtime_env, f)
            runtime_env_yaml = f.name

    # Start Ray Job
    ray_job.start(
        command=cmd,
        workdir=str(Path.cwd()) + "/",
        pre_ray_start_commands=setup_commands,
        runtime_env_yaml=runtime_env_yaml,
    )

    # Copy config to remote code directory
    remote_code_dir = f"{executor.tunnel.job_dir}/{job_name}/code"
    executor.tunnel.put(str(repo_config), f"{remote_code_dir}/{REMOTE_CONFIG}")

    # Recover job_id if not set (nemo-run bug workaround)
    if ray_job.backend.job_id is None:
        try:
            status = ray_job.backend.status(display=False)
            if status and status.get("job_id"):
                ray_job.backend.job_id = status["job_id"]
                typer.echo(f"[info] Recovered job_id {status['job_id']} from cluster status")
        except Exception as e:
            typer.echo(f"[warning] Slurm status check failed: {e}")

    # Attach to logs if requested
    if attached:
        try:
            ray_job.logs(follow=True, timeout=600)
        except KeyboardInterrupt:
            typer.echo("\n")
            job_id = ray_job.backend.job_id
            typer.echo(f"[info] Ctrl-C detected. Job {job_id} is still running.")
            typer.echo("")
            typer.echo("  [d] Detach - keep job running in background")
            typer.echo("  [c] Cancel - stop the job")
            typer.echo("  [enter] Detach (default)")
            typer.echo("")

            try:
                choice = input("Choice [d/c]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = "d"

            if choice == "c":
                typer.echo("[info] Cancelling job...")
                try:
                    ray_job.stop()
                    typer.echo(f"[info] Job {job_id} cancelled")
                except Exception as e:
                    typer.echo(f"[warning] Failed to cancel job: {e}")
                raise typer.Exit(130)
            else:
                typer.echo(f"[info] Detaching. Job {job_id} continues running.")
                typer.echo(f"[info] To view logs: squeue -u $USER | grep {job_id}")
                typer.echo(f"[info] To cancel: scancel {job_id}")
                raise typer.Exit(0)


# =============================================================================
# CLI Entry Point
# =============================================================================


def rl(ctx: typer.Context) -> None:
    """Run reinforcement learning with NeMo-RL GRPO (stage2).

    This command runs GRPO RL training using NeMo-RL. Unlike pretrain/sft,
    this uses Ray for distributed execution. The execution logic is visible
    in this file - see _execute_ray() for the Ray submission setup.
    """
    cfg = parse_recipe_config(ctx)
    _execute_rl(cfg)
