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

"""@recipe decorator for defining CLI commands.

The decorator attaches metadata and standardizes the execution flow
for recipe commands.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from nemotron.kit.cli.config import ConfigBuilder
from nemotron.kit.cli.display import display_job_config, display_job_submission
from nemotron.kit.cli.globals import GlobalContext, split_unknown_args

console = Console()


def _get_startup_commands(env_config: dict | None) -> list[str]:
    """Extract and validate startup_commands from env config.

    Args:
        env_config: Environment configuration dict from run.env

    Returns:
        List of shell commands to run before training, or empty list
    """
    if not env_config:
        return []
    commands = env_config.get("startup_commands")
    if not commands:
        return []
    if not isinstance(commands, list):
        raise typer.Exit(1)
    for cmd in commands:
        if not isinstance(cmd, str):
            typer.echo(
                f"Error: startup_commands must be a list of strings, got {type(cmd).__name__}",
                err=True,
            )
            raise typer.Exit(1)
    return commands


def _prepend_startup_to_cmd(startup_commands: list[str], cmd: str) -> str:
    """Prepend startup commands to a shell command string.

    Args:
        startup_commands: List of shell commands to run first
        cmd: The main command to run after startup

    Returns:
        Combined command string with startup commands prepended
    """
    if not startup_commands:
        return cmd
    # Join with && for fail-fast behavior
    startup_block = " && ".join(startup_commands)
    return f"{{ {startup_block}; }} && {cmd}"


def _run_startup_commands_local(startup_commands: list[str]) -> None:
    """Run startup commands locally before training.

    Args:
        startup_commands: List of shell commands to run

    Raises:
        typer.Exit: If any command fails
    """
    for cmd in startup_commands:
        typer.echo(f"[startup] {cmd}")
        result = subprocess.run(cmd, shell=True, executable="/bin/bash")
        if result.returncode != 0:
            typer.echo(f"Error: startup command failed with code {result.returncode}", err=True)
            raise typer.Exit(result.returncode)


@dataclass
class RecipeMetadata:
    """Metadata attached to a recipe command function.

    Attributes:
        name: Recipe identifier (e.g., "nano3/pretrain")
        script_path: Path to training script relative to repo root
        config_dir: Path to config directory relative to repo root
        artifacts: Artifact slot definitions for resolution
        torchrun: Whether to use torchrun launcher
        ray: Whether this recipe requires Ray
        packager: Packager type ("pattern", "code", "self_contained")
        workdir: Container working directory for Ray jobs (e.g., "/opt/nemo-rl")
        pre_ray_start_commands: Shell commands to run before Ray starts
        run_command: Custom command template to run the script (e.g., "python {script} --config {config}")
    """

    name: str
    script_path: str
    config_dir: str
    default_config: str = "default"
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    torchrun: bool = True
    ray: bool = False
    packager: str = "pattern"
    workdir: str | None = None
    pre_ray_start_commands: list[str] | None = None
    run_command: str | None = None


def recipe(
    name: str,
    script_path: str,
    config_dir: str,
    default_config: str = "default",
    artifacts: dict[str, dict[str, Any]] | None = None,
    *,
    torchrun: bool = True,
    ray: bool = False,
    packager: str = "pattern",
    workdir: str | None = None,
    pre_ray_start_commands: list[str] | None = None,
    run_command: str | None = None,
) -> Callable:
    """Decorator marking a function as a recipe command.

    The decorated function becomes a typer command that:
    1. Loads and merges configuration
    2. Resolves env profile (if --run/--batch)
    3. Optionally resolves artifacts
    4. Saves job.yaml and train.yaml
    5. Executes the training script

    Args:
        name: Recipe identifier (e.g., "nano3/pretrain")
        script_path: Path to Python script for execution
                    (e.g., "src/nemotron/recipes/nano3/stage0_pretrain/train.py")
        config_dir: Path to config directory
                   (e.g., "src/nemotron/recipes/nano3/stage0_pretrain/config")
        artifacts: Optional artifact slot definitions:
                  {"data": {"default": "DataBlendsArtifact-pretrain",
                            "mappings": {"path": "recipe.per_split_data_args_path"}}}
        default_config: Default config name (stem) or path used when -c/--config
            is not provided (default: "default").
        torchrun: Whether to use torchrun launcher (default: True)
        ray: Whether this recipe requires Ray for execution (default: False)
        packager: Packager type ("pattern", "code", "self_contained") (default: "pattern")
        workdir: Container working directory for Ray jobs (e.g., "/opt/nemo-rl").
            Can be overridden via YAML config at `run.env.workdir`.
        pre_ray_start_commands: Shell commands to run before Ray starts.
            Can be overridden via YAML config at `run.env.pre_ray_start_commands`.
        run_command: Custom command template to run the script. Supports placeholders:
            {script} - the script path (e.g., "main.py")
            {config} - the config path (e.g., "config.yaml")
            Example: "python {script} --config {config}"
            Can be overridden via YAML config at `run.env.run_command`.

    YAML config options (under `run.env`):
        startup_commands: List of shell commands to run immediately before training.
            Commands run after infrastructure is ready (container started, Ray cluster up)
            but before the training script executes. Useful for:
            - Cloning a git branch: "git clone --branch feature https://... /workspace/custom"
            - Installing packages: "pip install /workspace/custom"
            - Creating directories: "mkdir -p /nemo_run/outputs"
            Commands execute with fail-fast behavior (stops on first error).

    Example:
        @recipe(
            name="nano3/pretrain",
            script_path="src/nemotron/recipes/nano3/stage0_pretrain/train.py",
            config_dir="src/nemotron/recipes/nano3/stage0_pretrain/config",
        )
        def pretrain(ctx: typer.Context):
            '''Run pretraining with Megatron-Bridge.'''
            ...

        @recipe(
            name="nano3/rl",
            script_path="src/nemotron/recipes/nano3/stage2_rl/train.py",
            config_dir="src/nemotron/recipes/nano3/stage2_rl/config",
            torchrun=False,
            ray=True,
            packager="self_contained",
            workdir="/opt/nemo-rl",
            pre_ray_start_commands=[
                "find . -type d -name __pycache__ -delete 2>/dev/null || true",
            ],
            run_command="uv run python {script} --config {config}",
        )
        def rl(ctx: typer.Context):
            '''Run RL training with Ray.'''
            ...
    """
    if artifacts is None:
        artifacts = {}

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(ctx: typer.Context) -> None:
            # Get global context
            global_ctx: GlobalContext = ctx.obj
            if global_ctx is None:
                global_ctx = GlobalContext()

            # Split unknown args into dotlist and passthrough
            # Also extract any global options that appear after the subcommand
            dotlist, passthrough, global_ctx = split_unknown_args(ctx.args or [], global_ctx)
            global_ctx.dotlist = dotlist
            global_ctx.passthrough = passthrough

            # Validate options after split_unknown_args has extracted all global options
            if global_ctx.run and global_ctx.batch:
                typer.echo("Error: --run and --batch cannot both be set", err=True)
                raise typer.Exit(1)

            if global_ctx.stage and not global_ctx.profile:
                typer.echo(
                    "Error: --stage requires --run or --batch to specify target cluster", err=True
                )
                raise typer.Exit(1)

            # Build configuration
            builder = ConfigBuilder(
                recipe_name=name,
                script_path=script_path,
                config_dir=config_dir,
                default_config=default_config,
                ctx=global_ctx,
                argv=sys.argv,
            )

            # Load and merge config
            builder.load_and_merge()

            # TODO: Resolve artifacts if run.data etc. specified
            # This would apply mappings from artifact metadata to config

            # Build full job config
            builder.build_job_config()

            # Display compiled configuration
            # Show resolved paths for remote execution (--run/--batch), but not for "code" packager
            for_remote = global_ctx.mode in ("run", "batch") and packager != "code"
            display_job_config(builder.job_config, for_remote=for_remote)

            # Handle dry-run mode
            if global_ctx.dry_run:
                return

            # Save configs
            job_path, train_path = builder.save(packager=packager)

            # Handle stage-only mode
            if global_ctx.stage:
                _execute_stage_only(
                    script_path=script_path,
                    train_path=train_path,
                    job_dir=builder.job_dir,
                    job_config=builder.job_config,
                    packager=packager,
                    torchrun=torchrun,
                )
                return

            # Extract env config for building env vars
            env_config = None
            if hasattr(builder.job_config, "run") and hasattr(builder.job_config.run, "env"):
                from omegaconf import OmegaConf

                env_config = OmegaConf.to_container(builder.job_config.run.env, resolve=True)

            # Build env vars for display (needs job_config for wandb settings)
            env_vars = _build_env_vars(builder.job_config, env_config)

            # Display job submission summary
            display_job_submission(job_path, train_path, env_vars, global_ctx.mode)

            # Get startup commands from env config
            startup_commands = _get_startup_commands(env_config)

            # Execute based on mode
            if global_ctx.mode == "local":
                # Set env vars so subprocess inherits them (wandb, HF tokens, etc.)
                os.environ.update(env_vars)
                # Run startup commands before training
                if startup_commands:
                    _run_startup_commands_local(startup_commands)
                _execute_local(script_path, train_path, passthrough, torchrun=torchrun)
            else:
                _execute_nemo_run(
                    script_path=script_path,
                    train_path=train_path,
                    job_dir=builder.job_dir,
                    job_config=builder.job_config,
                    passthrough=passthrough,
                    attached=(global_ctx.mode == "run"),
                    env_vars=env_vars,
                    torchrun=torchrun,
                    ray=ray,
                    packager=packager,
                    workdir=workdir,
                    pre_ray_start_commands=pre_ray_start_commands,
                    run_command=run_command,
                    force_squash=global_ctx.force_squash,
                    startup_commands=startup_commands,
                )

        # Attach metadata to function for introspection
        wrapper._recipe_metadata = RecipeMetadata(
            name=name,
            script_path=script_path,
            config_dir=config_dir,
            default_config=default_config,
            artifacts=artifacts,
            torchrun=torchrun,
            ray=ray,
            packager=packager,
            workdir=workdir,
            pre_ray_start_commands=pre_ray_start_commands,
            run_command=run_command,
        )

        return wrapper

    return decorator


def _execute_local(
    script_path: str,
    train_path: Path,
    passthrough: list[str],
    *,
    torchrun: bool = True,
) -> None:
    """Execute script locally via subprocess.

    Args:
        script_path: Path to the training script
        train_path: Path to the saved train.yaml
        passthrough: Additional args to pass to script
        torchrun: Whether to use torchrun launcher
    """
    if torchrun:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            script_path,
            "--config",
            str(train_path),
            *passthrough,
        ]
    else:
        cmd = [
            sys.executable,
            script_path,
            "--config",
            str(train_path),
            *passthrough,
        ]

    typer.echo(f"Executing: {' '.join(cmd)}")

    result = subprocess.run(cmd)
    raise typer.Exit(result.returncode)


def _execute_nemo_run(
    script_path: str,
    train_path: Path,
    job_dir: Path,
    job_config: Any,
    passthrough: list[str],
    attached: bool,
    env_vars: dict[str, str],
    *,
    torchrun: bool = True,
    ray: bool = False,
    packager: str = "pattern",
    workdir: str | None = None,
    pre_ray_start_commands: list[str] | None = None,
    run_command: str | None = None,
    force_squash: bool = False,
    startup_commands: list[str] | None = None,
) -> None:
    """Execute script via nemo-run.

    Args:
        script_path: Path to the training script
        train_path: Path to the saved train.yaml
        job_dir: Path to the job directory
        job_config: Full job configuration (contains run.env)
        passthrough: Additional args to pass to script
        attached: If True, wait for completion; if False, detach
        env_vars: Pre-built environment variables
        torchrun: Whether to use torchrun launcher
        ray: Whether this recipe requires Ray
        packager: Packager type ("pattern", "code", "self_contained")
        workdir: Container working directory for Ray jobs (e.g., "/opt/nemo-rl")
        pre_ray_start_commands: Shell commands to run before Ray starts
        run_command: Custom command template (supports {script} and {config} placeholders)
        force_squash: Whether to force re-squash container image
        startup_commands: Shell commands to run immediately before training starts
    """
    import time

    try:
        import nemo_run as run
    except ImportError:
        typer.echo("Error: nemo-run is required for --run/--batch execution", err=True)
        typer.echo("Install with: pip install nemo-run", err=True)
        raise typer.Exit(1)

    from omegaconf import OmegaConf

    # Extract env config
    env_config = OmegaConf.to_container(job_config.run.env, resolve=True)

    # Build executor with flat file layout (main.py, config.yaml)
    executor = _build_executor(
        env_config,
        job_config,
        script_path,
        train_path,
        job_dir,
        env_vars,
        torchrun=torchrun,
        ray=ray,
        attached=attached,
        packager=packager,
        force_squash=force_squash,
    )

    # Script args use flat names on remote
    script_args = ["--config", "config.yaml", *passthrough]

    # Get experiment name from recipe
    recipe_name = job_config.run.recipe.name.replace("/", "-")

    if ray:
        # Use RayJob for Ray-based recipes
        import shutil

        from nemo_run.run.ray.job import RayJob

        # Generate unique job name to prevent directory collisions
        job_name = f"{recipe_name}_{int(time.time())}"
        ray_job = RayJob(name=job_name, executor=executor)

        # Copy train.yaml to repo root so it gets rsynced
        # This ensures the config with rewritten paths is available on remote
        repo_config = Path.cwd() / "config.yaml"
        shutil.copy2(train_path, repo_config)

        # Check for YAML overrides for workdir, pre_ray_start_commands, and run_command
        effective_workdir = workdir
        effective_pre_ray_start_commands = pre_ray_start_commands
        effective_run_command = run_command
        if env_config.get("workdir"):
            effective_workdir = env_config["workdir"]
        if env_config.get("pre_ray_start_commands"):
            effective_pre_ray_start_commands = env_config["pre_ray_start_commands"]
        if env_config.get("run_command"):
            effective_run_command = env_config["run_command"]

        # Build setup commands based on packager type and workdir
        if effective_pre_ray_start_commands is not None:
            # Use explicitly provided commands (user handles all setup)
            setup_commands = list(effective_pre_ray_start_commands)
        elif packager == "self_contained":
            # For self_contained packager, skip uv sync (dependencies in container)
            # and copy files from /nemo_run/code to workdir
            setup_commands = [
                "find . -type d -name __pycache__ -delete 2>/dev/null || true",
            ]
            if effective_workdir:
                setup_commands.extend(
                    [
                        f"cp /nemo_run/code/main.py {effective_workdir}/",
                        f"cp /nemo_run/code/config.yaml {effective_workdir}/",
                    ]
                )
            else:
                setup_commands.extend(
                    [
                        "cp /nemo_run/code/main.py .",
                        "cp /nemo_run/code/config.yaml .",
                    ]
                )
        else:
            # Default setup for other packagers
            setup_commands = [
                "find . -type d -name __pycache__ -delete 2>/dev/null || true",
                "uv sync --reinstall-package nemotron",
            ]

        # Determine remote script path
        if packager == "self_contained":
            remote_script = "main.py"
        else:
            # Get the actual script path from the recipe config
            remote_script = job_config.run.recipe.script

        # Build the command to run
        config_file = "config.yaml"
        if effective_run_command:
            # Use custom run command template with placeholders
            cmd = effective_run_command.format(script=remote_script, config=config_file)
            # Prepend cd to workdir if set (pre_ray_start_commands run before Ray, not in job)
            if effective_workdir:
                cmd = f"cd {effective_workdir} && {cmd}"
        elif effective_workdir and packager == "self_contained":
            # Already cd'd in setup_commands, run directly
            cmd = f"cd {effective_workdir} && python {remote_script} --config {config_file}"
        elif effective_workdir:
            cmd = f"cd {effective_workdir} && uv run python {remote_script} --config {config_file}"
        else:
            cmd = f"uv run python {remote_script} --config {config_file}"
        if passthrough:
            cmd += " " + " ".join(passthrough)

        # Prepend startup commands to run immediately before training
        if startup_commands:
            cmd = _prepend_startup_to_cmd(startup_commands, cmd)

        # Build runtime_env with environment variables for Ray workers
        runtime_env: dict = {"env_vars": dict(env_vars)}

        # Create temporary runtime_env YAML file
        import tempfile

        import yaml as pyyaml

        runtime_env_yaml = None
        if runtime_env["env_vars"]:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                pyyaml.dump(runtime_env, f)
                runtime_env_yaml = f.name

        ray_job.start(
            command=cmd,
            # Pass workdir with trailing slash to rsync the repo contents to remote.
            # nemo-run uses --filter=':- .gitignore' which respects .gitignore patterns.
            # The trailing slash ensures contents are synced (not the parent dir itself).
            workdir=str(Path.cwd()) + "/",
            pre_ray_start_commands=setup_commands,
            runtime_env_yaml=runtime_env_yaml,
        )

        # Copy config.yaml to remote code directory since it's excluded by .gitignore
        # This ensures the config file with rewritten paths is available remotely
        remote_code_dir = f"{executor.tunnel.job_dir}/{job_name}/code"
        executor.tunnel.put(str(repo_config), f"{remote_code_dir}/config.yaml")

        # Workaround for nemo-run bug: when reusing an existing cluster,
        # SlurmRayCluster.create() returns None instead of the job_id.
        if ray_job.backend.job_id is None:
            try:
                status = ray_job.backend.status(display=False)
            except Exception as e:
                # Slurm controller may be temporarily unavailable (e.g., backup controller
                # in standby mode). Continue without recovered job_id rather than failing.
                typer.echo(
                    f"[warning] Slurm status check failed; continuing without recovered job_id: {e}"
                )
                status = None

            if status and status.get("job_id"):
                ray_job.backend.job_id = status["job_id"]
                typer.echo(f"[info] Recovered job_id {status['job_id']} from cluster status")

        if attached:
            try:
                # Wait up to 10 minutes for log file to appear
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
                    # Second Ctrl+C or EOF means detach
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
    else:
        # Standard execution via nemo-run Script
        if startup_commands:
            # When startup commands are specified, wrap the training command in bash
            # This runs startup commands immediately before training
            import shlex

            train_cmd = shlex.join(["python", "main.py", *script_args])
            full_cmd = _prepend_startup_to_cmd(startup_commands, train_cmd)
            script_task = run.Script(
                path="bash",
                args=["-lc", full_cmd],
            )
        else:
            script_task = run.Script(
                path="main.py",  # Flat name on remote
                args=script_args,
                entrypoint="python",
            )

        with run.Experiment(recipe_name) as exp:
            exp.add(
                script_task,
                executor=executor,
                name=recipe_name,
            )
            exp.run(detach=not attached)


def _build_executor(
    env_config: dict,
    job_config: Any,
    script_path: str,
    train_path: Path,
    job_dir: Path,
    env_vars: dict[str, str],
    *,
    torchrun: bool = True,
    ray: bool = False,
    attached: bool = True,
    packager: str = "pattern",
    force_squash: bool = False,
) -> Any:
    """Build nemo-run executor from env config.

    Args:
        env_config: Environment configuration dict
        job_config: Full job config (unused, kept for future extensions)
        script_path: Path to the training script
        train_path: Path to the train.yaml config
        job_dir: Path to the job directory for staging files
        env_vars: Pre-built environment variables
        torchrun: Whether to use torchrun launcher
        ray: Whether this recipe requires Ray
        attached: Whether running in attached mode (--run vs --batch)
        force_squash: Whether to force re-squash container image

    Returns:
        nemo-run Executor instance
    """
    import nemo_run as run

    # Apply patches to nemo-run before building executor
    from nemotron.kit.run import (
        patch_nemo_run_ray_template_for_cpu,
        patch_nemo_run_rsync_accept_new_host_keys,
    )
    patch_nemo_run_rsync_accept_new_host_keys()
    patch_nemo_run_ray_template_for_cpu()

    executor_type = env_config.get("executor", "local")

    # Determine launcher
    launcher = "torchrun" if torchrun else None

    if executor_type == "local":
        return run.LocalExecutor(
            ntasks_per_node=env_config.get("nproc_per_node", 1),
            launcher=launcher,
            env_vars=env_vars,
        )

    elif executor_type == "slurm":
        # Build tunnel if configured
        tunnel = None
        remote_job_dir = env_config.get("remote_job_dir")
        if env_config.get("tunnel") == "ssh":
            tunnel = run.SSHTunnel(
                host=env_config.get("host", "localhost"),
                user=env_config.get("user"),
                job_dir=remote_job_dir,
            )

        # Build packager with flat file layout (main.py, config.yaml)
        packager = _build_packager(
            script_path,
            train_path,
            job_dir,
            packager=packager,
        )

        # Container image can be specified as "container" or "container_image"
        container_image = env_config.get("container_image") or env_config.get("container")

        # Ensure container image is squashed on the cluster
        if container_image and tunnel and remote_job_dir:
            # Connect tunnel to check/create squashed image
            tunnel.connect()
            container_image = _ensure_squashed_image(
                tunnel, container_image, remote_job_dir, env_config, force=force_squash
            )

        # Select partition based on mode (--run uses run_partition, --batch uses batch_partition)
        if attached:
            partition = env_config.get("run_partition") or env_config.get("partition")
        else:
            partition = env_config.get("batch_partition") or env_config.get("partition")

        # Build container mounts, adding /lustre and Ray temp directory
        mounts = list(env_config.get("mounts") or [])
        # Mount /lustre for access to shared storage (HF cache, data, etc.)
        mounts.append("/lustre:/lustre")
        remote_job_dir = env_config.get("remote_job_dir")
        if remote_job_dir:
            # Ray temp directory mount (avoids filling container storage with Ray logs)
            ray_temp_path = f"{remote_job_dir}/ray_temp"
            mounts.append(f"{ray_temp_path}:/ray-cluster")
            # Ensure the ray_temp directory exists on the remote filesystem
            if tunnel:
                tunnel.run(f"mkdir -p {ray_temp_path}", hide=True)

        # Build executor kwargs, only including exclusive if True
        executor_kwargs: dict[str, Any] = {
            "account": env_config.get("account"),
            "partition": partition,
            "nodes": env_config.get("nodes", 1),
            "ntasks_per_node": env_config.get("ntasks_per_node", 1),
            "gpus_per_node": env_config.get("gpus_per_node"),
            "cpus_per_task": env_config.get("cpus_per_task"),
            "time": env_config.get("time", "04:00:00"),
            "container_image": container_image,
            "container_mounts": mounts,
            "tunnel": tunnel,
            "packager": packager,
            "mem": env_config.get("mem"),
            "env_vars": env_vars,
            "launcher": launcher,
        }

        # Only add exclusive if explicitly True (avoids slurm error)
        if env_config.get("exclusive"):
            executor_kwargs["exclusive"] = True

        # TODO: Add Ray support when ray=True
        # This would configure the executor for Ray cluster setup

        return run.SlurmExecutor(**executor_kwargs)

    else:
        raise ValueError(f"Unknown executor type: {executor_type}")


def _build_env_vars(job_config: Any, env_config: dict | None = None) -> dict:
    """Build environment variables for nemo-run execution.

    Sets up:
    - NEMO_RUN_DIR for output paths
    - HF_HOME for HuggingFace cache (defaults to remote_job_dir/hf)
    - HF_TOKEN if logged in to HuggingFace
    - WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT if logged in to W&B

    Args:
        job_config: Full job configuration (contains run.wandb section)
        env_config: Environment configuration from env.toml (contains remote_job_dir)

    Returns:
        Dictionary of environment variables
    """
    import os

    from omegaconf import OmegaConf

    env_vars: dict[str, str] = {}

    # Set NEMO_RUN_DIR to actual lustre path for output paths
    # This ensures artifacts store the real path, not /nemo_run container mount
    # Only set for remote execution - local execution uses default paths
    if env_config and env_config.get("remote_job_dir"):
        env_vars["NEMO_RUN_DIR"] = env_config["remote_job_dir"]

    # Set HF_HOME to remote_job_dir/hf if not explicitly set by user
    # This ensures HuggingFace downloads go to Lustre storage with sufficient space
    if os.environ.get("HF_HOME"):
        # Respect user's explicit HF_HOME setting
        env_vars["HF_HOME"] = os.environ["HF_HOME"]
    elif env_config and env_config.get("remote_job_dir"):
        env_vars["HF_HOME"] = f"{env_config['remote_job_dir']}/hf"

    # Auto-detect HuggingFace token
    try:
        from huggingface_hub import HfFolder

        token = HfFolder.get_token()
        if token:
            env_vars["HF_TOKEN"] = token
    except Exception:
        pass

    # Auto-detect Weights & Biases API key
    try:
        import wandb

        api_key = wandb.api.api_key
        if api_key:
            env_vars["WANDB_API_KEY"] = api_key
    except Exception:
        pass

    # Extract W&B entity and project from job config
    try:
        if hasattr(job_config, "run") and hasattr(job_config.run, "wandb"):
            wandb_config = OmegaConf.to_container(job_config.run.wandb, resolve=True)
            if wandb_config.get("entity"):
                env_vars["WANDB_ENTITY"] = str(wandb_config["entity"])
            if wandb_config.get("project"):
                env_vars["WANDB_PROJECT"] = str(wandb_config["project"])
    except Exception:
        pass

    return env_vars


def _build_packager(
    script_path: str,
    train_path: Path,
    job_dir: Path,
    *,
    packager: str = "pattern",
) -> Any:
    """Build a packager for file syncing.

    Packager types:
    - "pattern": Minimal sync of `main.py` + `config.yaml` only (default)
    - "code": Full codebase sync with exclusions (for Ray jobs needing local imports)
    - "self_contained": Inlines `nemotron.*` imports into a single script
    """
    import shutil

    from nemo_run.core.packaging import PatternPackager

    if packager == "self_contained":
        from nemotron.kit.packaging import SelfContainedPackager

        return SelfContainedPackager(
            script_path=script_path,
            train_path=str(train_path),
        )

    if packager == "code":
        from nemotron.kit.packaging import CodePackager

        return CodePackager(
            script_path=script_path,
            train_path=str(train_path),
            exclude_dirs=("usage-cookbook", "use-case-examples"),
        )

    if packager != "pattern":
        raise ValueError(f"Unknown packager: {packager}")

    code_dir = job_dir / "code"
    code_dir.mkdir(exist_ok=True)

    shutil.copy2(script_path, code_dir / "main.py")
    shutil.copy2(train_path, code_dir / "config.yaml")

    main_path = str(code_dir / "main.py")
    config_path = str(code_dir / "config.yaml")
    return PatternPackager(
        include_pattern=[main_path, config_path],
        relative_path=[str(code_dir), str(code_dir)],
    )


def _get_squash_path(container_image: str, remote_job_dir: str) -> str:
    """Get the path to the squashed container image.

    Creates a deterministic filename based on the container image name.
    For example: nvcr.io/nvidian/nemo:25.11-nano-v3.rc2 -> nemo-25.11-nano-v3.rc2.sqsh

    Args:
        container_image: Docker container image (e.g., nvcr.io/nvidian/nemo:25.11-nano-v3.rc2)
        remote_job_dir: Remote directory for squashed images

    Returns:
        Full path to squashed image file
    """
    # Extract image name and tag for readable filename
    # nvcr.io/nvidian/nemo:25.11-nano-v3.rc2 -> nemo:25.11-nano-v3.rc2
    image_name = container_image.split("/")[-1]
    # nemo:25.11-nano-v3.rc2 -> nemo-25.11-nano-v3.rc2.sqsh
    sqsh_name = image_name.replace(":", "-") + ".sqsh"

    return f"{remote_job_dir}/{sqsh_name}"


def _ensure_squashed_image(
    tunnel: Any,
    container_image: str,
    remote_job_dir: str,
    env_config: dict,
    *,
    force: bool = False,
) -> str:
    """Ensure the container image is squashed on the remote cluster.

    Checks if a squashed version exists, and if not, creates it using enroot
    on a compute node via salloc.

    Args:
        tunnel: SSHTunnel instance (already connected)
        container_image: Docker container image to squash
        remote_job_dir: Remote directory for squashed images
        env_config: Environment config with slurm settings (account, partition, time)
        force: If True, re-squash even if file already exists

    Returns:
        Path to the squashed image file
    """
    sqsh_path = _get_squash_path(container_image, remote_job_dir)

    # Check if squashed image already exists (unless force is set)
    if not force:
        with console.status("[bold blue]Checking for squashed image..."):
            result = tunnel.run(f"test -f {sqsh_path} && echo exists", hide=True, warn=True)

        if result.ok and "exists" in result.stdout:
            console.print(
                f"[green]✓[/green] Using existing squashed image: [cyan]{sqsh_path}[/cyan]"
            )
            return sqsh_path

    # Need to create the squashed image
    if force:
        console.print("[yellow]![/yellow] Force re-squash requested, removing existing file...")
        tunnel.run(f"rm -f {sqsh_path}", hide=True)
    else:
        console.print("[yellow]![/yellow] Squashed image not found, creating...")
    console.print(f"  [dim]Image:[/dim] {container_image}")
    console.print(f"  [dim]Output:[/dim] {sqsh_path}")
    console.print()

    # Ensure directory exists
    tunnel.run(f"mkdir -p {remote_job_dir}", hide=True)

    # Build salloc command to run enroot import on a compute node
    # (login nodes don't have enough memory for enroot import)
    account = env_config.get("account")
    partition = env_config.get("run_partition") or env_config.get("partition")
    time_limit = env_config.get("time", "04:00:00")
    gpus_per_node = env_config.get("gpus_per_node")

    salloc_args = []
    if account:
        salloc_args.append(f"--account={account}")
    if partition:
        salloc_args.append(f"--partition={partition}")
    salloc_args.append("--nodes=1")
    salloc_args.append("--ntasks-per-node=1")
    if gpus_per_node:
        salloc_args.append(f"--gpus-per-node={gpus_per_node}")
    salloc_args.append(f"--time={time_limit}")

    # Set up writable enroot paths (default /raid/enroot may not be user-writable)
    enroot_runtime = f"{remote_job_dir}/.enroot"
    enroot_env = (
        f"export ENROOT_RUNTIME_PATH={enroot_runtime} "
        f"ENROOT_CACHE_PATH={enroot_runtime}/cache "
        f"ENROOT_DATA_PATH={enroot_runtime}/data && "
        f"mkdir -p {enroot_runtime}/cache {enroot_runtime}/data && "
    )
    enroot_cmd = f"{enroot_env}enroot import --output {sqsh_path} docker://{container_image}"
    cmd = f"salloc {' '.join(salloc_args)} srun --export=ALL bash -c '{enroot_cmd}'"

    # Run enroot import via salloc (this can take a while)
    console.print(
        "[bold blue]Allocating compute node and importing container "
        "(this may take several minutes)...[/bold blue]"
    )
    console.print(f"[dim]$ {cmd}[/dim]")
    console.print()
    result = tunnel.run(cmd, hide=False, warn=True)

    if not result.ok:
        raise RuntimeError(
            f"Failed to squash container image.\n"
            f"Command: {cmd}\n"
            f"Error: {result.stderr or 'Unknown error'}"
        )

    console.print(f"[green]✓[/green] Created squashed image: [cyan]{sqsh_path}[/cyan]")
    return sqsh_path


def _execute_stage_only(
    script_path: str,
    train_path: Path,
    job_dir: Path,
    job_config: Any,
    packager: str = "pattern",
    *,
    torchrun: bool = True,
) -> None:
    """Stage script + config to remote cluster without execution.

    Stages files to a fixed location in remote_job_dir and prints
    commands for interactive debugging.

    Args:
        script_path: Path to training script
        train_path: Path to train.yaml config
        job_dir: Local job directory
        job_config: Full job configuration (contains run.env)
        packager: Packager type ("pattern", "code", "self_contained")
        torchrun: Whether to use torchrun launcher
    """

    try:
        import nemo_run as run
    except ImportError:
        typer.echo("Error: nemo-run is required for --stage", err=True)
        typer.echo("Install with: pip install nemo-run", err=True)
        raise typer.Exit(1)

    from omegaconf import OmegaConf

    # Extract env config
    env_config = OmegaConf.to_container(job_config.run.env, resolve=True)

    # Only support SSH tunnel for now
    tunnel_type = env_config.get("tunnel")
    if tunnel_type != "ssh":
        console.print("[red]Error:[/red] --stage requires SSH tunnel configuration (tunnel: ssh)")
        raise typer.Exit(1)

    remote_job_dir = env_config.get("remote_job_dir")
    if not remote_job_dir:
        console.print("[red]Error:[/red] remote_job_dir not configured in env profile")
        raise typer.Exit(1)

    # Fixed staging location for interactive debugging
    stage_dir = f"{remote_job_dir}/interactive"

    # Build tunnel
    tunnel = run.SSHTunnel(
        host=env_config.get("host", "localhost"),
        user=env_config.get("user"),
        job_dir=remote_job_dir,
    )

    # Connect
    with console.status("[bold blue]Connecting to remote cluster..."):
        tunnel.connect()

    # Create remote directories
    console.print(f"\n[cyan]Creating remote directory:[/cyan] {stage_dir}")
    tunnel.run(f"mkdir -p {stage_dir}", hide=True)

    # Stage files locally using the packager
    # For self_contained packager, this will inline nemotron imports
    code_dir = job_dir / "code"
    code_dir.mkdir(exist_ok=True)

    if packager == "self_contained":
        from nemotron.kit.packaging.self_contained_packager import inline_imports

        # Inline imports to create main.py
        script_file = Path(script_path)
        if not script_file.is_absolute():
            script_file = Path.cwd() / script_path

        inlined = inline_imports(
            script_file,
            repo_root=Path.cwd(),
            package_prefix="nemotron",
        )
        (code_dir / "main.py").write_text(inlined, encoding="utf-8")
        shutil.copy2(train_path, code_dir / "config.yaml")
    else:
        # For pattern/code packagers, just copy files
        shutil.copy2(script_path, code_dir / "main.py")
        shutil.copy2(train_path, code_dir / "config.yaml")

    local_script = code_dir / "main.py"
    local_config = code_dir / "config.yaml"

    # Build environment variables (same as _execute_nemo_run)
    env_vars = _build_env_vars(job_config, env_config)

    # Override NEMO_RUN_DIR for --stage mode: /nemo_run maps to stage_dir, not remote_job_dir
    env_vars["NEMO_RUN_DIR"] = stage_dir

    # Get GPU count for torchrun
    gpus = env_config.get("gpus_per_node") or env_config.get("ntasks_per_node", 8)

    # Create run.sh script that sets env vars and runs training
    run_script_lines = [
        "#!/bin/bash",
        "# Auto-generated training script with environment setup",
        "",
    ]
    run_script_lines.append("# Environment variables for W&B and HuggingFace")
    for key, value in env_vars.items():
        # Escape single quotes in values
        escaped_value = value.replace("'", "'\"'\"'")
        run_script_lines.append(f"export {key}='{escaped_value}'")
    run_script_lines.append("")

    # Add startup commands if configured
    startup_commands = _get_startup_commands(env_config)
    if startup_commands:
        run_script_lines.append("# Startup commands (run before training)")
        run_script_lines.append("set -e  # Exit on error")
        for cmd in startup_commands:
            run_script_lines.append(cmd)
        run_script_lines.append("")

    run_script_lines.append("# Run training")
    if torchrun:
        run_script_lines.append(
            f'torchrun --nproc_per_node={gpus} main.py --config config.yaml "$@"'
        )
    else:
        run_script_lines.append('python main.py --config config.yaml "$@"')
    run_script = "\n".join(run_script_lines) + "\n"
    run_script_path = code_dir / "run.sh"
    run_script_path.write_text(run_script)

    # Upload files via scp/sftp
    console.print(f"[cyan]Uploading files to:[/cyan] {stage_dir}")
    with console.status("[bold blue]Uploading script, config, and run.sh..."):
        tunnel.put(str(local_script), f"{stage_dir}/main.py")
        tunnel.put(str(local_config), f"{stage_dir}/config.yaml")
        tunnel.put(str(run_script_path), f"{stage_dir}/run.sh")
        # Make run.sh executable
        tunnel.run(f"chmod +x {stage_dir}/run.sh", hide=True)

    console.print("[green]✓[/green] Files staged successfully\n")

    # Build and display commands
    _print_stage_commands(env_config, stage_dir, env_vars=env_vars, torchrun=torchrun)


def _print_stage_commands(
    env_config: dict,
    stage_dir: str,
    *,
    env_vars: dict[str, str] | None = None,
    torchrun: bool = True,
) -> None:
    """Print commands for interactive debugging after staging.

    Args:
        env_config: Environment configuration dict
        stage_dir: Remote directory where files were staged
        env_vars: Environment variables for W&B/HF (displayed to user)
        torchrun: Whether to use torchrun launcher
    """
    from rich.panel import Panel

    host = env_config.get("host", "localhost")
    user = env_config.get("user", "")
    partition = env_config.get("run_partition") or env_config.get("partition", "interactive")
    nodes = env_config.get("nodes", 1)
    gpus = env_config.get("gpus_per_node") or env_config.get("ntasks_per_node", 8)
    time_limit = env_config.get("time", "04:00:00")
    container = env_config.get("container_image") or env_config.get("container")
    account = env_config.get("account")
    remote_job_dir = env_config.get("remote_job_dir")

    # Get squashed container path
    sqsh_path = None
    if container and remote_job_dir:
        sqsh_path = _get_squash_path(container, remote_job_dir)

    # Mount to /workspace for simpler commands inside container
    container_mount_path = "/workspace"

    # Build srun command (multi-line for display)
    srun_parts = ["srun"]
    if account:
        srun_parts.append(f"--account={account}")
    srun_parts.extend(
        [
            f"--partition={partition}",
            f"--nodes={nodes}",
            f"--ntasks-per-node={gpus}",
            f"--gpus-per-node={gpus}",
            f"--time={time_limit}",
        ]
    )
    if sqsh_path:
        srun_parts.append(f"--container-image={sqsh_path}")
        # Mount stage_dir to both /workspace and /nemo_run to match real nemo-run behavior
        srun_parts.append(
            f"--container-mounts={stage_dir}:{container_mount_path},{stage_dir}:/nemo_run,/lustre:/lustre"
        )
        srun_parts.append(f"--container-workdir={container_mount_path}")

    # Add NEMO_RUN_DIR to srun command - critical for resolving /nemo_run mount to actual Lustre path
    if env_vars and "NEMO_RUN_DIR" in env_vars:
        srun_parts.append(f"--export=ALL,NEMO_RUN_DIR={env_vars['NEMO_RUN_DIR']}")

    srun_parts.append("--pty bash")
    srun_cmd_display = " \\\n    ".join(srun_parts)
    srun_cmd_oneline = " ".join(srun_parts)

    # Build environment info for display
    env_info = ""
    if env_vars:
        env_keys = []
        for key in env_vars:
            if key in ("WANDB_API_KEY", "HF_TOKEN"):
                env_keys.append(f"{key}=***")
            else:
                env_keys.append(f"{key}={env_vars[key]}")
        env_info = f"[bold cyan]Environment:[/bold cyan] {', '.join(env_keys)}\n"

    # Display
    console.print(
        Panel.fit(
            f"[bold cyan]Files staged to:[/bold cyan] {stage_dir}\n"
            f"[bold cyan]Mounted at:[/bold cyan] {container_mount_path} and /nemo_run\n"
            f"{env_info}\n"
            f"[bold cyan]1. SSH to cluster:[/bold cyan]\n"
            f"   [green]ssh {user}@{host}[/green]\n\n"
            f"[bold cyan]2. Start interactive job:[/bold cyan]\n"
            f"   [green]{srun_cmd_display}[/green]\n\n"
            f"[bold cyan]3. Run training:[/bold cyan]\n"
            f"   [green]./run.sh[/green]\n\n"
            f"[dim]Tip: Keep the srun session alive while iterating. "
            f"Re-run with --stage to update files, then run ./run.sh again.[/dim]",
            title="[bold]Interactive Debugging[/bold]",
            border_style="green",
        )
    )

    # Print single-line srun command for easy copying
    # Use print() instead of console.print() to avoid Rich text wrapping
    console.print("\n[bold cyan]Copy-paste srun command:[/bold cyan]")
    print(srun_cmd_oneline)
    print()
