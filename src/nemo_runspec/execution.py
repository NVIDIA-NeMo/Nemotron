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

"""Execution utilities for recipe commands.

Provides the shared building blocks for running recipes: startup commands,
environment variable setup, executor creation, git repo cloning, and local
subprocess execution.

Design principle: extract only utilities, keep policy visible.
Commands should show exactly how they build executors and run experiments.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

console = Console()


# =============================================================================
# Startup Commands
# =============================================================================


def get_startup_commands(env_config: dict | None) -> list[str]:
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
        typer.echo(
            f"Error: startup_commands must be a list, got {type(commands).__name__}",
            err=True,
        )
        raise typer.Exit(1)
    for cmd in commands:
        if not isinstance(cmd, str):
            typer.echo(
                f"Error: startup_commands must be a list of strings, got {type(cmd).__name__}",
                err=True,
            )
            raise typer.Exit(1)
    return commands


def prepend_startup_to_cmd(startup_commands: list[str], cmd: str) -> str:
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


def run_startup_commands_local(startup_commands: list[str]) -> None:
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


# =============================================================================
# Environment Variables
# =============================================================================


def build_env_vars(job_config: Any, env_config: dict | None = None) -> dict[str, str]:
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
    from omegaconf import OmegaConf

    env_vars: dict[str, str] = {}

    # Set NEMO_RUN_DIR to remote_job_dir for shared filesystem operations
    # (e.g., artifact marker files for multi-node sync).
    # NOTE: This is the root job dir, NOT the exact /nemo_run mount source.
    # For resolving /nemo_run paths to Lustre, _resolve_to_lustre_path()
    # prefers /proc/mounts which gives the exact bind mount source.
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

    # Auto-detect Weights & Biases API key and validate it before forwarding.
    # Validating early avoids wasting time on Slurm allocation + container import
    # only to fail with a 401 inside the container.
    api_key = None
    try:
        import wandb

        api_key = wandb.api.api_key
        if api_key:
            # Quick auth check — this is what the container will do later
            test_api = wandb.Api(timeout=10)
            _ = test_api.viewer  # triggers the actual auth request
            env_vars["WANDB_API_KEY"] = api_key
    except Exception as e:
        import sys

        err_str = str(e)
        err_type = type(e).__name__
        if "401" in err_str or "Unauthorized" in err_str or "AuthenticationError" in err_type:
            raise RuntimeError(
                "WANDB_API_KEY is set but authentication failed (401 Unauthorized). "
                "Artifact resolution will fail inside the container. "
                "Fix: run 'wandb login --relogin' to refresh your credentials."
            ) from e
        # For non-auth errors (network timeout, etc.), still pass the key through
        if api_key:
            env_vars["WANDB_API_KEY"] = api_key

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

    # Merge explicit env_vars from run.env config (YAML or env.toml).
    # These are applied last so they can override auto-detected values above.
    if env_config:
        extra = env_config.get("env_vars") if hasattr(env_config, "get") else getattr(env_config, "env_vars", None)
        if extra and hasattr(extra, "items"):
            env_vars.update({str(k): str(v) for k, v in extra.items()})

    return env_vars


# =============================================================================
# Git Repo Cloning
# =============================================================================


def clone_git_repos_via_tunnel(tunnel: Any, remote_job_dir: str) -> list[str]:
    """Clone git repos on the remote side via SSH tunnel.

    This runs during executor setup, before job submission. The cloned repos
    are then mounted into the container.

    Args:
        tunnel: Connected SSH tunnel
        remote_job_dir: Remote directory for git cache

    Returns:
        List of container mount strings (e.g., "/path/to/repo:/opt/Target")
    """
    from nemo_runspec.config.resolvers import get_git_mounts

    git_mounts = get_git_mounts()
    if not git_mounts:
        return []

    cache_dir = f"{remote_job_dir}/git-cache"
    mounts = []

    # Ensure cache directory exists
    tunnel.run(f"mkdir -p {cache_dir}", hide=True)

    for repo_name, repo_info in git_mounts.items():
        url = repo_info["url"]
        ref = repo_info["ref"]
        target = repo_info.get("target", "")

        repo_cache = f"{cache_dir}/{repo_name}"

        # Clone or update the repo
        typer.echo(f"[auto_mount] Syncing {repo_name}@{ref}...")

        # Check if repo already exists
        result = tunnel.run(f"test -d {repo_cache}/.git && echo exists", hide=True, warn=True)

        # Check if ref is a full commit SHA (40 hex chars) - these are immutable
        is_commit_sha = len(ref) == 40 and all(c in "0123456789abcdef" for c in ref.lower())

        if result.ok and "exists" in result.stdout:
            # Repo exists in cache
            if is_commit_sha:
                # For exact commits, check if we already have it
                have_commit = tunnel.run(
                    f"git -C {repo_cache} cat-file -t {ref} 2>/dev/null", hide=True, warn=True
                )
                if have_commit.ok:
                    typer.echo(f"[auto_mount] Using cached {repo_name}@{ref[:8]}...")
                else:
                    # Need to fetch to get this commit
                    typer.echo(f"[auto_mount] Fetching {repo_name} to get commit {ref[:8]}...")
                    tunnel.run(f"git -C {repo_cache} fetch origin", hide=True, warn=True)
            else:
                # For branches/tags, always fetch to get latest
                typer.echo(f"[auto_mount] Updating {repo_name}@{ref}...")
                fetch_result = tunnel.run(f"git -C {repo_cache} fetch origin", hide=True, warn=True)
                if not fetch_result.ok:
                    typer.echo(f"[auto_mount] Warning: fetch failed, will re-clone")
                    tunnel.run(f"rm -rf {repo_cache}", hide=True)
                    # Fall through to clone

        # Check again if we need to clone (either didn't exist or was removed)
        result = tunnel.run(f"test -d {repo_cache}/.git && echo exists", hide=True, warn=True)
        if not (result.ok and "exists" in result.stdout):
            # Fresh clone
            typer.echo(f"[auto_mount] Cloning {repo_name}...")
            clone_result = tunnel.run(f"git clone {url} {repo_cache}", hide=False, warn=True)
            if not clone_result.ok:
                typer.echo(f"Error: git clone failed for {repo_name}", err=True)
                raise typer.Exit(1)

        # Checkout the specific ref
        # For branches, use origin/{ref} to get latest remote version
        # For tags/commits, fall back to just {ref}
        checkout_result = tunnel.run(
            f"git -C {repo_cache} checkout origin/{ref} 2>/dev/null || git -C {repo_cache} checkout {ref}",
            hide=True,
            warn=True,
        )
        if not checkout_result.ok:
            typer.echo(f"Error: git checkout {ref} failed for {repo_name}", err=True)
            raise typer.Exit(1)

        # Reset to ensure clean state (discard any local changes)
        tunnel.run(f"git -C {repo_cache} reset --hard HEAD", hide=True, warn=True)

        typer.echo(f"[auto_mount] {repo_name} ready at {repo_cache}")

        # Add container mount if target specified
        if target:
            mounts.append(f"{repo_cache}:{target}")

    return mounts


# =============================================================================
# Executor Creation
# =============================================================================


def _get_env(env: Any, key: str, default: Any = None) -> Any:
    """Get value from env config (OmegaConf or dict).

    Args:
        env: OmegaConf DictConfig or dict
        key: Key to look up
        default: Default value if key not found

    Returns:
        Value or default
    """
    if env is None:
        return default
    # Works for both OmegaConf and dict
    return env.get(key, default) if hasattr(env, "get") else getattr(env, key, default)


def create_executor(
    env: Any,
    env_vars: dict[str, str],
    packager: Any,
    *,
    attached: bool = False,
    force_squash: bool = False,
    default_image: str | None = None,
    script_resources: Any | None = None,
) -> Any:
    """Create a nemo-run executor based on env config.

    This handles the common pattern of building LocalExecutor or SlurmExecutor.
    For Ray executors, see the RL command implementation.

    Args:
        env: Environment configuration (OmegaConf DictConfig from parse_env, or dict)
        env_vars: Environment variables to pass to executor
        packager: Packager object (e.g., SelfContainedPackager)
        attached: Whether running in attached (--run) vs batch (--batch) mode
        force_squash: Force re-squash of container image
        default_image: Fallback container image (e.g., from SPEC.image) if env
            config doesn't specify one
        script_resources: RunspecResources from the script's [tool.runspec.resources].
            Used as defaults when env config doesn't specify nodes/gpus.

    Returns:
        Configured executor (LocalExecutor or SlurmExecutor)
    """
    import nemo_run as run

    executor_type = _get_env(env, "executor", "local")

    if executor_type == "local":
        return run.LocalExecutor(
            ntasks_per_node=_get_env(env, "nproc_per_node", 1),
            launcher="torchrun",
            env_vars=env_vars,
        )

    if executor_type == "docker":
        container_image = _get_env(env, "container_image") or _get_env(env, "container") or default_image
        if not container_image:
            raise ValueError("container_image required for docker executor")

        # Resolve relative paths and expand env vars in mounts
        mounts = _get_env(env, "mounts") or []
        resolved_mounts = []
        for mount in mounts:
            if ":" in mount:
                host_path, container_path = mount.split(":", 1)
                expanded = os.path.expandvars(host_path)
                if "$" in expanded:
                    typer.echo(
                        f"[warning] Skipping mount {mount!r}: environment variable not set",
                        err=True,
                    )
                    continue
                host_path = str(Path(expanded).expanduser())
                if not host_path.startswith("/"):
                    host_path = str(Path.cwd() / host_path)
                resolved_mounts.append(f"{host_path}:{container_path}")
            else:
                resolved_mounts.append(mount)

        return run.DockerExecutor(
            container_image=container_image,
            num_gpus=_get_env(env, "gpus_per_node") or _get_env(env, "nproc_per_node"),
            runtime=_get_env(env, "runtime", "nvidia"),
            ipc_mode=_get_env(env, "ipc_mode"),
            shm_size=_get_env(env, "shm_size"),
            volumes=resolved_mounts,
            env_vars=env_vars,
            packager=packager,
        )

    if executor_type != "slurm":
        raise ValueError(f"Unknown executor type: {executor_type}")

    # Slurm executor setup
    remote_job_dir = _get_env(env, "remote_job_dir")

    # Build SSH tunnel if configured
    tunnel = None
    if _get_env(env, "tunnel") == "ssh":
        tunnel = run.SSHTunnel(
            host=_get_env(env, "host", "localhost"),
            user=_get_env(env, "user"),
            job_dir=remote_job_dir,
        )

    # Container image handling (env.toml > config YAML > SPEC.image fallback)
    container_image = _get_env(env, "container_image") or _get_env(env, "container") or default_image

    # Ensure container is squashed on cluster
    if container_image and tunnel and remote_job_dir:
        tunnel.connect()
        from nemo_runspec.squash import ensure_squashed_image

        # Convert env to dict for ensure_squashed_image (it uses .get internally)
        env_dict = dict(env) if env else {}
        container_image = ensure_squashed_image(
            tunnel, container_image, remote_job_dir, env_dict, force=force_squash
        )

    # Clone git repos via tunnel
    git_mounts = []
    if tunnel and remote_job_dir:
        tunnel.connect()
        git_mounts = clone_git_repos_via_tunnel(tunnel, remote_job_dir)

    # Select partition based on mode
    if attached:
        partition = _get_env(env, "run_partition") or _get_env(env, "partition")
    else:
        partition = _get_env(env, "batch_partition") or _get_env(env, "partition")

    # Build container mounts
    raw_mounts = list(_get_env(env, "mounts") or [])
    mounts = [m for m in raw_mounts if not m.startswith("__auto_mount__:")]
    mounts.extend(git_mounts)
    mounts.append("/lustre:/lustre")

    if remote_job_dir:
        ray_temp_path = f"{remote_job_dir}/ray_temp"
        mounts.append(f"{ray_temp_path}:/ray-cluster")
        if tunnel:
            tunnel.run(f"mkdir -p {ray_temp_path}", hide=True)

    # Build executor kwargs
    # Use script's runspec resources as defaults when env doesn't specify them
    default_nodes = script_resources.nodes if script_resources else 1
    default_gpus = script_resources.gpus_per_node if script_resources else None
    executor_kwargs = {
        "account": _get_env(env, "account"),
        "partition": partition,
        "nodes": _get_env(env, "nodes", default_nodes),
        "ntasks_per_node": _get_env(env, "ntasks_per_node", default_gpus or 1),
        "gpus_per_node": _get_env(env, "gpus_per_node", default_gpus),
        "cpus_per_task": _get_env(env, "cpus_per_task"),
        "time": _get_env(env, "time", "04:00:00"),
        "container_image": container_image,
        "container_mounts": mounts,
        "tunnel": tunnel,
        "packager": packager,
        "mem": _get_env(env, "mem"),
        "env_vars": env_vars,
        "launcher": "torchrun",
    }

    if _get_env(env, "exclusive"):
        executor_kwargs["exclusive"] = True

    return run.SlurmExecutor(**executor_kwargs)


# =============================================================================
# Local Execution
# =============================================================================


def execute_local(
    script_path: str,
    train_path: Path,
    passthrough: list[str],
    *,
    torchrun: bool = True,
    env_vars: dict[str, str] | None = None,
    startup_commands: list[str] | None = None,
) -> None:
    """Execute script locally via subprocess.

    Args:
        script_path: Path to the training script
        train_path: Path to the saved train.yaml
        passthrough: Additional args to pass to script
        torchrun: Whether to use torchrun launcher
        env_vars: Environment variables to set
        startup_commands: Shell commands to run before training
    """
    import sys

    # Set env vars so subprocess inherits them (wandb, HF tokens, etc.)
    if env_vars:
        os.environ.update(env_vars)

    # Run startup commands before training
    if startup_commands:
        run_startup_commands_local(startup_commands)

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


def execute_uv_local(
    *,
    script_path: str,
    stage_dir: Path,
    repo_root: Path,
    train_path: Path,
    passthrough: list[str],
    extra_with: list[str] | None = None,
    extras: list[str] | None = None,
    pre_script_args: list[str] | None = None,
) -> None:
    """Execute a stage script locally via UV, handling container torch correctly.

    When torch is already importable in the current Python (e.g., inside an
    NVIDIA container), creates a venv with ``--system-site-packages`` and
    excludes torch from UV resolution. This avoids the CUDA version mismatch
    where UV's torch-backend detects the kernel driver's CUDA version but the
    container's ``libcuda.so`` is a different version.

    When torch is NOT importable (bare machine), runs ``uv run --project
    <stage>``. If the stage's ``pyproject.toml`` declares mutually-exclusive
    cuXXX optional-dependencies (the standard UV multi-CUDA pattern from
    https://docs.astral.sh/uv/guides/integration/pytorch/), this function
    auto-detects the host's NVIDIA driver and passes ``--extra cuXXX`` so UV
    selects a torch wheel that matches the driver. If the stage doesn't
    declare those extras, no extra is injected and resolution proceeds via
    the stage's normal dependencies (i.e., transitive torch from the lock).

    Args:
        script_path: Relative or absolute path to the stage script.
        stage_dir: Absolute path to the stage directory (contains pyproject.toml).
        repo_root: Absolute path to the repo root (installed via ``--with``).
        train_path: Path to the resolved training config YAML.
        passthrough: Extra CLI arguments to forward to the script.
        extra_with: Additional ``--with`` packages for uv run (e.g., ["tensorrt"]).
        extras: ``[project.optional-dependencies]`` groups to activate on the
            stage project (e.g., ["tensorrt"] → ``--extra tensorrt``).
        pre_script_args: Arguments inserted before the script path
            (e.g., ["-m", "torch.distributed.run", "--nproc_per_node=gpu"]).

    Raises:
        typer.Exit: with the script's exit code.
    """
    import shutil

    uv_cmd = shutil.which("uv")
    if not uv_cmd:
        typer.echo("Error: 'uv' command not found. Please install uv.", err=True)
        raise typer.Exit(1)

    script_abs = (
        (stage_dir / Path(script_path).name)
        if not Path(script_path).is_absolute()
        else Path(script_path)
    )
    extras = list(extras or [])

    if _torch_is_importable():
        rc = _execute_with_system_torch(
            uv_cmd=uv_cmd,
            stage_dir=stage_dir,
            repo_root=repo_root,
            script_abs=script_abs,
            train_path=train_path,
            passthrough=passthrough,
            extras=extras,
            pre_script_args=pre_script_args or [],
        )
    else:
        cuda_extra = _pick_cuda_extra(_stage_optional_extras(stage_dir))
        if cuda_extra and cuda_extra not in extras:
            extras.append(cuda_extra)
            typer.echo(f"Auto-detected CUDA torch extra: --extra {cuda_extra}")
        rc = _execute_with_uv_torch(
            uv_cmd=uv_cmd,
            stage_dir=stage_dir,
            repo_root=repo_root,
            script_abs=script_abs,
            train_path=train_path,
            passthrough=passthrough,
            extra_with=extra_with or [],
            extras=extras,
            pre_script_args=pre_script_args or [],
        )

    raise typer.Exit(rc)


def _torch_is_importable() -> bool:
    """Check if torch is importable in the current Python."""
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import torch"],
        capture_output=True,
    )
    return result.returncode == 0



def _execute_with_system_torch(
    *,
    uv_cmd: str,
    stage_dir: Path,
    repo_root: Path,
    script_abs: Path,
    train_path: Path,
    passthrough: list[str],
    extras: list[str],
    pre_script_args: list[str],
) -> int:
    """Execute using system torch via --system-site-packages venv."""
    import tempfile
    import tomllib

    from nemo_runspec._pyproject import _write_temp_pyproject

    typer.echo(
        "Detected system torch — using system-site-packages to avoid CUDA mismatch"
    )

    pyproject_path = stage_dir / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)

    nemotron_cfg = pyproject_data.get("tool", {}).get("nemotron", {})
    exclude_deps = nemotron_cfg.get(
        "container-exclude-dependencies",
        [
            "torch", "torchvision", "flash-attn", "triton",
            "pyarrow", "scipy", "opencv-python-headless",
        ],
    )

    temp_dir = _write_temp_pyproject(pyproject_data, stage_dir, exclude_deps)

    venv_path = Path(tempfile.mkdtemp()) / "venv"
    result = subprocess.run(
        [uv_cmd, "venv", "--system-site-packages", "--seed", str(venv_path)]
    )
    if result.returncode != 0:
        typer.echo("Error: Failed to create venv", err=True)
        return 1

    venv_python = str(venv_path / "bin" / "python3")
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env["VIRTUAL_ENV"] = str(venv_path)
    env["UV_PROJECT_ENVIRONMENT"] = str(venv_path)
    env["PATH"] = f"{venv_path / 'bin'}:{env.get('PATH', '')}"

    sync_cmd = [uv_cmd, "sync", "--active", "--project", str(temp_dir)]
    for extra in extras:
        sync_cmd += ["--extra", extra]
    typer.echo(f"Syncing dependencies (torch excluded): {' '.join(sync_cmd)}")
    result = subprocess.run(sync_cmd, env=env, cwd=str(temp_dir))
    if result.returncode != 0:
        typer.echo("Error: Package sync failed", err=True)
        return 1

    result = subprocess.run(
        [uv_cmd, "pip", "install", "--no-deps", str(repo_root)], env=env,
    )
    if result.returncode != 0:
        typer.echo("Error: Failed to install repo package", err=True)
        return 1

    cmd = [
        venv_python, *pre_script_args, str(script_abs),
        "--config", str(train_path), *passthrough,
    ]
    typer.echo(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode


def _execute_with_uv_torch(
    *,
    uv_cmd: str,
    stage_dir: Path,
    repo_root: Path,
    script_abs: Path,
    train_path: Path,
    passthrough: list[str],
    extra_with: list[str],
    extras: list[str],
    pre_script_args: list[str],
) -> int:
    """Execute via ``uv run --project`` against the stage's pyproject + lock.

    Torch resolution is driven by the stage's declared dependencies and
    (optionally) a ``--extra cuXXX`` injected by the caller. We do NOT pass
    ``--with torch`` or set ``UV_TORCH_BACKEND``: ``--with`` triggers a
    separate ephemeral resolution that bypasses the project's lock and is
    not honored by ``UV_TORCH_BACKEND`` (only ``uv pip``/``uv add``/``uv
    sync`` honor it). See the multi-CUDA pattern in
    https://docs.astral.sh/uv/guides/integration/pytorch/.
    """
    cmd = [uv_cmd, "run", "--with", str(repo_root)]
    for pkg in extra_with:
        cmd += ["--with", pkg]
    cmd += ["--project", str(stage_dir)]
    for extra in extras:
        cmd += ["--extra", extra]

    if pre_script_args:
        cmd += [*pre_script_args]
    else:
        cmd += ["python"]

    cmd += [str(script_abs), "--config", str(train_path), *passthrough]

    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)

    typer.echo(f"Executing with uv isolated environment: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    return result.returncode


# =============================================================================
# CUDA Driver Detection
# =============================================================================
#
# Ported from astral-sh/uv (MIT/Apache-2.0):
#   - crates/uv-torch/src/accelerator.rs (detection order)
#   - crates/uv-torch/src/backend.rs     (driver→cuXXX table)
#
# Used to pick the highest cuXXX optional-dependency extra that the host
# driver supports AND the stage's pyproject.toml declares. Matches the
# documented pattern at
# https://docs.astral.sh/uv/guides/integration/pytorch/#configuring-accelerators-with-optional-dependencies.

# (extra-name, minimum-driver-version-for-this-CUDA-toolkit). Walk descending;
# first row whose minimum is <= host driver wins. Source: NVIDIA CUDA Toolkit
# Release Notes Table 2/Table 1.
_LINUX_CUDA_DRIVERS: list[tuple[str, tuple[int, ...]]] = [
    ("cu130", (580,)),
    ("cu129", (525, 60, 13)),
    ("cu128", (525, 60, 13)),
    ("cu126", (525, 60, 13)),
    ("cu125", (525, 60, 13)),
    ("cu124", (525, 60, 13)),
    ("cu123", (525, 60, 13)),
    ("cu122", (525, 60, 13)),
    ("cu121", (525, 60, 13)),
    ("cu120", (525, 60, 13)),
    ("cu118", (450, 80, 2)),
    ("cu117", (450, 80, 2)),
    ("cu116", (450, 80, 2)),
    ("cu115", (450, 80, 2)),
    ("cu114", (450, 80, 2)),
    ("cu113", (450, 80, 2)),
    ("cu112", (450, 80, 2)),
    ("cu111", (450, 80, 2)),
    ("cu110", (450, 36, 6)),
    ("cu102", (440, 33)),
    ("cu101", (418, 39)),
    ("cu100", (410, 48)),
    ("cu92", (396, 26)),
    ("cu91", (390, 46)),
    ("cu90", (384, 81)),
    ("cu80", (375, 26)),
]


def _parse_driver_version(text: str) -> tuple[int, ...] | None:
    """Parse a driver version string like '550.144.03' into a tuple of ints."""
    import re

    text = text.strip()
    if not text:
        return None
    parts = [p for p in re.split(r"[.\-]", text) if p]
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return None


def _read_nvidia_driver_version() -> tuple[int, ...] | None:
    """Detect NVIDIA driver version, mirroring uv's source order.

    Order: ``UV_CUDA_DRIVER_VERSION`` env (escape hatch) →
    ``/sys/module/nvidia/version`` → ``/proc/driver/nvidia/version`` →
    ``nvidia-smi --query-gpu=driver_version --format=csv,noheader``.
    Returns ``None`` if no NVIDIA driver is detectable.
    """
    override = os.environ.get("UV_CUDA_DRIVER_VERSION")
    if override:
        return _parse_driver_version(override)

    try:
        return _parse_driver_version(Path("/sys/module/nvidia/version").read_text())
    except OSError:
        pass

    try:
        # Format: "NVRM version: NVIDIA UNIX … x86_64  550.144.03  Release Build  …"
        # uv splits on two-space and takes index 1.
        content = Path("/proc/driver/nvidia/version").read_text()
        parts = content.split("  ")
        if len(parts) >= 2:
            v = _parse_driver_version(parts[1])
            if v:
                return v
    except OSError:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            first_line = result.stdout.strip().split("\n")[0]
            return _parse_driver_version(first_line)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def _pick_cuda_extra(stage_extras: set[str]) -> str | None:
    """Highest cuXXX the host driver supports AND the stage declares, else None."""
    if not stage_extras:
        return None
    driver = _read_nvidia_driver_version()
    if driver is None:
        return None
    for variant, min_driver in _LINUX_CUDA_DRIVERS:
        if driver >= min_driver and variant in stage_extras:
            return variant
    return None


def _stage_optional_extras(stage_dir: Path) -> set[str]:
    """Read ``[project.optional-dependencies]`` keys from the stage pyproject.toml."""
    import tomllib

    try:
        with open(stage_dir / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
    except OSError:
        return set()
    return set(data.get("project", {}).get("optional-dependencies", {}).keys())
