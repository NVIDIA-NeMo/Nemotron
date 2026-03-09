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
    executor_kwargs = {
        "account": _get_env(env, "account"),
        "partition": partition,
        "nodes": _get_env(env, "nodes", 1),
        "ntasks_per_node": _get_env(env, "ntasks_per_node", 1),
        "gpus_per_node": _get_env(env, "gpus_per_node"),
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
