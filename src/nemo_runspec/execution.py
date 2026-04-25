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

    Dispatches on the ``executor`` field in the env.toml profile:
      - ``local`` (default): torchrun on local GPUs
      - ``docker``: DockerExecutor for local container execution
      - ``slurm``: SlurmExecutor for HPC clusters
      - ``lepton``: LeptonExecutor for DGX Cloud via Lepton API
      - ``runai``: KubeflowExecutor configured for Run:AI Kubernetes clusters

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
        Configured executor
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

    if executor_type == "lepton":
        return _create_lepton_executor(
            env,
            env_vars=env_vars,
            packager=packager,
            default_image=default_image,
            script_resources=script_resources,
        )

    if executor_type == "runai":
        return _create_runai_executor(
            env,
            env_vars=env_vars,
            packager=packager,
            default_image=default_image,
            script_resources=script_resources,
        )

    if executor_type != "slurm":
        raise ValueError(
            f"Unknown executor type: {executor_type!r}. "
            "Supported: local, docker, slurm, lepton, runai"
        )

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


def _create_lepton_executor(
    env: Any,
    *,
    env_vars: dict[str, str],
    packager: Any,
    default_image: str | None = None,
    script_resources: Any | None = None,
) -> Any:
    """Create a LeptonExecutor for DGX Cloud Lepton.

    Required env.toml fields:
        container_image (or container): Container image for the job
        node_group: Lepton dedicated node group name

    Optional env.toml fields:
        resource_shape: GPU shape (default: "gpu.8xh100-80gb")
        nemo_run_dir: Remote code directory (default: "/nemo_run/code")
        nodes: Number of nodes (default: 1)
        gpus_per_node: GPUs per node (default: 8)
        mounts: List of mount dicts with 'path' and 'mount_path' keys
        node_reservation: Reservation ID for dedicated capacity
        pre_launch_commands: Shell commands to run before launch
        image_pull_secrets: Container registry auth secrets
        ray_version: Ray version (for LeptonRayCluster)
        head_resource_shape: Head node shape (for LeptonRayCluster)

    Args:
        env: Environment configuration (OmegaConf DictConfig or dict)
        env_vars: Environment variables to pass to executor
        packager: Packager object for code shipping
        default_image: Fallback container image if env doesn't specify one
        script_resources: RunspecResources defaults for nodes/gpus

    Returns:
        Configured LeptonExecutor
    """
    import nemo_run as run

    # Container image (required)
    container_image = (
        _get_env(env, "container_image")
        or _get_env(env, "container")
        or default_image
    )
    if not container_image:
        raise ValueError(
            "container_image is required for lepton executor. "
            "Set it in your env.toml profile or in the recipe's [tool.runspec] image."
        )

    # Node group (required)
    node_group = _get_env(env, "node_group")
    if not node_group:
        raise ValueError(
            "node_group is required for lepton executor. "
            "Set it in your env.toml profile, e.g.: node_group = \"my-dgx-group\""
        )

    # Resource defaults from script metadata
    default_nodes = script_resources.nodes if script_resources else 1
    default_gpus = script_resources.gpus_per_node if script_resources else 8

    executor_kwargs: dict[str, Any] = {
        "container_image": container_image,
        "node_group": node_group,
        "resource_shape": _get_env(env, "resource_shape", "gpu.8xh100-80gb"),
        "nemo_run_dir": _get_env(env, "nemo_run_dir", "/nemo_run/code"),
        "nodes": _get_env(env, "nodes", default_nodes),
        "nprocs_per_node": _get_env(env, "gpus_per_node", default_gpus),
        "mounts": list(_get_env(env, "mounts") or []),
        "pre_launch_commands": list(_get_env(env, "pre_launch_commands") or []),
        "image_pull_secrets": list(_get_env(env, "image_pull_secrets") or []),
        "packager": packager,
        "env_vars": env_vars,
        "launcher": run.Torchrun(rdzv_backend="c10d", rdzv_port=29500),
    }

    # Optional fields
    node_reservation = _get_env(env, "node_reservation")
    if node_reservation:
        executor_kwargs["node_reservation"] = node_reservation

    ray_version = _get_env(env, "ray_version")
    if ray_version:
        executor_kwargs["ray_version"] = ray_version

    head_resource_shape = _get_env(env, "head_resource_shape")
    if head_resource_shape:
        executor_kwargs["head_resource_shape"] = head_resource_shape

    return run.LeptonExecutor(**executor_kwargs)


def _create_runai_executor(
    env: Any,
    *,
    env_vars: dict[str, str],
    packager: Any,
    default_image: str | None = None,
    script_resources: Any | None = None,
) -> Any:
    """Create a KubeflowExecutor configured for Run:AI clusters.

    Run:AI provides a Kubernetes-based GPU orchestration platform. Since
    nemo-run does not ship a dedicated RunAIExecutor, we use the
    KubeflowExecutor (Kubeflow Training Operator v2) which targets the
    same Kubernetes API surface that Run:AI exposes.

    Required env.toml fields:
        container_image (or container): Container image for the job
        cluster: Kubernetes context name for the Run:AI cluster
        project: Run:AI project name (maps to Kubernetes namespace)

    Optional env.toml fields:
        nodes: Number of nodes (default: 1)
        gpus_per_node: GPUs per node
        pvc_mounts: List of PVC mount dicts, each with keys:
            name: PVC name
            mount_path: Container mount path
            sub_path: (optional) Sub-path within the PVC
        node_pool: Run:AI node pool name
        runtime_ref: Kubeflow runtime reference (default: "torch-distributed")

    Args:
        env: Environment configuration (OmegaConf DictConfig or dict)
        env_vars: Environment variables to pass to executor
        packager: Packager object for code shipping
        default_image: Fallback container image if env doesn't specify one
        script_resources: RunspecResources defaults for nodes/gpus

    Returns:
        Configured KubeflowExecutor
    """
    import nemo_run as run

    # Container image (required)
    container_image = (
        _get_env(env, "container_image")
        or _get_env(env, "container")
        or default_image
    )
    if not container_image:
        raise ValueError(
            "container_image is required for runai executor. "
            "Set it in your env.toml profile or in the recipe's [tool.runspec] image."
        )

    # Cluster / project (required)
    cluster = _get_env(env, "cluster")
    if not cluster:
        raise ValueError(
            "cluster is required for runai executor. "
            "Set it in your env.toml profile, e.g.: cluster = \"my-runai-cluster\""
        )

    project = _get_env(env, "project")
    if not project:
        raise ValueError(
            "project is required for runai executor. "
            "Set it in your env.toml profile, e.g.: project = \"my-team\""
        )

    # Resource defaults from script metadata
    default_nodes = script_resources.nodes if script_resources else 1
    default_gpus = script_resources.gpus_per_node if script_resources else None

    nodes = _get_env(env, "nodes", default_nodes)
    gpus_per_node = _get_env(env, "gpus_per_node", default_gpus)

    # Build PVC volume / volumeMount specs from pvc_mounts shorthand
    pvc_mounts = list(_get_env(env, "pvc_mounts") or [])
    volumes: list[dict[str, Any]] = []
    volume_mounts: list[dict[str, Any]] = []
    for pvc in pvc_mounts:
        pvc_name = pvc.get("name") if hasattr(pvc, "get") else getattr(pvc, "name", None)
        mount_path = pvc.get("mount_path") if hasattr(pvc, "get") else getattr(pvc, "mount_path", None)
        sub_path = pvc.get("sub_path", "") if hasattr(pvc, "get") else getattr(pvc, "sub_path", "")
        if not pvc_name or not mount_path:
            raise ValueError(
                "Each pvc_mounts entry must have 'name' and 'mount_path'. "
                f"Got: {pvc}"
            )
        vol_name = f"pvc-{pvc_name}"
        volumes.append({
            "name": vol_name,
            "persistentVolumeClaim": {"claimName": pvc_name},
        })
        vm: dict[str, Any] = {"name": vol_name, "mountPath": mount_path}
        if sub_path:
            vm["subPath"] = sub_path
        volume_mounts.append(vm)

    # Node pool annotation (Run:AI-specific scheduling)
    node_pool = _get_env(env, "node_pool")

    executor_kwargs: dict[str, Any] = {
        "runtime_ref": _get_env(env, "runtime_ref", "torch-distributed"),
        "namespace": project,
        "image": container_image,
        "num_nodes": nodes,
        "gpus_per_node": gpus_per_node,
        "volumes": volumes,
        "volume_mounts": volume_mounts,
        "packager": packager,
        "env_vars": env_vars,
    }

    if node_pool:
        executor_kwargs["annotations"] = {
            "run.ai/node-pool": node_pool,
        }

    console.print(
        f"[dim]Run:AI executor: cluster={cluster}, project={project}, "
        f"nodes={nodes}, gpus_per_node={gpus_per_node}[/dim]"
    )

    return run.KubeflowExecutor(**executor_kwargs)


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
