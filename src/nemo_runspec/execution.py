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

Supported executor types:
- local: LocalExecutor for single-machine runs
- docker: DockerExecutor for containerized local runs
- slurm: SlurmExecutor for HPC clusters via SSH tunnel
- dgxcloud: DGXCloudExecutor for NVIDIA DGX Cloud (run:ai API)
- lepton: LeptonExecutor for NVIDIA DGX Cloud Lepton clusters

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

from nemo_runspec import data_mover

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
                    typer.echo("[auto_mount] Warning: fetch failed, will re-clone")
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


def _to_plain(value: Any) -> Any:
    """Convert OmegaConf containers to plain Python dicts/lists.

    nemo-run's serialization (fiddle) cannot handle OmegaConf DictConfig
    objects. This function recursively converts them so env.toml values
    like pvcs, mounts, and custom_spec work transparently.
    """
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


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


def get_executor_type(env: Any, default: str = "slurm") -> str:
    """Return the ``executor`` key from env config, defaulting to ``slurm``.

    Shared helper so CLI commands don't each reimplement the OmegaConf/dict
    lookup for deciding which execution backend to dispatch to.
    """
    return _get_env(env, "executor", default)


def _resolve_container_image(env: Any, default_image: str | None) -> str | None:
    """Three-way container-image fallback shared by slurm/lepton/dgxcloud.

    Priority: ``env.container_image`` > ``env.container`` (legacy name) >
    caller-provided default (typically ``SPEC.image``).
    """
    return _get_env(env, "container_image") or _get_env(env, "container") or default_image


def _resolve_nodes_gpus(
    env: Any, script_resources: Any | None
) -> tuple[int, int | None]:
    """Defaults for ``nodes`` / ``gpus_per_node`` — env.toml overrides the
    script's ``[tool.runspec.resources]`` which overrides 1/None."""
    default_nodes = script_resources.nodes if script_resources else 1
    default_gpus = script_resources.gpus_per_node if script_resources else None
    return (
        _get_env(env, "nodes", default_nodes),
        _get_env(env, "gpus_per_node", default_gpus),
    )


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
    executor_type = _get_env(env, "executor", "local")
    if executor_type == "local":
        return _create_local_executor(env, env_vars)
    if executor_type == "docker":
        return _create_docker_executor(env, env_vars, packager, default_image)
    if executor_type == "slurm":
        return create_slurm_executor(
            env, env_vars, packager,
            default_image=default_image,
            script_resources=script_resources,
            attached=attached,
            force_squash=force_squash,
        )
    if executor_type == "dgxcloud":
        return _create_dgxcloud_executor(env, env_vars, packager, default_image, script_resources)
    if executor_type == "lepton":
        return _create_lepton_executor(env, env_vars, packager, default_image, script_resources)
    raise ValueError(
        f"Unknown executor type: {executor_type!r}. "
        "Supported: local, docker, slurm, dgxcloud, lepton"
    )


# =============================================================================
# Local / Docker Executors
# =============================================================================


def _create_local_executor(env: Any, env_vars: dict[str, str]) -> Any:
    """LocalExecutor for single-machine runs (torchrun-based)."""
    import nemo_run as run

    return run.LocalExecutor(
        ntasks_per_node=_get_env(env, "nproc_per_node", 1),
        launcher="torchrun",
        env_vars=env_vars,
    )


def _create_docker_executor(
    env: Any,
    env_vars: dict[str, str],
    packager: Any,
    default_image: str | None,
) -> Any:
    """DockerExecutor with env-var-expanded, path-resolved host mounts."""
    import nemo_run as run

    container_image = _resolve_container_image(env, default_image)
    if not container_image:
        raise ValueError("container_image required for docker executor")

    # Resolve relative paths and expand env vars in mounts
    resolved_mounts: list[str] = []
    for mount in _get_env(env, "mounts") or []:
        if ":" not in mount:
            resolved_mounts.append(mount)
            continue
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


# =============================================================================
# Slurm Executor
# =============================================================================


def create_slurm_executor(
    env: Any,
    env_vars: dict[str, str],
    packager: Any,
    *,
    default_image: str | None = None,
    script_resources: Any | None = None,
    attached: bool = False,
    force_squash: bool = False,
    launcher: str | None = "torchrun",
) -> Any:
    """SlurmExecutor for HPC clusters via SSH tunnel.

    Handles SSH tunneling, container image squashing, git-repo mounting,
    partition selection (attached vs batch), and lustre / ray-temp mounts.

    Pass ``launcher=None`` for Ray-based flows (data prep, RL) that wrap the
    command themselves and don't want nemo-run to inject a torchrun prefix.
    """
    import nemo_run as run

    remote_job_dir = _get_env(env, "remote_job_dir")

    # SSH tunnel (optional). ``identity`` lets the user point paramiko at a
    # specific key when ssh-agent isn't running / no default key matches.
    tunnel = None
    if _get_env(env, "tunnel") == "ssh":
        tunnel_kwargs: dict[str, Any] = {
            "host": _get_env(env, "host", "localhost"),
            "user": _get_env(env, "user"),
            "job_dir": remote_job_dir,
        }
        identity = _get_env(env, "identity")
        if identity:
            tunnel_kwargs["identity"] = os.path.expanduser(str(identity))
        tunnel = run.SSHTunnel(**tunnel_kwargs)

    container_image = _resolve_container_image(env, default_image)

    # Force-resolve ``env.mounts`` BEFORE cloning: any ``${auto_mount:git+...}``
    # entries only register themselves in the global ``get_git_mounts()``
    # registry at resolution time, and ``clone_git_repos_via_tunnel`` reads
    # from that registry. Accessing the list here triggers OmegaConf to walk
    # and resolve each element. Cache the result so the later step reuses it.
    raw_mounts = list(_get_env(env, "mounts") or [])

    # One tunnel.connect() block covers both squashing and git-repo cloning.
    git_mounts: list[str] = []
    if tunnel and remote_job_dir:
        tunnel.connect()
        if container_image:
            from nemo_runspec.squash import ensure_squashed_image

            env_dict = dict(env) if env else {}
            container_image = ensure_squashed_image(
                tunnel, container_image, remote_job_dir, env_dict, force=force_squash
            )
        git_mounts = clone_git_repos_via_tunnel(tunnel, remote_job_dir)

    # Partition selection: attached (--run) vs batch (--batch) can route to
    # different Slurm queues, falling back to a single shared ``partition``.
    partition_key = "run_partition" if attached else "batch_partition"
    partition = _get_env(env, partition_key) or _get_env(env, "partition")

    # Container mounts = explicit mounts (minus auto-mount placeholders)
    # + git-cloned repos + optional /lustre + optional ray-temp dir.
    mounts = [m for m in raw_mounts if not m.startswith("__auto_mount__:")]
    mounts.extend(git_mounts)
    lustre_mount = _get_env(env, "lustre_mount", "/lustre:/lustre")
    if lustre_mount:
        mounts.append(lustre_mount)
    if remote_job_dir:
        ray_temp_path = f"{remote_job_dir}/ray_temp"
        mounts.append(f"{ray_temp_path}:/ray-cluster")
        if tunnel:
            tunnel.run(f"mkdir -p {ray_temp_path}", hide=True)

    nodes, gpus_per_node = _resolve_nodes_gpus(env, script_resources)
    executor_kwargs: dict[str, Any] = {
        "account": _get_env(env, "account"),
        "partition": partition,
        "nodes": nodes,
        "ntasks_per_node": _get_env(env, "ntasks_per_node", gpus_per_node or 1),
        "gpus_per_node": gpus_per_node,
        "cpus_per_task": _get_env(env, "cpus_per_task"),
        "time": _get_env(env, "time", "04:00:00"),
        "container_image": container_image,
        "container_mounts": mounts,
        "tunnel": tunnel,
        "packager": packager,
        "mem": _get_env(env, "mem"),
        "env_vars": env_vars,
        "launcher": launcher,
    }
    if _get_env(env, "exclusive"):
        executor_kwargs["exclusive"] = True

    return run.SlurmExecutor(**executor_kwargs)


# =============================================================================
# DGX Cloud Executor
# =============================================================================


def _create_dgxcloud_executor(
    env: Any,
    env_vars: dict[str, str],
    packager: Any,
    default_image: str | None,
    script_resources: Any | None,
) -> Any:
    """Create a DGXCloudExecutor for NVIDIA DGX Cloud (run:ai).

    Required env.toml fields:
        base_url: DGX Cloud API base URL
        kube_apiserver_url: Run:ai Kubernetes API server URL
        client_id: OAuth client ID (or legacy app_id)
        client_secret: OAuth client secret (or legacy app_secret)
        project_name: DGX Cloud project name
        pvc_nemo_run_dir: PVC path for nemo-run job directory

    Optional fields:
        container_image, nodes, gpus_per_node, nprocs_per_node,
        pvcs (list of PVC mount dicts), distributed_framework,
        custom_spec (dict for additional workload spec overrides)
    """
    import nemo_run as run

    container_image = _resolve_container_image(env, default_image)
    if not container_image:
        raise ValueError("container_image required for dgxcloud executor")

    base_url = _get_env(env, "base_url")
    if not base_url:
        raise ValueError("base_url required for dgxcloud executor")

    kube_apiserver_url = _get_env(env, "kube_apiserver_url")
    if not kube_apiserver_url:
        raise ValueError("kube_apiserver_url required for dgxcloud executor")

    # Accept both old (app_id/app_secret) and new (client_id/client_secret)
    # names in env.toml. DGXCloudExecutor itself now requires client_id /
    # client_secret (post nemo-run PR #480 / 0.10rc).
    client_id = _get_env(env, "client_id") or _get_env(env, "app_id")
    client_secret = _get_env(env, "client_secret") or _get_env(env, "app_secret")
    if not client_id or not client_secret:
        raise ValueError("client_id/client_secret (or legacy app_id/app_secret) required for dgxcloud executor")

    project_name = _get_env(env, "project_name")
    if not project_name:
        raise ValueError("project_name required for dgxcloud executor")

    pvc_nemo_run_dir = _get_env(env, "pvc_nemo_run_dir")
    if not pvc_nemo_run_dir:
        raise ValueError("pvc_nemo_run_dir required for dgxcloud executor")

    nodes, gpus_per_node = _resolve_nodes_gpus(env, script_resources)
    executor_kwargs: dict[str, Any] = {
        "base_url": base_url,
        "kube_apiserver_url": kube_apiserver_url,
        "client_id": client_id,
        "client_secret": client_secret,
        "project_name": project_name,
        "container_image": container_image,
        "pvc_nemo_run_dir": pvc_nemo_run_dir,
        "nodes": nodes,
        "gpus_per_node": gpus_per_node or 0,
        "nprocs_per_node": _get_env(env, "nprocs_per_node") or _get_env(env, "ntasks_per_node", 1),
        "packager": packager,
        "env_vars": env_vars,
    }

    pvcs = _get_env(env, "pvcs")
    if pvcs:
        executor_kwargs["pvcs"] = _to_plain(pvcs)

    distributed_framework = _get_env(env, "distributed_framework")
    if distributed_framework:
        executor_kwargs["distributed_framework"] = distributed_framework

    custom_spec = _get_env(env, "custom_spec")
    if custom_spec:
        executor_kwargs["custom_spec"] = _to_plain(custom_spec)

    launcher = _get_env(env, "launcher", "torchrun")
    if launcher:
        executor_kwargs["launcher"] = launcher

    executor = run.DGXCloudExecutor(**executor_kwargs)
    # run:AI's Args field is hard-capped at 10 000 chars (HTTP 400 above).
    # nemo-run's fork defaults to 9500 for ~500 char headroom. Expose a knob
    # for advanced tuning but keep the safe default.
    max_args = _get_env(env, "dgxcloud_max_args_chars")
    if max_args is not None:
        executor.MAX_ARGS_CHARS = int(max_args)
    return executor


# =============================================================================
# Lepton Executor
# =============================================================================


def _create_lepton_executor(
    env: Any,
    env_vars: dict[str, str],
    packager: Any,
    default_image: str | None,
    script_resources: Any | None,
) -> Any:
    """Create a LeptonExecutor for NVIDIA DGX Cloud Lepton clusters.

    Required env.toml fields:
        nemo_run_dir: Remote directory for nemo-run job files on the Lepton cluster

    Optional fields:
        container_image, nodes, gpus_per_node, nprocs_per_node,
        resource_shape, node_group, node_reservation,
        shared_memory_size, mounts (list of mount dicts),
        image_pull_secrets, custom_spec, pre_launch_commands
    """
    import nemo_run as run

    container_image = _resolve_container_image(env, default_image)
    if not container_image:
        raise ValueError("container_image required for lepton executor")

    nemo_run_dir = _get_env(env, "nemo_run_dir")
    if not nemo_run_dir:
        raise ValueError("nemo_run_dir required for lepton executor")

    nodes, gpus_per_node = _resolve_nodes_gpus(env, script_resources)
    executor_kwargs: dict[str, Any] = {
        "container_image": container_image,
        "nemo_run_dir": nemo_run_dir,
        "nodes": nodes,
        "gpus_per_node": gpus_per_node or 0,
        "nprocs_per_node": _get_env(env, "nprocs_per_node") or _get_env(env, "ntasks_per_node", 1),
        "packager": packager,
        "env_vars": env_vars,
    }

    resource_shape = _get_env(env, "resource_shape")
    if resource_shape:
        executor_kwargs["resource_shape"] = resource_shape

    node_group = _get_env(env, "node_group")
    if node_group:
        executor_kwargs["node_group"] = node_group

    node_reservation = _get_env(env, "node_reservation")
    if node_reservation:
        executor_kwargs["node_reservation"] = node_reservation

    shared_memory_size = _get_env(env, "shared_memory_size")
    if shared_memory_size is not None:
        executor_kwargs["shared_memory_size"] = int(shared_memory_size)

    mounts = _get_env(env, "mounts")
    if mounts:
        # Filter out __auto_mount__ strings (Slurm-specific, not valid on cloud)
        plain = _to_plain(mounts)
        executor_kwargs["mounts"] = [
            m for m in plain
            if not (isinstance(m, str) and m.startswith("__auto_mount__"))
        ]

    image_pull_secrets = _get_env(env, "image_pull_secrets")
    if image_pull_secrets:
        executor_kwargs["image_pull_secrets"] = list(image_pull_secrets)

    # LeptonRayCluster needs a plain version ("2.48.0"), not the default
    # image-tag string 'ray:2.48.0-py312-gpu' that nemo-run supplies.
    ray_version = _get_env(env, "ray_version")
    if ray_version:
        executor_kwargs["ray_version"] = ray_version

    custom_spec = _get_env(env, "custom_spec")
    if custom_spec:
        executor_kwargs["custom_spec"] = _to_plain(custom_spec)

    # Pre-launch commands: user-defined + auto_mount git repos
    pre_launch = list(_get_env(env, "pre_launch_commands") or [])
    pre_launch.extend(_git_mount_commands())
    if pre_launch:
        executor_kwargs["pre_launch_commands"] = pre_launch

    launcher = _get_env(env, "launcher", "torchrun")
    if launcher:
        executor_kwargs["launcher"] = launcher

    return run.LeptonExecutor(**executor_kwargs)


# =============================================================================
# Cloud Execution (Lepton / DGX Cloud)
# =============================================================================


def _git_mount_commands() -> list[str]:
    """Convert registered auto_mount git repos to shell clone commands.

    On Slurm, auto_mount repos are cloned via SSH tunnel and bind-mounted
    into the container. On cloud executors (Lepton/DGXCloud), bind mounts
    aren't available, so we clone the repos inside the container instead.

    Returns:
        List of shell commands like:
          "rm -rf /opt/megatron-lm && git clone --depth 1 -b <ref> <url> /opt/megatron-lm"
    """
    from nemo_runspec.config.resolvers import get_git_mounts

    commands = []
    for repo_name, info in get_git_mounts().items():
        url = info["url"]
        ref = info.get("ref", "main")
        target = info.get("target", "")
        if not target:
            continue

        # Check if ref is a commit SHA (40 hex chars) vs branch/tag name.
        # git clone --depth 1 -b only works with branch/tag names.
        is_sha = len(ref) == 40 and all(c in "0123456789abcdef" for c in ref.lower())
        if is_sha:
            # For specific commits: try shallow fetch, but don't delete the
            # existing target if fetch fails (keeps container's built-in version).
            # GitHub often blocks fetching arbitrary SHAs with --depth 1.
            commands.append(
                f"(git init /tmp/_git_{repo_name} && git -C /tmp/_git_{repo_name} remote add origin {url}"
                f" && git -C /tmp/_git_{repo_name} fetch --depth 1 origin {ref}"
                f" && git -C /tmp/_git_{repo_name} checkout FETCH_HEAD"
                f" && rm -rf {target} && mv /tmp/_git_{repo_name} {target})"
                f" || echo '[auto_mount] WARNING: Could not fetch {ref[:12]}, using container built-in {target}'"
            )
        else:
            commands.append(
                f"rm -rf {target} && git clone --depth 1 -b {ref} {url} {target}"
            )
    return commands


def _derive_cloud_workspace(env: Any) -> str:
    """Derive persistent storage root from env config.

    Priority: explicit ``workspace`` > first mount/PVC path > ``/tmp``.
    """
    explicit = _get_env(env, "workspace")
    if explicit:
        return explicit

    executor_type = _get_env(env, "executor", "")
    if executor_type == "lepton":
        mounts = _to_plain(_get_env(env, "mounts") or [])
        # Skip auto_mount strings — only real mount dicts have mount_path.
        for m in mounts:
            if isinstance(m, dict) and m.get("mount_path"):
                return m["mount_path"]
    else:
        pvcs = _to_plain(_get_env(env, "pvcs") or [])
        for p in pvcs:
            if isinstance(p, dict) and p.get("path"):
                return p["path"]

    import logging
    logging.getLogger(__name__).warning(
        "No workspace, mounts, or pvcs configured — output goes to ephemeral /tmp"
    )
    return "/tmp"


def execute_cloud(
    script_path: str,
    train_path: Path,
    env: Any,
    env_vars: dict[str, str],
    passthrough: list[str],
    attached: bool,
    *,
    default_image: str | None = None,
    script_resources: Any | None = None,
    startup_commands: list[str] | None = None,
    run_command: str | None = None,
    setup_commands: list[str] | None = None,
    launch: str | None = None,
) -> None:
    """Execute a recipe script on Lepton or DGX Cloud.

    Source distribution: ``PatternPackager`` tars the local ``src/nemotron``
    and ``src/nemo_runspec`` directories and extracts them to
    ``/nemo_run/code/src/...`` on the remote pod. This is airgap-friendly and
    picks up local uncommitted edits — no ``git clone`` is required on the
    remote.

    How it works:
    1. ``PatternPackager`` uploads ``src/`` to ``/nemo_run/code/src`` on the pod.
    2. Config YAML is passed as a base64 env var and decoded into the workspace.
    3. Extra packages from ``pip_extras`` are installed (CLI deps, etc.).
    4. Symlinks at ``{workspace}/_nemotron/src/`` point at ``/nemo_run/code/src``
       so ``${oc.env:PWD}/src/nemotron/...`` config paths resolve.
    5. ``PWD`` is set to ``{workspace}/_nemotron`` so outputs
       (``${oc.env:PWD}/../output/...``) land on persistent storage.
    """
    import base64

    import nemo_run as run

    executor_type = _get_env(env, "executor")

    # ── 1. Workspace & paths ─────────────────────────────────────────
    workspace = _derive_cloud_workspace(env)
    nemotron_home = f"{workspace}/_nemotron"
    config_path = f"{nemotron_home}/config.yaml"

    # ── 2. Config + source transport ────────────────────────────────
    env_vars["_NEMOTRON_CONFIG_B64"] = base64.b64encode(
        train_path.read_bytes()
    ).decode("ascii")
    transport = data_mover.plan_for(
        executor_type=executor_type or "",
        env_vars=env_vars,
        script_path=script_path,
        pod_nemotron_home=nemotron_home,
        repo_root=_get_env(env, "repo_root"),
    )

    # ── 3. Executor ──────────────────────────────────────────────────
    if executor_type == "lepton":
        executor = _create_lepton_executor(
            env, env_vars, transport.packager, default_image, script_resources
        )
    else:
        executor = _create_dgxcloud_executor(
            env, env_vars, transport.packager, default_image, script_resources
        )
    # We wrap the final command ourselves; never let nemo-run inject a launcher.
    executor.launcher = None

    # ── 4. Config from env.toml ──────────────────────────────────────
    pip_extras = _to_plain(_get_env(env, "pip_extras") or [])

    # ── 5. Run command ───────────────────────────────────────────────
    module_path = script_path.replace("src/", "").replace("/", ".").removesuffix(".py")
    if launch == "torchrun":
        # Multi-process training. Lepton/DGXCloud populate NODE_RANK /
        # MASTER_ADDR / MASTER_PORT per worker pod.
        nnodes = _get_env(env, "nodes") or (script_resources.nodes if script_resources else 1)
        nproc = (
            _get_env(env, "nprocs_per_node")
            or _get_env(env, "ntasks_per_node")
            or _get_env(env, "gpus_per_node")
            or (script_resources.gpus_per_node if script_resources else 1)
        )
        default_cmd = (
            f"torchrun --nnodes {nnodes} --nproc-per-node {nproc}"
            ' --node-rank ${{NODE_RANK:-0}}'
            ' --master-addr ${{MASTER_ADDR:-127.0.0.1}}'
            ' --master-port ${{MASTER_PORT:-29500}}'
            f" -m {module_path} --config {{config}}"
        )
    else:
        default_cmd = f"python -m {module_path} --config {{config}}"
    effective_cmd = run_command or _get_env(env, "run_command") or default_cmd
    script_cmd = effective_cmd.format(script=script_path, config=config_path)
    if passthrough:
        script_cmd += " " + " ".join(passthrough)

    # ── 6. Inline launch script ──────────────────────────────────────
    parts: list[str] = [
        # Decode config to persistent workspace
        f"mkdir -p {nemotron_home}",
        f"echo $_NEMOTRON_CONFIG_B64 | base64 -d > {config_path}",
    ]
    # Per-transport extraction (env-var chunks, job_dir tarball, or nothing).
    parts.extend(transport.pre_script_cmds)
    # Extra pip packages from env.toml (CLI deps, experimental libs, etc.)
    for pkg in pip_extras:
        parts.append(f"pip install -q {pkg} 2>/dev/null || true")
    # Clone auto_mount git repos (Megatron-LM, Megatron-Bridge, etc.)
    # On Slurm these are bind-mounted; on cloud we clone inside the container.
    parts.extend(_git_mount_commands())
    # Caller-provided setup & env.toml startup commands
    if setup_commands:
        parts.extend(setup_commands)
    if startup_commands:
        parts.extend(startup_commands)

    # Final line: activate source + run. Native-packager source lives under
    # ``/nemo_run/code/src``; symlink it under ${oc.env:PWD}/src so OmegaConf
    # interpolations resolve. Chunked / job_dir transports already extracted
    # directly into ``nemotron_home/src``.
    launch = f"export PYTHONPATH={transport.pod_src_root}:${{PYTHONPATH:-}}"
    if transport.needs_pwd_symlinks:
        launch += (
            f" && mkdir -p {nemotron_home}/src"
            f" && ln -sfn {transport.pod_src_root}/nemotron {nemotron_home}/src/nemotron"
            f" && ln -sfn {transport.pod_src_root}/nemo_runspec {nemotron_home}/src/nemo_runspec"
        )
    launch += f" && export PWD={nemotron_home} && cd {nemotron_home} && {script_cmd}"
    parts.append(launch)

    # ── 7. Submit ────────────────────────────────────────────────────
    script_task = run.Script(inline=" && ".join(parts))
    recipe_name = (
        script_path
        .replace("src/nemotron/recipes/", "")
        .replace("/", "-")
        .removesuffix(".py")
    )
    with run.Experiment(recipe_name) as exp:
        exp.add(script_task, executor=executor, name=recipe_name)
        exp.run(detach=not attached)


# =============================================================================
# Cloud Ray Execution (Lepton + DGX Cloud, via nemo-run RayCluster/RayJob)
# =============================================================================


def execute_cloud_ray(
    script_path: str,
    train_path: Path,
    env: Any,
    env_vars: dict[str, str],
    passthrough: list[str],
    attached: bool,
    *,
    default_image: str | None = None,
    script_resources: Any | None = None,
    startup_commands: list[str] | None = None,
    run_command: str | None = None,
    setup_commands: list[str] | None = None,
) -> None:
    """Execute a Ray recipe (launch='ray') on Lepton / DGX Cloud via nemo-run's
    ``RayCluster`` + ``RayJob`` classes — the same pattern Slurm already uses.

    Source distribution: same chunked-env-var transport as :func:`execute_cloud`.
    Envs propagate to every Ray pod (head + workers) and are inherited by Ray
    actors automatically. No Storage API, no git clone, no PatternPackager.
    """
    import base64
    import time

    from nemo_run.run.ray.cluster import RayCluster
    from nemo_run.run.ray.job import RayJob

    executor_type = _get_env(env, "executor")

    workspace = _derive_cloud_workspace(env)
    nemotron_home = f"{workspace}/_nemotron"
    config_path = f"{nemotron_home}/config.yaml"

    env_vars["_NEMOTRON_CONFIG_B64"] = base64.b64encode(
        train_path.read_bytes()
    ).decode("ascii")

    # Same source-transport strategy selection as the non-Ray path.
    transport = data_mover.plan_for(
        executor_type=executor_type or "",
        env_vars=env_vars,
        script_path=script_path,
        pod_nemotron_home=nemotron_home,
        repo_root=_get_env(env, "repo_root"),
    )
    # Make source importable on every pod (head + workers) without a per-pod
    # install step; Ray actors inherit this via env_vars.
    env_vars["PYTHONPATH"] = transport.pod_src_root + ":" + env_vars.get("PYTHONPATH", "")

    if executor_type == "lepton":
        executor = _create_lepton_executor(
            env, env_vars, transport.packager, default_image, script_resources
        )
    else:
        executor = _create_dgxcloud_executor(
            env, env_vars, transport.packager, default_image, script_resources
        )
    executor.launcher = None

    pip_extras = _to_plain(_get_env(env, "pip_extras") or [])

    # Per-pod setup (runs BEFORE ray start). The transport's extraction
    # commands run here so workers have source on disk before Ray actors start.
    pre_ray_commands: list[str] = list(transport.pre_script_cmds)
    for pkg in pip_extras:
        pre_ray_commands.append(f"pip install -q {pkg} 2>/dev/null || true")
    pre_ray_commands.extend(_git_mount_commands())
    if setup_commands:
        pre_ray_commands.extend(setup_commands)

    # Head command (runs after Ray cluster is up).
    module_path = script_path.replace("src/", "").replace("/", ".").removesuffix(".py")
    default_cmd = f"python -m {module_path} --config {{config}}"
    effective_cmd = run_command or _get_env(env, "run_command") or default_cmd
    script_cmd = effective_cmd.format(script=script_path, config=config_path)
    if passthrough:
        script_cmd += " " + " ".join(passthrough)

    head_pip = ["typer", "rich", "pydantic-settings", "shellingham", *pip_extras]
    head_setup = [
        f"pip install -q {' '.join(head_pip)} 2>/dev/null || true",
        f"export PYTHONPATH={transport.pod_src_root}:${{PYTHONPATH:-}}",
        f"mkdir -p {nemotron_home}",
        f"echo $_NEMOTRON_CONFIG_B64 | base64 -d > {config_path}",
    ]
    if transport.needs_pwd_symlinks:
        # Native-packager path: source sits at /nemo_run/code/src; symlink it
        # under nemotron_home/src so ${oc.env:PWD}/src/... still resolves.
        head_setup.append(
            f"mkdir -p {nemotron_home}/src"
            f" && ln -sfn {transport.pod_src_root}/nemotron {nemotron_home}/src/nemotron"
            f" && ln -sfn {transport.pod_src_root}/nemo_runspec {nemotron_home}/src/nemo_runspec"
        )
    if startup_commands:
        head_setup.extend(startup_commands)
    full_cmd = " && ".join([
        *head_setup,
        f"export PWD={nemotron_home} && cd {nemotron_home}",
        script_cmd,
    ])

    # Note: we deliberately do NOT forward env_vars through Ray's job
    # ``runtime_env``. nemo-rl's ``VllmAsyncGenerationWorker`` calls
    # ``ray.init(runtime_env=...)`` itself, and Ray can't merge two
    # runtime_envs that both set env_vars. Env vars are already present on
    # the head pod via ``LeptonExecutor.env_vars``, so the entrypoint
    # inherits them naturally; Ray actors inherit from there.

    recipe_name = (
        script_path
        .replace("src/nemotron/recipes/", "")
        .replace("/", "-")
        .removesuffix(".py")
    )
    stamp = int(time.time())
    cluster_name = f"{recipe_name}-{stamp}".replace("_", "-").replace(".", "-").lower()[:35]
    job_name = f"{recipe_name}-job-{stamp}".replace("_", "-").replace(".", "-").lower()[:35]

    # For Lepton: spin a RayCluster and submit a RayJob to it.
    # For DGXCloud: RayJob alone (DGXCloudRayCluster is a no-op placeholder in
    # nemo-run; DGXCloudRayJob creates its own distributed workload).
    cluster = None
    if executor_type == "lepton":
        cluster = RayCluster(name=cluster_name, executor=executor)
        typer.echo(f"[ray] starting RayCluster {cluster_name} (timeout 1800s)...")
        try:
            cluster.start(pre_ray_start_commands=pre_ray_commands, timeout=1800)
        except Exception as e:  # noqa: BLE001
            typer.echo(f"[ray] cluster.start warning ({type(e).__name__}: {e}); continuing")

    if executor_type == "lepton":
        ray_job = RayJob(name=job_name, executor=executor, cluster_name=cluster_name)
    else:
        ray_job = RayJob(
            name=job_name,
            executor=executor,
            pre_ray_start_commands=pre_ray_commands,
        )

    # Passing a non-empty workdir makes Ray's JobSubmissionClient add
    # ``working_dir`` to the job's runtime_env. vLLM's internal ``ray.init``
    # inside each generation actor has its OWN runtime_env that Ray can't
    # merge with one that also has ``working_dir`` — it fails with
    # "Failed to merge the Job's runtime env". Workaround: pass empty
    # workdir so the job's runtime_env stays empty. Source is already on
    # every pod via the chunked env-var transport (or the fallback packager),
    # so Ray actors can import ``nemotron.*`` directly.
    typer.echo(f"[ray] submitting RayJob {job_name} (no workdir; source via env-var chunks)")
    ray_job.start(
        command=full_cmd,
        workdir="",
    )
    typer.echo(f"[ray] submission_id={getattr(ray_job, 'submission_id', None)}")

    if attached:
        # Retry log streaming on transient disconnects (Ray dashboard/WebSocket
        # can drop mid-stream during heavy log bursts like vLLM compile).
        while True:
            try:
                ray_job.logs(follow=True)
                break
            except KeyboardInterrupt:
                typer.echo("\n[info] Detaching. Job continues running.")
                raise typer.Exit(0)
            except Exception as e:  # noqa: BLE001
                typer.echo(f"[ray] log stream dropped ({type(e).__name__}: {e}); reconnecting...")
                import time as _time
                _time.sleep(5)


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
