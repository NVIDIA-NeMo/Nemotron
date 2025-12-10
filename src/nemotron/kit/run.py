# Copyright (c) Nemotron Contributors
# SPDX-License-Identifier: MIT

"""NeMo-Run integration for nemotron.kit.

Provides RunConfig dataclass and executor builders for all nemo-run executors:
- local: Local execution with torchrun
- docker: Docker container execution
- slurm: Slurm cluster execution
- skypilot: Cloud execution via SkyPilot
- dgxcloud: NVIDIA DGX Cloud execution
- lepton: Lepton AI execution

Example:
    >>> from nemotron.kit.run import RunConfig, build_executor
    >>>
    >>> config = RunConfig(executor="slurm", account="my-account", partition="gpu")
    >>> executor = build_executor(config, env_vars={"NCCL_DEBUG": "INFO"})

Wandb configuration can also be stored in run.toml:
    >>> # run.toml
    >>> # [wandb]
    >>> # project = "my-project"
    >>> # entity = "my-team"
    >>>
    >>> from nemotron.kit.run import load_wandb_config
    >>> wandb_config = load_wandb_config()  # Returns WandbConfig or None
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from nemotron.kit.wandb import WandbConfig

Executor = Literal["local", "docker", "slurm", "skypilot", "dgxcloud", "lepton"]
"""Supported nemo-run executor types."""


@dataclass
class RunConfig:
    """Configuration for nemo-run execution (infrastructure only).

    Ray usage is determined by the recipe (ray=True in cli()), not here.
    Supports all nemo-run executors.

    Attributes:
        executor: Executor type (local, docker, slurm, skypilot, dgxcloud, lepton)
        nproc_per_node: Number of processes (GPUs) per node
        nodes: Number of nodes for distributed execution

        container_image: Container image for docker/slurm/skypilot
        mounts: Container mount points (e.g., '/data:/data')

        account: Slurm account name
        partition: Slurm partition name
        time: Slurm job time limit (HH:MM:SS)
        job_name: Slurm job name

        tunnel: Tunnel type for Slurm (local or ssh)
        host: SSH host for remote job submission
        user: SSH user for remote job submission
        identity: SSH identity file path
        remote_job_dir: Remote directory for job files

        runtime: Docker runtime (e.g., 'nvidia')
        ipc_mode: Docker IPC mode
        shm_size: Docker shared memory size

        cloud: SkyPilot cloud provider (aws, gcp, azure)
        gpus: SkyPilot GPU spec (e.g., 'A100:8')
        cluster_name: SkyPilot cluster name
        setup: SkyPilot pre-launch commands

        base_url: DGX Cloud base URL
        app_id: DGX Cloud app ID
        app_secret: DGX Cloud app secret
        project_name: DGX Cloud project name
        pvcs: DGX Cloud persistent volume claims

        resource_shape: Lepton GPU pod type
        node_group: Lepton node group
        nemo_run_dir: Lepton nemo-run directory

        ray_working_dir: Working directory for Ray jobs

        env_vars: Environment variables (KEY=VALUE format)
        dry_run: Print commands without executing
        detach: Don't wait for job completion
    """

    # Executor type
    executor: Executor = "local"

    # Common resource settings
    nproc_per_node: int = 8
    nodes: int = 1

    # Container settings (docker, slurm, skypilot)
    container_image: str | None = None
    mounts: list[str] = field(default_factory=list)

    # Slurm settings
    account: str | None = None
    partition: str | None = None
    time: str = "04:00:00"
    job_name: str = "nemo-run"

    # SSH tunnel settings (for remote Slurm submission)
    tunnel: Literal["local", "ssh"] = "local"
    host: str | None = None
    user: str | None = None
    identity: str | None = None
    remote_job_dir: str | None = None

    # Docker settings
    runtime: str | None = None  # e.g., "nvidia"
    ipc_mode: str | None = None
    shm_size: str | None = None

    # Skypilot settings
    cloud: str | None = None  # e.g., "aws", "gcp", "azure"
    gpus: str | None = None  # e.g., "A100:8"
    cluster_name: str | None = None
    setup: str | None = None  # Pre-launch commands

    # DGX Cloud settings
    base_url: str | None = None
    app_id: str | None = None
    app_secret: str | None = None
    project_name: str | None = None
    pvcs: list[str] = field(default_factory=list)  # persistent volume claims

    # Lepton settings
    resource_shape: str | None = None  # GPU pod type
    node_group: str | None = None
    nemo_run_dir: str | None = None

    # Ray infrastructure settings (used when recipe has ray=True)
    ray_working_dir: str | None = None

    # Environment
    env_vars: list[str] = field(default_factory=list)

    # Execution options
    dry_run: bool = False
    detach: bool = False


def build_executor(config: RunConfig, env_vars: dict[str, str] | None = None) -> Any:
    """Build nemo-run executor from RunConfig.

    Args:
        config: Run configuration specifying executor type and settings.
        env_vars: Additional environment variables to merge.

    Returns:
        A nemo-run executor instance.

    Raises:
        ImportError: If nemo-run is not installed.
        ValueError: If required settings are missing for the executor type.
    """
    try:
        import nemo_run as run
    except ImportError as e:
        raise ImportError(
            "nemo-run not installed. Install with: pip install nemo-run\n"
            "Or use direct execution without --run"
        ) from e

    # Parse and merge environment variables
    merged_env = {}
    for env in config.env_vars:
        if "=" in env:
            key, value = env.split("=", 1)
            merged_env[key] = value
    if env_vars:
        merged_env.update(env_vars)

    match config.executor:
        case "local":
            return run.LocalExecutor(
                ntasks_per_node=config.nproc_per_node,
                launcher="torchrun",
                env_vars=merged_env,
            )

        case "docker":
            if not config.container_image:
                raise ValueError("container_image required for docker executor")
            return run.DockerExecutor(
                container_image=config.container_image,
                num_gpus=config.nproc_per_node,
                runtime=config.runtime or "nvidia",
                ipc_mode=config.ipc_mode,
                shm_size=config.shm_size,
                volumes=config.mounts,
                env_vars=merged_env,
            )

        case "slurm":
            if not config.account:
                raise ValueError("account required for slurm executor")
            if not config.partition:
                raise ValueError("partition required for slurm executor")

            tunnel = _build_tunnel(config)
            return run.SlurmExecutor(
                account=config.account,
                partition=config.partition,
                nodes=config.nodes,
                ntasks_per_node=config.nproc_per_node,
                gpus_per_node=config.nproc_per_node,
                time=config.time,
                mem="0",
                exclusive=True,
                container_image=config.container_image,
                container_mounts=config.mounts,
                tunnel=tunnel,
                env_vars=merged_env,
            )

        case "skypilot":
            return run.SkypilotExecutor(
                gpus=config.gpus,
                gpus_per_node=config.nproc_per_node,
                num_nodes=config.nodes,
                cloud=config.cloud,
                cluster_name=config.cluster_name,
                setup=config.setup,
                env_vars=merged_env,
            )

        case "dgxcloud":
            return run.DGXCloudExecutor(
                base_url=config.base_url,
                app_id=config.app_id,
                app_secret=config.app_secret,
                project_name=config.project_name,
                nodes=config.nodes,
                gpus_per_node=config.nproc_per_node,
                pvcs=config.pvcs,
                env_vars=merged_env,
            )

        case "lepton":
            return run.LeptonExecutor(
                resource_shape=config.resource_shape,
                node_group=config.node_group,
                nemo_run_dir=config.nemo_run_dir,
                mounts=config.mounts,
                env_vars=merged_env,
            )

        case _:
            raise ValueError(f"Unknown executor: {config.executor}")


def _build_tunnel(config: RunConfig) -> Any:
    """Build nemo-run tunnel for Slurm executor.

    Args:
        config: Run configuration with tunnel settings.

    Returns:
        A nemo-run tunnel instance (LocalTunnel or SSHTunnel).
    """
    import nemo_run as run

    if config.tunnel == "ssh":
        if not config.host or not config.user:
            raise ValueError("host and user required for SSH tunnel")
        return run.SSHTunnel(
            host=config.host,
            user=config.user,
            job_dir=config.remote_job_dir,
            identity=config.identity,
        )
    return run.LocalTunnel()


def _find_run_config() -> Path | None:
    """Find run config file in cwd or walking up to project root.

    Searches for: run.toml, run.yaml, run.yml, run.json

    Returns:
        Path to run config file, or None if not found.
    """
    filenames = ["run.toml", "run.yaml", "run.yml", "run.json"]
    for path in [Path.cwd(), *Path.cwd().parents]:
        for filename in filenames:
            run_file = path / filename
            if run_file.exists():
                return run_file
        # Stop at project root
        if (path / "pyproject.toml").exists():
            break
    return None


def _load_config_file(config_path: Path) -> dict[str, Any]:
    """Load configuration file (TOML, YAML, or JSON).

    Args:
        config_path: Path to config file.

    Returns:
        Dictionary of profile name -> profile settings.
    """
    suffix = config_path.suffix.lower()

    if suffix == ".toml":
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib
        with open(config_path, "rb") as f:
            return tomllib.load(f)

    elif suffix in (".yaml", ".yml"):
        import yaml

        with open(config_path) as f:
            return yaml.safe_load(f) or {}

    elif suffix == ".json":
        import json

        with open(config_path) as f:
            return json.load(f)

    else:
        raise ValueError(f"Unsupported config format: {suffix}")


def _resolve_profile(name: str, all_profiles: dict[str, Any], seen: set[str]) -> RunConfig:
    """Recursively resolve profile with extends inheritance.

    Args:
        name: Profile name to resolve.
        all_profiles: All profiles from config file.
        seen: Set of already visited profiles (for cycle detection).

    Returns:
        Resolved RunConfig instance.

    Raises:
        ValueError: If profile not found or circular inheritance detected.
    """
    if name in seen:
        raise ValueError(f"Circular inheritance detected: {name}")
    seen.add(name)

    if name not in all_profiles:
        raise ValueError(f"Profile '{name}' not found in run config")

    profile = all_profiles[name].copy()
    extends = profile.pop("extends", None)

    if extends:
        # Recursively resolve parent profile
        parent = _resolve_profile(extends, all_profiles, seen)
        # Merge: parent values as base, child values override
        parent_dict = {k: v for k, v in vars(parent).items() if not k.startswith("_")}
        merged = {**parent_dict, **profile}
        return RunConfig(**merged)

    return RunConfig(**profile)


def load_run_profile(name: str, config_path: Path | None = None) -> RunConfig:
    """Load a named profile from run config file.

    Args:
        name: Profile name to load.
        config_path: Optional explicit path to config file.

    Returns:
        RunConfig instance with resolved settings.

    Raises:
        FileNotFoundError: If no run config file found.
        ValueError: If profile not found or inheritance error.
    """
    if config_path is None:
        config_path = _find_run_config()
    if config_path is None:
        raise FileNotFoundError("No run config file found (run.toml/yaml/json)")

    all_profiles = _load_config_file(config_path)
    return _resolve_profile(name, all_profiles, seen=set())


def load_wandb_config(config_path: Path | None = None) -> "WandbConfig | None":
    """Load wandb configuration from run.toml [wandb] section.

    The [wandb] section is a top-level section in run.toml that configures
    W&B tracking for all profiles. This allows centralizing wandb settings
    alongside execution profiles.

    Example run.toml:
        [wandb]
        project = "my-project"
        entity = "my-team"
        tags = ["training", "v1"]

        [draco]
        executor = "slurm"
        account = "my-account"

    Args:
        config_path: Optional explicit path to config file.

    Returns:
        WandbConfig instance if [wandb] section exists, None otherwise.
    """
    from nemotron.kit.wandb import WandbConfig

    if config_path is None:
        config_path = _find_run_config()
    if config_path is None:
        return None

    all_sections = _load_config_file(config_path)
    wandb_section = all_sections.get("wandb")

    if wandb_section is None:
        return None

    # Convert tags from list to tuple if present
    if "tags" in wandb_section and isinstance(wandb_section["tags"], list):
        wandb_section["tags"] = tuple(wandb_section["tags"])

    # Map run_name from TOML (snake_case is more natural in TOML)
    if "run_name" in wandb_section:
        pass  # Already correct field name
    elif "name" in wandb_section:
        # Allow shorthand "name" in TOML
        wandb_section["run_name"] = wandb_section.pop("name")

    return WandbConfig(**wandb_section)


def run_with_nemo_run(
    script_path: str,
    script_args: list[str],
    run_config: RunConfig,
    ray: bool = False,
    pre_ray_start_commands: list[str] | None = None,
) -> int:
    """Execute script via nemo-run, optionally with Ray.

    Args:
        script_path: Path to Python script to execute.
        script_args: Arguments to pass to the script.
        run_config: Run configuration for executor.
        ray: Whether to use Ray for execution.
        pre_ray_start_commands: Commands to run before Ray starts.

    Returns:
        Exit code (0 = success).
    """
    try:
        import nemo_run as run
    except ImportError:
        sys.stderr.write(
            "[run] ERROR: nemo-run not installed. Install with: pip install nemo-run\n"
        )
        return 1

    # Handle dry-run
    if run_config.dry_run:
        _print_dry_run(script_path, script_args, run_config, ray, pre_ray_start_commands)
        return 0

    # Build executor
    executor = build_executor(run_config)

    if ray:
        # Recipe requires Ray - use RayJob
        ray_job = run.RayJob(executor=executor)
        cmd = f"python {script_path}"
        if script_args:
            cmd += " " + " ".join(script_args)
        ray_job.start(
            command=cmd,
            workdir=run_config.ray_working_dir,
            pre_ray_start_commands=pre_ray_start_commands,
        )
        ray_job.logs(follow=True)
    else:
        # Standard execution via nemo-run Script
        with run.Experiment(run_config.job_name) as exp:
            task = run.Script(path=script_path, args=script_args)
            exp.add(task, executor=executor)
            exp.run(detach=run_config.detach, tail_logs=not run_config.detach)

    return 0


def _print_dry_run(
    script_path: str,
    script_args: list[str],
    run_config: RunConfig,
    ray: bool,
    pre_ray_start_commands: list[str] | None,
) -> None:
    """Print dry-run information."""
    sys.stderr.write("[run] Dry-run mode - would execute:\n")
    sys.stderr.write(f"[run]   Script: {script_path}\n")
    sys.stderr.write(f"[run]   Args: {' '.join(script_args) if script_args else '(none)'}\n")
    sys.stderr.write(f"[run]   Executor: {run_config.executor}\n")
    sys.stderr.write(f"[run]   Nodes: {run_config.nodes}\n")
    sys.stderr.write(f"[run]   GPUs/node: {run_config.nproc_per_node}\n")
    sys.stderr.write(f"[run]   Ray: {ray}\n")

    if run_config.executor == "slurm":
        sys.stderr.write(f"[run]   Account: {run_config.account}\n")
        sys.stderr.write(f"[run]   Partition: {run_config.partition}\n")
        sys.stderr.write(f"[run]   Time: {run_config.time}\n")
        if run_config.container_image:
            sys.stderr.write(f"[run]   Container: {run_config.container_image}\n")
        if run_config.tunnel == "ssh":
            sys.stderr.write(f"[run]   SSH tunnel: {run_config.user}@{run_config.host}\n")

    if run_config.executor == "docker":
        sys.stderr.write(f"[run]   Container: {run_config.container_image}\n")
        sys.stderr.write(f"[run]   Runtime: {run_config.runtime or 'nvidia'}\n")

    if run_config.executor == "skypilot":
        sys.stderr.write(f"[run]   Cloud: {run_config.cloud}\n")
        sys.stderr.write(f"[run]   GPUs: {run_config.gpus}\n")
        if run_config.cluster_name:
            sys.stderr.write(f"[run]   Cluster: {run_config.cluster_name}\n")

    if ray and pre_ray_start_commands:
        sys.stderr.write(f"[run]   Pre-Ray commands: {pre_ray_start_commands}\n")

    sys.stderr.flush()
