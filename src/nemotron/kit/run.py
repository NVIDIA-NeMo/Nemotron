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

Wandb configuration can also be stored in env.toml:
    >>> # env.toml
    >>> # [wandb]
    >>> # project = "my-project"
    >>> # entity = "my-team"
    >>>
    >>> from nemotron.kit.run import load_wandb_config
    >>> wandb_config = load_wandb_config()  # Returns WandbConfig or None
"""

from __future__ import annotations

import sys
import time
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
        run_partition: Partition to use for attached execution (--run), overrides partition
        batch_partition: Partition to use for detached execution (--batch), overrides partition
        time: Slurm job time limit (HH:MM:SS)
        job_name: Slurm job name
        ntasks_per_node: Slurm tasks per node
        gpus_per_node: Slurm GPUs per node
        mem: Slurm memory request (e.g., '0' for all, '64G')
        exclusive: Request exclusive node access
        cpus_per_task: Slurm CPUs per task
        cpus_per_gpu: Slurm CPUs per GPU
        gpus_per_task: Slurm GPUs per task
        mem_per_gpu: Slurm memory per GPU (e.g., '32G')
        mem_per_cpu: Slurm memory per CPU (e.g., '4G')
        qos: Slurm quality of service
        constraint: Slurm node constraints (e.g., 'a100')
        exclude: Slurm nodes to exclude
        gres: Slurm generic resources
        array: Slurm job array specification

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
        ray_mode: Ray execution mode - "job" (ephemeral, auto-terminates) or
            "cluster" (persistent, for interactive use)

        env_vars: Environment variables (KEY=VALUE format)
        dry_run: Print commands without executing
        detach: Don't wait for job completion
    """

    # Executor type
    executor: Executor = "local"

    # Common resource settings
    nproc_per_node: int | None = None
    nodes: int | None = None

    # Container settings (docker, slurm, skypilot)
    container_image: str | None = None
    mounts: list[str] = field(default_factory=list)

    # Slurm settings
    account: str | None = None
    partition: str | None = None
    run_partition: str | None = None
    batch_partition: str | None = None
    time: str = "04:00:00"
    job_name: str = "nemo-run"
    ntasks_per_node: int | None = None
    gpus_per_node: int | None = None
    mem: str | None = None
    exclusive: bool | None = None
    cpus_per_task: int | None = None
    cpus_per_gpu: int | None = None
    gpus_per_task: int | None = None
    mem_per_gpu: str | None = None
    mem_per_cpu: str | None = None
    qos: str | None = None
    constraint: str | None = None
    exclude: str | None = None
    gres: str | None = None
    array: str | None = None

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
    ray_mode: Literal["job", "cluster"] = "job"  # "job" (ephemeral) or "cluster" (persistent)

    # Environment
    env_vars: list[str] = field(default_factory=list)

    # Execution options
    dry_run: bool = False
    detach: bool = False


def resolve_partition(config: RunConfig, is_launch: bool) -> str | None:
    """Resolve the effective partition based on execution mode.

    Selects the appropriate partition based on whether the job is being
    batched (detached) or run (attached):
    - For --batch (detached): use batch_partition if defined, else partition
    - For --run (attached): use run_partition if defined, else partition

    Args:
        config: RunConfig with partition settings.
        is_launch: True for detached execution (--batch), False for attached (--run).

    Returns:
        The effective partition name, or None if no partition is configured.

    Example:
        >>> config = RunConfig(partition="batch", batch_partition="interactive")
        >>> resolve_partition(config, is_launch=False)
        'batch'
        >>> resolve_partition(config, is_launch=True)
        'interactive'
    """
    if is_launch and config.batch_partition is not None:
        return config.batch_partition
    if not is_launch and config.run_partition is not None:
        return config.run_partition
    return config.partition


def patch_nemo_run_ray_template_for_cpu() -> None:
    """Patch nemo-run Ray template to properly handle CPU-only partitions.

    The default nemo_run Ray template hardcodes gpus_per_node=8 and calculates
    CPUs as 16*gpus_per_node, which results in 0 CPUs for CPU-only partitions.

    This patch modifies the template location to use our custom template that
    auto-detects CPUs from SLURM environment variables.
    """
    import os
    import tempfile
    from pathlib import Path

    try:
        # Use 'from ... import' syntax to avoid issues with 'run' being shadowed
        # by the nemo_run.run function when using 'import nemo_run.run.ray.slurm'
        from nemo_run.run.ray import slurm as slurm_mod
    except Exception:
        return

    if getattr(slurm_mod, "_nemotron_cpu_template_patched", False):
        return

    # Get the path to our custom template
    custom_template_dir = Path(__file__).parent / "templates"
    custom_template_name = "ray_cpu.sub.j2"

    # Check if our custom template exists
    template_path = custom_template_dir / custom_template_name
    if not template_path.exists():
        return

    def patched_create(
        self,
        pre_ray_start_commands=None,
        dryrun=False,
        command=None,
        workdir=None,
        command_groups=None,
    ):
        """Patched create that uses custom CPU-aware Ray template."""
        name = self.name
        executor = self.executor
        cluster_dir = os.path.join(executor.tunnel.job_dir, name)

        # Use custom template for CPU-aware Ray cluster
        ray_sbatch = slurm_mod.SlurmRayRequest(
            name=name,
            cluster_dir=cluster_dir,
            template_name=custom_template_name,
            template_dir=str(custom_template_dir),
            executor=executor,
            pre_ray_start_commands=pre_ray_start_commands,
            command=command,
            workdir=workdir,
            command_groups=command_groups,
            launch_cmd=["sbatch", "--requeue", "--parsable", "--dependency=singleton"],
        ).materialize()

        if dryrun:
            slurm_mod.logger.debug(f"Dry run: Ray cluster '{name}'")
            print(ray_sbatch)
            return None

        slurm_mod.logger.info(f"Creating Ray cluster '{name}'")
        # Check if a cluster with this name already exists
        status = self.status()

        if status["job_id"] is not None:
            job_state = status["state"]
            if job_state in ["PENDING", "RUNNING", "CONFIGURING"]:
                slurm_mod.logger.debug(
                    f"Ray cluster '{name}' already exists with ID {status['job_id']} "
                    f"and is currently in {job_state} state. "
                    f"Skipping creation."
                )
                return None
            elif job_state not in [
                "COMPLETING",
                "COMPLETED",
                "CANCELLED",
                "FAILED",
                "TIMEOUT",
                "NOT_FOUND",
            ]:
                slurm_mod.logger.warning(
                    f"Ray cluster '{name}' exists with ID {status['job_id']} "
                    f"in state {job_state}. Creating new cluster anyway."
                )

        # Submit to SLURM - same logic as original nemo-run
        executor.tunnel.connect()
        executor.tunnel.run(f"mkdir -p {cluster_dir}")

        with tempfile.NamedTemporaryFile(mode="w", delete=True) as f:
            f.write(ray_sbatch)
            f.flush()
            os.fsync(f.fileno())
            ray_sbatch_path = f.name
            executor.tunnel.put(ray_sbatch_path, os.path.join(cluster_dir, "ray.sub"))

        sbatch_cmd = ["sbatch", "--parsable", os.path.join(cluster_dir, "ray.sub")]
        job_id = executor.tunnel.run(" ".join(sbatch_cmd)).stdout.strip()

        # Store job_id in cluster_map
        self.cluster_map[name] = job_id

        slurm_mod.logger.info(f"Slurm job for Ray cluster '{name}' created with ID {job_id}")

        return job_id

    slurm_mod.SlurmRayCluster.create = patched_create
    slurm_mod._nemotron_cpu_template_patched = True


def patch_nemo_run_rsync_accept_new_host_keys() -> None:
    """Patch nemo-run rsync to avoid hanging on first-time host key prompts.

    nemo-run's SSH tunnel uses Paramiko for its control connection, but the
    rsync step shells out to the system `ssh`, which can block waiting for an
    interactive StrictHostKeyChecking prompt.

    We set `StrictHostKeyChecking=accept-new` unless the caller already
    provided a StrictHostKeyChecking option.
    """

    try:
        import nemo_run.core.tunnel.rsync as rsync_mod
    except Exception:
        return

    if getattr(rsync_mod.rsync, "_nemotron_patched", False):
        return

    orig = rsync_mod.rsync

    def patched(*args, **kwargs):
        ssh_opts = kwargs.get("ssh_opts", "") or ""
        if "StrictHostKeyChecking" not in ssh_opts:
            ssh_opts = (ssh_opts + " " if ssh_opts else "") + "-o StrictHostKeyChecking=accept-new"
        if "BatchMode" not in ssh_opts:
            ssh_opts = (ssh_opts + " " if ssh_opts else "") + "-o BatchMode=yes"
        if "PreferredAuthentications" not in ssh_opts:
            ssh_opts = (ssh_opts + " " if ssh_opts else "") + (
                "-o PreferredAuthentications=publickey"
            )
        if "ConnectTimeout" not in ssh_opts:
            ssh_opts = (ssh_opts + " " if ssh_opts else "") + "-o ConnectTimeout=30"
        kwargs["ssh_opts"] = ssh_opts

        rsync_opts = kwargs.get("rsync_opts", "") or ""
        # Note: --info=progress2 removed because older rsync versions on some clusters don't support it
        if "--timeout" not in rsync_opts:
            rsync_opts = (rsync_opts + " " if rsync_opts else "") + "--timeout=60"
        # Use --delete for faster incremental syncs (removes stale files on remote)
        if "--delete" not in rsync_opts:
            rsync_opts = (rsync_opts + " " if rsync_opts else "") + "--delete"
        kwargs["rsync_opts"] = rsync_opts

        # Default exclusions for our repo (avoid syncing large non-runtime dirs).
        # Users can override by passing `exclude=...` explicitly.
        # Note: Use patterns anchored at root (e.g., "/artifacts") to avoid
        # excluding source directories like src/nemotron/kit/artifacts.
        kwargs.setdefault(
            "exclude",
            (
                ".git",
                ".venv",
                "__pycache__",
                ".ruff_cache",
                ".pytest_cache",
                ".mypy_cache",
                ".nemotron",
                ".conductor",
                "/output",
                "/outputs",
                "/artifacts",
                "/wandb",
                "usage-cookbook",
                "use-case-examples",
            ),
        )

        # Show progress/errors instead of looking hung.
        kwargs.setdefault("hide_output", False)

        return orig(*args, **kwargs)

    patched._nemotron_patched = True  # type: ignore[attr-defined]
    rsync_mod.rsync = patched  # type: ignore[assignment]

    # Patch already-imported call sites that `from ... import rsync`.
    try:
        import nemo_run.run.experiment as exp

        exp.rsync = patched  # type: ignore[assignment]
    except Exception:
        pass

    try:
        import nemo_run.run.ray.slurm as slurm

        slurm.rsync = patched  # type: ignore[assignment]
    except Exception:
        pass


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

    patch_nemo_run_rsync_accept_new_host_keys()
    patch_nemo_run_ray_template_for_cpu()

    # Parse and merge environment variables
    merged_env = {}
    for env in config.env_vars:
        if "=" in env:
            key, value = env.split("=", 1)
            merged_env[key] = value
    if env_vars:
        merged_env.update(env_vars)

    # Auto-detect HuggingFace token if not already set
    if "HF_TOKEN" not in merged_env:
        try:
            from huggingface_hub import HfFolder

            token = HfFolder.get_token()
            if token:
                merged_env["HF_TOKEN"] = token
                sys.stderr.write(
                    "[info] Detected HuggingFace login, adding HF_TOKEN to environment\n"
                )
        except Exception:
            pass  # huggingface_hub not installed or no token

    # Auto-detect Weights & Biases API key if not already set
    if "WANDB_API_KEY" not in merged_env:
        try:
            import wandb

            api_key = wandb.api.api_key
            if api_key:
                merged_env["WANDB_API_KEY"] = api_key
                sys.stderr.write("[info] Detected W&B login, adding WANDB_API_KEY to environment\n")
        except Exception:
            pass  # wandb not installed or not logged in

    # Auto-detect Weights & Biases project/entity from env.toml [wandb] section
    wandb_config = load_wandb_config()
    if wandb_config is not None:
        if wandb_config.project and "WANDB_PROJECT" not in merged_env:
            merged_env["WANDB_PROJECT"] = wandb_config.project
        if wandb_config.entity and "WANDB_ENTITY" not in merged_env:
            merged_env["WANDB_ENTITY"] = wandb_config.entity

    # NOTE: PYTHONPATH and NEMO_RUN_DIR are NOT set here for Ray jobs.
    # Ray jobs use nemo-run's SlurmRayJob which rsyncs to {cluster_dir}/code
    # but does NOT mount to /nemo_run. The workdir is set to the actual path.
    # We handle this in run_with_nemo_run() by using relative paths.

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
            packager = _build_packager()

            # Build container mounts, adding /lustre and Ray temp directory
            mounts = list(config.mounts)
            # Mount /lustre for access to shared storage (HF cache, data, etc.)
            mounts.append("/lustre:/lustre")
            if config.remote_job_dir:
                # Ray temp directory mount (avoids filling container storage with Ray logs)
                ray_temp_path = f"{config.remote_job_dir}/ray_temp"
                mounts.append(f"{ray_temp_path}:/ray-cluster")

            return run.SlurmExecutor(
                account=config.account,
                partition=config.partition,
                nodes=config.nodes,
                ntasks_per_node=config.ntasks_per_node,
                gpus_per_node=config.gpus_per_node,
                time=config.time,
                mem=config.mem,
                exclusive=config.exclusive,
                cpus_per_task=config.cpus_per_task,
                cpus_per_gpu=config.cpus_per_gpu,
                gpus_per_task=config.gpus_per_task,
                mem_per_gpu=config.mem_per_gpu,
                mem_per_cpu=config.mem_per_cpu,
                qos=config.qos,
                constraint=config.constraint,
                exclude=config.exclude,
                gres=config.gres,
                array=config.array,
                container_image=config.container_image,
                container_mounts=mounts,
                tunnel=tunnel,
                packager=packager,
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


def _build_packager() -> Any:
    """Build a HybridPackager for selective file syncing.

    Packages only the necessary files for remote cluster sync:
    - src/ directory: only .py, .json, .jinja, .yaml, .typed files (excludes __pycache__)
    - Top-level files: pyproject.toml, env.toml

    This avoids packaging:
    - __pycache__/ directories with stale .pyc bytecode
    - Large unnecessary directories like usage-cookbook/, use-case-examples/
    - tests/ directory (not needed for execution)

    Returns:
        A HybridPackager instance configured for selective syncing.
    """
    from nemo_run.core.packaging import HybridPackager, PatternPackager

    return HybridPackager(
        extract_at_root=True,
        sub_packagers={
            # Package src/ with only source files, excluding __pycache__
            "src_py": PatternPackager(
                include_pattern='src -name "*.py"',
                relative_path=".",
            ),
            "src_json": PatternPackager(
                include_pattern='src -name "*.json"',
                relative_path=".",
            ),
            "src_jinja": PatternPackager(
                include_pattern='src -name "*.jinja"',
                relative_path=".",
            ),
            "src_yaml": PatternPackager(
                include_pattern='src -name "*.yaml"',
                relative_path=".",
            ),
            "src_typed": PatternPackager(
                include_pattern='src -name "py.typed"',
                relative_path=".",
            ),
            "pyproject": PatternPackager(
                include_pattern="pyproject.toml",
                relative_path=".",
            ),
            "env_toml": PatternPackager(
                include_pattern="env.toml",
                relative_path=".",
            ),
        },
    )


def _find_env_toml() -> Path | None:
    """Find env.toml in cwd or walking up to project root."""
    for path in [Path.cwd(), *Path.cwd().parents]:
        env_file = path / "env.toml"
        if env_file.exists():
            return env_file
        # Stop at project root
        if (path / "pyproject.toml").exists():
            break
    return None


def _load_env_toml(config_path: Path) -> dict[str, Any]:
    """Load env.toml configuration.

    Args:
        config_path: Path to config file.

    Returns:
        Dictionary of profile name -> profile settings.
    """
    if config_path.suffix.lower() != ".toml":
        raise ValueError(f"Expected a .toml config file, got: {config_path}")

    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    with open(config_path, "rb") as f:
        return tomllib.load(f)


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
        raise ValueError(f"Profile '{name}' not found in env.toml")

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
    """Load a named profile from env.toml.

    Args:
        name: Profile name to load.
        config_path: Optional explicit path to config file.

    Returns:
        RunConfig instance with resolved settings.

    Raises:
        FileNotFoundError: If no env.toml found.
        ValueError: If profile not found or inheritance error.
    """
    if config_path is None:
        config_path = _find_env_toml()
    if config_path is None:
        raise FileNotFoundError("No env.toml found")

    all_profiles = _load_env_toml(config_path)
    return _resolve_profile(name, all_profiles, seen=set())


def list_run_profiles(config_path: Path | None = None) -> list[str]:
    """List available run profiles from env.toml.

    Profiles are top-level sections in env.toml, excluding the special [wandb] section.

    Args:
        config_path: Optional explicit path to config file.

    Returns:
        Sorted list of profile names.
    """
    if config_path is None:
        config_path = _find_env_toml()
    if config_path is None:
        return []

    sections = _load_env_toml(config_path)
    profiles = [k for k in sections.keys() if k != "wandb"]
    return sorted(profiles)


def load_wandb_config(config_path: Path | None = None) -> WandbConfig | None:
    """Load wandb configuration from env.toml [wandb] section.

    The [wandb] section is a top-level section in env.toml that configures W&B tracking
    for all profiles. This allows centralizing wandb settings alongside execution profiles.

    Example env.toml:
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
        config_path = _find_env_toml()
    if config_path is None:
        return None

    all_sections = _load_env_toml(config_path)
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
    packager: str = "code",
    workdir: str | None = None,
    run_command: str | None = None,
) -> int:
    """Execute script via nemo-run, optionally with Ray.

    Args:
        script_path: Path to Python script to execute.
        script_args: Arguments to pass to the script.
        run_config: Run configuration for executor.
        ray: Whether to use Ray for execution.
        pre_ray_start_commands: Commands to run before Ray starts.
        packager: Packager type ("code", "self_contained", "pattern").
        workdir: Container working directory for Ray jobs (e.g., "/opt/nemo-rl").
        run_command: Custom command template (supports {script} and {config} placeholders).

    Returns:
        Exit code (0 = success).
    """
    try:
        import nemo_run as run
        from nemo_run.run.ray.job import RayJob
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
        import tempfile

        import yaml

        # Recipe requires Ray - use RayJob
        # Generate unique job name to prevent directory collisions
        # nemo-run's SlurmRayJob uses cluster_dir = tunnel.job_dir + name,
        # so jobs with the same name would overwrite each other's directories
        base_name = run_config.job_name or Path(script_path).stem
        job_name = f"{base_name}_{int(time.time())}"
        ray_job = RayJob(name=job_name, executor=executor)

        # Log the ray mode
        mode_desc = "ephemeral" if run_config.ray_mode == "job" else "persistent"
        sys.stderr.write(f"[run] Ray mode: {run_config.ray_mode} ({mode_desc} cluster)\n")

        # Build log clearing command to prevent old logs from appearing in output.
        # nemo-run reuses Ray clusters and appends to existing log files, so we
        # truncate the log file before starting to ensure clean output.
        log_clear_cmd = None
        if run_config.remote_job_dir:
            log_file = f"{run_config.remote_job_dir}/{job_name}/logs/ray-job.log"
            log_clear_cmd = f": > {log_file} 2>/dev/null || true"

        # Build setup commands based on packager type and workdir
        if pre_ray_start_commands is not None:
            # Use explicitly provided commands
            setup_commands = list(pre_ray_start_commands)
        elif packager == "self_contained":
            # For self_contained packager, skip uv sync (dependencies in container)
            setup_commands = [
                "find . -type d -name __pycache__ -delete 2>/dev/null || true",
            ]
        else:
            # Default setup for other packagers
            setup_commands = [
                "find . -type d -name __pycache__ -delete 2>/dev/null || true",
                "uv sync --reinstall-package nemotron",
            ]

        # For self_contained packager, copy files from /nemo_run/code to workdir
        # and use main.py instead of the original script path
        if packager == "self_contained":
            if workdir:
                # Copy files to workdir and cd there
                setup_commands.extend(
                    [
                        f"cp /nemo_run/code/main.py {workdir}/",
                        f"cp /nemo_run/code/config.yaml {workdir}/",
                        f"cd {workdir}",
                    ]
                )
            else:
                # Copy files to current directory
                setup_commands.extend(
                    [
                        "cp /nemo_run/code/main.py .",
                        "cp /nemo_run/code/config.yaml .",
                    ]
                )
            remote_script = "main.py"
        else:
            remote_script = script_path

        # Prepend log clearing if remote_job_dir is configured
        if log_clear_cmd:
            setup_commands.insert(0, log_clear_cmd)

        # Build the command to run
        # For self_contained packager, config is at config.yaml; otherwise use script_args
        config_file = "config.yaml" if packager == "self_contained" else None
        if run_command:
            # Use custom run command template with placeholders
            cmd = run_command.format(script=remote_script, config=config_file or "")
            # Prepend cd to workdir if set (pre_ray_start_commands run before Ray, not in job)
            if workdir:
                cmd = f"cd {workdir} && {cmd}"
        elif workdir and packager == "self_contained":
            cmd = f"cd {workdir} && python {remote_script}"
        elif workdir:
            cmd = f"cd {workdir} && uv run python {remote_script}"
        else:
            cmd = f"uv run python {remote_script}"
        if script_args:
            cmd += " " + " ".join(script_args)

        # Build runtime_env with environment variables for Ray workers
        # This ensures env vars like HF_TOKEN are available in Ray tasks/actors
        runtime_env: dict = {"env_vars": {}}

        # Auto-detect HuggingFace token for Ray workers
        try:
            from huggingface_hub import HfFolder

            hf_token = HfFolder.get_token()
            if hf_token:
                runtime_env["env_vars"]["HF_TOKEN"] = hf_token
        except Exception:
            pass

        # Set HF_HOME for persistent dataset caching on Lustre
        # Priority: env var > remote_job_dir/hf
        if os.environ.get("HF_HOME"):
            runtime_env["env_vars"]["HF_HOME"] = os.environ["HF_HOME"]
        elif run_config.remote_job_dir:
            runtime_env["env_vars"]["HF_HOME"] = f"{run_config.remote_job_dir}/hf"

        # Auto-detect Weights & Biases API key for Ray workers
        try:
            import wandb

            wandb_api_key = wandb.api.api_key
            if wandb_api_key:
                runtime_env["env_vars"]["WANDB_API_KEY"] = wandb_api_key
        except Exception:
            pass

        # Load wandb project/entity from env.toml [wandb] section
        wandb_config = load_wandb_config()
        if wandb_config is not None:
            if wandb_config.project:
                runtime_env["env_vars"]["WANDB_PROJECT"] = wandb_config.project
            if wandb_config.entity:
                runtime_env["env_vars"]["WANDB_ENTITY"] = wandb_config.entity

        # Create temporary runtime_env YAML file if we have env vars to pass
        runtime_env_yaml = None
        if runtime_env["env_vars"]:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump(runtime_env, f)
                runtime_env_yaml = f.name

        ray_workdir = run_config.ray_working_dir
        if not ray_workdir:
            # For Ray jobs, prefer nemo-run's native workdir rsync (respects .gitignore)
            # instead of the packager->tar->extract->rsync path.
            ray_workdir = "."

        # Display job submission summary
        from nemotron.kit.cli.display import display_ray_job_submission

        display_ray_job_submission(
            script_path=script_path,
            script_args=script_args or [],
            env_vars=runtime_env.get("env_vars", {}),
            mode="detached" if run_config.detach else "attached",
        )

        ray_job.start(
            command=cmd,
            workdir=ray_workdir,
            pre_ray_start_commands=setup_commands,
            runtime_env_yaml=runtime_env_yaml,
        )

        # Workaround for nemo-run bug: when reusing an existing cluster,
        # SlurmRayCluster.create() returns None instead of the job_id.
        # Fix by querying the backend status which has the actual job_id.
        if ray_job.backend.job_id is None:
            status = ray_job.backend.status(display=False)
            if status and status.get("job_id"):
                ray_job.backend.job_id = status["job_id"]
                sys.stderr.write(
                    f"[info] Recovered job_id {status['job_id']} from cluster status\n"
                )

        if not run_config.detach:
            try:
                # Wait up to 10 minutes for log file to appear (Slurm jobs can be slow to start)
                ray_job.logs(follow=True, timeout=600)
            except KeyboardInterrupt:
                if run_config.ray_mode == "cluster":
                    # In cluster mode, keep the cluster running for subsequent jobs
                    sys.stderr.write(
                        "\n[info] Ctrl-C detected. Cluster mode: leaving Ray cluster running.\n"
                        "[info] Use 'scancel <job_id>' to stop the cluster manually.\n"
                    )
                else:
                    # In job mode, stop the cluster
                    sys.stderr.write("\n[info] Ctrl-C detected, stopping Ray cluster...\n")
                    try:
                        ray_job.stop()
                        sys.stderr.write("[info] Ray cluster stopped\n")
                    except Exception as e:
                        sys.stderr.write(f"[warning] Failed to stop Ray cluster: {e}\n")
                raise
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
