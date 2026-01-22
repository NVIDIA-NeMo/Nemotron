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

"""Data preparation for Megatron training.

Processes raw text data from HuggingFace, S3, or local sources into
various training formats compatible with Megatron-Bridge and Megatron-Core.

Supported output formats:
- **binidx**: Tokenized .bin/.idx files (default, for pretraining)
- **jsonl**: JSONL files with optional transforms (for SFT/RL, no tokenization)
- **packed**: Packed sequences in .npy format (for efficient SFT training)

Quick Start:
    from nemotron.data_prep import DataPrepConfig, run_data_prep
    from pathlib import Path

    # Create config
    config = DataPrepConfig(
        blend_path=Path("data_blend.json"),
        output_dir=Path("./output"),
        tokenizer_model="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    )

    # Run data preparation
    artifact = run_data_prep(config)

    # Use output with Megatron-Bridge
    print(f"Blend path: {artifact.path}")

Low-Level API (last_mile_process):
    from nemotron.data_prep import last_mile_process, DataBlend, PipelineConfig
    from nemotron.data_prep.config import OutputConfig, JsonlOutputConfig
    from nemotron.data_prep.formats.transforms import sft

    blend = DataBlend.load("data_blend.json")

    # JSONL output for SFT
    config = PipelineConfig(
        output=OutputConfig(
            dir=Path("./sft_data"),
            format=JsonlOutputConfig(transform=sft(input="instruction", output="response")),
        ),
    )
    result = last_mile_process(blend, config)

Output Format:
    The generated blend.json is directly compatible with Megatron-Bridge's
    get_blend_fields_from_data_paths() function.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from nemotron.data_prep.blend import DataBlend, Dataset
from nemotron.data_prep.config import (
    BinIdxOutputConfig,
    ChatSftOutputConfig,
    JsonlOutputConfig,
    OutputConfig,
    PackedOutputConfig,
    PerSplitConfig,
    PipelineConfig,
    TokenizerConfig,
    Transform,
)
from nemotron.data_prep.discovery import get_dataset_metadata
from nemotron.data_prep.formats.transforms import (
    OpenAIChatRecord,
    SftRecord,
    ShareGPTRecord,
    openai_chat,
    passthrough,
    rename,
    select,
    sft,
    sharegpt,
)
from nemotron.data_prep.pipeline import (
    PipelineResult,
    SplitResult,
    last_mile_process,
    tokenize,
)
from nemotron.kit.artifact import DataBlendsArtifact, PretrainBlendsArtifact
from nemotron.kit.trackers import InputDatasetInfo, tokenizer_to_uri
from nemotron.kit.wandb import finish_wandb


@dataclass
class DataPrepConfig:
    """Configuration for data preparation.

    Generic configuration that can be customized per-recipe.

    Example:
        >>> from nemotron.data_prep import DataPrepConfig, run_data_prep
        >>> config = DataPrepConfig(
        ...     blend_path=Path("data_blend.json"),
        ...     output_dir=Path("./output"),
        ...     tokenizer_model="meta-llama/Llama-3.2-1B",
        ... )
        >>> artifact = run_data_prep(config)
    """

    # Data source
    blend_path: Path = field(default_factory=lambda: Path("data_blend.json"))
    """Path to data blend JSON file"""

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    """Output directory for tokenized data"""

    num_shards: int = 128
    """Number of output shards for parallel loading"""

    split: str | None = None
    """Deprecated: Train:valid:test ratio (e.g., '99990,8,2'). Use per_split instead."""

    per_split: PerSplitConfig | None = field(default_factory=PerSplitConfig)
    """Per-split output config. Produces {"train": [...], "valid": [...], "test": [...]} JSON
    compatible with Megatron-Bridge's per_split_data_args_path parameter.
    Set to None to use legacy split ratio mode."""

    # Tokenizer
    tokenizer_model: str = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    """HuggingFace tokenizer model name"""

    add_bos: bool = False
    """Prepend BOS token to documents"""

    add_eos: bool = True
    """Append EOS token to documents"""

    # Processing
    text_field: str = "text"
    """Default text field name in datasets"""

    min_doc_chars: int | None = None
    """Skip documents shorter than this"""

    max_doc_tokens: int | None = None
    """Truncate documents longer than this"""

    # Execution
    sample: int | None = None
    """Limit rows per dataset (for quick tests)"""

    force: bool = False
    """Force new run, ignoring cache"""

    artifact_name: str | None = None
    """Semantic artifact name (e.g., 'nano3/pretrain/data')"""

    # Ray Data execution
    ray_data_enabled: bool = True
    """Enable Ray Data executor for shard processing.
    Uses Ray Data's ActorPoolStrategy for automatic actor lifecycle
    management, resource accounting, and bottleneck metrics in W&B."""

    ray_data_min_actors: int = 2
    """Minimum actors for Ray Data executor (warm pool)"""

    ray_data_max_actors: int | None = None
    """Maximum actors for Ray Data executor (None = use all available CPUs)"""

    ray_data_cpus_per_actor: float = 1.0
    """CPUs per actor for Ray Data executor"""

    ray_data_max_tasks_in_flight: int = 4
    """Max tasks in flight per actor (pipelining depth for better I/O overlap)"""

    max_concurrent_downloads: int = 64
    """Maximum parallel HuggingFace file downloads during pre-download phase.
    Higher values increase throughput but may overwhelm HF servers or network."""

    cleanup_hf_cache: bool = False
    """Delete HuggingFace cache after processing. Useful for one-off jobs."""

    console_mode: Literal["rich", "simple"] = "simple"
    """Console output mode: 'rich' for animated progress bars, 'simple' for periodic text updates"""

    simple_log_interval_sec: int = 30
    """Interval in seconds between status updates in simple console mode (default: 30)"""

    execution_engine: Literal["ray", "xenna"] = "ray"
    """Execution backend for shard processing."""

    wandb_log_downloads: bool = False
    """Log download progress metrics to W&B (Xenna path only)."""

    wandb_download_log_interval_sec: int = 30
    """Interval (seconds) for W&B download progress logging."""

    hf_download_timeout_sec: int = 300
    """Per-file HF download timeout in seconds (Xenna path only)."""

    hf_download_max_retries: int = 3
    """Max retries for HF downloads before giving up (Xenna path only)."""


def _ensure_driver_hf_home() -> None:
    """Ensure HF_HOME is set for the driver process.

    In nemo-run Ray job mode, runtime_env_yaml env_vars apply to Ray workers,
    but not to the driver script. This function derives HF_HOME from NEMO_RUN_DIR
    (which IS set for the driver) so that:
    1. The driver's HF cache goes to shared storage (e.g., Lustre)
    2. The value propagates to Ray workers via ray.init(runtime_env=...)
    3. Xenna stages get it via _get_hf_runtime_env() -> env_info -> actor_pool

    This prevents "No space left on device" errors from HF downloads filling
    local node storage instead of shared Lustre.
    """
    if os.environ.get("HF_HOME"):
        return  # Already set, respect user's explicit setting

    nemo_run_dir = os.environ.get("NEMO_RUN_DIR")
    if not nemo_run_dir:
        return  # Not running via nemo-run, fall back to HF defaults

    # Use same convention as nemo-run's worker-side: <job_dir>/hf
    hf_home = str(Path(nemo_run_dir) / "hf")
    os.environ["HF_HOME"] = hf_home


def run_data_prep(
    config: DataPrepConfig, *, artifact_class: type = PretrainBlendsArtifact
) -> DataBlendsArtifact | PretrainBlendsArtifact:
    """Execute data preparation pipeline.

    Loads the data blend, tokenizes all datasets, and produces a
    Megatron-Bridge compatible blend.json.

    Args:
        config: Data preparation configuration
        artifact_class: Artifact class to use for output (default: PretrainDataArtifact)

    Returns:
        Artifact instance with blend.json path and metrics

    Example:
        >>> from nemotron.data_prep import DataPrepConfig, run_data_prep
        >>> config = DataPrepConfig(
        ...     blend_path=Path("data_blend.json"),
        ...     output_dir=Path("./output"),
        ... )
        >>> artifact = run_data_prep(config)
        >>> print(f"Blend path: {artifact.path}")
    """
    # Ensure HF_HOME is set for the driver process early.
    # In nemo-run Ray job mode, runtime_env_yaml env_vars apply to Ray workers only,
    # not the driver. We derive HF_HOME from NEMO_RUN_DIR (which IS set for the driver)
    # so that all downstream code sees a consistent cache directory on shared storage.
    _ensure_driver_hf_home()

    # Load data blend specification
    blend = DataBlend.load(config.blend_path)

    # Apply default text_field to datasets that use default
    for split_datasets in blend.splits.values():
        for dataset in split_datasets:
            if dataset.text_field == "text" and config.text_field != "text":
                # Use object.__setattr__ since Dataset is a Pydantic model
                object.__setattr__(dataset, "text_field", config.text_field)

    # Build pipeline config
    # When sampling, use 1 shard to get exactly `sample` rows per dataset
    num_shards = config.num_shards
    if config.sample is not None:
        num_shards = 1

    # Build output config with format that has num_shards
    # When num_shards is specified, clear shard_size to avoid conflict
    output_format = BinIdxOutputConfig(
        num_shards=num_shards,
        shard_size=None if num_shards is not None else "256MB",
    )

    # Resolve output_dir to absolute path for W&B artifact storage
    output_dir = config.output_dir.resolve() if hasattr(config.output_dir, 'resolve') else Path(config.output_dir).resolve()

    # Initialize Ray for download tasks (Xenna and Ray executors both use Ray)
    if config.execution_engine in ("ray", "xenna"):
        # Enable uv integration for Ray workers (Ray 2.43+)
        # Must be set BEFORE importing ray
        os.environ.setdefault("RAY_RUNTIME_ENV_HOOK", "ray._private.runtime_env.uv_runtime_env_hook.hook")
        import ray

        if not ray.is_initialized():
            runtime_env = {
                "excludes": [
                    "output/",
                    "outputs/",
                    "wandb/",
                    "data/",
                    "checkpoints/",
                    "*.bin",
                    "*.idx",
                    "*.npy",
                    "__pycache__/",
                    ".git/",
                    ".venv/",
                    "*.egg-info/",
                ],
                "env_vars": {},
            }
            # Pass HF_HOME to Ray actors for persistent dataset caching on Lustre
            if os.environ.get("HF_HOME"):
                runtime_env["env_vars"]["HF_HOME"] = os.environ["HF_HOME"]
            # Pass HF_TOKEN for private dataset access
            if os.environ.get("HF_TOKEN"):
                runtime_env["env_vars"]["HF_TOKEN"] = os.environ["HF_TOKEN"]

            # Set environment variables required by cosmos-xenna monitoring
            # These must be set BEFORE ray.init()
            os.environ.setdefault("RAY_MAX_LIMIT_FROM_API_SERVER", "40000")
            os.environ.setdefault("RAY_MAX_LIMIT_FROM_DATA_SOURCE", "40000")

            # Try connecting to existing cluster, fall back to local mode
            # include_dashboard=True is required for cosmos-xenna's State API monitoring
            try:
                ray.init(
                    address="auto",
                    ignore_reinit_error=True,
                    runtime_env=runtime_env,
                    include_dashboard=True,
                )
            except ConnectionError:
                # No cluster found - start Ray locally
                ray.init(
                    ignore_reinit_error=True,
                    runtime_env=runtime_env,
                    include_dashboard=True,
                )

    # Build Ray Data config if enabled, auto-detecting cluster resources
    ray_data_config = None
    if config.execution_engine == "ray" and config.ray_data_enabled:
        from nemotron.data_prep.config import RayDataConfig

        # Auto-detect available CPUs from Ray cluster
        # Fallback chain: Ray cluster -> SLURM env var -> os.cpu_count()
        cluster_resources = ray.cluster_resources()
        ray_cpus = cluster_resources.get("CPU", 0)
        slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
        os_cpus = os.cpu_count() or 4

        # Use the highest available CPU count (Ray may report fewer due to config issues)
        available_cpus = max(int(ray_cpus), slurm_cpus, os_cpus)

        # CPU-based limit (use 90% of CPUs)
        cpus_per_actor = config.ray_data_cpus_per_actor
        cpu_based_limit = int(available_cpus * 0.9 / cpus_per_actor)

        # Memory-based limit to prevent OOM
        # Each actor loads a tokenizer (~1GB) + needs working memory (~1GB) = ~2GB total
        # Ray's object_store_memory is pre-allocated from system RAM
        # Worker memory = total system RAM - object store - overhead
        ray_memory = cluster_resources.get("memory", 0)
        object_store = cluster_resources.get("object_store_memory", 0)

        if ray_memory > 0 and object_store > 0:
            # Worker memory is roughly: total - object_store - 10% overhead
            total_memory_gb = ray_memory / (1024**3)
            object_store_gb = object_store / (1024**3)
            worker_memory_gb = (total_memory_gb - object_store_gb) * 0.9
            # Estimate: tokenizer (~1-2GB) + working memory (~1GB) = ~3GB per actor
            memory_per_actor_gb = 3.0
            memory_based_limit = max(4, int(worker_memory_gb / memory_per_actor_gb))
        else:
            # Fallback: use conservative 50% of CPUs when memory info unavailable
            memory_based_limit = int(available_cpus * 0.5 / cpus_per_actor)
            total_memory_gb = 0
            object_store_gb = 0
            worker_memory_gb = 0

        # Use the more restrictive limit to prevent OOM
        auto_max_actors = min(cpu_based_limit, memory_based_limit)

        # Apply user override if specified
        if config.ray_data_max_actors is not None:
            max_actors = min(config.ray_data_max_actors, auto_max_actors)
        else:
            max_actors = auto_max_actors
        min_actors = min(config.ray_data_min_actors, max_actors)

        # Log resource detection for debugging
        print(f"Ray cluster resources: {cluster_resources}")
        print(f"CPU detection: Ray={ray_cpus}, SLURM={slurm_cpus}, os={os_cpus} -> using {available_cpus}")
        if ray_memory > 0:
            print(f"Memory detection: total={total_memory_gb:.1f}GB, object_store={object_store_gb:.1f}GB, worker={worker_memory_gb:.1f}GB")
        print(f"Actor limits: CPU-based={cpu_based_limit}, Memory-based={memory_based_limit}")
        print(f"Ray Data config: min_actors={min_actors}, max_actors={max_actors}")

        # Log W&B status for debugging
        try:
            import wandb
            if wandb.run is not None:
                print(f"[W&B] Active run: {wandb.run.name} (id={wandb.run.id})")
            else:
                print("[W&B] No active run - metrics will not be logged")
        except ImportError:
            print("[W&B] wandb not installed")

        ray_data_config = RayDataConfig(
            enabled=True,
            min_actors=min_actors,
            max_actors=max_actors,
            cpus_per_actor=cpus_per_actor,
            max_tasks_in_flight_per_actor=config.ray_data_max_tasks_in_flight,
            max_concurrent_downloads=config.max_concurrent_downloads,
            cleanup_hf_cache=config.cleanup_hf_cache,
        )

    pipeline_config = PipelineConfig(
        output=OutputConfig(
            dir=output_dir,
            format=output_format,
            min_doc_chars=config.min_doc_chars,
            max_doc_tokens=config.max_doc_tokens,
            max_rows=config.sample,
        ),
        tokenizer=TokenizerConfig(
            model=config.tokenizer_model,
            add_bos=config.add_bos,
            add_eos=config.add_eos,
        ),
        force=config.force,
        split=config.split,
        per_split=config.per_split,
        ray_data=ray_data_config,
        console_mode=config.console_mode,
        simple_log_interval_sec=config.simple_log_interval_sec,
        execution_engine=config.execution_engine,
        max_concurrent_downloads=config.max_concurrent_downloads,
        wandb_log_downloads=config.wandb_log_downloads,
        wandb_download_log_interval_sec=config.wandb_download_log_interval_sec,
        hf_download_timeout_sec=config.hf_download_timeout_sec,
        hf_download_max_retries=config.hf_download_max_retries,
    )

    # Run processing pipeline
    result = last_mile_process(blend, pipeline_config)

    # Collect source datasets with metadata for lineage tracking
    source_datasets: list[InputDatasetInfo] = []
    seen_keys: set[str] = set()
    for split_datasets in blend.splits.values():
        for dataset in split_datasets:
            # Use path+subset as key since same path can have different subsets
            key = f"{dataset.path}|{dataset.subset or ''}"
            if key not in seen_keys:
                seen_keys.add(key)
                # Build dataset config for metadata fetching
                from nemotron.data_prep.config import DatasetConfig

                ds_config = DatasetConfig(
                    name=dataset.name,
                    path=dataset.path,
                    split=dataset.split,
                    subset=dataset.subset,
                    text_field=dataset.text_field,
                )
                hf_metadata = get_dataset_metadata(ds_config)
                source_datasets.append(
                    InputDatasetInfo(
                        uri=dataset.path,
                        name=dataset.name,
                        weight=dataset.weight,
                        split=dataset.split,
                        subset=dataset.subset,
                        text_field=dataset.text_field,
                        num_rows=hf_metadata.num_rows,
                        size_bytes=hf_metadata.size_bytes,
                    )
                )

    # Create tokenizer URI for lineage tracking
    tok_uri = tokenizer_to_uri(config.tokenizer_model)

    # Build output artifact - path points to output directory, blend_path to blend.json
    blend_json_path = result.output_dir / "blend.json"
    artifact = artifact_class(
        path=result.output_dir,
        blend_path=str(blend_json_path),
        total_tokens=result.total_tokens,
        total_sequences=result.total_sequences,
        elapsed_sec=result.elapsed_sec,
        num_shards=num_shards,
        source_datasets=source_datasets,
        tokenizer_uri=tok_uri,
        name=config.artifact_name,  # Semantic name for W&B artifact naming
    )
    artifact.save()

    # Cleanup HuggingFace cache if requested
    if config.cleanup_hf_cache:
        hf_home = os.environ.get("HF_HOME")
        if hf_home and os.path.isdir(hf_home):
            import shutil

            print(f"Cleaning up HF cache: {hf_home}")
            try:
                shutil.rmtree(hf_home)
                print(f"HF cache deleted: {hf_home}")
            except Exception as e:
                print(f"Failed to delete HF cache: {e}")

    # Mark wandb run as successful (before Ray shutdown to avoid socket noise)
    finish_wandb(exit_code=0)

    # Gracefully shutdown Ray - suppress stderr during cleanup
    try:
        import ray

        if ray.is_initialized():
            import io
            import sys

            # Temporarily suppress stderr during Ray shutdown (socket cleanup noise)
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                ray.shutdown()
            finally:
                sys.stderr = old_stderr
    except Exception:
        pass

    return artifact


__all__ = [
    # Input specification
    "DataBlend",
    "Dataset",
    # High-level API
    "DataPrepConfig",
    "run_data_prep",
    "DataBlendsArtifact",
    # Low-level configuration
    "PipelineConfig",
    "PerSplitConfig",
    "TokenizerConfig",
    "OutputConfig",
    # Output format configs
    "BinIdxOutputConfig",
    "JsonlOutputConfig",
    "PackedOutputConfig",
    "ChatSftOutputConfig",
    "Transform",
    # Transform factories
    "sft",
    "openai_chat",
    "sharegpt",
    "passthrough",
    "select",
    "rename",
    # Transform type definitions
    "SftRecord",
    "OpenAIChatRecord",
    "ShareGPTRecord",
    # Execution
    "last_mile_process",
    "tokenize",  # Deprecated alias
    # Results
    "PipelineResult",
    "SplitResult",
]
