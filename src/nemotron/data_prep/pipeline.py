"""Pipeline orchestration for processing data blends into training formats."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import ray

from nemotron.data_prep.config import (
    DatasetConfig,
    InternalOutputConfig,
    InternalTokenizerConfig,
    PipelineConfig,
    PerSplitConfig,
    ShardPlan,
    SourceChangedError,
    TokenizerConfig,
    OutputConfig,
    BinIdxOutputConfig,
    JsonlOutputConfig,
    PackedOutputConfig,
    ChatSftOutputConfig,
)
from nemotron.data_prep.filesystem import (
    ensure_dir,
    get_filesystem,
    read_json,
    write_json,
)
from nemotron.data_prep.planning import (
    apply_shard_sampling,
    create_shard_plan,
    get_pending_shards,
    serialize_shard_plan,
)
from nemotron.data_prep.discovery import get_dataset_metadata
from nemotron.data_prep.tokenizer import ShardProcessor
from nemotron.data_prep import console as con

if TYPE_CHECKING:
    from nemotron.data_prep.blend import DataBlend, Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# W&B Logging Helpers
# =============================================================================


def log_pipeline_metrics_to_wandb(result: PipelineResult) -> None:
    """Log pipeline metrics to wandb if active.

    Args:
        result: The completed pipeline result
    """
    try:
        import wandb

        if wandb.run is None:
            return

        # Summary metrics
        wandb.log(
            {
                "data_prep/total_tokens": result.total_tokens,
                "data_prep/total_sequences": result.total_sequences,
                "data_prep/elapsed_sec": result.elapsed_sec,
                "data_prep/tokens_per_second": result.total_tokens
                / max(result.elapsed_sec, 0.001),
            }
        )

        # Per-split breakdown
        for split_name, split_result in result.splits.items():
            wandb.log(
                {
                    f"data_prep/{split_name}/tokens": split_result.total_tokens,
                    f"data_prep/{split_name}/sequences": split_result.total_sequences,
                }
            )
    except ImportError:
        pass


def log_cache_stats_to_wandb(
    cached_tokens: int,
    cached_sequences: int,
    pending_shards: int,
    total_shards: int,
) -> None:
    """Log cache hit statistics to wandb.

    Args:
        cached_tokens: Number of tokens served from cache
        cached_sequences: Number of sequences served from cache
        pending_shards: Number of shards that need processing
        total_shards: Total number of shards
    """
    try:
        import wandb

        if wandb.run is None:
            return

        cache_hit_rate = (total_shards - pending_shards) / max(total_shards, 1)
        wandb.log(
            {
                "data_prep/cache_hit_rate": cache_hit_rate,
                "data_prep/cached_tokens": cached_tokens,
                "data_prep/cached_sequences": cached_sequences,
                "data_prep/pending_shards": pending_shards,
            }
        )
    except ImportError:
        pass


# ============================================================================
# Result Types
# ============================================================================


@dataclass
class SplitResult:
    """Result for a single split.

    Attributes:
        name: Split name ("all", "train", "valid", "test")
        run_hash: Unique hash for this processing run
        output_dir: Directory containing tokenized shards
        data_paths: Megatron-Bridge format ["weight", "path", ...]
        num_shards: Number of shards produced
        total_tokens: Total tokens across all shards
        total_sequences: Total sequences (documents) processed
    """

    name: str
    run_hash: str
    output_dir: Path
    data_paths: list[str]
    num_shards: int
    total_tokens: int
    total_sequences: int


@dataclass
class PipelineResult:
    """Complete pipeline result.

    Attributes:
        output_dir: Root output directory
        blend_path: Path to generated blend.json
        splits: Results by split name
        is_per_split: True if per-split mode was used
        split_ratio: Split ratio if single-blend mode (e.g., "99990,8,2")
        elapsed_sec: Total processing time
        from_cache: True if all results were served from cache
    """

    output_dir: Path
    blend_path: Path
    splits: dict[str, SplitResult]
    is_per_split: bool
    split_ratio: str | None
    elapsed_sec: float
    from_cache: bool = False

    @property
    def total_tokens(self) -> int:
        """Total tokens across all splits."""
        return sum(s.total_tokens for s in self.splits.values())

    @property
    def total_sequences(self) -> int:
        """Total sequences across all splits."""
        return sum(s.total_sequences for s in self.splits.values())


# ============================================================================
# Public API
# ============================================================================


def get_default_num_actors() -> int:
    """Infer default number of Ray actors from available CPUs.

    Uses os.cpu_count() but caps at a reasonable maximum to avoid
    overwhelming the system. Leaves some headroom for the main process.

    Returns:
        Number of actors to use (min 2, max 32, ~75% of CPUs)
    """
    cpu_count = os.cpu_count() or 4
    # Use ~75% of CPUs, minimum 2, maximum 32
    return max(2, min(32, int(cpu_count * 0.75)))


def last_mile_process(
    blend: DataBlend,
    config: PipelineConfig,
) -> PipelineResult:
    """Process data blend into final training format.

    Dispatches to format-specific processing based on config.output.format:
    - binidx: Tokenize → Megatron .bin/.idx indexed dataset
    - jsonl: Transform → JSONL files (no tokenization)
    - packed: Tokenize → Pack → .npy packed sequences

    Args:
        blend: Data blend specification (datasets and weights)
        config: Pipeline configuration (output format, optional tokenizer)

    Returns:
        PipelineResult with paths to processed data and blend.json

    Output Format:
        The generated blend.json is directly compatible with Megatron-Bridge:

        Single blend mode:
            {"data_paths": ["1.0", "/path/shard", ...], "split": "99990,8,2"}

        Per-split mode:
            {"train_data_paths": [...], "valid_data_paths": [...], ...}

    Example:
        from nemotron.data_prep import last_mile_process, DataBlend, PipelineConfig
        from nemotron.data_prep.config import TokenizerConfig, OutputConfig, JsonlOutputConfig
        from nemotron.data_prep.formats.transforms import sft

        blend = DataBlend.load("data_blend.json")

        # JSONL output (no tokenization)
        config = PipelineConfig(
            output=OutputConfig(
                dir=Path("./sft_data"),
                format=JsonlOutputConfig(transform=sft(input="instruction", output="response")),
            ),
        )
        result = last_mile_process(blend, config)

        # BinIdx output (tokenization)
        config = PipelineConfig(
            tokenizer=TokenizerConfig(model="nvidia/NVIDIA-Nemotron-Nano-9B-v2"),
            output=OutputConfig(dir=Path("./output")),
        )
        result = last_mile_process(blend, config)
    """
    start = time.time()

    # Get format type
    format_config = config.output.format
    format_type = getattr(format_config, "format", "binidx")

    # Validate tokenizer requirement
    if format_type in ("binidx", "packed", "chat_sft") and config.tokenizer is None:
        raise ValueError(f"tokenizer is required for '{format_type}' output format")
    if format_type == "jsonl" and config.tokenizer is not None:
        logger.warning("Tokenizer ignored for JSONL format")

    # Dispatch to format-specific processing
    if format_type == "jsonl":
        result = _process_jsonl_blend(blend, config)
    elif format_type == "packed":
        result = _process_packed_blend(blend, config)
    elif format_type == "chat_sft":
        result = _process_chat_sft_blend(blend, config)
    else:
        # Default: binidx (tokenized)
        if blend.is_per_split:
            result = _tokenize_per_split(blend, config)
        else:
            result = _tokenize_single(blend, config)

    # Update elapsed time, preserving from_cache flag
    result = PipelineResult(
        output_dir=result.output_dir,
        blend_path=result.blend_path,
        splits=result.splits,
        is_per_split=result.is_per_split,
        split_ratio=result.split_ratio,
        elapsed_sec=time.time() - start,
        from_cache=result.from_cache,
    )

    # Log metrics to wandb if active
    if not result.from_cache:
        log_pipeline_metrics_to_wandb(result)

    return result


def tokenize(
    blend: DataBlend,
    config: PipelineConfig,
) -> PipelineResult:
    """Tokenize data blend to Megatron-Bridge format.

    .. deprecated::
        Use :func:`last_mile_process` instead. This function is provided
        for backward compatibility.

    Args:
        blend: Data blend specification
        config: Pipeline configuration (tokenizer, output settings)

    Returns:
        PipelineResult with paths to tokenized data
    """
    return last_mile_process(blend, config)


# ============================================================================
# Internal Processing Functions
# ============================================================================


def _tokenize_single(blend: DataBlend, config: PipelineConfig) -> PipelineResult:
    """Process single blend.

    If config.per_split is set, distributes shards into train/valid/test splits
    and outputs {"train": [...], "valid": [...], "test": [...]} JSON format
    compatible with Megatron-Bridge's per_split_data_args_path parameter.

    Otherwise, outputs {"data_paths": [...], "split": "..."} format for
    runtime split by ratio.
    """
    split_result = _process_split(
        datasets=blend.datasets,
        split_name="all",
        config=config,
    )

    # Check if per-split output mode is enabled
    if config.per_split is not None and config.per_split.enabled:
        blend_data = _distribute_shards_to_splits(
            data_paths=split_result.data_paths,
            num_shards=split_result.num_shards,
            valid_shards=config.per_split.valid_shards,
            test_shards=config.per_split.test_shards,
        )
        is_per_split = True
        split_ratio = None
    else:
        # Generate blend.json with data_paths and optional split ratio
        blend_data = {"data_paths": split_result.data_paths}
        if config.split:
            blend_data["split"] = config.split
        is_per_split = False
        split_ratio = config.split

    blend_path = config.output.dir / "blend.json"
    _write_json(blend_path, blend_data)

    return PipelineResult(
        output_dir=config.output.dir,
        blend_path=blend_path,
        splits={"all": split_result},
        is_per_split=is_per_split,
        split_ratio=split_ratio,
        elapsed_sec=0,
    )


def _distribute_shards_to_splits(
    data_paths: list[str],
    num_shards: int,
    valid_shards: int = 1,
    test_shards: int = 1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Distribute shard paths into train/valid/test splits.

    Collects all shards from all datasets into a pool, then randomly selects
    shards for test and valid splits. The remaining shards go to train.

    The data_paths format is: ["weight", "path", "weight", "path", ...]
    where paths are shard prefixes (e.g., /path/to/shard).

    Output format compatible with Megatron-Bridge's per_split_data_args_path:
    {"train": ["weight", "path_0000", "weight", "path_0001", ...], "valid": [...], "test": [...]}

    Args:
        data_paths: Megatron-Bridge format path list ["weight", "path", ...]
        num_shards: Total number of shards per dataset
        valid_shards: Number of shards for validation (total, not per-dataset)
        test_shards: Number of shards for test (total, not per-dataset)
        seed: Random seed for reproducible shard selection

    Returns:
        Dict with "train", "valid", "test" keys containing data_paths lists
    """
    import random

    # Parse weight/path pairs from data_paths
    # Format: ["1.0", "/path/dataset1/shard", "0.5", "/path/dataset2/shard", ...]
    pairs = []
    for i in range(0, len(data_paths), 2):
        if i + 1 < len(data_paths):
            weight = data_paths[i]
            prefix = data_paths[i + 1]
            pairs.append((weight, prefix))

    # Collect ALL shards from ALL datasets into one pool
    # Each entry is (weight, shard_path) where shard_path has the _XXXX suffix
    all_shards: list[tuple[str, str]] = []
    for weight, prefix in pairs:
        for shard_idx in range(num_shards):
            all_shards.append((weight, f"{prefix}_{shard_idx:06d}"))

    # Use seeded RNG for reproducibility
    rng = random.Random(seed)

    # Randomly select shards for test and valid
    # Ensure we don't request more shards than available
    total_shards = len(all_shards)
    actual_test_shards = min(test_shards, total_shards)
    remaining_after_test = total_shards - actual_test_shards
    actual_valid_shards = min(valid_shards, remaining_after_test)

    # Shuffle and partition
    shuffled = all_shards.copy()
    rng.shuffle(shuffled)

    test_selection = shuffled[:actual_test_shards]
    valid_selection = shuffled[actual_test_shards : actual_test_shards + actual_valid_shards]
    train_selection = shuffled[actual_test_shards + actual_valid_shards :]

    # Convert back to flat list format ["weight", "path", "weight", "path", ...]
    def flatten(shard_pairs: list[tuple[str, str]]) -> list[str]:
        result: list[str] = []
        for weight, path in shard_pairs:
            result.append(weight)
            result.append(path)
        return result

    return {
        "train": flatten(train_selection),
        "valid": flatten(valid_selection),
        "test": flatten(test_selection),
    }


def _tokenize_per_split(blend: DataBlend, config: PipelineConfig) -> PipelineResult:
    """Process each split separately (train/valid/test).

    Generates blend.json with {"train": [...], "valid": [...], "test": [...]}
    format compatible with Megatron-Bridge's per_split_data_args_path parameter.
    """
    splits: dict[str, SplitResult] = {}
    blend_data: dict[str, list[str]] = {}

    for split_name, datasets in blend.splits.items():
        # Create split-specific output config (preserve format from parent config)
        split_output = OutputConfig(
            dir=config.output.dir / split_name,
            format=config.output.format,
            min_doc_chars=config.output.min_doc_chars,
            max_doc_tokens=config.output.max_doc_tokens,
            max_rows=config.output.max_rows,
        )

        split_config = PipelineConfig(
            tokenizer=config.tokenizer,
            output=split_output,
            num_actors=config.num_actors,
            sample=config.sample,
            sample_seed=config.sample_seed,
            force=config.force,
            split=None,  # No split ratio for per-split mode
        )

        split_result = _process_split(
            datasets=datasets,
            split_name=split_name,
            config=split_config,
        )

        splits[split_name] = split_result
        # Use simple key names for Megatron-Bridge compatibility
        blend_data[split_name] = split_result.data_paths

    # Generate combined blend.json
    blend_path = config.output.dir / "blend.json"
    _write_json(blend_path, blend_data)

    return PipelineResult(
        output_dir=config.output.dir,
        blend_path=blend_path,
        splits=splits,
        is_per_split=True,
        split_ratio=None,
        elapsed_sec=0,
    )


def _process_split(
    datasets: list[Dataset],
    split_name: str,
    config: PipelineConfig,
) -> SplitResult:
    """Process a list of datasets into tokenized shards.

    This function orchestrates the full tokenization pipeline:
    1. Create shard plans for each dataset
    2. Process shards in parallel using Ray actors
    3. Aggregate results and build data_paths list
    """
    from nemotron.data_prep.blend import Dataset

    # Get filesystem
    fs, base_path = get_filesystem(str(config.output.dir))

    # Build internal config dict for planning/processing
    pipeline_dict = {
        "datasets": [
            {
                "name": d.name,
                "path": d.path,
                "weight": d.weight,
                "split": d.split,
                "subset": d.subset,
                "text_field": d.text_field,
            }
            for d in datasets
        ],
        "tokenizer": {
            "type": config.tokenizer.type,
            "model": config.tokenizer.model,
            "add_bos": config.tokenizer.add_bos,
            "add_eos": config.tokenizer.add_eos,
            "trust_remote_code": config.tokenizer.trust_remote_code,
        },
        "output": {
            "num_shards": config.output.format.num_shards,
            "dtype": config.output.format.dtype,
            "min_doc_chars": config.output.min_doc_chars,
            "max_doc_tokens": config.output.max_doc_tokens,
            "max_rows": config.output.max_rows,
        },
    }

    # Compute run hash (includes sampling params)
    run_config = pipeline_dict.copy()
    if config.sample is not None:
        run_config["_sample"] = {"spec": str(config.sample), "seed": config.sample_seed}
    config_hash = hashlib.sha256(
        json.dumps(run_config, sort_keys=True).encode()
    ).hexdigest()[:16]

    # Run namespace
    run_hash = config_hash if not config.force else f"{config_hash}_{int(time.time())}"
    run_dir = f"{base_path}/runs/{run_hash}"
    ensure_dir(fs, run_dir)

    # Freeze config
    write_json(fs, f"{run_dir}/config.json", run_config)

    tokenizer_config = InternalTokenizerConfig(**pipeline_dict["tokenizer"])
    output_config = InternalOutputConfig(**pipeline_dict["output"])

    # Planning phase
    con.planning_header()

    execution_plans: list[_DatasetExecutionPlan] = []
    plan_hashes = {}
    resolved_tokenizer = None
    plan_infos = []

    for dataset_entry in pipeline_dict["datasets"]:
        dataset_config = DatasetConfig(**dataset_entry)
        name = dataset_config.name

        # Create or load plan
        plan = _load_or_create_plan(
            dataset_config=dataset_config,
            output_config=output_config,
            tokenizer_config=tokenizer_config,
            config_hash=config_hash,
            run_dir=run_dir,
            fs=fs,
            force=config.force,
        )

        plan_hashes[name] = plan.plan_hash

        if resolved_tokenizer is None:
            resolved_tokenizer = plan.resolved_tokenizer

        # Paths for this plan
        dataset_dir = f"{run_dir}/datasets/{name}/{plan.plan_hash}"
        receipts_dir = f"{dataset_dir}/receipts"
        ensure_dir(fs, dataset_dir)
        ensure_dir(fs, receipts_dir)

        # Get pending shards and cached stats
        all_pending = get_pending_shards(plan, receipts_dir, fs)
        cached_stats = _aggregate_stats_from_receipts(receipts_dir, plan, fs)

        # Apply sampling
        sampled_count = None
        if config.sample is not None:
            pending_indices = apply_shard_sampling(all_pending, plan, config.sample, config.sample_seed)
            sampled_count = len(pending_indices)
        else:
            pending_indices = all_pending

        # Fetch HuggingFace metadata (non-blocking, best-effort)
        hf_metadata = get_dataset_metadata(dataset_config)

        # Build plan info for display
        plan_infos.append(
            con.DatasetPlanInfo(
                name=name,
                plan_hash=plan.plan_hash,
                num_shards=plan.num_shards,
                num_files=sum(len(a.files) for a in plan.file_assignments),
                pending=len(all_pending),
                cached=cached_stats["num_shards_completed"],
                cached_tokens=cached_stats["total_tokens"],
                cached_sequences=cached_stats["total_sequences"],
                sampled=sampled_count,
                hf_rows=hf_metadata.num_rows_str,
                hf_size=hf_metadata.size_str,
            )
        )

        # Store execution plan
        execution_plans.append(
            _DatasetExecutionPlan(
                name=name,
                config=dataset_config,
                plan=plan,
                dataset_dir=dataset_dir,
                receipts_dir=receipts_dir,
                pending_indices=pending_indices,
                cached_stats=cached_stats,
            )
        )

    # Show plan summary
    con.plan_summary(plan_infos, run_hash, num_actors=config.num_actors)

    # Execution phase
    results = {}
    has_work = any(ep.pending_indices for ep in execution_plans)

    if has_work:
        con.execution_header()

        # Create live status panel with all datasets
        live_status = con.create_live_status(
            datasets=[
                (ep.name, len(ep.pending_indices) or ep.cached_stats["num_shards_completed"])
                for ep in execution_plans
            ],
            run_hash=run_hash,
        )
        live_status.start()

        try:
            # Handle cached datasets first
            for ep in execution_plans:
                if not ep.pending_indices:
                    results[ep.name] = ep.cached_stats
                    live_status.report_tokens(ep.name, ep.cached_stats.get("total_tokens", 0))
                    live_status.cache_dataset(ep.name)

            # Process ALL pending shards from ALL datasets in parallel
            _process_all_shards_parallel(
                execution_plans=[ep for ep in execution_plans if ep.pending_indices],
                output_config=output_config,
                fs=fs,
                num_actors=config.num_actors,
                live_status=live_status,
                results=results,
            )
        finally:
            live_status.stop()
    else:
        # All cached, no live display needed
        for ep in execution_plans:
            results[ep.name] = ep.cached_stats

    # Generate outputs
    _generate_manifest(run_dir, pipeline_dict, results, plan_hashes, run_hash, resolved_tokenizer, fs)

    # Build data_paths in Megatron-Bridge format
    data_paths: list[str] = []
    for dataset_entry in pipeline_dict["datasets"]:
        name = dataset_entry["name"]
        weight = dataset_entry.get("weight", 1.0)

        if weight > 0 and name in plan_hashes:
            plan_hash = plan_hashes[name]
            prefix = f"{run_dir}/datasets/{name}/{plan_hash}/shard"
            data_paths.append(str(weight))
            data_paths.append(prefix)

    return SplitResult(
        name=split_name,
        run_hash=run_hash,
        output_dir=Path(config.output.dir),
        data_paths=data_paths,
        num_shards=config.output.format.num_shards,
        total_tokens=sum(r.get("total_tokens", 0) for r in results.values()),
        total_sequences=sum(r.get("total_sequences", 0) for r in results.values()),
    )


def _write_json(path: Path, data: dict) -> None:
    """Write JSON file with atomic write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".json.tmp")
    with open(temp_path, "w") as f:
        json.dump(data, f, indent=2)
    temp_path.rename(path)


# ============================================================================
# Internal Helper Classes and Functions
# ============================================================================


@dataclass
class _DatasetExecutionPlan:
    """Execution plan for a single dataset."""

    name: str
    config: DatasetConfig
    plan: ShardPlan
    dataset_dir: str
    receipts_dir: str
    pending_indices: list[int]
    cached_stats: dict


class _PlanDriftError(Exception):
    """Raised when a new plan would create drift from existing plans."""

    pass


def _load_or_create_plan(
    dataset_config: DatasetConfig,
    output_config: InternalOutputConfig,
    tokenizer_config: InternalTokenizerConfig,
    config_hash: str,
    run_dir: str,
    fs,
    force: bool = False,
) -> ShardPlan:
    """Load existing plan or create new one.

    Enforces single active plan per dataset unless force=True.
    This prevents silent drift where source changes create orphaned plans.
    """
    # Create plan to get hash
    plan = create_shard_plan(
        dataset_config=dataset_config,
        output_config=output_config,
        tokenizer_config=tokenizer_config,
        config_hash=config_hash,
        fs=fs,
    )

    dataset_plans_dir = f"{run_dir}/datasets/{dataset_config.name}"
    plan_path = f"{dataset_plans_dir}/{plan.plan_hash}/plan.json"

    if fs.exists(plan_path):
        # Load and verify
        existing_data = read_json(fs, plan_path)
        if existing_data.get("source_fingerprint") != plan.source_fingerprint:
            raise SourceChangedError(f"Source data changed for {dataset_config.name}")
        return ShardPlan.from_dict(existing_data)

    # Check for existing plans with different hashes (drift detection)
    if not force:
        try:
            existing_plan_dirs = [
                d for d in fs.ls(dataset_plans_dir) if fs.isdir(d) and fs.exists(f"{d}/plan.json")
            ]
            if existing_plan_dirs:
                existing_hashes = [d.split("/")[-1] for d in existing_plan_dirs]
                raise _PlanDriftError(
                    f"Dataset '{dataset_config.name}' has existing plan(s): {existing_hashes}. "
                    f"New plan hash {plan.plan_hash} would create drift. "
                    f"Use --force to create a new run namespace, or delete the existing run."
                )
        except FileNotFoundError:
            pass  # No existing plans, OK to create

    # Save new plan
    plan_dir = f"{dataset_plans_dir}/{plan.plan_hash}"
    ensure_dir(fs, plan_dir)
    write_json(fs, plan_path, serialize_shard_plan(plan))

    return plan


def _process_all_shards_parallel(
    execution_plans: list[_DatasetExecutionPlan],
    output_config: InternalOutputConfig,
    fs,
    num_actors: int,
    live_status,
    results: dict,
) -> None:
    """Process ALL pending shards from ALL datasets in parallel.

    This maximizes parallelism by submitting shards from all datasets
    to a shared actor pool, rather than processing datasets sequentially.
    """
    from nemotron.data_prep.shard_processor import ShardProcessor

    if not execution_plans:
        return

    # Determine filesystem protocol
    protocol = fs.protocol
    if isinstance(protocol, tuple):
        protocol = protocol[0]
    fs_protocol = protocol if protocol != "file" else "file"

    # Create shared actor pool
    # Use the first plan's tokenizer config (should be same for all)
    first_plan = execution_plans[0].plan
    actors = [
        ShardProcessor.remote(
            resolved_tokenizer=first_plan.resolved_tokenizer,
            text_field=execution_plans[0].config.text_field,
            min_doc_chars=output_config.min_doc_chars,
            max_doc_tokens=output_config.max_doc_tokens,
            dtype=output_config.dtype,
            max_rows=output_config.max_rows,
        )
        for _ in range(num_actors)
    ]

    try:
        # Build all tasks from all datasets
        # Task: (dataset_name, shard_index, assignment_dict, plan_hash, dataset_dir, receipts_dir)
        all_tasks: list[tuple] = []
        dataset_pending_counts: dict[str, int] = {}
        dataset_completed_counts: dict[str, int] = {}

        for ep in execution_plans:
            live_status.start_dataset(ep.name)
            dataset_pending_counts[ep.name] = len(ep.pending_indices)
            dataset_completed_counts[ep.name] = 0

            # Convert assignments to dicts for Ray serialization
            assignment_dicts = {}
            for a in ep.plan.file_assignments:
                assignment_dicts[a.shard_index] = {
                    "shard_index": a.shard_index,
                    "files": [asdict(f) for f in a.files],
                    "total_bytes": a.total_bytes,
                }

            for shard_idx in ep.pending_indices:
                all_tasks.append((
                    ep.name,
                    shard_idx,
                    assignment_dicts[shard_idx],
                    ep.plan.plan_hash,
                    ep.dataset_dir,
                    ep.receipts_dir,
                    ep,  # Keep reference to execution plan for aggregation
                ))

        # Submit tasks with backpressure
        max_in_flight = num_actors * 2
        task_queue = list(all_tasks)
        actor_idx = 0
        pending_list: list = []
        future_to_task: dict = {}

        def submit_task(task: tuple) -> None:
            nonlocal actor_idx
            name, shard_idx, assignment, plan_hash, dataset_dir, receipts_dir, ep = task
            actor = actors[actor_idx % num_actors]
            actor_idx += 1
            future = actor.process_shard.remote(
                shard_index=shard_idx,
                assignment=assignment,
                plan_hash=plan_hash,
                output_dir=dataset_dir,
                receipts_dir=receipts_dir,
                fs_protocol=fs_protocol,
            )
            pending_list.append(future)
            future_to_task[future] = task

        # Initial submission
        while task_queue and len(pending_list) < max_in_flight:
            submit_task(task_queue.pop(0))

        # Process with backpressure
        while pending_list:
            done, pending_list = ray.wait(pending_list, num_returns=1, timeout=60)
            for future in done:
                task = future_to_task.pop(future)
                name = task[0]
                ep = task[6]

                try:
                    ray.get(future)
                except Exception as e:
                    logger.error(f"Shard {task[1]} for {name} failed: {e}")

                # Update progress
                live_status.advance_dataset(name)
                dataset_completed_counts[name] += 1

                # Check if dataset is complete
                if dataset_completed_counts[name] >= dataset_pending_counts[name]:
                    # Aggregate final stats for this dataset
                    results[name] = _aggregate_stats_from_receipts(ep.receipts_dir, ep.plan, fs)
                    live_status.report_metrics(
                        name,
                        rows=results[name].get("total_sequences", 0),
                        tokens=results[name].get("total_tokens", 0),
                    )
                    live_status.complete_dataset(name)

                # Submit next task if available
                if task_queue:
                    submit_task(task_queue.pop(0))
    finally:
        # Clean up actors
        for actor in actors:
            ray.kill(actor)


def _process_shards_with_actors(
    pending_indices: list[int],
    plan: ShardPlan,
    dataset_dir: str,
    receipts_dir: str,
    dataset_config: DatasetConfig,
    output_config: InternalOutputConfig,
    fs,
    num_actors: int,
    on_progress: Callable[[], None] | None = None,
):
    """Process pending shards using actor pool.

    Args:
        on_progress: Optional callback called when a shard completes
    """
    # Determine filesystem protocol from fs object (not from path which has scheme stripped)
    protocol = fs.protocol
    if isinstance(protocol, tuple):
        protocol = protocol[0]
    # Map protocol names to fsspec protocol identifiers
    fs_protocol = protocol if protocol != "file" else "file"

    # Create actor pool
    actors = [
        ShardProcessor.remote(
            resolved_tokenizer=plan.resolved_tokenizer,
            text_field=dataset_config.text_field,
            min_doc_chars=output_config.min_doc_chars,
            max_doc_tokens=output_config.max_doc_tokens,
            dtype=output_config.dtype,
            max_rows=output_config.max_rows,
        )
        for _ in range(num_actors)
    ]

    # Convert assignments to dicts for Ray serialization
    assignment_dicts = {}
    for a in plan.file_assignments:
        assignment_dicts[a.shard_index] = {
            "shard_index": a.shard_index,
            "files": [asdict(f) for f in a.files],
            "total_bytes": a.total_bytes,
        }

    # Use backpressure loop: keep at most 2*num_actors tasks in flight
    # This prevents memory bloat from submitting all tasks at once
    max_in_flight = num_actors * 2
    shard_queue = list(pending_indices)
    actor_idx = 0
    total = len(pending_indices)

    # Track futures as list directly to avoid repeated dict->list conversion in ray.wait
    pending_list: list = []
    future_to_shard: dict = {}
    future_to_start_time: dict = {}  # Track when each task was submitted

    # Timeout configuration
    task_warn_timeout = 120  # Log warning after 2 minutes
    task_cancel_timeout = 600  # Cancel task after 10 minutes

    def submit_task(shard_index: int) -> None:
        nonlocal actor_idx
        actor = actors[actor_idx % num_actors]
        actor_idx += 1
        future = actor.process_shard.remote(
            shard_index=shard_index,
            assignment=assignment_dicts[shard_index],
            plan_hash=plan.plan_hash,
            output_dir=dataset_dir,
            receipts_dir=receipts_dir,
            fs_protocol=fs_protocol,
        )
        pending_list.append(future)
        future_to_shard[future] = shard_index
        future_to_start_time[future] = time.time()

    # Initial submission up to max_in_flight
    while shard_queue and len(pending_list) < max_in_flight:
        submit_task(shard_queue.pop(0))

    # Track which tasks we've already warned about
    warned_tasks: set = set()

    # Process with backpressure
    def process_loop(advance_fn: Callable[[], None]) -> None:
        nonlocal pending_list, shard_queue
        while pending_list:
            # ray.wait returns (done, remaining) - use remaining directly
            done, pending_list = ray.wait(pending_list, num_returns=1, timeout=60)

            # Handle timeout: log info about long-running tasks
            if not done and pending_list:
                current_time = time.time()
                tasks_to_cancel = []

                for future in pending_list:
                    shard_index = future_to_shard.get(future)
                    start_time = future_to_start_time.get(future, current_time)
                    elapsed = current_time - start_time

                    # Check for tasks that should be cancelled
                    if elapsed > task_cancel_timeout:
                        tasks_to_cancel.append((future, shard_index, elapsed))
                    # Log warning for long-running tasks (only once per task)
                    elif elapsed > task_warn_timeout and future not in warned_tasks:
                        warned_tasks.add(future)
                        logger.warning(
                            f"Shard {shard_index} has been running for {elapsed:.0f}s "
                            f"(likely slow HuggingFace download)"
                        )

                # Cancel truly stuck tasks
                for future, shard_index, elapsed in tasks_to_cancel:
                    logger.error(
                        f"Cancelling shard {shard_index} after {elapsed:.0f}s timeout"
                    )
                    try:
                        ray.cancel(future, force=True)
                    except Exception as e:
                        logger.debug(f"Failed to cancel task: {e}")

                    # Remove from tracking
                    pending_list.remove(future)
                    future_to_shard.pop(future, None)
                    future_to_start_time.pop(future, None)
                    warned_tasks.discard(future)

                    # Count as completed (with error) to unblock progress
                    advance_fn()

                    # Submit replacement task if queue has more
                    if shard_queue:
                        submit_task(shard_queue.pop(0))

                continue

            for future in done:
                shard_index = future_to_shard.pop(future)
                future_to_start_time.pop(future, None)
                warned_tasks.discard(future)
                try:
                    ray.get(future)
                    advance_fn()
                except Exception as e:
                    logger.error(f"Shard {shard_index} failed: {e}")
                    advance_fn()

                # Submit next task if queue has more
                if shard_queue:
                    submit_task(shard_queue.pop(0))

    if on_progress is not None:
        # Use external progress callback (for live status panel)
        process_loop(on_progress)
    else:
        # Use standalone progress bar
        with con.create_progress() as progress:
            task = progress.add_task("Processing shards", total=total)
            process_loop(lambda: progress.advance(task))


def _aggregate_stats_from_receipts(
    receipts_dir: str,
    plan: ShardPlan,
    fs,
) -> dict:
    """Aggregate statistics from all shard receipts."""
    stats = {
        "num_shards_completed": 0,
        "total_sequences": 0,
        "total_tokens": 0,
        "total_bin_bytes": 0,
        "total_idx_bytes": 0,
    }

    try:
        receipt_files = fs.glob(f"{receipts_dir}/shard_*.json")
    except FileNotFoundError:
        return stats
    except Exception:
        return stats

    for receipt_file in receipt_files:
        try:
            receipt = read_json(fs, receipt_file)
            if receipt.get("status") == "completed" and receipt.get("plan_hash") == plan.plan_hash:
                stats["num_shards_completed"] += 1
                stats["total_sequences"] += receipt["stats"]["num_sequences"]
                stats["total_tokens"] += receipt["stats"]["total_tokens"]
                stats["total_bin_bytes"] += receipt["files"]["bin"]["bytes"]
                stats["total_idx_bytes"] += receipt["files"]["idx"]["bytes"]
        except Exception:
            pass

    return stats


def _generate_manifest(
    run_dir: str,
    config: dict,
    results: dict,
    plan_hashes: dict[str, str],
    run_hash: str,
    resolved_tokenizer: dict | None,
    fs,
):
    """Generate manifest summary."""
    # Use resolved tokenizer (with SHA) if available, otherwise fall back to config
    tokenizer_info = resolved_tokenizer if resolved_tokenizer else config["tokenizer"]

    manifest = {
        "version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_hash": run_hash,
        "tokenizer": tokenizer_info,
        "datasets": {},
    }

    for name, stats in results.items():
        num_shards = config["output"]["num_shards"]
        completed = stats.get("num_shards_completed", 0)
        status = "completed" if completed == num_shards else "in_progress"

        manifest["datasets"][name] = {
            "status": status,
            "plan_hash": plan_hashes.get(name),
            "num_shards": num_shards,
            **stats,
        }

    write_json(fs, f"{run_dir}/manifest.json", manifest)


# ============================================================================
# JSONL Processing (No Tokenization)
# ============================================================================


def _process_jsonl_blend(blend: DataBlend, config: PipelineConfig) -> PipelineResult:
    """Process blend to JSONL output (no tokenization).

    Transforms records according to the configured transform function
    and writes to JSONL files (optionally compressed).
    """
    from nemotron.data_prep.jsonl_processor import JsonlShardProcessor

    format_config = config.output.format
    assert isinstance(format_config, JsonlOutputConfig)

    # Get filesystem
    fs, base_path = get_filesystem(str(config.output.dir))

    # Compute run hash (different from tokenization - no tokenizer info)
    run_config = {
        "datasets": [
            {
                "name": d.name,
                "path": d.path,
                "weight": d.weight,
                "split": d.split,
                "subset": d.subset,
            }
            for d in blend.datasets
        ],
        "output": {
            "format": "jsonl",
            "compression": format_config.compression,
        },
    }
    if config.sample is not None:
        run_config["_sample"] = {"spec": str(config.sample), "seed": config.sample_seed}

    config_hash = hashlib.sha256(
        json.dumps(run_config, sort_keys=True).encode()
    ).hexdigest()[:16]

    run_hash = config_hash if not config.force else f"{config_hash}_{int(time.time())}"
    run_dir = f"{base_path}/runs/{run_hash}"
    ensure_dir(fs, run_dir)

    # Freeze config
    write_json(fs, f"{run_dir}/config.json", run_config)

    # Determine num_shards from format config
    num_shards = _resolve_num_shards(format_config, blend, fs)

    # For JSONL, we use a simpler processing model:
    # Each dataset's files are distributed across shards and written directly
    results = {}
    data_paths: list[str] = []

    con.planning_header()

    # Planning phase: discover files and check cache for all datasets
    from nemotron.data_prep.discovery import discover_input_files, get_dataset_metadata

    dataset_plans: list[tuple] = []  # (dataset, dataset_dir, files, cached_stats)
    plan_infos = []

    for dataset in blend.datasets:
        name = dataset.name

        # Create dataset directory
        dataset_dir = f"{run_dir}/datasets/{name}"
        ensure_dir(fs, dataset_dir)

        # Get files for this dataset
        dataset_config = DatasetConfig(
            name=dataset.name,
            path=dataset.path,
            split=dataset.split,
            subset=dataset.subset,
            text_field=dataset.text_field,
        )
        files = discover_input_files(dataset_config, fs)

        # Check cached stats
        cached_stats = _aggregate_jsonl_stats(dataset_dir, num_shards, fs)
        cached_shards = cached_stats.get("num_shards_completed", 0)
        pending_shards = num_shards - cached_shards

        # Fetch HuggingFace metadata (non-blocking, best-effort)
        hf_metadata = get_dataset_metadata(dataset_config)

        # Build plan info for display
        plan_infos.append(
            con.DatasetPlanInfo(
                name=name,
                plan_hash=run_hash[:8],
                num_shards=num_shards,
                num_files=len(files),
                pending=pending_shards,
                cached=cached_shards,
                cached_tokens=0,  # JSONL doesn't track tokens
                cached_sequences=cached_stats.get("num_records", 0),
                sampled=num_shards if config.output.max_rows else None,
                hf_rows=hf_metadata.num_rows_str,
                hf_size=hf_metadata.size_str,
            )
        )

        dataset_plans.append((dataset, dataset_dir, files, cached_stats))

    # Show plan summary
    con.plan_summary(plan_infos, run_hash, num_actors=config.num_actors)

    # Execution phase
    has_work = any(
        num_shards - cached_stats.get("num_shards_completed", 0) > 0
        for _, _, _, cached_stats in dataset_plans
    )

    if has_work:
        con.execution_header()

    for dataset, dataset_dir, files, cached_stats in dataset_plans:
        name = dataset.name

        # Process with actors
        if files:
            _process_jsonl_shards_with_actors(
                files=files,
                num_shards=num_shards,
                dataset_dir=dataset_dir,
                text_field=dataset.text_field,
                transform=format_config.transform,
                compression=format_config.compression,
                max_rows=config.output.max_rows,
                fs=fs,
                num_actors=config.num_actors,
            )

        # Aggregate stats
        stats = _aggregate_jsonl_stats(dataset_dir, num_shards, fs)
        results[name] = stats

        # Build data_paths
        weight = dataset.weight
        if weight > 0:
            prefix = f"{dataset_dir}/shard"
            data_paths.append(str(weight))
            data_paths.append(prefix)

    # Generate blend.json
    blend_data: dict = {"data_paths": data_paths}
    if config.split:
        blend_data["split"] = config.split

    blend_path = config.output.dir / "blend.json"
    _write_json(blend_path, blend_data)

    return PipelineResult(
        output_dir=config.output.dir,
        blend_path=blend_path,
        splits={
            "all": SplitResult(
                name="all",
                run_hash=run_hash,
                output_dir=config.output.dir,
                data_paths=data_paths,
                num_shards=num_shards,
                total_tokens=0,  # No tokenization
                total_sequences=sum(r.get("num_records", 0) for r in results.values()),
            )
        },
        is_per_split=False,
        split_ratio=config.split,
        elapsed_sec=0,
    )


def _resolve_num_shards(format_config, blend: DataBlend, fs) -> int:
    """Resolve num_shards from format config (shard_size or explicit num_shards)."""
    from nemotron.data_prep.utils.size import compute_num_shards, parse_byte_size

    if format_config.num_shards is not None:
        return format_config.num_shards

    # Compute from shard_size
    if format_config.shard_size is not None:
        # Estimate total bytes from blend
        total_bytes = _estimate_blend_bytes(blend, fs)
        return compute_num_shards(total_bytes, format_config.shard_size)

    # Default fallback
    return 128


def _estimate_blend_bytes(blend: DataBlend, fs) -> int:
    """Estimate total bytes in blend for shard planning."""
    from nemotron.data_prep.discovery import discover_input_files

    total = 0
    for dataset in blend.datasets:
        try:
            dataset_config = DatasetConfig(
                name=dataset.name,
                path=dataset.path,
                split=dataset.split,
                subset=dataset.subset,
                text_field=dataset.text_field,
            )
            files = discover_input_files(dataset_config, fs)
            total += sum(f.size for f in files)
        except Exception:
            pass
    return total or 1  # Avoid division by zero


def _process_jsonl_shards_with_actors(
    files: list,
    num_shards: int,
    dataset_dir: str,
    text_field: str,
    transform,
    compression: str,
    max_rows: int | None,
    fs,
    num_actors: int,
) -> None:
    """Process files to JSONL shards using Ray actors."""
    from nemotron.data_prep.jsonl_processor import JsonlShardProcessor

    # Determine filesystem protocol
    protocol = fs.protocol
    if isinstance(protocol, tuple):
        protocol = protocol[0]
    fs_protocol = protocol if protocol != "file" else "file"

    # Create actor pool
    actors = [
        JsonlShardProcessor.remote(
            text_field=text_field,
            transform=transform,
            compression=compression,
            max_rows=max_rows,
        )
        for _ in range(num_actors)
    ]

    # Distribute files across shards (round-robin for now)
    # TODO: Could use smarter distribution based on file sizes
    shard_assignments: dict[int, list] = {i: [] for i in range(num_shards)}
    for i, file_info in enumerate(files):
        shard_idx = i % num_shards
        shard_assignments[shard_idx].append(file_info)

    # Submit tasks with backpressure
    max_in_flight = num_actors * 2
    shard_queue = list(range(num_shards))
    actor_idx = 0
    pending_list: list = []
    future_to_shard: dict = {}

    def submit_task(shard_index: int) -> None:
        nonlocal actor_idx
        actor = actors[actor_idx % num_actors]
        actor_idx += 1
        future = actor.process_shard.remote(
            shard_index=shard_index,
            files=[f.__dict__ if hasattr(f, "__dict__") else f for f in shard_assignments[shard_index]],
            output_dir=dataset_dir,
            fs_protocol=fs_protocol,
        )
        pending_list.append(future)
        future_to_shard[future] = shard_index

    # Initial submission
    while shard_queue and len(pending_list) < max_in_flight:
        submit_task(shard_queue.pop(0))

    # Process with backpressure
    while pending_list:
        done, pending_list = ray.wait(pending_list, num_returns=1, timeout=60)
        for future in done:
            shard_index = future_to_shard.pop(future)
            try:
                ray.get(future)
            except Exception as e:
                logger.error(f"JSONL shard {shard_index} failed: {e}")

            if shard_queue:
                submit_task(shard_queue.pop(0))


def _aggregate_jsonl_stats(dataset_dir: str, num_shards: int, fs) -> dict:
    """Aggregate statistics from JSONL receipts."""
    stats = {
        "num_shards_completed": 0,
        "num_records": 0,
        "num_skipped": 0,
        "total_bytes": 0,
    }

    try:
        receipt_files = fs.glob(f"{dataset_dir}/shard_*.receipt.json")
    except Exception:
        return stats

    for receipt_file in receipt_files:
        try:
            receipt = read_json(fs, receipt_file)
            if receipt.get("status") == "completed":
                stats["num_shards_completed"] += 1
                stats["num_records"] += receipt.get("num_records", 0)
                stats["num_skipped"] += receipt.get("num_skipped", 0)
                stats["total_bytes"] += receipt.get("total_bytes", 0)
        except Exception:
            pass

    return stats


# ============================================================================
# Packed Sequence Processing (Tokenization + Packing)
# ============================================================================


def _process_packed_blend(blend: DataBlend, config: PipelineConfig) -> PipelineResult:
    """Process blend to packed sequence output (.npy files).

    Tokenizes records and packs them into efficient batches compatible with
    Megatron-Bridge's GPTSFTPackedDataset.
    """
    from nemotron.data_prep.packed_processor import PackedShardProcessor
    from nemotron.data_prep.discovery import discover_input_files

    format_config = config.output.format
    assert isinstance(format_config, PackedOutputConfig)

    # Get filesystem
    fs, base_path = get_filesystem(str(config.output.dir))

    # Compute run hash (includes tokenizer, pack_size, algorithm)
    run_config = {
        "datasets": [
            {
                "name": d.name,
                "path": d.path,
                "weight": d.weight,
                "split": d.split,
                "subset": d.subset,
                "text_field": d.text_field,
            }
            for d in blend.datasets
        ],
        "tokenizer": {
            "type": config.tokenizer.type,
            "model": config.tokenizer.model,
            "add_bos": config.tokenizer.add_bos,
            "add_eos": config.tokenizer.add_eos,
            "trust_remote_code": config.tokenizer.trust_remote_code,
        },
        "output": {
            "format": "packed",
            "pack_size": format_config.pack_size,
            "algorithm": format_config.algorithm,
            "dtype": format_config.dtype,
        },
    }
    if config.sample is not None:
        run_config["_sample"] = {"spec": str(config.sample), "seed": config.sample_seed}

    config_hash = hashlib.sha256(
        json.dumps(run_config, sort_keys=True).encode()
    ).hexdigest()[:16]

    run_hash = config_hash if not config.force else f"{config_hash}_{int(time.time())}"
    run_dir = f"{base_path}/runs/{run_hash}"
    ensure_dir(fs, run_dir)

    # Freeze config
    write_json(fs, f"{run_dir}/config.json", run_config)

    # Determine num_shards from format config
    num_shards = _resolve_num_shards(format_config, blend, fs)

    # Resolve tokenizer to get SHA for determinism
    from nemotron.data_prep.planning import resolve_tokenizer

    tokenizer_config = InternalTokenizerConfig(**run_config["tokenizer"])
    resolved_tokenizer = resolve_tokenizer(tokenizer_config)

    # Process each dataset
    results = {}
    data_paths: list[str] = []

    con.planning_header()

    for dataset in blend.datasets:
        name = dataset.name

        # Create dataset directory structure
        dataset_dir = f"{run_dir}/datasets/{name}"
        receipts_dir = f"{dataset_dir}/receipts"
        ensure_dir(fs, dataset_dir)
        ensure_dir(fs, receipts_dir)

        # Get files for this dataset
        dataset_config = DatasetConfig(
            name=dataset.name,
            path=dataset.path,
            split=dataset.split,
            subset=dataset.subset,
            text_field=dataset.text_field,
        )
        files = discover_input_files(dataset_config, fs)

        # Display info
        logger.info(
            f"Processing dataset '{name}' with {len(files)} files -> "
            f"{num_shards} packed shards (pack_size={format_config.pack_size})"
        )

        # Process with actors
        if files:
            _process_packed_shards_with_actors(
                files=files,
                num_shards=num_shards,
                dataset_dir=dataset_dir,
                receipts_dir=receipts_dir,
                text_field=dataset.text_field,
                resolved_tokenizer=resolved_tokenizer,
                format_config=format_config,
                min_doc_chars=config.output.min_doc_chars,
                max_doc_tokens=config.output.max_doc_tokens,
                max_rows=config.output.max_rows,
                fs=fs,
                num_actors=config.num_actors,
            )

        # Aggregate stats
        stats = _aggregate_packed_stats(dataset_dir, receipts_dir, fs)
        results[name] = stats

        # Build data_paths
        weight = dataset.weight
        if weight > 0:
            prefix = f"{dataset_dir}/shard"
            data_paths.append(str(weight))
            data_paths.append(prefix)

    # Generate blend.json
    blend_data: dict = {"data_paths": data_paths}
    if config.split:
        blend_data["split"] = config.split

    blend_path = config.output.dir / "blend.json"
    _write_json(blend_path, blend_data)

    return PipelineResult(
        output_dir=config.output.dir,
        blend_path=blend_path,
        splits={
            "all": SplitResult(
                name="all",
                run_hash=run_hash,
                output_dir=config.output.dir,
                data_paths=data_paths,
                num_shards=num_shards,
                total_tokens=sum(r.get("total_tokens", 0) for r in results.values()),
                total_sequences=sum(r.get("num_sequences", 0) for r in results.values()),
            )
        },
        is_per_split=False,
        split_ratio=config.split,
        elapsed_sec=0,
    )


def _process_packed_shards_with_actors(
    files: list,
    num_shards: int,
    dataset_dir: str,
    receipts_dir: str,
    text_field: str,
    resolved_tokenizer: dict,
    format_config: PackedOutputConfig,
    min_doc_chars: int | None,
    max_doc_tokens: int | None,
    max_rows: int | None,
    fs,
    num_actors: int,
) -> None:
    """Process files to packed shards using Ray actors."""
    from nemotron.data_prep.packed_processor import PackedShardProcessor
    from dataclasses import asdict

    # Determine filesystem protocol
    protocol = fs.protocol
    if isinstance(protocol, tuple):
        protocol = protocol[0]
    fs_protocol = protocol if protocol != "file" else "file"

    # Create actor pool
    actors = [
        PackedShardProcessor.remote(
            resolved_tokenizer=resolved_tokenizer,
            text_field=text_field,
            pack_size=format_config.pack_size,
            algorithm=format_config.algorithm,
            dtype=format_config.dtype,
            min_doc_chars=min_doc_chars,
            max_doc_tokens=max_doc_tokens,
            max_rows=max_rows,
            seed=42,  # Fixed seed for reproducibility
        )
        for _ in range(num_actors)
    ]

    # Distribute files across shards (round-robin)
    shard_assignments: dict[int, list] = {i: [] for i in range(num_shards)}
    for i, file_info in enumerate(files):
        shard_idx = i % num_shards
        # Convert FileInfo to dict for Ray serialization
        if hasattr(file_info, "__dict__"):
            shard_assignments[shard_idx].append(asdict(file_info))
        else:
            shard_assignments[shard_idx].append(file_info)

    # Submit tasks with backpressure
    max_in_flight = num_actors * 2
    shard_queue = list(range(num_shards))
    actor_idx = 0
    pending_list: list = []
    future_to_shard: dict = {}

    def submit_task(shard_index: int) -> None:
        nonlocal actor_idx
        actor = actors[actor_idx % num_actors]
        actor_idx += 1
        future = actor.process_shard.remote(
            shard_index=shard_index,
            files=shard_assignments[shard_index],
            output_dir=dataset_dir,
            receipts_dir=receipts_dir,
            fs_protocol=fs_protocol,
        )
        pending_list.append(future)
        future_to_shard[future] = shard_index

    # Initial submission
    while shard_queue and len(pending_list) < max_in_flight:
        submit_task(shard_queue.pop(0))

    # Process with backpressure
    while pending_list:
        done, pending_list = ray.wait(pending_list, num_returns=1, timeout=60)
        for future in done:
            shard_index = future_to_shard.pop(future)
            try:
                ray.get(future)
            except Exception as e:
                logger.error(f"Packed shard {shard_index} failed: {e}")

            if shard_queue:
                submit_task(shard_queue.pop(0))


def _aggregate_packed_stats(dataset_dir: str, receipts_dir: str, fs) -> dict:
    """Aggregate statistics from packed receipts."""
    stats = {
        "num_shards_completed": 0,
        "num_sequences": 0,
        "num_packed_sequences": 0,
        "total_tokens": 0,
        "total_npy_bytes": 0,
    }

    try:
        receipt_files = fs.glob(f"{receipts_dir}/shard_*.json")
    except Exception:
        return stats

    for receipt_file in receipt_files:
        try:
            receipt = read_json(fs, receipt_file)
            if receipt.get("status") == "completed":
                stats["num_shards_completed"] += 1
                stats["num_sequences"] += receipt["stats"].get("num_sequences", 0)
                stats["num_packed_sequences"] += receipt["stats"].get("num_packed_sequences", 0)
                stats["total_tokens"] += receipt["stats"].get("total_tokens", 0)
                stats["total_npy_bytes"] += receipt.get("npy_bytes", 0)
        except Exception:
            pass

    return stats


# ============================================================================
# Chat SFT Processing (Tokenization + Loss Masking + Packing)
# ============================================================================


def _process_chat_sft_blend(blend: DataBlend, config: PipelineConfig) -> PipelineResult:
    """Process blend to chat-templated SFT output (.npy files with loss masks).

    Applies materialize.py chat template logic, tokenizes with role-based
    loss masking, and packs sequences into .npy files compatible with
    GPTSFTPackedDataset.
    """
    from nemotron.data_prep.chat_sft_processor import ChatSftShardProcessor
    from nemotron.data_prep.discovery import discover_input_files

    format_config = config.output.format
    assert isinstance(format_config, ChatSftOutputConfig)

    # Get filesystem
    fs, base_path = get_filesystem(str(config.output.dir))

    # Compute run hash (includes tokenizer, pack_size, algorithm, chat_template)
    run_config = {
        "datasets": [
            {
                "name": d.name,
                "path": d.path,
                "weight": d.weight,
                "split": d.split,
                "subset": d.subset,
            }
            for d in blend.datasets
        ],
        "tokenizer": {
            "type": config.tokenizer.type,
            "model": config.tokenizer.model,
            "add_bos": config.tokenizer.add_bos,
            "add_eos": config.tokenizer.add_eos,
            "trust_remote_code": config.tokenizer.trust_remote_code,
        },
        "output": {
            "format": "chat_sft",
            "pack_size": format_config.pack_size,
            "algorithm": format_config.algorithm,
            "dtype": format_config.dtype,
            "chat_template": format_config.chat_template,
            "messages_field": format_config.messages_field,
            "tools_field": format_config.tools_field,
        },
    }
    if config.sample is not None:
        run_config["_sample"] = {"spec": str(config.sample), "seed": config.sample_seed}

    config_hash = hashlib.sha256(
        json.dumps(run_config, sort_keys=True).encode()
    ).hexdigest()[:16]

    run_hash = config_hash if not config.force else f"{config_hash}_{int(time.time())}"
    run_dir = f"{base_path}/runs/{run_hash}"
    ensure_dir(fs, run_dir)

    # Freeze config
    write_json(fs, f"{run_dir}/config.json", run_config)

    # Determine num_shards from format config
    num_shards = _resolve_num_shards(format_config, blend, fs)

    # Resolve tokenizer to get SHA for determinism
    from nemotron.data_prep.planning import resolve_tokenizer

    tokenizer_config = InternalTokenizerConfig(**run_config["tokenizer"])
    resolved_tokenizer = resolve_tokenizer(tokenizer_config)

    # Planning phase: discover files for all datasets first
    results = {}
    data_paths: list[str] = []

    con.planning_header()

    # Discover files for all datasets
    dataset_plans: list[tuple] = []  # (dataset, dataset_dir, receipts_dir, files)
    for dataset in blend.datasets:
        name = dataset.name

        # Create dataset directory structure
        dataset_dir = f"{run_dir}/datasets/{name}"
        receipts_dir = f"{dataset_dir}/receipts"
        ensure_dir(fs, dataset_dir)
        ensure_dir(fs, receipts_dir)

        # Get files for this dataset
        dataset_config = DatasetConfig(
            name=dataset.name,
            path=dataset.path,
            split=dataset.split,
            subset=dataset.subset,
            text_field=dataset.text_field,
        )
        files = discover_input_files(dataset_config, fs)

        # Display discovered info
        logger.info(
            f"Discovered dataset '{name}' with {len(files)} files"
        )

        dataset_plans.append((dataset, dataset_dir, receipts_dir, files))

    # Build plan info for display
    plan_infos = []
    for dataset, dataset_dir, receipts_dir, files in dataset_plans:
        # Check cached stats
        cached_stats = _aggregate_packed_stats(dataset_dir, receipts_dir, fs)
        cached_shards = cached_stats.get("num_shards_completed", 0)

        plan_infos.append(
            con.DatasetPlanInfo(
                name=dataset.name,
                plan_hash=run_hash[:8],
                num_shards=num_shards,
                num_files=len(files),
                pending=num_shards - cached_shards if files else 0,
                cached=cached_shards,
                cached_tokens=cached_stats.get("total_tokens", 0),
                cached_sequences=cached_stats.get("num_sequences", 0),
                sampled=num_shards if config.output.max_rows else None,
                hf_rows=None,
                hf_size=None,
            )
        )

    # Show plan summary
    con.plan_summary(plan_infos, run_hash, num_actors=config.num_actors)

    # Execution phase
    has_work = any(len(files) > 0 for _, _, _, files in dataset_plans)

    if has_work:
        con.execution_header()

        # Create actor pool ONCE and reuse across all datasets
        from nemotron.data_prep.chat_sft_processor import ChatSftShardProcessor
        from dataclasses import asdict

        actors = [
            ChatSftShardProcessor.remote(
                resolved_tokenizer=resolved_tokenizer,
                messages_field=format_config.messages_field,
                tools_field=format_config.tools_field,
                pack_size=format_config.pack_size,
                algorithm=format_config.algorithm,
                dtype=format_config.dtype,
                chat_template=format_config.chat_template,
                max_doc_tokens=config.output.max_doc_tokens,
                max_rows=config.output.max_rows,
                seed=42,
                used_in_filter=format_config.used_in_filter,
                used_in_field=format_config.used_in_field,
            )
            for _ in range(config.num_actors)
        ]

        # Create live status panel with all datasets
        live_status = con.create_live_status(
            datasets=[
                (dataset.name, num_shards)
                for dataset, _, _, files in dataset_plans
                if files
            ],
            run_hash=run_hash,
        )
        live_status.start()

        try:
            # Determine filesystem protocol
            protocol = fs.protocol
            if isinstance(protocol, tuple):
                protocol = protocol[0]
            fs_protocol = protocol if protocol != "file" else "file"

            # Build all tasks upfront - process ALL datasets in parallel
            all_tasks: list[tuple[str, str, str, int, list]] = []  # (name, dataset_dir, receipts_dir, shard_idx, files)
            for dataset, dataset_dir, receipts_dir, files in dataset_plans:
                if not files:
                    continue
                # Each dataset gets 1 shard (since num_shards is computed per-dataset with 1 file)
                # Convert files to dicts for Ray serialization
                files_as_dicts = [
                    asdict(f) if hasattr(f, "__dict__") else f for f in files
                ]
                all_tasks.append((dataset.name, dataset_dir, receipts_dir, 0, files_as_dicts))
                live_status.start_dataset(dataset.name)

            # Submit all tasks with backpressure
            num_actors = len(actors)
            max_in_flight = num_actors * 2
            task_queue = list(all_tasks)
            actor_idx = 0
            pending_list: list = []
            future_to_task: dict = {}

            def submit_task(task: tuple) -> None:
                nonlocal actor_idx
                name, dataset_dir, receipts_dir, shard_idx, files_dicts = task
                actor = actors[actor_idx % num_actors]
                actor_idx += 1
                future = actor.process_shard.remote(
                    shard_index=shard_idx,
                    files=files_dicts,
                    output_dir=dataset_dir,
                    receipts_dir=receipts_dir,
                    fs_protocol=fs_protocol,
                )
                pending_list.append(future)
                future_to_task[future] = task

            # Initial submission
            while task_queue and len(pending_list) < max_in_flight:
                submit_task(task_queue.pop(0))

            # Process with backpressure
            while pending_list:
                done, pending_list = ray.wait(pending_list, num_returns=1, timeout=60)
                for future in done:
                    task = future_to_task.pop(future)
                    name = task[0]
                    dataset_dir = task[1]
                    receipts_dir = task[2]
                    try:
                        ray.get(future)
                    except Exception as e:
                        logger.error(f"Chat SFT shard for {name} failed: {e}")

                    # Update progress
                    live_status.advance_dataset(name)

                    # Aggregate stats for this dataset
                    stats = _aggregate_packed_stats(dataset_dir, receipts_dir, fs)
                    results[name] = stats
                    live_status.report_metrics(
                        name,
                        rows=stats.get("num_sequences", 0),
                        tokens=stats.get("total_tokens", 0),
                    )
                    live_status.complete_dataset(name)

                    # Submit next task if available
                    if task_queue:
                        submit_task(task_queue.pop(0))

            # Build data_paths for all completed datasets
            for dataset, dataset_dir, receipts_dir, files in dataset_plans:
                if not files:
                    continue
                weight = dataset.weight
                if weight > 0:
                    prefix = f"{dataset_dir}/shard"
                    data_paths.append(str(weight))
                    data_paths.append(prefix)
        finally:
            live_status.stop()
            # Clean up actors
            for actor in actors:
                ray.kill(actor)
    else:
        # No work to do - all datasets empty or cached
        for dataset, dataset_dir, receipts_dir, files in dataset_plans:
            stats = _aggregate_packed_stats(dataset_dir, receipts_dir, fs)
            results[dataset.name] = stats
            weight = dataset.weight
            if weight > 0:
                prefix = f"{dataset_dir}/shard"
                data_paths.append(str(weight))
                data_paths.append(prefix)

    # Generate blend.json
    # Check if per-split output mode is enabled
    if config.per_split is not None and config.per_split.enabled:
        blend_data = _distribute_shards_to_splits(
            data_paths=data_paths,
            num_shards=num_shards,
            valid_shards=config.per_split.valid_shards,
            test_shards=config.per_split.test_shards,
        )
        is_per_split = True
        split_ratio = None
    else:
        blend_data = {"data_paths": data_paths}
        if config.split:
            blend_data["split"] = config.split
        is_per_split = False
        split_ratio = config.split

    blend_path = config.output.dir / "blend.json"
    _write_json(blend_path, blend_data)

    return PipelineResult(
        output_dir=config.output.dir,
        blend_path=blend_path,
        splits={
            "all": SplitResult(
                name="all",
                run_hash=run_hash,
                output_dir=config.output.dir,
                data_paths=data_paths,
                num_shards=num_shards,
                total_tokens=sum(r.get("total_tokens", 0) for r in results.values()),
                total_sequences=sum(r.get("num_sequences", 0) for r in results.values()),
            )
        },
        is_per_split=is_per_split,
        split_ratio=split_ratio,
        elapsed_sec=0,
    )


def _process_chat_sft_shards_with_actors_pool(
    actors: list,
    files: list,
    num_shards: int,
    dataset_dir: str,
    receipts_dir: str,
    max_rows: int | None,
    fs,
    on_progress: Callable[[], None] | None = None,
) -> None:
    """Process files to chat SFT packed shards using existing Ray actor pool.

    This version takes a pre-created actor pool to allow reuse across datasets.
    """
    from dataclasses import asdict

    # Determine filesystem protocol
    protocol = fs.protocol
    if isinstance(protocol, tuple):
        protocol = protocol[0]
    fs_protocol = protocol if protocol != "file" else "file"

    num_actors = len(actors)

    # Distribute files across shards (round-robin)
    shard_assignments: dict[int, list] = {i: [] for i in range(num_shards)}
    for i, file_info in enumerate(files):
        shard_idx = i % num_shards
        # Convert FileInfo to dict for Ray serialization
        if hasattr(file_info, "__dict__"):
            shard_assignments[shard_idx].append(asdict(file_info))
        else:
            shard_assignments[shard_idx].append(file_info)

    # Submit tasks with backpressure
    max_in_flight = num_actors * 2
    shard_queue = list(range(num_shards))
    actor_idx = 0
    pending_list: list = []
    future_to_shard: dict = {}

    def submit_task(shard_index: int) -> None:
        nonlocal actor_idx
        actor = actors[actor_idx % num_actors]
        actor_idx += 1
        future = actor.process_shard.remote(
            shard_index=shard_index,
            files=shard_assignments[shard_index],
            output_dir=dataset_dir,
            receipts_dir=receipts_dir,
            fs_protocol=fs_protocol,
        )
        pending_list.append(future)
        future_to_shard[future] = shard_index

    # Initial submission
    while shard_queue and len(pending_list) < max_in_flight:
        submit_task(shard_queue.pop(0))

    # Process with backpressure
    while pending_list:
        done, pending_list = ray.wait(pending_list, num_returns=1, timeout=60)
        for future in done:
            shard_index = future_to_shard.pop(future)
            try:
                ray.get(future)
                if on_progress:
                    on_progress()
            except Exception as e:
                logger.error(f"Chat SFT shard {shard_index} failed: {e}")
                if on_progress:
                    on_progress()

            if shard_queue:
                submit_task(shard_queue.pop(0))


def _process_chat_sft_shards_with_actors(
    files: list,
    num_shards: int,
    dataset_dir: str,
    receipts_dir: str,
    resolved_tokenizer: dict,
    format_config: ChatSftOutputConfig,
    max_doc_tokens: int | None,
    max_rows: int | None,
    fs,
    num_actors: int,
    on_progress: Callable[[], None] | None = None,
) -> None:
    """Process files to chat SFT packed shards using Ray actors.

    This creates its own actor pool - for single dataset processing.
    For processing multiple datasets, use _process_chat_sft_shards_with_actors_pool
    with a shared actor pool.
    """
    from nemotron.data_prep.chat_sft_processor import ChatSftShardProcessor

    # Create actor pool
    actors = [
        ChatSftShardProcessor.remote(
            resolved_tokenizer=resolved_tokenizer,
            messages_field=format_config.messages_field,
            tools_field=format_config.tools_field,
            pack_size=format_config.pack_size,
            algorithm=format_config.algorithm,
            dtype=format_config.dtype,
            chat_template=format_config.chat_template,
            max_doc_tokens=max_doc_tokens,
            max_rows=max_rows,
            seed=42,  # Fixed seed for reproducibility
            used_in_filter=format_config.used_in_filter,
            used_in_field=format_config.used_in_field,
        )
        for _ in range(num_actors)
    ]

    try:
        _process_chat_sft_shards_with_actors_pool(
            actors=actors,
            files=files,
            num_shards=num_shards,
            dataset_dir=dataset_dir,
            receipts_dir=receipts_dir,
            max_rows=max_rows,
            fs=fs,
            on_progress=on_progress,
        )
    finally:
        # Clean up actors
        for actor in actors:
            ray.kill(actor)
