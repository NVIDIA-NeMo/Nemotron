"""Main orchestration for the tokenization pipeline."""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import ray

from nemotron.data_prep.config import (
    DatasetConfig,
    OutputConfig,
    ShardPlan,
    SourceChangedError,
    TokenizerConfig,
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

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a pipeline run."""

    output_dir: str
    run_hash: str
    datasets: dict[str, dict]
    total_tokens: int
    total_sequences: int
    elapsed_sec: float


@dataclass
class DatasetExecutionPlan:
    """Execution plan for a single dataset."""

    name: str
    config: DatasetConfig
    plan: ShardPlan
    dataset_dir: str
    receipts_dir: str
    pending_indices: list[int]
    cached_stats: dict


def tokenize_to_binidx(
    config_path: str,
    output_dir: str,
    sample: str | int | None = None,
    sample_seed: int = 42,
    num_actors: int = 4,
    force: bool = False,
) -> PipelineResult:
    """
    Run tokenization pipeline with true resume support.

    Args:
        config_path: Path to blend config JSON
        output_dir: Output directory (local or cloud URI)
        sample: Shard-level sample spec ("10%" or count)
        sample_seed: Random seed for sampling
        num_actors: Number of ShardProcessor actors
        force: Create new run (new namespace)

    Returns:
        PipelineResult with statistics and output location
    """
    start_time = time.time()

    # Get filesystem
    fs, base_path = get_filesystem(output_dir)

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Validate
    if "num_shards" not in config.get("output", {}):
        raise ValueError("output.num_shards is required")

    # Compute run hash (includes sampling params)
    run_config = config.copy()
    if sample is not None:
        run_config["_sample"] = {"spec": str(sample), "seed": sample_seed}
    config_hash = hashlib.sha256(
        json.dumps(run_config, sort_keys=True).encode()
    ).hexdigest()[:16]

    # Run namespace
    run_hash = config_hash if not force else f"{config_hash}_{int(time.time())}"
    run_dir = f"{base_path}/runs/{run_hash}"
    ensure_dir(fs, run_dir)

    # Freeze config
    write_json(fs, f"{run_dir}/config.json", run_config)

    tokenizer_config = TokenizerConfig(**config["tokenizer"])
    output_config = OutputConfig(**config["output"])

    # ========================================
    # PHASE 1: Planning
    # ========================================
    con.planning_header()

    execution_plans: list[DatasetExecutionPlan] = []
    plan_hashes = {}
    resolved_tokenizer = None
    plan_infos = []

    for dataset_entry in config["datasets"]:
        dataset_config = DatasetConfig(**dataset_entry)
        name = dataset_config.name

        # Create or load plan
        plan = load_or_create_plan(
            dataset_config=dataset_config,
            output_config=output_config,
            tokenizer_config=tokenizer_config,
            config_hash=config_hash,
            run_dir=run_dir,
            fs=fs,
            force=force,
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
        cached_stats = aggregate_stats_from_receipts(receipts_dir, plan, fs)

        # Apply sampling
        sampled_count = None
        if sample is not None:
            pending_indices = apply_shard_sampling(all_pending, plan, sample, sample_seed)
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
            DatasetExecutionPlan(
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
    con.plan_summary(plan_infos, run_hash)

    # ========================================
    # PHASE 2: Execution
    # ========================================
    results = {}
    has_work = any(ep.pending_indices for ep in execution_plans)

    if has_work:
        con.execution_header()

        # Create live status panel with all datasets
        live_status = con.create_live_status(
            datasets=[(ep.name, len(ep.pending_indices) or ep.cached_stats["num_shards_completed"])
                      for ep in execution_plans],
            run_hash=run_hash,
        )
        live_status.start()

        try:
            for ep in execution_plans:
                if not ep.pending_indices:
                    # All cached
                    results[ep.name] = ep.cached_stats
                    live_status.cache_dataset(ep.name)
                    continue

                live_status.start_dataset(ep.name)

                # Process shards with actor pool
                process_shards_with_actors(
                    pending_indices=ep.pending_indices,
                    plan=ep.plan,
                    dataset_dir=ep.dataset_dir,
                    receipts_dir=ep.receipts_dir,
                    dataset_config=ep.config,
                    output_config=output_config,
                    fs=fs,
                    num_actors=num_actors,
                    on_progress=lambda name=ep.name: live_status.advance_dataset(name),
                )

                # Aggregate final stats
                results[ep.name] = aggregate_stats_from_receipts(ep.receipts_dir, ep.plan, fs)
                live_status.complete_dataset(ep.name)
        finally:
            live_status.stop()
    else:
        # All cached, no live display needed
        for ep in execution_plans:
            results[ep.name] = ep.cached_stats

    # Generate outputs
    generate_blend_file(run_dir, config, plan_hashes, fs)
    generate_manifest(run_dir, config, results, plan_hashes, run_hash, resolved_tokenizer, fs)

    elapsed = time.time() - start_time

    return PipelineResult(
        output_dir=output_dir,
        run_hash=run_hash,
        datasets=results,
        total_tokens=sum(r.get("total_tokens", 0) for r in results.values()),
        total_sequences=sum(r.get("total_sequences", 0) for r in results.values()),
        elapsed_sec=elapsed,
    )


def process_shards_with_actors(
    pending_indices: list[int],
    plan: ShardPlan,
    dataset_dir: str,
    receipts_dir: str,
    dataset_config: DatasetConfig,
    output_config: OutputConfig,
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

    # Initial submission up to max_in_flight
    while shard_queue and len(pending_list) < max_in_flight:
        submit_task(shard_queue.pop(0))

    # Process with backpressure
    def process_loop(advance_fn: Callable[[], None]) -> None:
        nonlocal pending_list, shard_queue
        while pending_list:
            # ray.wait returns (done, remaining) - use remaining directly
            done, pending_list = ray.wait(pending_list, num_returns=1, timeout=60)

            for future in done:
                shard_index = future_to_shard.pop(future)
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


class PlanDriftError(Exception):
    """Raised when a new plan would create drift from existing plans."""

    pass


def load_or_create_plan(
    dataset_config: DatasetConfig,
    output_config: OutputConfig,
    tokenizer_config: TokenizerConfig,
    config_hash: str,
    run_dir: str,
    fs,
    force: bool = False,
) -> ShardPlan:
    """
    Load existing plan or create new one.

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
                d for d in fs.ls(dataset_plans_dir)
                if fs.isdir(d) and fs.exists(f"{d}/plan.json")
            ]
            if existing_plan_dirs:
                existing_hashes = [d.split("/")[-1] for d in existing_plan_dirs]
                raise PlanDriftError(
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


def aggregate_stats_from_receipts(
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
            if (
                receipt.get("status") == "completed"
                and receipt.get("plan_hash") == plan.plan_hash
            ):
                stats["num_shards_completed"] += 1
                stats["total_sequences"] += receipt["stats"]["num_sequences"]
                stats["total_tokens"] += receipt["stats"]["total_tokens"]
                stats["total_bin_bytes"] += receipt["files"]["bin"]["bytes"]
                stats["total_idx_bytes"] += receipt["files"]["idx"]["bytes"]
        except Exception:
            pass

    return stats


def generate_blend_file(
    run_dir: str,
    config: dict,
    plan_hashes: dict[str, str],
    fs,
):
    """Generate Megatron blend file using tracked plan hashes."""
    blend = []

    for dataset_entry in config["datasets"]:
        name = dataset_entry["name"]
        weight = dataset_entry.get("weight", 1.0)
        include = dataset_entry.get("include_in_blend", True)

        if weight > 0 and include and name in plan_hashes:
            plan_hash = plan_hashes[name]
            prefix = f"{run_dir}/datasets/{name}/{plan_hash}/shard"
            blend.append([weight, prefix])

    write_json(fs, f"{run_dir}/blend.json", {"datasets": blend})


def generate_manifest(
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
