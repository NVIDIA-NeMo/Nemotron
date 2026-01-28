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

"""
RL Resolve Pipeline Recipe - JSONL processing with HF placeholder resolution.

This recipe composes reusable stages into a complete JSONL data pipeline
for RL training data:

    [JsonlDatasetWorkItem] → JsonlPlanStage → DownloadStage → JsonlShardStage
                              (fan-out)        (HF/S3/GCS)    (transform + write)

    + Driver-side finalize (scan receipts, write manifest.json)

Key Design Decisions:
    - Dedicated JSONL path (not shoehorned into pretrain PlanStage which requires tokenizer)
    - Same 3-stage pattern as pretrain: Plan → Download → Process
    - Manifest contract preserved: {train, val, test} absolute paths
    - Transform fingerprint in plan_hash so toggling placeholder resolution
      invalidates cached results
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cosmos_xenna.pipelines.v1 as pipelines_v1

from nemotron.data_prep.config import ObservabilityConfig
from nemotron.data_prep.observability.wandb_hook import (
    compute_dataset_input_bytes,
    log_plan_table_to_wandb,
    make_wandb_stats_hook,
)
from nemotron.data_prep.utils.filesystem import ensure_dir, get_filesystem, read_json, write_json
from nemotron.data_prep.stages import (
    DownloadStage,
    DownloadStageConfig,
    PipelineContext,
)
from nemotron.data_prep.stages.jsonl_plan import JsonlPlanStage, JsonlPlanStageConfig
from nemotron.data_prep.stages.jsonl_write import JsonlShardStage, JsonlShardStageConfig
from nemotron.data_prep.utils.hf_env import detect_hf_env_vars
from nemotron.data_prep.core.work_items import JsonlDatasetWorkItem
from nemotron.data_prep.recipes.execution_mode import ExecutionModeRequest, decide_execution_mode_for_stages

if TYPE_CHECKING:
    from nemotron.data_prep.blend import DataBlend

logger = logging.getLogger(__name__)


# =============================================================================
# Result Type
# =============================================================================


@dataclass(frozen=True)
class RlResolveResult:
    """Result from running the RL resolve pipeline.

    Attributes:
        run_hash: Deterministic hash identifying this run configuration.
        run_dir: Path to the runs/{run_hash} directory.
        split_paths: Mapping of split name (train/val/test) to absolute JSONL path.
        total_records: Total records written across all splits.
        manifest_path: Path to the manifest.json file.
    """

    run_hash: str
    run_dir: str
    split_paths: dict[str, str]
    total_records: int
    manifest_path: str


# =============================================================================
# Driver: Setup + Finalize
# =============================================================================


def _setup_run(
    blend: "DataBlend",
    output_dir: str | Path,
    *,
    sample: int | None,
    force: bool,
    compression: Literal["none", "zstd"],
    num_shards_per_split: int,
    resolve_hf_placeholders: bool,
) -> tuple[list[JsonlDatasetWorkItem], str, str, str, list[str]]:
    """
    Setup an RL resolve run: discover splits, compute run_hash, create work items.

    Returns:
        - List of JsonlDatasetWorkItems (input to pipeline)
        - run_hash
        - run_dir
        - config_hash
        - available_splits (HF split names)
    """
    from datasets import get_dataset_split_names

    if len(blend.datasets) != 1:
        raise ValueError(
            f"RL resolve pipeline expects exactly one dataset in blend, got {len(blend.datasets)}"
        )

    dataset = blend.datasets[0]
    fs, base_path = get_filesystem(str(output_dir))

    # Handle hf:// prefix
    dataset_path = dataset.path
    if dataset_path.startswith("hf://"):
        dataset_path = dataset_path[5:]

    # Discover available splits from HF
    available_splits = get_dataset_split_names(dataset_path)

    # Normalize split names for output directories
    split_name_mapping = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }

    # Build deterministic run config for hashing
    # Include transform fingerprint so toggling placeholder resolution changes the hash
    transform_fingerprint = hashlib.sha256(
        json.dumps({"resolve_hf_placeholders": resolve_hf_placeholders}, sort_keys=True).encode()
    ).hexdigest()[:16]

    run_config = {
        "datasets": [{
            "name": dataset.name,
            "path": dataset.path,
            "split": None,  # We process all splits
            "subset": dataset.subset,
            "text_field": getattr(dataset, "text_field", None) or "text",
        }],
        "output": {
            "format": "jsonl",
            "num_shards_per_split": num_shards_per_split,
            "compression": compression,
        },
        "available_splits": sorted(available_splits),
        "transform_fingerprint": transform_fingerprint,
    }
    if sample is not None:
        run_config["_sample"] = sample

    # Compute run hash
    config_hash = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()[:16]
    run_hash = config_hash if not force else f"{config_hash}_{int(time.time())}"

    # Create run directory
    run_dir = f"{base_path.rstrip('/')}/runs/{run_hash}"
    ensure_dir(fs, run_dir)
    write_json(fs, f"{run_dir}/config.json", run_config)

    # Build JsonlDatasetWorkItems (one per split)
    dataset_items: list[JsonlDatasetWorkItem] = []
    for hf_split in available_splits:
        output_split_name = split_name_mapping.get(hf_split, hf_split)
        # Use split-specific dataset name for filesystem safety
        split_dataset_name = f"{dataset.name}__{output_split_name}"

        dataset_items.append(
            JsonlDatasetWorkItem(
                dataset_name=split_dataset_name,
                path=dataset.path,
                weight=dataset.weight,
                split=hf_split,
                subset=dataset.subset,
                text_field=getattr(dataset, "text_field", None) or "text",
                run_hash=run_hash,
                run_dir=run_dir,
                config_hash=config_hash,
                num_shards=num_shards_per_split,
                compression=compression,
                max_rows=sample,
                resolve_hf_placeholders=resolve_hf_placeholders,
            )
        )

    return dataset_items, run_hash, run_dir, config_hash, available_splits


def _finalize_run(
    run_dir: str,
    output_dir: str | Path,
    available_splits: list[str],
    dataset_name_base: str,
) -> RlResolveResult:
    """
    Finalize an RL resolve run: scan receipts, write manifest.json.

    Scans the run directory for completed receipts and builds a manifest
    mapping split names to absolute JSONL paths.
    """
    fs, _ = get_filesystem(str(output_dir))

    split_name_mapping = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }

    split_paths: dict[str, str] = {}
    total_records = 0
    run_hash = Path(run_dir).name

    for hf_split in available_splits:
        output_split_name = split_name_mapping.get(hf_split, hf_split)
        split_dataset_name = f"{dataset_name_base}__{output_split_name}"

        # Find plan_hash directory
        dataset_base = f"{run_dir}/datasets/{split_dataset_name}"
        plan_hash = None
        try:
            subdirs = [p for p in fs.ls(dataset_base) if fs.isdir(p)]
            for subdir in subdirs:
                plan_path = f"{subdir}/plan.json"
                if fs.exists(plan_path):
                    plan_hash = subdir.split("/")[-1]
                    break
        except Exception:
            logger.warning(f"Could not find plan directory for {split_dataset_name}")
            continue

        if not plan_hash:
            continue

        # Read receipts to get total records and find shard paths
        receipts_dir = f"{dataset_base}/{plan_hash}/receipts"
        split_records = 0
        shard_paths: list[str] = []

        try:
            receipt_files = fs.glob(f"{receipts_dir}/shard_*.json")
        except Exception:
            receipt_files = []

        for receipt_path in receipt_files:
            try:
                receipt = read_json(fs, receipt_path)
                if receipt.get("status") != "completed":
                    continue
                if receipt.get("plan_hash") != plan_hash:
                    continue

                num_records = receipt.get("stats", {}).get("num_records", 0)
                split_records += num_records

                output_file = receipt.get("output_file")
                if output_file and num_records > 0:
                    shard_path = f"{dataset_base}/{plan_hash}/{output_file}"
                    shard_paths.append(shard_path)
            except Exception as e:
                logger.warning(f"Failed to parse receipt {receipt_path}: {e}")

        total_records += split_records

        if shard_paths:
            # For single-shard mode (num_shards=1), use the shard path directly
            # Convert to absolute path
            abs_path = str(Path(shard_paths[0]).resolve()) if not shard_paths[0].startswith("/") else shard_paths[0]
            split_paths[output_split_name] = abs_path
            logger.info(f"Split {output_split_name}: {split_records} records at {abs_path}")

    # Write manifest.json at the output_dir root
    output_dir_str = str(output_dir).rstrip("/")
    manifest = {
        "train": split_paths.get("train", ""),
        "val": split_paths.get("val", ""),
        "test": split_paths.get("test", ""),
        "mode": "resolve",
        "source_splits": available_splits,
        "run_hash": run_hash,
    }

    manifest_path = f"{output_dir_str}/manifest.json"
    write_json(fs, manifest_path, manifest)

    return RlResolveResult(
        run_hash=run_hash,
        run_dir=run_dir,
        split_paths=split_paths,
        total_records=total_records,
        manifest_path=manifest_path,
    )


# =============================================================================
# Main Entry Point
# =============================================================================


def run_rl_resolve_pipeline(
    *,
    blend: "DataBlend",
    output_dir: str | Path,
    sample: int | None = None,
    force: bool = False,
    compression: Literal["none", "zstd"] = "none",
    num_shards_per_split: int = 1,
    resolve_hf_placeholders: bool = True,
    execution_mode: ExecutionModeRequest = "auto",
    plan_stage: JsonlPlanStageConfig | None = None,
    download_stage: DownloadStageConfig | None = None,
    jsonl_stage: JsonlShardStageConfig | None = None,
    observability: ObservabilityConfig | None = None,
) -> RlResolveResult:
    """
    RL resolve pipeline (3-stage design).

    Architecture:
        [JsonlDatasetWorkItem] → JsonlPlanStage → DownloadStage → JsonlShardStage
                                  (fan-out)                        (transform + write)
        + Driver-side finalize (scan receipts, write manifest)

    This pipeline processes a single HF dataset across all available splits
    (train/validation/test), optionally resolving HF placeholder records.

    Args:
        blend: DataBlend with exactly one dataset to process.
        output_dir: Root output directory.
        sample: Limit rows per split (for quick tests).
        force: Create new run namespace even if config matches.
        compression: Output compression ("none" or "zstd").
        num_shards_per_split: Number of output shards per split.
        resolve_hf_placeholders: Whether to resolve HF placeholder records.
        execution_mode: Execution mode ('auto', 'streaming', 'batch', or pipelines_v1.ExecutionMode).
            'auto' (default) uses STREAMING if cluster CPUs suffice, BATCH otherwise.
        plan_stage: Config for JSONL planning stage.
        download_stage: Config for download stage.
        jsonl_stage: Config for JSONL shard writing stage.
        observability: Pipeline observability config.

    Returns:
        RlResolveResult with run metadata, split paths, and manifest path.
    """
    plan_stage_cfg = plan_stage or JsonlPlanStageConfig()
    download_stage_cfg = download_stage or DownloadStageConfig()
    jsonl_stage_cfg = jsonl_stage or JsonlShardStageConfig()
    observability_cfg = observability or ObservabilityConfig()

    # Phase 1: Setup (driver-side)
    dataset_items, run_hash, run_dir, config_hash, available_splits = _setup_run(
        blend=blend,
        output_dir=output_dir,
        sample=sample,
        force=force,
        compression=compression,
        num_shards_per_split=num_shards_per_split,
        resolve_hf_placeholders=resolve_hf_placeholders,
    )

    # Phase 2: Execute 3-stage pipeline via xenna
    if dataset_items:
        pipeline_ctx = PipelineContext(
            output_root=str(output_dir),
            run_hash=run_hash,
            run_dir=run_dir,
            config_hash=config_hash,
            resolved_tokenizer=None,  # No tokenizer for JSONL
            observability=observability_cfg,
            hf_env=detect_hf_env_vars(),
        )

        # Log plan table to W&B before pipeline runs
        log_plan_table_to_wandb(
            observability=observability_cfg,
            pipeline_kind="rl",
            dataset_items=dataset_items,
            run_hash=run_hash,
        )

        # Build dataset_num_shards mapping for progress tracking
        dataset_num_shards = {item.dataset_name: item.num_shards for item in dataset_items}

        # Pre-compute dataset sizes from HF metadata (available before plan.json exists)
        dataset_input_bytes = compute_dataset_input_bytes(dataset_items)

        # Setup W&B stats hook
        wandb_hook = make_wandb_stats_hook(
            observability=observability_cfg,
            pipeline_kind="rl",
            run_hash=run_hash,
            run_dir=run_dir,
            dataset_names=[item.dataset_name for item in dataset_items],
            dataset_num_shards=dataset_num_shards,
            dataset_input_bytes=dataset_input_bytes,
        )

        stage_specs = [
            # Stage 1: Plan (fan-out datasets/splits to shards)
            pipelines_v1.StageSpec(
                JsonlPlanStage(plan_stage_cfg, pipeline_ctx),
                num_workers=1,  # Single planner
            ),
            # Stage 2: Download files (HF, S3, GCS, etc.)
            pipelines_v1.StageSpec(
                DownloadStage(download_stage_cfg, pipeline_ctx),
                num_workers_per_node=1,  # One downloader per node
            ),
            # Stage 3: Write JSONL shards
            pipelines_v1.StageSpec(
                JsonlShardStage(jsonl_stage_cfg, pipeline_ctx),
                slots_per_actor=1,  # Prevents 2x memory from concurrent tasks
            ),
        ]

        decision = decide_execution_mode_for_stages(
            requested=execution_mode,
            stage_specs=stage_specs,
            pipeline_name="rl",
            logger=logger,
        )

        pipeline_spec = pipelines_v1.PipelineSpec(
            input_data=dataset_items,
            stages=stage_specs,
            config=pipelines_v1.PipelineConfig(
                execution_mode=decision.resolved,
                return_last_stage_outputs=False,
                logging_interval_s=observability_cfg.pipeline_logging_interval_s,
            ),
        )

        # Run pipeline with optional W&B stats logging
        if wandb_hook:
            with wandb_hook:
                pipelines_v1.run_pipeline(pipeline_spec)
        else:
            pipelines_v1.run_pipeline(pipeline_spec)

    # Phase 3: Finalize (driver-side)
    dataset_name_base = blend.datasets[0].name
    return _finalize_run(run_dir, output_dir, available_splits, dataset_name_base)
