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

    [JsonlDatasetWorkItem] → PlanStage(JsonlPlanAdapter) → DownloadStage → JsonlShardStage
                              (fan-out)                      (HF/S3/GCS)    (transform + write)

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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import cosmos_xenna.pipelines.v1 as pipelines_v1
from fsspec import AbstractFileSystem

from nemotron.data_prep.config import DatasetConfig, ObservabilityConfig
from nemotron.data_prep.core.planning import PlanRequest, verify_jsonl_output
from nemotron.data_prep.observability import pipeline_wandb_hook
from nemotron.data_prep.utils.filesystem import ensure_dir, get_filesystem, write_json
from nemotron.data_prep.core.finalize import scan_dataset_receipts
from nemotron.data_prep.stages import (
    DownloadStage,
    DownloadStageConfig,
    PlanStage,
    PipelineContext,
)
from nemotron.data_prep.stages.jsonl_plan import JsonlPlanStageConfig
from nemotron.data_prep.stages.jsonl_write import JsonlShardStage, JsonlShardStageConfig
from nemotron.data_prep.utils.hf_env import detect_hf_env_vars
from nemotron.data_prep.core.work_items import JsonlDatasetWorkItem, JsonlShardWorkItem
from nemotron.data_prep.recipes.execution_mode import ExecutionModeRequest, resolve_execution_mode

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
# Adapter: JSONL planning
# =============================================================================


class JsonlPlanAdapter:
    """Adapter for JSONL dataset and shard work items.

    PlanStage calls these methods to:
    1. to_plan_request — build a PlanRequest from dataset config (with transform fingerprint)
    2. to_shard_item — create JsonlShardWorkItems for each pending shard
    3. get_output_verifier — check JSONL output files exist on resume
    """

    def to_plan_request(self, item: JsonlDatasetWorkItem) -> PlanRequest:
        transform_fingerprint = hashlib.sha256(
            json.dumps({"resolve_hf_placeholders": item.resolve_hf_placeholders}, sort_keys=True).encode()
        ).hexdigest()[:16]

        return PlanRequest(
            dataset_config=DatasetConfig(
                name=item.dataset_name,
                path=item.path,
                weight=item.weight,
                split=item.split,
                subset=item.subset,
                text_field=item.text_field,
            ),
            num_shards=item.num_shards,
            config_hash=item.config_hash,
            tokenizer_config=None,
            transform_fingerprint=transform_fingerprint,
        )

    def to_shard_item(
        self,
        item: JsonlDatasetWorkItem,
        *,
        plan_hash: str,
        shard_index: int,
        assignment: dict[str, Any],
        output_dir: str,
        receipts_dir: str,
    ) -> JsonlShardWorkItem:
        return JsonlShardWorkItem(
            dataset_name=item.dataset_name,
            plan_hash=plan_hash,
            shard_index=shard_index,
            assignment=assignment,
            output_dir=output_dir,
            receipts_dir=receipts_dir,
            text_field=item.text_field,
            compression=item.compression,
            max_rows=item.max_rows,
            resolve_hf_placeholders=item.resolve_hf_placeholders,
        )

    def get_output_verifier(
        self, fs: AbstractFileSystem
    ) -> Callable[[dict, str, AbstractFileSystem], bool] | None:
        return verify_jsonl_output


# =============================================================================
# Driver: Setup + Finalize
# =============================================================================


def setup_rl_run(
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


def finalize_rl_run(
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
    split_dataset_names = [f"{dataset_name_base}__{split_name_mapping.get(split, split)}" for split in available_splits]
    by_dataset = scan_dataset_receipts(run_dir, split_dataset_names, fs)

    split_paths: dict[str, str] = {}
    total_records = 0
    run_hash = Path(run_dir).name

    for hf_split in available_splits:
        output_split_name = split_name_mapping.get(hf_split, hf_split)
        split_dataset_name = f"{dataset_name_base}__{output_split_name}"
        dataset_receipts = by_dataset.get(split_dataset_name)
        if not dataset_receipts:
            logger.warning(f"Could not find receipts for {split_dataset_name}")
            continue

        split_records = 0
        shard_paths: list[str] = []
        for receipt in sorted(
            dataset_receipts.completed,
            key=lambda r: int(r.get("shard_index", 0)),
        ):
            num_records = int(receipt.get("stats", {}).get("num_records", 0) or 0)
            split_records += num_records
            output_file = receipt.get("output_file")
            if output_file and num_records > 0:
                dataset_prefix = dataset_receipts.prefix.rsplit("/", 1)[0]
                shard_paths.append(f"{dataset_prefix}/{output_file}")

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
# Convenience Entry Point
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
    """Convenience wrapper: setup → execute → finalize in one call.

    For full control over the pipeline stages, use setup_rl_run
    and finalize_rl_run with explicit PipelineSpec construction.
    """
    plan_stage_cfg = plan_stage or JsonlPlanStageConfig()
    download_stage_cfg = download_stage or DownloadStageConfig()
    jsonl_stage_cfg = jsonl_stage or JsonlShardStageConfig()
    observability_cfg = observability or ObservabilityConfig()

    # Phase 1: Setup
    dataset_items, run_hash, run_dir, config_hash, available_splits = setup_rl_run(
        blend=blend,
        output_dir=output_dir,
        sample=sample,
        force=force,
        compression=compression,
        num_shards_per_split=num_shards_per_split,
        resolve_hf_placeholders=resolve_hf_placeholders,
    )

    # Phase 2: Execute 3-stage pipeline
    if dataset_items:
        pipeline_ctx = PipelineContext(
            output_root=str(output_dir),
            run_hash=run_hash,
            run_dir=run_dir,
            config_hash=config_hash,
            resolved_tokenizer=None,
            observability=observability_cfg,
            hf_env=detect_hf_env_vars(),
        )
        stage_specs = [
            pipelines_v1.StageSpec(
                PlanStage(plan_stage_cfg, pipeline_ctx, JsonlPlanAdapter()),
                num_workers=1,
            ),
            pipelines_v1.StageSpec(
                DownloadStage(download_stage_cfg, pipeline_ctx),
                num_workers_per_node=1,
            ),
            pipelines_v1.StageSpec(
                JsonlShardStage(jsonl_stage_cfg, pipeline_ctx),
                slots_per_actor=1,
            ),
        ]
        spec = pipelines_v1.PipelineSpec(
            input_data=dataset_items,
            stages=stage_specs,
            config=pipelines_v1.PipelineConfig(
                execution_mode=resolve_execution_mode(stage_specs, execution_mode),
                return_last_stage_outputs=False,
                logging_interval_s=observability_cfg.pipeline_logging_interval_s,
            ),
        )
        with pipeline_wandb_hook(dataset_items, pipeline_ctx, "rl"):
            pipelines_v1.run_pipeline(spec)

    # Phase 3: Finalize
    dataset_name_base = blend.datasets[0].name
    return finalize_rl_run(run_dir, output_dir, available_splits, dataset_name_base)


__all__ = [
    "JsonlPlanAdapter",
    "RlResolveResult",
    "finalize_rl_run",
    "run_rl_resolve_pipeline",
    "setup_rl_run",
]
