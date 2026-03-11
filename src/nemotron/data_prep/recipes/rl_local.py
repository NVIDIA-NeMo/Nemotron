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
RL Local Pipeline Recipe — process local JSONL files for RL training.

Supports two paths:

1. **Direct path** (SWE, RLHF): Read local JSONL, split into train/val
   by holding out the last N rows for validation.

2. **Pipeline path** (RLVR): Run the xenna 3-stage pipeline
   (Plan → Download → JsonlShard) with HF placeholder resolution,
   then split the resolved output.

Both paths produce the same output: train/val JSONL files + manifest.json,
wrapped in a SplitJsonlDataArtifact.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cosmos_xenna.pipelines.v1 as pipelines_v1

from nemotron.data_prep.config import ObservabilityConfig
from nemotron.data_prep.observability import pipeline_wandb_hook
from nemotron.data_prep.recipes.execution_mode import resolve_execution_mode
from nemotron.data_prep.recipes.rl import (
    JsonlPlanAdapter,
    finalize_rl_run,
)
from nemotron.data_prep.stages import (
    DownloadStage,
    DownloadStageConfig,
    PipelineContext,
    PlanStage,
)
from nemotron.data_prep.stages.jsonl_plan import JsonlPlanStageConfig
from nemotron.data_prep.stages.jsonl_write import JsonlShardStage, JsonlShardStageConfig
from nemotron.data_prep.utils.filesystem import ensure_dir, get_filesystem, write_json
from nemotron.data_prep.utils.hf_env import detect_hf_env_vars
from nemotron.data_prep.core.work_items import JsonlDatasetWorkItem

logger = logging.getLogger(__name__)


# =============================================================================
# Result Type
# =============================================================================


@dataclass(frozen=True)
class LocalSplitResult:
    """Result from splitting a local JSONL file into train/val."""

    train_path: str
    val_path: str | None
    train_rows: int
    val_rows: int
    run_dir: str
    manifest_path: str


# =============================================================================
# Direct Path: split_local_jsonl
# =============================================================================


def split_local_jsonl(
    input_path: Path,
    output_dir: Path,
    *,
    val_holdout: int = 100,
    sample: int | None = None,
    force: bool = False,
) -> LocalSplitResult:
    """Split a local JSONL file into train/val by holding out the last N rows.

    This is the direct path for SWE/RLHF stages that don't need
    placeholder resolution — just read, split, write.

    Args:
        input_path: Path to source JSONL file.
        output_dir: Root output directory.
        val_holdout: Number of rows to hold out for validation (from end of file).
        sample: If set, limit total rows to this count.
        force: If True, ignore cached results and re-run.

    Returns:
        LocalSplitResult with paths to train/val files.
    """
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL file not found: {input_path}")

    # Compute deterministic run hash for caching
    stat = input_path.stat()
    run_config = {
        "input_path": str(input_path),
        "input_mtime": stat.st_mtime,
        "input_size": stat.st_size,
        "val_holdout": val_holdout,
        "sample": sample,
    }
    config_hash = hashlib.sha256(
        json.dumps(run_config, sort_keys=True).encode()
    ).hexdigest()[:16]
    run_hash = config_hash if not force else f"{config_hash}_{int(time.time())}"

    # Setup run directory
    run_dir = output_dir / "runs" / run_hash
    run_dir.mkdir(parents=True, exist_ok=True)

    # Check for cached result
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not force:
        try:
            existing = json.loads(manifest_path.read_text())
            if existing.get("run_hash") == run_hash:
                # Verify cached output files still exist
                cached_train = existing.get("train", "")
                cached_val = existing.get("val")
                if cached_train and Path(cached_train).exists() and (
                    not cached_val or Path(cached_val).exists()
                ):
                    logger.info(f"Using cached result from {run_dir}")
                    return LocalSplitResult(
                    train_path=existing.get("train", ""),
                    val_path=existing.get("val"),
                    train_rows=existing.get("train_rows", 0),
                    val_rows=existing.get("val_rows", 0),
                    run_dir=str(run_dir),
                    manifest_path=str(manifest_path),
                )
        except (json.JSONDecodeError, KeyError):
            pass  # Re-run if cache is corrupt

    # Save run config
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))

    # Count total lines (streaming, memory-efficient)
    logger.info(f"Counting rows in {input_path}...")
    total_rows = 0
    with open(input_path, "r") as f:
        for _ in f:
            total_rows += 1
    logger.info(f"Total rows: {total_rows}")

    # Apply sample limit
    effective_total = min(total_rows, sample) if sample else total_rows

    # Compute split point
    if effective_total <= val_holdout:
        logger.warning(
            f"Total rows ({effective_total}) <= val_holdout ({val_holdout}). "
            f"All rows go to train, no validation split."
        )
        train_end = effective_total
        val_start = effective_total  # No val rows
    else:
        train_end = effective_total - val_holdout
        val_start = train_end

    # Create output directories
    train_dir = run_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    train_file = train_dir / "train.jsonl"

    val_dir = run_dir / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    val_file = val_dir / "val.jsonl"

    # Single-pass write: read lines, route to train or val
    logger.info(f"Splitting: train=[0, {train_end}), val=[{val_start}, {effective_total})")
    train_count = 0
    val_count = 0

    with (
        open(input_path, "r") as fin,
        open(train_file, "w") as f_train,
        open(val_file, "w") as f_val,
    ):
        for i, line in enumerate(fin):
            if i >= effective_total:
                break
            if i < train_end:
                f_train.write(line)
                train_count += 1
            else:
                f_val.write(line)
                val_count += 1

    logger.info(f"Written: train={train_count} rows, val={val_count} rows")

    # Remove empty val file if no val rows
    val_path_str: str | None = None
    if val_count > 0:
        val_path_str = str(val_file.resolve())
    else:
        val_file.unlink(missing_ok=True)

    train_path_str = str(train_file.resolve())

    # Write manifest
    manifest = {
        "train": train_path_str,
        "val": val_path_str or "",
        "test": "",
        "mode": "local_split",
        "source": str(input_path),
        "run_hash": run_hash,
        "train_rows": train_count,
        "val_rows": val_count,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info(f"Manifest written to {manifest_path}")

    return LocalSplitResult(
        train_path=train_path_str,
        val_path=val_path_str,
        train_rows=train_count,
        val_rows=val_count,
        run_dir=str(run_dir),
        manifest_path=str(manifest_path),
    )


# =============================================================================
# Pipeline Path: resolve placeholders then split
# =============================================================================


def run_resolve_and_split(
    input_path: Path,
    output_dir: Path,
    *,
    val_holdout: int = 100,
    sample: int | None = None,
    force: bool = False,
    execution_mode: str = "auto",
) -> LocalSplitResult:
    """Run xenna pipeline to resolve HF placeholders, then split into train/val.

    For RLVR stages where the JSONL files contain `_hf_placeholder` entries
    that need to be resolved from DAPO-Math-17k and Skywork-OR1-RL-Data.

    Two-phase approach:
    1. Run the existing rl.py pipeline to resolve placeholders → single resolved JSONL
    2. Apply split_local_jsonl() on the resolved output

    Args:
        input_path: Path to source JSONL file with placeholders.
        output_dir: Root output directory.
        val_holdout: Number of rows to hold out for validation.
        sample: If set, limit total rows.
        force: If True, ignore cached results.
        execution_mode: Pipeline execution mode ('auto', 'streaming', 'batch').

    Returns:
        LocalSplitResult with paths to train/val files.
    """
    from nemotron.data_prep.utils.hf_placeholder import SUPER3_TARGET_DATASETS

    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL file not found: {input_path}")

    # Phase 1: Resolve placeholders via xenna pipeline
    # Output goes to a 'resolved' subdirectory
    resolved_dir = output_dir / "resolved"

    # Compute run hash
    stat = input_path.stat()
    resolve_config = {
        "input_path": str(input_path),
        "input_mtime": stat.st_mtime,
        "input_size": stat.st_size,
        "resolve_hf_placeholders": True,
        "sample": sample,
    }
    config_hash = hashlib.sha256(
        json.dumps(resolve_config, sort_keys=True).encode()
    ).hexdigest()[:16]
    run_hash = config_hash if not force else f"{config_hash}_{int(time.time())}"

    # Setup run directory for resolution
    fs, base_path = get_filesystem(str(resolved_dir))
    run_dir = f"{str(resolved_dir).rstrip('/')}/runs/{run_hash}"
    ensure_dir(fs, run_dir)
    write_json(fs, f"{run_dir}/config.json", resolve_config)

    # Create work item — single file, single "shard", treat whole file as one split
    # Fields are set directly rather than going through DataBlend since we have
    # a single local file (no HF split discovery needed).
    dataset_item = JsonlDatasetWorkItem(
        dataset_name=input_path.stem,
        path=str(input_path),
        weight=1.0,
        split=None,
        subset=None,
        text_field="text",
        run_hash=run_hash,
        run_dir=run_dir,
        config_hash=config_hash,
        num_shards=1,
        compression="none",
        max_rows=sample,
        resolve_hf_placeholders=True,
    )

    # Run 3-stage pipeline
    pipeline_ctx = PipelineContext(
        output_root=str(resolved_dir),
        run_hash=run_hash,
        run_dir=run_dir,
        config_hash=config_hash,
        resolved_tokenizer=None,
        observability=ObservabilityConfig(),
        hf_env=detect_hf_env_vars(),
        hf_placeholder_targets=SUPER3_TARGET_DATASETS,
    )
    stage_specs = [
        pipelines_v1.StageSpec(
            PlanStage(JsonlPlanStageConfig(), pipeline_ctx, JsonlPlanAdapter()),
            num_workers=1,
        ),
        pipelines_v1.StageSpec(
            DownloadStage(DownloadStageConfig(), pipeline_ctx),
            num_workers_per_node=1,
        ),
        pipelines_v1.StageSpec(
            JsonlShardStage(JsonlShardStageConfig(), pipeline_ctx),
            slots_per_actor=1,
        ),
    ]
    spec = pipelines_v1.PipelineSpec(
        input_data=[dataset_item],
        stages=stage_specs,
        config=pipelines_v1.PipelineConfig(
            execution_mode=resolve_execution_mode(stage_specs, execution_mode),
        ),
    )
    with pipeline_wandb_hook([dataset_item], pipeline_ctx, "rl"):
        pipelines_v1.run_pipeline(spec)

    # Finalize — find the resolved output file
    result = finalize_rl_run(
        run_dir, resolved_dir, [input_path.stem], input_path.stem
    )

    # Find the resolved JSONL file from the pipeline output
    # finalize_rl_run returns split_paths with our dataset name as key
    resolved_path = None
    for split_name, path_str in result.split_paths.items():
        if path_str and Path(path_str).exists():
            resolved_path = Path(path_str)
            break

    if resolved_path is None:
        raise RuntimeError(
            f"Pipeline produced no output files. Check logs. run_dir={run_dir}"
        )

    logger.info(f"Placeholder resolution complete: {resolved_path}")

    # Phase 2: Split the resolved output into train/val
    return split_local_jsonl(
        resolved_path,
        output_dir,
        val_holdout=val_holdout,
        sample=None,  # Already applied during resolution
        force=force,
    )


__all__ = [
    "LocalSplitResult",
    "run_resolve_and_split",
    "split_local_jsonl",
]
