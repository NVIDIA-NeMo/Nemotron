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
Pretrain Pipeline Recipe - Tokenization to Megatron .bin/.idx.

This recipe composes reusable stages into a complete pretrain data pipeline:

    [DatasetWorkItem] → PlanStage → DownloadStage → BinIdxTokenizationStage
                       (fan-out)    (HF/S3/GCS)     (tokenize + receipts)

    + Driver-side finalize (scan receipts after pipeline completes)

Key Design Decisions:
    - 3 stages: PlanStage fans out datasets to shards, then parallel work
    - Memory proxy: Use Resources(cpus=K) instead of hardcoding max_workers
    - slots_per_actor=1: Prevents 2x memory from concurrent tasks
    - Finalize in driver: Scan receipts after run_pipeline() returns
    - Single receipt writer: BinIdxTokenizationStage owns all checkpoint logic

Usage:
    from nemotron.data_prep.recipes import run_pretrain_pipeline
    from nemotron.data_prep.blend import DataBlend

    blend = DataBlend.load("blend.json")
    result = run_pretrain_pipeline(
        blend=blend,
        output_dir="/output",
        tokenizer="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        num_shards=128,
    )
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import cosmos_xenna.pipelines.v1 as pipelines_v1

from collections.abc import Callable
from fsspec import AbstractFileSystem

from nemotron.data_prep.config import (
    DatasetConfig,
    FormatResult,
    InternalOutputConfig,
    InternalTokenizerConfig,
    ObservabilityConfig,
    TokenizerConfig,
)
from nemotron.data_prep.observability import pipeline_wandb_hook
from nemotron.data_prep.utils.filesystem import ensure_dir, get_filesystem, write_json
from nemotron.data_prep.core.finalize import scan_dataset_receipts
from nemotron.data_prep.core.planning import PlanRequest, resolve_tokenizer, verify_binidx_output
from nemotron.data_prep.stages import (
    BinIdxTokenizationStage,
    BinIdxTokenizationStageConfig,
    DownloadStage,
    DownloadStageConfig,
    PipelineContext,
    PlanStage,
    PlanStageConfig,
)
from nemotron.data_prep.utils.hf_env import detect_hf_env_vars
from nemotron.data_prep.core.work_items import DatasetWorkItem, ShardWorkItem
from nemotron.data_prep.recipes.execution_mode import ExecutionModeRequest, resolve_execution_mode

if TYPE_CHECKING:
    from nemotron.data_prep.blend import DataBlend


# =============================================================================
# Run Context
# =============================================================================


@dataclass(frozen=True)
class PretrainRunContext:
    """Metadata for the run - passed to finalize."""

    run_hash: str
    run_dir: str
    num_shards: int
    dataset_names: list[str]


# =============================================================================
# Adapter: Pretrain planning
# =============================================================================


class PretrainPlanAdapter:
    """Adapter for pretrain dataset and shard work items.

    PlanStage calls these methods to:
    1. to_plan_request — build a PlanRequest from dataset config
    2. to_shard_item — create ShardWorkItems for each pending shard
    3. get_output_verifier — check bin/idx files exist on resume
    """

    def to_plan_request(self, item: DatasetWorkItem) -> PlanRequest:
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
            tokenizer_config=InternalTokenizerConfig(**item.tokenizer_config),
            output_config=InternalOutputConfig(
                num_shards=item.num_shards,
                dtype=item.dtype,
                min_doc_chars=item.min_doc_chars,
                max_doc_tokens=item.max_doc_tokens,
                max_rows=item.max_rows,
            ),
        )

    def to_shard_item(
        self,
        item: DatasetWorkItem,
        *,
        plan_hash: str,
        shard_index: int,
        assignment: dict[str, Any],
        output_dir: str,
        receipts_dir: str,
    ) -> ShardWorkItem:
        assignment["hf_subset"] = item.subset
        assignment["hf_split"] = item.split
        return ShardWorkItem(
            dataset_name=item.dataset_name,
            plan_hash=plan_hash,
            shard_index=shard_index,
            assignment=assignment,
            output_dir=output_dir,
            receipts_dir=receipts_dir,
            text_field=item.text_field,
            dtype=item.dtype,
            min_doc_chars=item.min_doc_chars,
            max_doc_tokens=item.max_doc_tokens,
            max_rows=item.max_rows,
        )

    def get_output_verifier(
        self, fs: AbstractFileSystem
    ) -> Callable[[dict, str, AbstractFileSystem], bool] | None:
        return verify_binidx_output


# =============================================================================
# Driver: Setup + Finalize
# =============================================================================


def _normalize_tokenizer(tokenizer: TokenizerConfig | Mapping[str, Any] | str) -> TokenizerConfig:
    """Convert various tokenizer specs to TokenizerConfig."""
    if isinstance(tokenizer, TokenizerConfig):
        return tokenizer
    if isinstance(tokenizer, str):
        return TokenizerConfig(model=tokenizer)
    return TokenizerConfig(**dict(tokenizer))


def setup_pretrain_run(
    blend: "DataBlend",
    output_dir: str | Path,
    tokenizer: TokenizerConfig | Mapping[str, Any] | str,
    *,
    num_shards: int,
    dtype: str = "int32",
    text_field_default: str = "text",
    min_doc_chars: int | None = None,
    max_doc_tokens: int | None = None,
    max_rows: int | None = None,
    sample: str | int | None = None,
    sample_seed: int = 42,
    force: bool = False,
) -> tuple[list[DatasetWorkItem], PretrainRunContext, dict[str, Any]]:
    """
    Setup a pretrain run: compute run_hash, create DatasetWorkItems.

    Returns:
        - List of DatasetWorkItems (input to pipeline)
        - PretrainRunContext (for finalize)
        - Resolved tokenizer dict
    """
    if getattr(blend, "datasets", None) is None:
        raise ValueError("run_pretrain_pipeline expects single-blend mode: blend.datasets != None")
    if num_shards <= 0:
        raise ValueError(f"num_shards must be > 0, got {num_shards}")

    fs, base_path = get_filesystem(str(output_dir))
    tok_cfg = _normalize_tokenizer(tokenizer)

    # Resolve tokenizer to get SHA for determinism
    # Pass user-specified revision if provided; resolve_tokenizer will resolve to SHA
    tokenizer_cfg = InternalTokenizerConfig(
        type=tok_cfg.type,
        model=tok_cfg.model,
        revision=getattr(tok_cfg, "revision", None),
        add_eos=tok_cfg.add_eos,
        add_bos=tok_cfg.add_bos,
        trust_remote_code=tok_cfg.trust_remote_code,
    )
    resolved_tokenizer = resolve_tokenizer(tokenizer_cfg)

    # Build deterministic run config for hashing
    run_config: dict[str, Any] = {
        "datasets": [
            {
                "name": d.name,
                "path": d.path,
                "weight": d.weight,
                "split": d.split,
                "subset": d.subset,
                "text_field": getattr(d, "text_field", None) or text_field_default,
            }
            for d in blend.datasets
        ],
        "tokenizer": resolved_tokenizer,
        "output": {
            "format": "binidx",
            "num_shards": int(num_shards),
            "dtype": dtype,
            "min_doc_chars": min_doc_chars,
            "max_doc_tokens": max_doc_tokens,
            "max_rows": max_rows,
        },
    }
    if sample is not None:
        run_config["_sample"] = {"spec": str(sample), "seed": int(sample_seed)}

    # Compute run hash
    config_hash = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()[:16]
    run_hash = config_hash if not force else f"{config_hash}_{int(time.time())}"

    # Create run directory
    run_dir = f"{base_path.rstrip('/')}/runs/{run_hash}"
    ensure_dir(fs, run_dir)
    write_json(fs, f"{run_dir}/config.json", run_config)

    # Build DatasetWorkItems
    dataset_items: list[DatasetWorkItem] = []
    for d in blend.datasets:
        dataset_items.append(
            DatasetWorkItem(
                dataset_name=d.name,
                path=d.path,
                weight=d.weight,
                split=d.split,
                subset=d.subset,
                text_field=getattr(d, "text_field", None) or text_field_default,
                run_hash=run_hash,
                run_dir=run_dir,
                config_hash=config_hash,
                num_shards=num_shards,
                dtype=dtype,
                min_doc_chars=min_doc_chars,
                max_doc_tokens=max_doc_tokens,
                max_rows=max_rows,
                sample=sample,
                sample_seed=sample_seed,
                tokenizer_config=asdict(tokenizer_cfg),
            )
        )

    context = PretrainRunContext(
        run_hash=run_hash,
        run_dir=run_dir,
        num_shards=num_shards,
        dataset_names=[d.name for d in blend.datasets],
    )

    return dataset_items, context, resolved_tokenizer


def finalize_pretrain_run(
    context: PretrainRunContext,
    blend: "DataBlend",
    output_dir: str | Path,
) -> FormatResult:
    """
    Finalize a pretrain run: scan receipts, aggregate stats, build data_paths.

    This runs after the pipeline completes - scans all receipts to compute
    final statistics and build the Megatron-compatible data_paths list.
    """
    fs, _ = get_filesystem(str(output_dir))
    by_dataset = scan_dataset_receipts(context.run_dir, context.dataset_names, fs)

    # Aggregate stats and build data_paths
    dataset_stats: dict[str, dict[str, Any]] = {}
    data_paths: list[str] = []

    for d in blend.datasets:
        dataset_receipts = by_dataset.get(d.name)
        if not dataset_receipts:
            continue

        stats: dict[str, Any] = {
            "num_shards_completed": 0,
            "total_sequences": 0,
            "total_tokens": 0,
            "total_bin_bytes": 0,
            "total_idx_bytes": 0,
        }
        for receipt in dataset_receipts.completed:
            st = receipt.get("stats", {}) or {}
            stats["num_shards_completed"] += 1
            stats["total_sequences"] += int(st.get("num_sequences", 0) or 0)
            stats["total_tokens"] += int(st.get("total_tokens", 0) or 0)
            files = receipt.get("files", {}) or {}
            stats["total_bin_bytes"] += int(((files.get("bin") or {}).get("bytes", 0)) or 0)
            stats["total_idx_bytes"] += int(((files.get("idx") or {}).get("bytes", 0)) or 0)
        dataset_stats[d.name] = stats

        if d.weight > 0:
            data_paths.extend([str(d.weight), dataset_receipts.prefix])

    total_tokens = sum(int(s.get("total_tokens", 0)) for s in dataset_stats.values())
    total_sequences = sum(int(s.get("total_sequences", 0)) for s in dataset_stats.values())

    return FormatResult(
        run_hash=context.run_hash,
        run_dir=context.run_dir,
        output_dir=Path(output_dir),
        num_shards=context.num_shards,
        data_paths=data_paths,
        dataset_stats=dataset_stats,
        from_cache=(total_sequences == 0),  # All cached if no new work
        total_tokens=total_tokens,
        total_sequences=total_sequences,
    )


# =============================================================================
# Convenience Entry Point
# =============================================================================


def run_pretrain_pipeline(
    blend: "DataBlend",
    output_dir: str | Path,
    tokenizer: TokenizerConfig | Mapping[str, Any] | str,
    *,
    num_shards: int,
    dtype: str = "int32",
    text_field_default: str = "text",
    min_doc_chars: int | None = None,
    max_doc_tokens: int | None = None,
    max_rows: int | None = None,
    sample: str | int | None = None,
    sample_seed: int = 42,
    force: bool = False,
    execution_mode: ExecutionModeRequest = "auto",
    plan_stage: PlanStageConfig | None = None,
    download_stage: DownloadStageConfig | None = None,
    tokenization_stage: BinIdxTokenizationStageConfig | None = None,
    observability: ObservabilityConfig | None = None,
) -> FormatResult:
    """Convenience wrapper: setup → execute → finalize in one call.

    For full control over the pipeline stages, use setup_pretrain_run
    and finalize_pretrain_run with explicit PipelineSpec construction.
    """
    plan_stage_cfg = plan_stage or PlanStageConfig()
    download_stage_cfg = download_stage or DownloadStageConfig()
    tokenization_stage_cfg = tokenization_stage or BinIdxTokenizationStageConfig()
    observability_cfg = observability or ObservabilityConfig()

    # Phase 1: Setup
    dataset_items, context, resolved_tokenizer = setup_pretrain_run(
        blend=blend,
        output_dir=output_dir,
        tokenizer=tokenizer,
        num_shards=num_shards,
        dtype=dtype,
        text_field_default=text_field_default,
        min_doc_chars=min_doc_chars,
        max_doc_tokens=max_doc_tokens,
        max_rows=max_rows,
        sample=sample,
        sample_seed=sample_seed,
        force=force,
    )

    # Phase 2: Execute 3-stage pipeline
    if dataset_items:
        pipeline_ctx = PipelineContext(
            output_root=str(output_dir),
            run_hash=context.run_hash,
            run_dir=context.run_dir,
            config_hash=None,
            resolved_tokenizer=resolved_tokenizer,
            observability=observability_cfg,
            hf_env=detect_hf_env_vars(),
        )
        stage_specs = [
            pipelines_v1.StageSpec(
                PlanStage(plan_stage_cfg, pipeline_ctx, PretrainPlanAdapter()),
                num_workers=1,
            ),
            pipelines_v1.StageSpec(
                DownloadStage(download_stage_cfg, pipeline_ctx),
                num_workers_per_node=1,
            ),
            pipelines_v1.StageSpec(
                BinIdxTokenizationStage(tokenization_stage_cfg, pipeline_ctx),
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
        with pipeline_wandb_hook(dataset_items, pipeline_ctx, "pretrain"):
            pipelines_v1.run_pipeline(spec)

    # Phase 3: Finalize
    return finalize_pretrain_run(context, blend, output_dir)


__all__ = [
    "PretrainPlanAdapter",
    "PretrainRunContext",
    "finalize_pretrain_run",
    "run_pretrain_pipeline",
    "setup_pretrain_run",
]
