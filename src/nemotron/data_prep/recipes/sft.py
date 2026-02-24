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
SFT Pipeline Recipe - ChatSFT to packed Parquet shards.

This recipe composes reusable stages into a complete SFT data pipeline:

    [SftDatasetWorkItem] → PlanStage(SftPlanAdapter) → DownloadStage → PackedSftParquetStage
                           (fan-out)                    (HF/S3/GCS)    (spool + parquet + receipts)

    + Driver-side finalize (scan receipts after pipeline completes)

Key Design Decisions:
    - 3 stages: PlanStage fans out datasets to shards, then parallel work
    - Finalize in driver: Scan receipts after run_pipeline() returns
    - Single receipt writer: PackedSftParquetStage owns all checkpoint logic
    - Output format: Packed Parquet per docs/packed-sft-impl-parquet-nemotron.md

Usage:
    from nemotron.data_prep.recipes import run_sft_pipeline
    from nemotron.data_prep.blend import DataBlend

    blend = DataBlend.load("blend.json")
    result = run_sft_pipeline(
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
from nemotron.data_prep.core.planning import PlanRequest, resolve_tokenizer, verify_parquet_output
from nemotron.data_prep.stages import (
    DownloadStage,
    DownloadStageConfig,
    PackedSftParquetStage,
    PackedSftParquetStageConfig,
    PipelineContext,
    PlanStage,
    SftPlanStageConfig,
)
from nemotron.data_prep.utils.hf_env import detect_hf_env_vars
from nemotron.data_prep.core.work_items import SftDatasetWorkItem, SftShardWorkItem
from nemotron.data_prep.recipes.execution_mode import ExecutionModeRequest, resolve_execution_mode

if TYPE_CHECKING:
    from nemotron.data_prep.blend import DataBlend


# =============================================================================
# Run Context
# =============================================================================


@dataclass(frozen=True)
class SftRunContext:
    """Metadata for the run - passed to finalize."""

    run_hash: str
    run_dir: str
    num_shards: int
    dataset_names: list[str]


# =============================================================================
# Adapter: SFT planning
# =============================================================================


class SftPlanAdapter:
    """Adapter for SFT dataset and shard work items.

    PlanStage calls these methods to:
    1. to_plan_request — build a PlanRequest from dataset config
    2. to_shard_item — create SftShardWorkItems for each pending shard
    3. get_output_verifier — check parquet files exist on resume
    """

    def to_plan_request(self, item: SftDatasetWorkItem) -> PlanRequest:
        return PlanRequest(
            dataset_config=DatasetConfig(
                name=item.dataset_name,
                path=item.path,
                weight=item.weight,
                split=item.split,
                subset=item.subset,
                text_field=item.messages_field,
            ),
            num_shards=item.num_shards,
            config_hash=item.config_hash,
            tokenizer_config=InternalTokenizerConfig(**item.tokenizer_config),
            output_config=InternalOutputConfig(
                num_shards=item.num_shards,
                dtype=item.dtype,
                max_doc_tokens=item.max_doc_tokens,
                max_rows=item.max_rows,
            ),
        )

    def to_shard_item(
        self,
        item: SftDatasetWorkItem,
        *,
        plan_hash: str,
        shard_index: int,
        assignment: dict[str, Any],
        output_dir: str,
        receipts_dir: str,
    ) -> SftShardWorkItem:
        spool_dir = f"{output_dir.rstrip('/')}/spool/shard_{shard_index:06d}"
        return SftShardWorkItem(
            dataset_name=item.dataset_name,
            plan_hash=plan_hash,
            shard_index=shard_index,
            assignment=assignment,
            output_dir=output_dir,
            receipts_dir=receipts_dir,
            spool_dir=spool_dir,
            dtype=item.dtype,
            messages_field=item.messages_field,
            tools_field=item.tools_field,
            chat_template=item.chat_template,
            max_doc_tokens=item.max_doc_tokens,
            max_rows=item.max_rows,
            used_in_filter=item.used_in_filter,
            used_in_field=item.used_in_field,
            pack_size=item.pack_size,
            algorithm=item.algorithm,
            seed=item.seed,
            parquet_row_group_size=item.parquet_row_group_size,
            parquet_compression=item.parquet_compression,
        )

    def get_output_verifier(
        self, fs: AbstractFileSystem
    ) -> Callable[[dict, str, AbstractFileSystem], bool] | None:
        return verify_parquet_output


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


def setup_sft_run(
    blend: "DataBlend",
    output_dir: str | Path,
    tokenizer: TokenizerConfig | Mapping[str, Any] | str,
    *,
    num_shards: int,
    dtype: str = "int32",
    messages_field_default: str = "messages",
    tools_field_default: str = "tools",
    chat_template: str | None = None,
    used_in_filter: str | None = None,
    used_in_field: str = "used_in",
    pack_size: int = 2048,
    algorithm: str = "first_fit_shuffle",
    seed: int | None = None,
    parquet_row_group_size: int = 1000,
    parquet_compression: str = "zstd",
    max_doc_tokens: int | None = None,
    max_rows: int | None = None,
    sample: str | int | None = None,
    sample_seed: int = 42,
    force: bool = False,
) -> tuple[list[SftDatasetWorkItem], SftRunContext, dict[str, Any]]:
    """
    Setup an SFT run: compute run_hash, create SftDatasetWorkItems.

    Returns:
        - List of SftDatasetWorkItems (input to pipeline)
        - SftRunContext (for finalize)
        - Resolved tokenizer dict
    """
    if getattr(blend, "datasets", None) is None:
        raise ValueError("run_sft_pipeline expects single-blend mode: blend.datasets != None")
    if num_shards <= 0:
        raise ValueError(f"num_shards must be > 0, got {num_shards}")
    if pack_size <= 0:
        raise ValueError(f"pack_size must be > 0, got {pack_size}")
    if parquet_row_group_size <= 0:
        raise ValueError(f"parquet_row_group_size must be > 0, got {parquet_row_group_size}")

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

    # Ensure deterministic packing seed for shuffle-based algorithms by default
    packing_seed = int(sample_seed) if seed is None else int(seed)

    # Build deterministic run config for hashing
    run_config: dict[str, Any] = {
        "datasets": [
            {
                "name": d.name,
                "path": d.path,
                "weight": d.weight,
                "split": d.split,
                "subset": d.subset,
                "messages_field": getattr(d, "messages_field", None) or messages_field_default,
                "tools_field": getattr(d, "tools_field", None) or tools_field_default,
            }
            for d in blend.datasets
        ],
        "tokenizer": resolved_tokenizer,
        "output": {
            "format": "packed_sft_parquet",
            "num_shards": int(num_shards),
            "dtype": dtype,
            "messages_field": messages_field_default,
            "tools_field": tools_field_default,
            "chat_template": chat_template,
            "used_in_filter": used_in_filter,
            "used_in_field": used_in_field,
            "max_doc_tokens": max_doc_tokens,
            "max_rows": max_rows,
            "pack_size": int(pack_size),
            "algorithm": str(algorithm),
            "seed": packing_seed,
            "parquet_row_group_size": int(parquet_row_group_size),
            "parquet_compression": str(parquet_compression),
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

    # Build SftDatasetWorkItems
    dataset_items: list[SftDatasetWorkItem] = []
    for d in blend.datasets:
        dataset_items.append(
            SftDatasetWorkItem(
                dataset_name=d.name,
                path=d.path,
                weight=d.weight,
                split=d.split,
                subset=d.subset,
                run_hash=run_hash,
                run_dir=run_dir,
                config_hash=config_hash,
                num_shards=int(num_shards),
                dtype=dtype,
                max_doc_tokens=max_doc_tokens,
                max_rows=max_rows,
                sample=sample,
                sample_seed=int(sample_seed),
                tokenizer_config=asdict(tokenizer_cfg),
                messages_field=getattr(d, "messages_field", None) or messages_field_default,
                tools_field=getattr(d, "tools_field", None) or tools_field_default,
                chat_template=chat_template,
                used_in_filter=used_in_filter,
                used_in_field=used_in_field,
                pack_size=int(pack_size),
                algorithm=str(algorithm),
                seed=packing_seed,
                parquet_row_group_size=int(parquet_row_group_size),
                parquet_compression=str(parquet_compression),
            )
        )

    context = SftRunContext(
        run_hash=run_hash,
        run_dir=run_dir,
        num_shards=int(num_shards),
        dataset_names=[d.name for d in blend.datasets],
    )

    return dataset_items, context, resolved_tokenizer


def finalize_sft_run(
    context: SftRunContext,
    blend: "DataBlend",
    output_dir: str | Path,
) -> FormatResult:
    """
    Finalize an SFT run: scan receipts, aggregate stats, build data_paths.

    This runs after the pipeline completes - scans all receipts to compute
    final statistics and build data_paths prefixes for training.
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
            "total_packed_sequences": 0,
            "total_tokens": 0,
            "total_parquet_bytes": 0,
        }
        for receipt in dataset_receipts.completed:
            st = receipt.get("stats", {}) or {}
            stats["num_shards_completed"] += 1
            stats["total_sequences"] += int(st.get("num_sequences", 0) or 0)
            stats["total_packed_sequences"] += int(st.get("num_packed_sequences", 0) or 0)
            stats["total_tokens"] += int(st.get("total_tokens", 0) or 0)
            files = receipt.get("files", {}) or {}
            stats["total_parquet_bytes"] += int(((files.get("parquet") or {}).get("bytes", 0)) or 0)
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
        from_cache=(total_sequences == 0),
        total_tokens=total_tokens,
        total_sequences=total_sequences,
    )


# =============================================================================
# Convenience Entry Point
# =============================================================================


def run_sft_pipeline(
    blend: "DataBlend",
    output_dir: str | Path,
    tokenizer: TokenizerConfig | Mapping[str, Any] | str,
    *,
    num_shards: int,
    dtype: str = "int32",
    messages_field_default: str = "messages",
    tools_field_default: str = "tools",
    chat_template: str | None = None,
    used_in_filter: str | None = None,
    used_in_field: str = "used_in",
    pack_size: int = 2048,
    algorithm: str = "first_fit_shuffle",
    seed: int | None = None,
    parquet_row_group_size: int = 1000,
    parquet_compression: str = "zstd",
    max_doc_tokens: int | None = None,
    max_rows: int | None = None,
    sample: str | int | None = None,
    sample_seed: int = 42,
    force: bool = False,
    execution_mode: ExecutionModeRequest = "auto",
    plan_stage: SftPlanStageConfig | None = None,
    download_stage: DownloadStageConfig | None = None,
    tokenization_stage: PackedSftParquetStageConfig | None = None,
    observability: ObservabilityConfig | None = None,
) -> FormatResult:
    """Convenience wrapper: setup → execute → finalize in one call.

    For full control over the pipeline stages, use setup_sft_run
    and finalize_sft_run with explicit PipelineSpec construction.
    """
    plan_stage_cfg = plan_stage or SftPlanStageConfig()
    download_stage_cfg = download_stage or DownloadStageConfig()
    tokenization_stage_cfg = tokenization_stage or PackedSftParquetStageConfig()
    observability_cfg = observability or ObservabilityConfig()

    # Phase 1: Setup
    dataset_items, context, resolved_tokenizer = setup_sft_run(
        blend=blend,
        output_dir=output_dir,
        tokenizer=tokenizer,
        num_shards=num_shards,
        dtype=dtype,
        messages_field_default=messages_field_default,
        tools_field_default=tools_field_default,
        chat_template=chat_template,
        used_in_filter=used_in_filter,
        used_in_field=used_in_field,
        pack_size=pack_size,
        algorithm=algorithm,
        seed=seed,
        parquet_row_group_size=parquet_row_group_size,
        parquet_compression=parquet_compression,
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
                PlanStage(plan_stage_cfg, pipeline_ctx, SftPlanAdapter()),
                num_workers=1,
            ),
            pipelines_v1.StageSpec(
                DownloadStage(download_stage_cfg, pipeline_ctx),
                num_workers_per_node=1,
            ),
            pipelines_v1.StageSpec(
                PackedSftParquetStage(tokenization_stage_cfg, pipeline_ctx),
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
        with pipeline_wandb_hook(dataset_items, pipeline_ctx, "sft"):
            pipelines_v1.run_pipeline(spec)

    # Phase 3: Finalize
    return finalize_sft_run(context, blend, output_dir)


__all__ = [
    "SftPlanAdapter",
    "SftRunContext",
    "finalize_sft_run",
    "run_sft_pipeline",
    "setup_sft_run",
]
