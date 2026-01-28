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

    [SftDatasetWorkItem] → SftPlanStage → DownloadStage → PackedSftParquetStage
                           (fan-out)      (HF/S3/GCS)    (spool + parquet + receipts)

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
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import cosmos_xenna.pipelines.v1 as pipelines_v1

from nemotron.data_prep.config import (
    FormatResult,
    InternalTokenizerConfig,
    ObservabilityConfig,
    TokenizerConfig,
)
from nemotron.data_prep.observability.wandb_hook import (
    compute_dataset_input_bytes,
    log_plan_table_to_wandb,
    make_wandb_stats_hook,
)
from nemotron.data_prep.utils.filesystem import ensure_dir, get_filesystem, read_json, write_json
from nemotron.data_prep.core.planning import resolve_tokenizer
from nemotron.data_prep.stages import (
    DownloadStage,
    DownloadStageConfig,
    PackedSftParquetStage,
    PackedSftParquetStageConfig,
    PipelineContext,
    SftPlanStage,
    SftPlanStageConfig,
)
from nemotron.data_prep.utils.hf_env import detect_hf_env_vars
from nemotron.data_prep.core.work_items import SftDatasetWorkItem
from nemotron.data_prep.recipes.execution_mode import ExecutionModeRequest, decide_execution_mode_for_stages

if TYPE_CHECKING:
    from nemotron.data_prep.blend import DataBlend

logger = logging.getLogger(__name__)


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
# Driver: Setup + Finalize
# =============================================================================


def _normalize_tokenizer(tokenizer: TokenizerConfig | Mapping[str, Any] | str) -> TokenizerConfig:
    """Convert various tokenizer specs to TokenizerConfig."""
    if isinstance(tokenizer, TokenizerConfig):
        return tokenizer
    if isinstance(tokenizer, str):
        return TokenizerConfig(model=tokenizer)
    return TokenizerConfig(**dict(tokenizer))


def _setup_run(
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


def _finalize_run(
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

    def _find_plan_hash(dataset_name: str) -> str | None:
        """Find the plan_hash for a dataset by scanning its directory."""
        dataset_base = f"{context.run_dir}/datasets/{dataset_name}"
        try:
            subdirs = [p for p in fs.ls(dataset_base) if fs.isdir(p)]
            for subdir in subdirs:
                plan_path = f"{subdir}/plan.json"
                if fs.exists(plan_path):
                    return subdir.split("/")[-1]
        except Exception:
            pass
        return None

    def _aggregate_dataset(dataset_name: str, plan_hash: str) -> dict[str, Any]:
        """Aggregate statistics from all completed receipts for one dataset."""
        receipts_dir = f"{context.run_dir}/datasets/{dataset_name}/{plan_hash}/receipts"
        out = {
            "num_shards_completed": 0,
            "total_sequences": 0,
            "total_packed_sequences": 0,
            "total_tokens": 0,
            "total_parquet_bytes": 0,
        }
        try:
            receipt_files = fs.glob(f"{receipts_dir}/shard_*.json")
        except Exception:
            return out

        seen: set[int] = set()
        for p in receipt_files:
            try:
                r = read_json(fs, p)
                if r.get("status") != "completed" or r.get("plan_hash") != plan_hash:
                    continue
                shard_index = int(r.get("shard_index", -1))
                if shard_index in seen:
                    continue
                seen.add(shard_index)

                st = r.get("stats", {}) or {}
                out["num_shards_completed"] += 1
                out["total_sequences"] += int(st.get("num_sequences", 0) or 0)
                out["total_packed_sequences"] += int(st.get("num_packed_sequences", 0) or 0)
                out["total_tokens"] += int(st.get("total_tokens", 0) or 0)

                files = r.get("files", {}) or {}
                out["total_parquet_bytes"] += int(((files.get("parquet") or {}).get("bytes", 0)) or 0)
            except Exception:
                continue
        return out

    # Aggregate stats and build data_paths
    dataset_stats: dict[str, dict[str, Any]] = {}
    dataset_plan_hashes: dict[str, str] = {}
    data_paths: list[str] = []

    for d in blend.datasets:
        plan_hash = _find_plan_hash(d.name)
        if not plan_hash:
            continue

        dataset_plan_hashes[d.name] = plan_hash
        dataset_stats[d.name] = _aggregate_dataset(d.name, plan_hash)

        if d.weight > 0:
            prefix = f"{context.run_dir}/datasets/{d.name}/{plan_hash}/shard"
            data_paths.extend([str(d.weight), prefix])

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
# Main Entry Point
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
    # Stage configs (optional, uses defaults if not provided)
    plan_stage: SftPlanStageConfig | None = None,
    download_stage: DownloadStageConfig | None = None,
    tokenization_stage: PackedSftParquetStageConfig | None = None,
    # Pipeline config (optional, uses defaults if not provided)
    observability: ObservabilityConfig | None = None,
) -> FormatResult:
    """
    SFT pipeline (3-stage design).

    Architecture:
        [SftDatasetWorkItem] → SftPlanStage → DownloadStage → PackedSftParquetStage
                              (fan-out)                     (receipts)
        + Driver-side finalize (scan receipts)

    Args:
        blend: DataBlend with datasets to process
        output_dir: Root output directory
        tokenizer: Tokenizer specification
        num_shards: Number of output shards per dataset
        dtype: Token dtype (kept for API symmetry; Parquet spec uses int32 tokens)
        messages_field_default: Default messages field name
        tools_field_default: Default tools field name
        chat_template: Chat template (name/path/inline)
        used_in_filter: Optional used_in filter value
        used_in_field: Field name containing used_in
        pack_size: Maximum tokens per packed sequence
        algorithm: Packing algorithm
        seed: Packing seed (defaults to sample_seed when None)
        parquet_row_group_size: Parquet row group size (bins per group)
        parquet_compression: Parquet compression codec
        max_doc_tokens: Truncate documents longer than this
        max_rows: Maximum rows per shard
        sample: Sampling specification
        sample_seed: Random seed for sampling (and default for packing when seed=None)
        force: Create new run namespace even if config matches
        execution_mode: Execution mode ('auto', 'streaming', 'batch', or pipelines_v1.ExecutionMode).
            'auto' (default) uses STREAMING if cluster CPUs suffice, BATCH otherwise.
        plan_stage: Config for planning stage (defaults to SftPlanStageConfig())
        download_stage: Config for download stage (defaults to DownloadStageConfig())
        tokenization_stage: Config for tokenization stage (defaults to PackedSftParquetStageConfig())
        observability: Pipeline observability config (defaults to ObservabilityConfig())

    Returns:
        FormatResult with run metadata, data paths, and statistics
    """
    # Use provided configs or defaults
    plan_stage_cfg = plan_stage or SftPlanStageConfig()
    download_stage_cfg = download_stage or DownloadStageConfig()
    tokenization_stage_cfg = tokenization_stage or PackedSftParquetStageConfig()
    observability_cfg = observability or ObservabilityConfig()

    # Phase 1: Setup (driver-side)
    dataset_items, context, resolved_tokenizer = _setup_run(
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

    # Phase 2: Execute 3-stage pipeline via xenna
    if dataset_items:
        # Build shared pipeline context
        pipeline_ctx = PipelineContext(
            output_root=str(output_dir),
            run_hash=context.run_hash,
            run_dir=context.run_dir,
            config_hash=None,  # Not needed for these stages
            resolved_tokenizer=resolved_tokenizer,
            observability=observability_cfg,
            hf_env=detect_hf_env_vars(),
        )

        # Log plan table to W&B before pipeline runs
        log_plan_table_to_wandb(
            observability=observability_cfg,
            pipeline_kind="sft",
            dataset_items=dataset_items,
            run_hash=context.run_hash,
        )

        # Build dataset_num_shards mapping for progress tracking
        dataset_num_shards = {item.dataset_name: item.num_shards for item in dataset_items}

        # Pre-compute dataset sizes from HF metadata (available before plan.json exists)
        dataset_input_bytes = compute_dataset_input_bytes(dataset_items)

        # Setup W&B stats hook for real-time logging
        # This patches PipelineMonitor._make_stats to intercept stats
        wandb_hook = make_wandb_stats_hook(
            observability=observability_cfg,
            pipeline_kind="sft",
            run_hash=context.run_hash,
            run_dir=context.run_dir,
            dataset_names=context.dataset_names,
            dataset_num_shards=dataset_num_shards,
            dataset_input_bytes=dataset_input_bytes,
        )

        stage_specs = [
            # Stage 1: Plan (fan-out datasets to shards)
            pipelines_v1.StageSpec(
                SftPlanStage(plan_stage_cfg, pipeline_ctx),
                num_workers=1,
            ),
            # Stage 2: Download files (HF, S3, GCS, etc.)
            pipelines_v1.StageSpec(
                DownloadStage(download_stage_cfg, pipeline_ctx),
                num_workers_per_node=1,
            ),
            # Stage 3: Spool + Pack to Parquet + Receipts
            pipelines_v1.StageSpec(
                PackedSftParquetStage(tokenization_stage_cfg, pipeline_ctx),
                slots_per_actor=1,
            ),
        ]

        decision = decide_execution_mode_for_stages(
            requested=execution_mode,
            stage_specs=stage_specs,
            pipeline_name="sft",
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
    return _finalize_run(context, blend, output_dir)


__all__ = ["run_sft_pipeline"]