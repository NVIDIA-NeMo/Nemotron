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
Reusable Xenna pipeline stages for data preparation.

This module provides composable stages that can be assembled into
different pipeline recipes (pretrain, SFT, JSONL processing, etc.)

Stage Construction Pattern:
    Each stage is constructed with (stage_config, pipeline_context) where:
    - stage_config: Stage-specific settings (e.g., PlanStageConfig, DownloadStageConfig)
    - pipeline_context: Shared runtime config (PipelineContext)

    This pattern separates stage-specific tuning from shared runtime config.

Stages:
    PlanStage: Discovers files, creates shard plans, fans out to work items (requires PlanAdapter)
    DownloadStage: Ensures files are available locally (HF, S3, GCS, local)
    BinIdxTokenizationStage: Tokenizes text to Megatron .bin/.idx format
    PackedSftParquetStage: Tokenizes+packs SFT data to Parquet format

Protocol:
    PlanAdapter: Pipeline-specific adapter for PlanStage (see recipes/ for implementations)

Configuration:
    PipelineContext: Shared runtime context (output paths, tokenizer, credentials)
    PlanStageConfig: Config for PlanStage
    SftPlanStageConfig: Config for SFT plan stage
    DownloadStageConfig: Config for DownloadStage (batch size, hf_xet tuning)
    BinIdxTokenizationStageConfig: Config for BinIdxTokenizationStage
    PackedSftParquetStageConfig: Config for PackedSftParquetStage

Usage:
    from nemotron.data_prep.stages import (
        PipelineContext,
        PlanStage, PlanStageConfig,
        DownloadStage, DownloadStageConfig,
        BinIdxTokenizationStage, BinIdxTokenizationStageConfig,
    )

    ctx = PipelineContext(
        output_root="/data/output",
        run_hash="abc123",
        resolved_tokenizer={"model": "gpt2", ...},
        hf_env={"HF_HOME": "/cache/hf"},
    )

    pipeline = PipelineSpec(
        input_data=[...],
        stages=[
            StageSpec(PlanStage(PlanStageConfig(), ctx), num_workers=1),
            StageSpec(DownloadStage(DownloadStageConfig(batch_size=64), ctx), num_workers_per_node=1),
            StageSpec(BinIdxTokenizationStage(BinIdxTokenizationStageConfig(), ctx), slots_per_actor=1),
        ],
    )
"""

from nemotron.data_prep.stages.context import PipelineContext
from nemotron.data_prep.stages.download import DownloadStage, DownloadStageConfig
from nemotron.data_prep.stages.jsonl_plan import JsonlPlanStageConfig
from nemotron.data_prep.stages.jsonl_write import JsonlShardStage, JsonlShardStageConfig
from nemotron.data_prep.stages.megatron_bin_idx import BinIdxTokenizationStage, BinIdxTokenizationStageConfig
from nemotron.data_prep.stages.packed_sft_parquet import PackedSftParquetStage, PackedSftParquetStageConfig
from nemotron.data_prep.stages.plan import PlanAdapter, PlanStage, PlanStageConfig
from nemotron.data_prep.stages.sft_plan import SftPlanStageConfig

__all__ = [
    # Context
    "PipelineContext",
    # Protocol
    "PlanAdapter",
    # Stage configs
    "PlanStageConfig",
    "SftPlanStageConfig",
    "JsonlPlanStageConfig",
    "JsonlShardStageConfig",
    "DownloadStageConfig",
    "BinIdxTokenizationStageConfig",
    "PackedSftParquetStageConfig",
    # Stages
    "DownloadStage",
    "PlanStage",
    "JsonlShardStage",
    "BinIdxTokenizationStage",
    "PackedSftParquetStage",
]
