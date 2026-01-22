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

"""PipelineSpec factories for Xenna pipeline types.

This module provides factory functions to build cosmos_xenna PipelineSpec objects
for each pipeline type (pretrain, jsonl, chat_sft). The factories are pure functions
that do not run pipelines or handle observability - that's the executor's job.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import cosmos_xenna.pipelines.v1 as pipelines_v1

from nemotron.data_prep.xenna.stages import (
    ChatSftCentralPackStage,
    ChatSftSpoolStage,
    HfPredownloadStage,
    JsonlShardStage,
    PretrainShardStage,
)
from nemotron.data_prep.xenna.work_items import (
    ChatSftShardWorkItem,
    JsonlShardWorkItem,
    ShardWorkItem,
)

if TYPE_CHECKING:
    from nemotron.data_prep.config import XennaConfig


def _make_pipeline_config(*, logging_interval_s: float) -> pipelines_v1.PipelineConfig:
    """Create standard PipelineConfig for Xenna pipelines."""
    return pipelines_v1.PipelineConfig(
        execution_mode=pipelines_v1.ExecutionMode.STREAMING,
        return_last_stage_outputs=False,
        logging_interval_s=logging_interval_s,
        monitoring_verbosity_level=pipelines_v1.VerbosityLevel.INFO,
    )


def build_pretrain_pipeline_spec(
    *,
    tasks: Sequence[ShardWorkItem],
    resolved_tokenizer: dict,
    output_root: str,
    xenna_cfg: "XennaConfig",
) -> pipelines_v1.PipelineSpec:
    """Build PipelineSpec for pretrain binidx processing.

    Args:
        tasks: List of ShardWorkItem to process
        resolved_tokenizer: Resolved tokenizer configuration dict
        output_root: Root output directory
        xenna_cfg: Xenna configuration

    Returns:
        PipelineSpec ready to be executed by run_pipeline()
    """
    return pipelines_v1.PipelineSpec(
        input_data=list(tasks),
        stages=[
            pipelines_v1.StageSpec(
                HfPredownloadStage(
                    max_concurrent_downloads=xenna_cfg.max_concurrent_downloads,
                    output_root=output_root,
                    download_timeout_sec=xenna_cfg.hf_download_timeout_sec,
                    max_retries=xenna_cfg.hf_download_max_retries,
                ),
                num_workers_per_node=1,
            ),
            pipelines_v1.StageSpec(
                PretrainShardStage(
                    resolved_tokenizer=resolved_tokenizer,
                    output_root=output_root,
                ),
                # Limit workers to prevent OOM (each worker ~4GB)
                **({"num_workers": xenna_cfg.max_shard_workers} if xenna_cfg.max_shard_workers else {}),
            ),
        ],
        config=_make_pipeline_config(logging_interval_s=xenna_cfg.pipeline_logging_interval_s),
    )


def build_jsonl_pipeline_spec(
    *,
    tasks: Sequence[JsonlShardWorkItem],
    output_root: str,
    text_field: str,
    transform,
    compression: str,
    max_rows: int | None,
    resolve_hf_placeholders: bool,
    xenna_cfg: "XennaConfig",
) -> pipelines_v1.PipelineSpec:
    """Build PipelineSpec for JSONL processing.

    Args:
        tasks: List of JsonlShardWorkItem to process
        output_root: Root output directory
        text_field: Field name containing text in input records
        transform: Optional transform function for records
        compression: Output compression ("none" or "zstd")
        max_rows: Maximum rows per shard (None for unlimited)
        resolve_hf_placeholders: Whether to resolve HuggingFace placeholders
        xenna_cfg: Xenna configuration

    Returns:
        PipelineSpec ready to be executed by run_pipeline()
    """
    return pipelines_v1.PipelineSpec(
        input_data=list(tasks),
        stages=[
            pipelines_v1.StageSpec(
                HfPredownloadStage(
                    max_concurrent_downloads=xenna_cfg.max_concurrent_downloads,
                    output_root=output_root,
                    download_timeout_sec=xenna_cfg.hf_download_timeout_sec,
                    max_retries=xenna_cfg.hf_download_max_retries,
                ),
                num_workers_per_node=1,
            ),
            pipelines_v1.StageSpec(
                JsonlShardStage(
                    output_root=output_root,
                    text_field=text_field,
                    transform=transform,
                    compression=compression,
                    max_rows=max_rows,
                    resolve_hf_placeholders=resolve_hf_placeholders,
                ),
                # Limit workers to prevent OOM on large datasets
                num_workers=4,
            ),
        ],
        config=_make_pipeline_config(logging_interval_s=xenna_cfg.pipeline_logging_interval_s),
    )


def build_chat_sft_pipeline_spec(
    *,
    tasks: Sequence[ChatSftShardWorkItem],
    output_root: str,
    resolved_tokenizer: dict,
    messages_field: str,
    tools_field: str,
    pack_size: int,
    algorithm: str,
    dtype: str,
    chat_template: str | None,
    max_doc_tokens: int | None,
    max_rows: int | None,
    seed: int | None,
    used_in_filter: str | None,
    used_in_field: str,
    xenna_cfg: "XennaConfig",
) -> pipelines_v1.PipelineSpec:
    """Build PipelineSpec for Chat SFT processing.

    Args:
        tasks: List of ChatSftShardWorkItem to process
        output_root: Root output directory
        resolved_tokenizer: Resolved tokenizer configuration dict
        messages_field: Field name for messages in input records
        tools_field: Field name for tools in input records
        pack_size: Maximum tokens per packed sequence
        algorithm: Packing algorithm
        dtype: Token dtype
        chat_template: Chat template (name, path, or inline)
        max_doc_tokens: Maximum tokens per document
        max_rows: Maximum rows per shard
        seed: Random seed for packing
        used_in_filter: Filter for used_in field
        used_in_field: Field name for used_in filtering
        xenna_cfg: Xenna configuration

    Returns:
        PipelineSpec ready to be executed by run_pipeline()
    """
    return pipelines_v1.PipelineSpec(
        input_data=list(tasks),
        stages=[
            pipelines_v1.StageSpec(
                HfPredownloadStage(
                    max_concurrent_downloads=xenna_cfg.max_concurrent_downloads,
                    output_root=output_root,
                    download_timeout_sec=xenna_cfg.hf_download_timeout_sec,
                    max_retries=xenna_cfg.hf_download_max_retries,
                ),
                num_workers_per_node=1,
            ),
            pipelines_v1.StageSpec(
                ChatSftSpoolStage(
                    resolved_tokenizer=resolved_tokenizer,
                    output_root=output_root,
                    messages_field=messages_field,
                    tools_field=tools_field,
                    pack_size=pack_size,
                    algorithm=algorithm,
                    dtype=dtype,
                    chat_template=chat_template,
                    max_doc_tokens=max_doc_tokens,
                    max_rows=max_rows,
                    seed=seed,
                    used_in_filter=used_in_filter,
                    used_in_field=used_in_field,
                ),
                # Let Xenna auto-scale - spool stage is memory-efficient
            ),
            pipelines_v1.StageSpec(
                ChatSftCentralPackStage(
                    output_root=output_root,
                    pack_size=pack_size,
                    algorithm=algorithm,
                    dtype=dtype,
                    seed=seed,
                ),
                num_workers=1,  # Must be single-worker for centralized packing
            ),
        ],
        config=_make_pipeline_config(logging_interval_s=xenna_cfg.pipeline_logging_interval_s),
    )
