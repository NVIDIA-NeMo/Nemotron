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

"""Work item types passed through pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class DatasetWorkItem:
    """
    Input to PlanStage - one per dataset in a blend.

    This work item carries all information needed for PlanStage to:
    - Discover input files
    - Create shard plan
    - Fan out to ShardWorkItems
    """

    dataset_name: str
    path: str
    weight: float
    split: str | None
    subset: str | None
    text_field: str

    # Run context (set by driver)
    run_hash: str
    run_dir: str
    config_hash: str
    num_shards: int
    dtype: str
    min_doc_chars: int | None
    max_doc_tokens: int | None
    max_rows: int | None
    sample: str | int | None
    sample_seed: int

    # Resolved tokenizer config (for plan creation)
    tokenizer_config: dict = field(default_factory=dict)


@dataclass
class ShardWorkItem:
    """Payload for shard processing."""

    dataset_name: str
    plan_hash: str
    shard_index: int
    assignment: dict[str, Any]
    output_dir: str
    receipts_dir: str
    text_field: str
    dtype: str
    min_doc_chars: int | None
    max_doc_tokens: int | None
    max_rows: int | None


@dataclass
class SftDatasetWorkItem:
    """Input to SFT plan stage - one per dataset."""

    # Dataset identity (matches DatasetWorkItem pattern)
    dataset_name: str
    path: str
    weight: float
    split: str | None
    subset: str | None

    # Run context (set by driver)
    run_hash: str
    run_dir: str
    config_hash: str

    # Planning/output partitioning
    num_shards: int
    dtype: str
    max_doc_tokens: int | None
    max_rows: int | None
    sample: str | int | None
    sample_seed: int

    # Resolved tokenizer config (for plan creation)
    tokenizer_config: dict = field(default_factory=dict)

    # ChatSFT parsing/tokenization options (consumed by stage 3)
    messages_field: str = "messages"
    tools_field: str = "tools"
    chat_template: str | None = None
    used_in_filter: str | None = None
    used_in_field: str = "used_in"

    # Packing options
    pack_size: int = 2048
    algorithm: str = "first_fit_shuffle"
    seed: int | None = None

    # Packed Parquet output options (per packed-sft-impl-parquet-nemotron.md)
    parquet_row_group_size: int = 1000
    parquet_compression: str = "zstd"


@dataclass
class SftShardWorkItem:
    """Payload for SFT shard processing (packed Parquet output)."""

    dataset_name: str
    plan_hash: str
    shard_index: int
    assignment: dict[str, Any]

    # Output locations
    output_dir: str
    receipts_dir: str
    spool_dir: str | None = None

    # Tokenization and filtering
    dtype: str = "int32"
    messages_field: str = "messages"
    tools_field: str = "tools"
    chat_template: str | None = None
    max_doc_tokens: int | None = None
    max_rows: int | None = None
    used_in_filter: str | None = None
    used_in_field: str = "used_in"

    # Packing
    pack_size: int = 2048
    algorithm: str = "first_fit_shuffle"
    seed: int | None = None

    # Packed Parquet output options
    parquet_row_group_size: int = 1000
    parquet_compression: str = "zstd"


@dataclass
class JsonlDatasetWorkItem:
    """
    Input to JsonlPlanStage - one per dataset/split in a JSONL pipeline.

    This work item carries all information needed for JsonlPlanStage to:
    - Discover input files
    - Create JSONL shard plan (without tokenizer resolution)
    - Fan out to JsonlShardWorkItems
    """

    dataset_name: str
    path: str
    weight: float
    split: str | None
    subset: str | None
    text_field: str

    # Run context (set by driver)
    run_hash: str
    run_dir: str
    config_hash: str

    num_shards: int
    compression: Literal["none", "zstd"] = "none"
    max_rows: int | None = None
    resolve_hf_placeholders: bool = False


@dataclass
class JsonlShardWorkItem:
    """Payload for JSONL shard processing."""

    dataset_name: str
    plan_hash: str
    shard_index: int
    assignment: dict[str, Any]
    output_dir: str
    receipts_dir: str
    text_field: str
    compression: str
    max_rows: int | None
    resolve_hf_placeholders: bool = False


__all__ = [
    "DatasetWorkItem",
    "ShardWorkItem",
    "SftDatasetWorkItem",
    "SftShardWorkItem",
    "JsonlDatasetWorkItem",
    "JsonlShardWorkItem",
]
