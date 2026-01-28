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

"""Core processing functions for data preparation.

This package provides consolidated access to the core shard processing functions,
planning utilities, tokenizer providers, and work item definitions.

Processing Functions:
    - process_binidx_shard_core: Tokenize text to Megatron bin/idx format
    - process_jsonl_shard_core: Transform and write JSONL records
    - process_chat_sft_spool_core: Tokenize chat messages to spool intermediate
    - process_chat_sft_parquet_core: Pack spool to Parquet output

Planning:
    - create_shard_plan: Create size-balanced shard assignments
    - get_pending_shards: Find shards that still need processing
    - resolve_tokenizer: Convert tokenizer config to internal format

Work Items:
    - DatasetWorkItem: Work item for pretrain datasets
    - ShardWorkItem: Work item for individual pretrain shards
    - SftDatasetWorkItem: Work item for SFT datasets
    - SftShardWorkItem: Work item for individual SFT shards

Usage:
    >>> from nemotron.data_prep.core import process_binidx_shard_core
    >>> stats, files = process_binidx_shard_core(
    ...     tokenize=tokenizer_fn,
    ...     text_field="text",
    ...     # ... other params
    ... )

Note:
    For most use cases, prefer using the recipe entry points
    (run_pretrain_pipeline, run_sft_pipeline) rather than calling
    these core functions directly.
"""

# Binidx tokenization (Megatron .bin/.idx format)
from nemotron.data_prep.core.shard_processor import (
    process_binidx_shard_files_core,
)

# JSONL processing (with transforms)
from nemotron.data_prep.core.jsonl_shard_core import (
    process_jsonl_shard_core,
)

# Chat SFT processing (tokenization + packing to Parquet)
from nemotron.data_prep.core.chat_sft_shard_core import (
    process_chat_sft_parquet_from_spool_core,
    process_chat_sft_spool_core,
)

# Planning utilities
from nemotron.data_prep.core.planning import (
    apply_shard_sampling,
    create_jsonl_shard_plan,
    create_shard_plan,
    get_pending_jsonl_shards,
    get_pending_shards,
    resolve_tokenizer,
    serialize_shard_plan,
)

# Tokenizer providers
from nemotron.data_prep.core.providers import (
    create_tokenizer,
)

# Work item dataclasses
from nemotron.data_prep.core.work_items import (
    DatasetWorkItem,
    JsonlDatasetWorkItem,
    JsonlShardWorkItem,
    ShardWorkItem,
    SftDatasetWorkItem,
    SftShardWorkItem,
)

# Standardized aliases following naming convention
# process_<format>_shard_core pattern
process_binidx_shard_core = process_binidx_shard_files_core
process_chat_sft_parquet_core = process_chat_sft_parquet_from_spool_core

__all__ = [
    # Binidx (pretrain tokenization)
    "process_binidx_shard_files_core",  # Original name (canonical)
    "process_binidx_shard_core",  # Standardized alias
    # JSONL (already follows pattern)
    "process_jsonl_shard_core",
    # Chat SFT (two-phase: spool + parquet)
    "process_chat_sft_spool_core",  # Phase 1: tokenize to spool
    "process_chat_sft_parquet_from_spool_core",  # Original name
    "process_chat_sft_parquet_core",  # Standardized alias
    # Planning
    "create_shard_plan",
    "create_jsonl_shard_plan",
    "get_pending_shards",
    "get_pending_jsonl_shards",
    "apply_shard_sampling",
    "serialize_shard_plan",
    "resolve_tokenizer",
    # Tokenizer providers
    "create_tokenizer",
    # Work items
    "DatasetWorkItem",
    "ShardWorkItem",
    "SftDatasetWorkItem",
    "SftShardWorkItem",
    "JsonlDatasetWorkItem",
    "JsonlShardWorkItem",
]
