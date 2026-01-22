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

"""Work item types passed through Xenna pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ShardWorkItem:
    """Payload for Xenna shard processing."""

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
class JsonlShardWorkItem:
    """Payload for Xenna JSONL shard processing."""

    dataset_name: str
    shard_index: int
    assignment: dict[str, Any]
    output_dir: str
    receipts_dir: str
    text_field: str
    compression: str
    max_rows: int | None
    resolve_hf_placeholders: bool = False


@dataclass
class ChatSftShardWorkItem:
    """Payload for Xenna ChatSFT shard processing."""

    dataset_name: str
    shard_index: int
    assignment: dict[str, Any]
    output_dir: str
    receipts_dir: str
    max_rows: int | None


@dataclass
class ChatSftSpoolWorkItem:
    """Payload for Xenna ChatSFT SequenceSpool generation (tokenize-only)."""

    dataset_name: str
    shard_index: int
    assignment: dict[str, Any]
    output_dir: str
    receipts_dir: str
    spool_dir: str
    max_rows: int | None
