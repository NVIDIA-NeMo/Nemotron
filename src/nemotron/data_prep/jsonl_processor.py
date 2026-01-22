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

"""JsonlShardProcessor Ray actor for parallel JSONL output processing."""

import logging
from collections.abc import Callable

import ray
from fsspec import filesystem

from nemotron.data_prep.config import FileInfo
from nemotron.data_prep.jsonl_shard_core import process_jsonl_shard_core

logger = logging.getLogger(__name__)


@ray.remote
class JsonlShardProcessor:
    """Ray actor for processing data files to JSONL output."""

    def __init__(
        self,
        text_field: str,
        transform: Callable[[dict], dict | None] | None = None,
        compression: str = "none",
        max_rows: int | None = None,
    ):
        self.text_field = text_field
        self.transform = transform
        self.compression = compression
        self.max_rows = max_rows

    def process_shard(
        self,
        shard_index: int,
        files: list[dict],  # FileInfo as dicts for Ray serialization
        output_dir: str,
        fs_protocol: str,
        receipts_dir: str | None = None,
    ) -> dict:
        """Process files to a single JSONL shard."""
        fs = filesystem(fs_protocol)
        resolved_receipts_dir = receipts_dir or f"{output_dir}/receipts"

        return process_jsonl_shard_core(
            shard_index=shard_index,
            files=[FileInfo(**f) for f in files],
            output_dir=output_dir,
            receipts_dir=resolved_receipts_dir,
            output_fs=fs,
            text_field=self.text_field,
            transform=self.transform,
            compression=self.compression,
            max_rows=self.max_rows,
        )
