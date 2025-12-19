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

"""BinIdx shard task UDF for Ray Data execution."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from fsspec import filesystem

from nemotron.data_prep.providers import create_tokenizer

logger = logging.getLogger(__name__)


class BinIdxShardTaskUDF:
    """Ray Data UDF for processing binidx shard tasks.

    This UDF is used with ActorPoolStrategy, meaning:
    - __init__ is called once per actor (tokenizer loaded once)
    - __call__ is called for each batch (batch_size=1 means one shard)

    The UDF delegates to process_binidx_shard_core, which contains
    the actual processing logic shared with the legacy actor.

    IMPORTANT: Tokenizer is created in __init__ (once per actor), not
    per-shard, to amortize initialization cost across all shards
    processed by this actor.
    """

    def __init__(
        self,
        resolved_tokenizer: dict,
        min_doc_chars: int | None = None,
        max_doc_tokens: int | None = None,
        dtype: str = "int32",
        max_rows: int | None = None,
    ):
        """Initialize actor with tokenizer (amortized across shards).

        Args:
            resolved_tokenizer: Tokenizer config dict with resolved SHA
            min_doc_chars: Skip documents shorter than this
            max_doc_tokens: Truncate documents longer than this
            dtype: Token dtype (e.g., "int32")
            max_rows: Limit rows processed per shard (for testing)
        """
        self.min_doc_chars = min_doc_chars
        self.max_doc_tokens = max_doc_tokens
        self.dtype = dtype
        self.max_rows = max_rows

        # Create tokenizer ONCE per actor (this is the perf win)
        logger.debug("BinIdxShardTaskUDF: initializing tokenizer...")
        self.tokenize = create_tokenizer(resolved_tokenizer)
        logger.debug("BinIdxShardTaskUDF: tokenizer ready")

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process a batch of shard tasks.

        With batch_size=1, we receive exactly one task per call.
        Ray Data passes dict-of-numpy-arrays (default batch_format).

        Args:
            batch: Dict with numpy arrays, each of length 1 (batch_size=1)

        Returns:
            Dict of numpy arrays with stats (same batch format)
        """
        # Import here to avoid circular imports
        from nemotron.data_prep.shard_processor import process_binidx_shard_core

        # Extract single task from batch (batch_size=1)
        # Ray Data passes columns as numpy arrays
        dataset_name = str(batch["dataset_name"][0])
        shard_index = int(batch["shard_index"][0])
        plan_hash = str(batch["plan_hash"][0])
        assignment_json = str(batch["assignment_json"][0])
        output_dir = str(batch["output_dir"][0])
        receipts_dir = str(batch["receipts_dir"][0])
        fs_protocol = str(batch["fs_protocol"][0])
        text_field = str(batch["text_field"][0])

        # Deserialize assignment from JSON
        assignment = json.loads(assignment_json)

        logger.debug(f"Processing shard {shard_index} for dataset {dataset_name}")

        # Create output filesystem
        output_fs = filesystem(fs_protocol)

        try:
            stats = process_binidx_shard_core(
                tokenize=self.tokenize,  # Pre-initialized in __init__
                text_field=text_field,
                min_doc_chars=self.min_doc_chars,
                max_doc_tokens=self.max_doc_tokens,
                dtype=self.dtype,
                max_rows=self.max_rows,
                shard_index=shard_index,
                assignment=assignment,
                plan_hash=plan_hash,
                output_dir=output_dir,
                receipts_dir=receipts_dir,
                output_fs=output_fs,
            )

            # Return dict-of-numpy-arrays (Ray Data's expected format)
            # Include task identity for tracking + key stats
            return {
                "dataset_name": np.array([dataset_name], dtype=object),
                "shard_index": np.array([shard_index], dtype=np.int64),
                "plan_hash": np.array([plan_hash], dtype=object),
                "total_tokens": np.array([stats.get("total_tokens", 0)], dtype=np.int64),
                "num_sequences": np.array([stats.get("num_sequences", 0)], dtype=np.int64),
                "num_filtered": np.array([stats.get("num_filtered", 0)], dtype=np.int64),
                "num_errors": np.array([stats.get("num_errors", 0)], dtype=np.int64),
                # Timing metrics for bottleneck identification
                "time_total_sec": np.array(
                    [stats.get("time_total_sec", 0.0)], dtype=np.float64
                ),
                "time_download_sec": np.array(
                    [stats.get("time_download_sec", 0.0)], dtype=np.float64
                ),
                "time_read_sec": np.array([stats.get("time_read_sec", 0.0)], dtype=np.float64),
                "time_tokenize_sec": np.array(
                    [stats.get("time_tokenize_sec", 0.0)], dtype=np.float64
                ),
                "time_write_sec": np.array([stats.get("time_write_sec", 0.0)], dtype=np.float64),
                "error": np.array([""], dtype=object),
            }
        except Exception as e:
            logger.error(f"Shard {shard_index} for {dataset_name} failed: {e}")
            # Return error stats (no receipt written = will be retried on resume)
            return {
                "dataset_name": np.array([dataset_name], dtype=object),
                "shard_index": np.array([shard_index], dtype=np.int64),
                "plan_hash": np.array([plan_hash], dtype=object),
                "total_tokens": np.array([0], dtype=np.int64),
                "num_sequences": np.array([0], dtype=np.int64),
                "num_filtered": np.array([0], dtype=np.int64),
                "num_errors": np.array([1], dtype=np.int64),
                "time_total_sec": np.array([0.0], dtype=np.float64),
                "time_download_sec": np.array([0.0], dtype=np.float64),
                "time_read_sec": np.array([0.0], dtype=np.float64),
                "time_tokenize_sec": np.array([0.0], dtype=np.float64),
                "time_write_sec": np.array([0.0], dtype=np.float64),
                "error": np.array([str(e)], dtype=object),
            }
