# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests that binidx pretrain processing applies the per-shard row partition.

Regression for the bug where a file split across N shards was re-read in full by
every shard, duplicating each document N times.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from nemotron.data_prep.core.shard_processor import _iter_parquet_batches_internal


def _collect(path: str, modulus: int | None, remainder: int | None) -> list[str]:
    parquet_file = pq.ParquetFile(path)
    texts: list[str] = []
    for batch_texts, _ in _iter_parquet_batches_internal(parquet_file, "text", None, modulus, remainder):
        texts.extend(batch_texts)
    return texts


def test_row_partition_is_disjoint_and_complete(tmp_path) -> None:
    # More rows than the internal batch size (10000) to exercise the cross-batch offset.
    n, modulus = 25000, 8
    path = str(tmp_path / "data.parquet")
    pq.write_table(pa.table({"text": [f"doc-{i}" for i in range(n)]}), path)

    seen: list[str] = []
    for remainder in range(modulus):
        part = _collect(path, modulus, remainder)
        assert part == [f"doc-{i}" for i in range(n) if i % modulus == remainder]
        seen.extend(part)

    # Union over all shards must equal every row exactly once (no duplication, no loss).
    assert sorted(seen) == sorted(f"doc-{i}" for i in range(n))
    assert len(seen) == n


def test_no_partition_processes_all_rows(tmp_path) -> None:
    n = 100
    path = str(tmp_path / "data.parquet")
    pq.write_table(pa.table({"text": [f"doc-{i}" for i in range(n)]}), path)
    assert _collect(path, None, None) == [f"doc-{i}" for i in range(n)]
