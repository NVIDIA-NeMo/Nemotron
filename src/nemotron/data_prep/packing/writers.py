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

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _normalize_compression(compression: str | None) -> str | None:
    if compression is None:
        return None
    c = str(compression).strip().lower()
    if c in ("none", ""):
        return None
    return c


def _move_atomic(
    *,
    src: str,
    dst: str,
    filesystem: pa.fs.FileSystem | None,
) -> None:
    if filesystem is None:
        os.replace(src, dst)
        return

    move = getattr(filesystem, "move", None)
    if callable(move):
        move(src, dst)
        return

    copy_file = getattr(filesystem, "copy_file", None)
    delete_file = getattr(filesystem, "delete_file", None)
    if callable(copy_file) and callable(delete_file):
        copy_file(src, dst)
        delete_file(src)
        return

    raise RuntimeError(f"PyArrow filesystem does not support move/copy_file+delete_file for: {type(filesystem)}")


class ParquetShardWriter:
    """Memory-efficient Parquet writer for packed SFT shards.

    Format:
      shard_000000.parquet
        - input_ids: list<int32>
        - loss_mask: list<uint8>
        - seq_start_id: list<int32>

    Writes bins incrementally, flushing row groups every `row_group_size` bins.
    Uses a tmp file then atomically promotes to final output path.
    """

    SCHEMA = pa.schema(
        [
            ("input_ids", pa.list_(pa.int32())),
            ("loss_mask", pa.list_(pa.uint8())),
            ("seq_start_id", pa.list_(pa.int32())),
        ]
    )

    def __init__(
        self,
        output_path: str,
        row_group_size: int = 1000,
        compression: str = "zstd",
        filesystem: pa.fs.FileSystem | None = None,
    ) -> None:
        if row_group_size <= 0:
            raise ValueError(f"row_group_size must be > 0, got {row_group_size}")

        self.output_path = output_path
        self.tmp_path = output_path + ".tmp"
        self.row_group_size = int(row_group_size)
        self.compression = _normalize_compression(compression)
        self.filesystem = filesystem

        self._input_ids_values: list[np.ndarray] = []
        self._loss_mask_values: list[np.ndarray] = []
        self._seq_start_values: list[np.ndarray] = []
        self._count = 0
        self._total_bins = 0
        self._closed = False

        self._writer = pq.ParquetWriter(
            self.tmp_path,
            self.SCHEMA,
            compression=self.compression,
            filesystem=self.filesystem,
        )

    def write_bin(
        self,
        bin_id: int,
        input_ids: np.ndarray,
        loss_mask: np.ndarray,
        seq_start_id: np.ndarray,
    ) -> None:
        if self._closed:
            raise RuntimeError("ParquetShardWriter is closed")

        self._input_ids_values.append(np.asarray(input_ids, dtype=np.int32))
        self._loss_mask_values.append(np.asarray(loss_mask, dtype=np.uint8))
        self._seq_start_values.append(np.asarray(seq_start_id, dtype=np.int32))
        self._count += 1
        self._total_bins += 1

        if self._count >= self.row_group_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        if self._count == 0:
            return

        input_ids_arr = pa.array(self._input_ids_values, type=pa.list_(pa.int32()))
        loss_mask_arr = pa.array(self._loss_mask_values, type=pa.list_(pa.uint8()))
        seq_start_arr = pa.array(self._seq_start_values, type=pa.list_(pa.int32()))

        table = pa.Table.from_arrays(
            [input_ids_arr, loss_mask_arr, seq_start_arr],
            schema=self.SCHEMA,
        )
        self._writer.write_table(table)

        self._input_ids_values.clear()
        self._loss_mask_values.clear()
        self._seq_start_values.clear()
        self._count = 0

    def finalize(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("ParquetShardWriter is already finalized/closed")
        self._closed = True

        self._flush_buffer()
        self._writer.close()

        _move_atomic(src=self.tmp_path, dst=self.output_path, filesystem=self.filesystem)

        return {
            "format": "parquet",
            "compression": self.compression or "none",
            "num_bins": int(self._total_bins),
            "row_group_size": int(self.row_group_size),
        }


__all__ = [
    "ParquetShardWriter",
]
