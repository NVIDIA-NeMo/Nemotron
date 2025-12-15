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

"""Streaming builder for Megatron-style .bin/.idx indexed dataset files."""

import struct
from typing import BinaryIO

import numpy as np
import xxhash

_INDEX_HEADER = b"MMIDIDX\x00\x00"

DTYPE_CODES = {
    np.dtype(np.int32): 4,
    np.dtype(np.int64): 5,
    np.dtype(np.uint16): 8,
}


class IndexedDatasetBuilder:
    """
    Streaming builder for .bin/.idx files.

    Writes documents incrementally with streaming checksums.
    Uses buffered writes to reduce syscall overhead.
    Never holds full shard in memory.
    """

    def __init__(
        self,
        bin_file: BinaryIO,
        dtype: np.dtype = np.int32,
        buffer_size: int = 64 * 1024,
    ):
        self.bin_file = bin_file
        self.dtype = np.dtype(dtype)
        self._sequence_lengths: list[int] = []
        self._document_indices: list[int] = [0]
        self._total_tokens = 0
        self._bin_hasher = xxhash.xxh64()
        self._bin_bytes = 0
        self._min_length: float = float("inf")
        self._max_length = 0

        # Write buffer for reducing syscalls
        self._write_buffer = bytearray()
        self._buffer_size = buffer_size

    def add_document(self, tokens: list[int]) -> None:
        """Add single document with buffered write."""
        if not tokens:
            return

        arr = np.asarray(tokens, dtype=self.dtype)
        data = arr.tobytes(order="C")

        self._write_buffer.extend(data)
        self._bin_hasher.update(data)
        self._bin_bytes += len(data)

        length = len(arr)
        self._sequence_lengths.append(length)
        self._document_indices.append(len(self._sequence_lengths))
        self._total_tokens += length
        self._min_length = min(self._min_length, length)
        self._max_length = max(self._max_length, length)

        # Flush when buffer is full
        if len(self._write_buffer) >= self._buffer_size:
            self._flush_buffer()

    def add_documents(self, token_lists: list[list[int]]) -> None:
        """
        Batch add documents - reduces numpy allocation overhead.

        Concatenates all tokens for a single numpy conversion,
        then writes to buffer in one operation.
        """
        if not token_lists:
            return

        # Filter empty lists and collect lengths
        all_tokens: list[int] = []
        lengths: list[int] = []
        for tokens in token_lists:
            if tokens:
                all_tokens.extend(tokens)
                lengths.append(len(tokens))

        if not lengths:
            return

        # Single numpy allocation for all tokens
        arr = np.array(all_tokens, dtype=self.dtype)
        data = arr.tobytes(order="C")

        self._write_buffer.extend(data)
        self._bin_hasher.update(data)
        self._bin_bytes += len(data)

        # Track metadata for each document
        for length in lengths:
            self._sequence_lengths.append(length)
            self._document_indices.append(len(self._sequence_lengths))
            self._total_tokens += length
            self._min_length = min(self._min_length, length)
            self._max_length = max(self._max_length, length)

        # Flush when buffer is full
        if len(self._write_buffer) >= self._buffer_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush write buffer to file."""
        if self._write_buffer:
            self.bin_file.write(self._write_buffer)
            self._write_buffer.clear()

    def finalize(self) -> None:
        """Flush remaining buffer. Call before get_bin_info or write_index."""
        self._flush_buffer()

    def get_bin_info(self) -> tuple[int, str]:
        """Get bin file size and checksum. Flushes buffer first."""
        self._flush_buffer()
        return self._bin_bytes, f"xxh64:{self._bin_hasher.hexdigest()}"

    def write_index(self, idx_file: BinaryIO) -> tuple[int, str]:
        """Write index file, return size and checksum."""
        sequence_count = len(self._sequence_lengths)
        document_count = len(self._document_indices)

        # Calculate pointers
        lengths = np.array(self._sequence_lengths, dtype=np.int64)
        pointers = np.zeros(sequence_count, dtype=np.int64)
        if sequence_count > 1:
            pointers[1:] = np.cumsum(lengths[:-1] * self.dtype.itemsize)

        idx_hasher = xxhash.xxh64()
        idx_bytes = 0

        def write_and_hash(data: bytes) -> None:
            nonlocal idx_bytes
            idx_file.write(data)
            idx_hasher.update(data)
            idx_bytes += len(data)

        write_and_hash(_INDEX_HEADER)
        write_and_hash(struct.pack("<Q", 1))  # version
        write_and_hash(struct.pack("<B", DTYPE_CODES[self.dtype]))
        write_and_hash(struct.pack("<Q", sequence_count))
        write_and_hash(struct.pack("<Q", document_count))
        write_and_hash(np.array(self._sequence_lengths, dtype=np.int32).tobytes())
        write_and_hash(pointers.tobytes())
        write_and_hash(np.array(self._document_indices, dtype=np.int64).tobytes())

        return idx_bytes, f"xxh64:{idx_hasher.hexdigest()}"

    def get_stats(self) -> dict:
        """Get statistics."""
        return {
            "num_sequences": len(self._sequence_lengths),
            "num_documents": max(0, len(self._document_indices) - 1),
            "total_tokens": self._total_tokens,
            "min_length": int(self._min_length) if self._sequence_lengths else 0,
            "max_length": int(self._max_length) if self._sequence_lengths else 0,
        }
