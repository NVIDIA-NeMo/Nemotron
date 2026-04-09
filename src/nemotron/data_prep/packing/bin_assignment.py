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

"""Compact bin assignment representation for packed sequence materialization.

BinAssignment stores a list-of-lists bin structure (bin_id -> [seq_index...])
in a CSR-like representation:

- bin_offsets: int64 array of length (num_bins + 1)
- bin_seq_indices: int32 array of length (total_assigned_sequences)

For bin i, the sequence indices live in:
    bin_seq_indices[bin_offsets[i] : bin_offsets[i+1]]

This structure is designed to:
- avoid large nested Python lists during the materialization phase
- allow cheap slicing per bin
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BinAssignment:
    """CSR-like representation of bin assignments."""

    bin_offsets: np.ndarray
    bin_seq_indices: np.ndarray
    num_bins: int
    num_sequences: int

    @classmethod
    def from_bins(cls, *, bins: Sequence[Sequence[int]], num_sequences: int) -> "BinAssignment":
        """Build a BinAssignment from a Python bins structure.

        Args:
            bins: Sequence of bins, each bin is a sequence of seq indices.
            num_sequences: Total number of sequences in the shard/spool (for metadata).

        Returns:
            BinAssignment with int64 offsets and int32 indices.

        Raises:
            ValueError: If indices are out of range.
        """
        num_bins = int(len(bins))
        if num_bins == 0:
            return cls(
                bin_offsets=np.zeros((1,), dtype=np.int64),
                bin_seq_indices=np.zeros((0,), dtype=np.int32),
                num_bins=0,
                num_sequences=int(num_sequences),
            )

        total_entries = 0
        for b in bins:
            total_entries += len(b)

        offsets = np.zeros((num_bins + 1,), dtype=np.int64)
        indices = np.zeros((total_entries,), dtype=np.int32)

        cursor = 0
        for i, b in enumerate(bins):
            offsets[i] = cursor
            for idx in b:
                if idx < 0 or idx >= num_sequences:
                    raise ValueError(f"Sequence index out of range in bins: {idx} (num_sequences={num_sequences})")
                indices[cursor] = np.int32(idx)
                cursor += 1
        offsets[num_bins] = cursor

        return cls(
            bin_offsets=offsets,
            bin_seq_indices=indices,
            num_bins=num_bins,
            num_sequences=int(num_sequences),
        )

    def bin_indices(self, bin_id: int) -> np.ndarray:
        """Return the seq indices for a given bin as a view."""
        if bin_id < 0 or bin_id >= self.num_bins:
            raise IndexError(f"bin_id out of range: {bin_id}")
        start = int(self.bin_offsets[bin_id])
        end = int(self.bin_offsets[bin_id + 1])
        return self.bin_seq_indices[start:end]


__all__ = ["BinAssignment"]