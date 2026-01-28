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

"""Tests for BinAssignment CSR representation."""

import numpy as np
import pytest

from nemotron.data_prep.packing.bin_assignment import BinAssignment


class TestBinAssignmentEmpty:
    def test_empty_bins(self) -> None:
        ba = BinAssignment.from_bins(bins=[], num_sequences=0)
        assert ba.num_bins == 0
        assert ba.num_sequences == 0
        assert len(ba.bin_offsets) == 1
        assert ba.bin_offsets[0] == 0
        assert len(ba.bin_seq_indices) == 0


class TestBinAssignmentFromBins:
    def test_single_bin_single_seq(self) -> None:
        ba = BinAssignment.from_bins(bins=[[0]], num_sequences=1)
        assert ba.num_bins == 1
        assert ba.num_sequences == 1
        indices = ba.bin_indices(0)
        assert list(indices) == [0]

    def test_single_bin_multiple_seqs(self) -> None:
        ba = BinAssignment.from_bins(bins=[[0, 1, 2]], num_sequences=3)
        assert ba.num_bins == 1
        indices = ba.bin_indices(0)
        assert list(indices) == [0, 1, 2]

    def test_multiple_bins(self) -> None:
        ba = BinAssignment.from_bins(
            bins=[[0, 1], [2, 3], [4]],
            num_sequences=5,
        )
        assert ba.num_bins == 3
        assert list(ba.bin_indices(0)) == [0, 1]
        assert list(ba.bin_indices(1)) == [2, 3]
        assert list(ba.bin_indices(2)) == [4]

    def test_offsets_correct(self) -> None:
        ba = BinAssignment.from_bins(
            bins=[[0, 1], [2]],
            num_sequences=3,
        )
        assert list(ba.bin_offsets) == [0, 2, 3]

    def test_dtypes(self) -> None:
        ba = BinAssignment.from_bins(bins=[[0]], num_sequences=1)
        assert ba.bin_offsets.dtype == np.int64
        assert ba.bin_seq_indices.dtype == np.int32

    def test_out_of_range_index(self) -> None:
        with pytest.raises(ValueError, match="Sequence index out of range"):
            BinAssignment.from_bins(bins=[[5]], num_sequences=3)

    def test_negative_index(self) -> None:
        with pytest.raises(ValueError, match="Sequence index out of range"):
            BinAssignment.from_bins(bins=[[-1]], num_sequences=3)


class TestBinAssignmentBinIndices:
    def test_valid_bin_id(self) -> None:
        ba = BinAssignment.from_bins(bins=[[0], [1]], num_sequences=2)
        assert list(ba.bin_indices(0)) == [0]
        assert list(ba.bin_indices(1)) == [1]

    def test_out_of_range_bin_id(self) -> None:
        ba = BinAssignment.from_bins(bins=[[0]], num_sequences=1)
        with pytest.raises(IndexError, match="bin_id out of range"):
            ba.bin_indices(1)

    def test_negative_bin_id(self) -> None:
        ba = BinAssignment.from_bins(bins=[[0]], num_sequences=1)
        with pytest.raises(IndexError, match="bin_id out of range"):
            ba.bin_indices(-1)

    def test_returns_view(self) -> None:
        ba = BinAssignment.from_bins(bins=[[0, 1, 2]], num_sequences=3)
        indices = ba.bin_indices(0)
        assert isinstance(indices, np.ndarray)
        assert len(indices) == 3
