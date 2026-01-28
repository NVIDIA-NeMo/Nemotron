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

"""Tests for packing algorithms."""

import pytest

from nemotron.data_prep.packing.algorithms import (
    ConcatenativePacker,
    FirstFitDecreasingPacker,
    FirstFitShufflePacker,
    PackingAlgorithm,
    PackingMetrics,
    SequencePacker,
    get_packer,
)


# =============================================================================
# PackingMetrics
# =============================================================================


class TestPackingMetrics:
    def test_packing_factor(self) -> None:
        m = PackingMetrics(num_sequences=10, num_bins=5, total_tokens=100, bin_capacity=50)
        assert m.packing_factor == 2.0

    def test_packing_factor_zero_bins(self) -> None:
        m = PackingMetrics(num_sequences=0, num_bins=0, total_tokens=0, bin_capacity=100)
        assert m.packing_factor == 0

    def test_packing_efficiency(self) -> None:
        m = PackingMetrics(num_sequences=4, num_bins=2, total_tokens=200, bin_capacity=100)
        assert m.packing_efficiency == 100.0

    def test_packing_efficiency_half(self) -> None:
        m = PackingMetrics(num_sequences=2, num_bins=2, total_tokens=100, bin_capacity=100)
        assert m.packing_efficiency == 50.0

    def test_packing_efficiency_zero_capacity(self) -> None:
        m = PackingMetrics(num_sequences=0, num_bins=0, total_tokens=0, bin_capacity=0)
        assert m.packing_efficiency == 0


# =============================================================================
# SequencePacker base
# =============================================================================


class TestSequencePackerBase:
    def test_invalid_bin_capacity_zero(self) -> None:
        with pytest.raises(ValueError, match="bin_capacity must be positive"):
            FirstFitDecreasingPacker(0)

    def test_invalid_bin_capacity_negative(self) -> None:
        with pytest.raises(ValueError, match="bin_capacity must be positive"):
            ConcatenativePacker(-10)


# =============================================================================
# FirstFitDecreasingPacker
# =============================================================================


class TestFirstFitDecreasingPacker:
    def test_empty_input(self) -> None:
        packer = FirstFitDecreasingPacker(100)
        bins, metrics = packer.pack([])
        assert bins == []
        assert metrics.num_sequences == 0
        assert metrics.num_bins == 0

    def test_single_sequence_fits(self) -> None:
        packer = FirstFitDecreasingPacker(100)
        bins, metrics = packer.pack([50])
        assert len(bins) == 1
        assert bins[0] == [0]
        assert metrics.num_sequences == 1
        assert metrics.total_tokens == 50

    def test_exact_fit(self) -> None:
        packer = FirstFitDecreasingPacker(100)
        bins, metrics = packer.pack([100])
        assert len(bins) == 1
        assert metrics.total_tokens == 100

    def test_two_sequences_fit_one_bin(self) -> None:
        packer = FirstFitDecreasingPacker(100)
        bins, metrics = packer.pack([40, 60])
        assert len(bins) == 1
        assert metrics.num_sequences == 2

    def test_sequences_split_across_bins(self) -> None:
        packer = FirstFitDecreasingPacker(100)
        bins, metrics = packer.pack([60, 60])
        assert len(bins) == 2
        assert metrics.num_sequences == 2

    def test_decreasing_order_packs_efficiently(self) -> None:
        packer = FirstFitDecreasingPacker(100)
        # Lengths: 80, 20, 70, 30 → sorted desc: 80, 70, 30, 20
        # Bin 0: 80, 20 (index 0, 1) → remaining 0
        # Bin 1: 70, 30 (index 2, 3) → remaining 0
        bins, metrics = packer.pack([80, 20, 70, 30])
        assert len(bins) == 2
        assert metrics.packing_efficiency == 100.0

    def test_truncation(self) -> None:
        packer = FirstFitDecreasingPacker(100)
        bins, metrics = packer.pack([150])
        assert len(bins) == 1
        assert metrics.num_truncated == 1
        assert metrics.total_tokens == 100  # truncated to bin_capacity

    def test_deterministic(self) -> None:
        packer = FirstFitDecreasingPacker(100)
        lengths = [30, 50, 20, 80, 10, 60]
        bins1, _ = packer.pack(lengths)
        bins2, _ = packer.pack(lengths)
        assert bins1 == bins2

    def test_all_indices_present(self) -> None:
        packer = FirstFitDecreasingPacker(100)
        lengths = [30, 50, 20, 80, 10]
        bins, _ = packer.pack(lengths)
        all_indices = sorted(idx for b in bins for idx in b)
        assert all_indices == [0, 1, 2, 3, 4]


# =============================================================================
# FirstFitShufflePacker
# =============================================================================


class TestFirstFitShufflePacker:
    def test_empty_input(self) -> None:
        packer = FirstFitShufflePacker(100, seed=42)
        bins, metrics = packer.pack([])
        assert bins == []
        assert metrics.num_sequences == 0

    def test_single_sequence(self) -> None:
        packer = FirstFitShufflePacker(100, seed=42)
        bins, metrics = packer.pack([50])
        assert len(bins) == 1

    def test_seed_reproducibility(self) -> None:
        lengths = [30, 50, 20, 80, 10, 60, 40, 70]
        bins1, _ = FirstFitShufflePacker(100, seed=42).pack(lengths)
        bins2, _ = FirstFitShufflePacker(100, seed=42).pack(lengths)
        assert bins1 == bins2

    def test_different_seeds_different_results(self) -> None:
        lengths = [30, 50, 20, 80, 10, 60, 40, 70]
        bins1, _ = FirstFitShufflePacker(100, seed=1).pack(lengths)
        bins2, _ = FirstFitShufflePacker(100, seed=2).pack(lengths)
        # Different seeds should (almost certainly) produce different bin orderings
        # We can't guarantee different structure, but order of indices should differ
        flat1 = [idx for b in bins1 for idx in b]
        flat2 = [idx for b in bins2 for idx in b]
        assert flat1 != flat2

    def test_all_indices_present(self) -> None:
        packer = FirstFitShufflePacker(100, seed=42)
        lengths = [30, 50, 20, 80, 10]
        bins, _ = packer.pack(lengths)
        all_indices = sorted(idx for b in bins for idx in b)
        assert all_indices == [0, 1, 2, 3, 4]

    def test_truncation(self) -> None:
        packer = FirstFitShufflePacker(50, seed=42)
        bins, metrics = packer.pack([100])
        assert metrics.num_truncated == 1
        assert metrics.total_tokens == 50


# =============================================================================
# ConcatenativePacker
# =============================================================================


class TestConcatenativePacker:
    def test_empty_input(self) -> None:
        packer = ConcatenativePacker(100)
        bins, metrics = packer.pack([])
        assert bins == []
        assert metrics.num_sequences == 0

    def test_single_sequence_fits(self) -> None:
        packer = ConcatenativePacker(100)
        bins, metrics = packer.pack([50])
        assert len(bins) == 1
        assert bins[0] == [0]

    def test_preserves_order(self) -> None:
        packer = ConcatenativePacker(100)
        bins, _ = packer.pack([30, 30, 30])
        # All fit in one bin, in order
        assert bins == [[0, 1, 2]]

    def test_splits_when_full(self) -> None:
        packer = ConcatenativePacker(100)
        bins, _ = packer.pack([60, 60])
        assert len(bins) == 2
        assert bins[0] == [0]
        assert bins[1] == [1]

    def test_concatenates_in_order(self) -> None:
        packer = ConcatenativePacker(100)
        bins, _ = packer.pack([40, 40, 40])
        # 40+40=80 fits, then 40 doesn't fit → new bin
        assert bins == [[0, 1], [2]]

    def test_truncation(self) -> None:
        packer = ConcatenativePacker(50)
        bins, metrics = packer.pack([100])
        assert metrics.num_truncated == 1
        assert metrics.total_tokens == 50

    def test_all_indices_present(self) -> None:
        packer = ConcatenativePacker(100)
        lengths = [30, 50, 20, 80, 10]
        bins, _ = packer.pack(lengths)
        all_indices = sorted(idx for b in bins for idx in b)
        assert all_indices == [0, 1, 2, 3, 4]


# =============================================================================
# get_packer factory
# =============================================================================


class TestGetPacker:
    def test_ffd_from_enum(self) -> None:
        packer = get_packer(PackingAlgorithm.FIRST_FIT_DECREASING, 100)
        assert isinstance(packer, FirstFitDecreasingPacker)

    def test_ffd_from_string(self) -> None:
        packer = get_packer("first_fit_decreasing", 100)
        assert isinstance(packer, FirstFitDecreasingPacker)

    def test_ffs_from_string(self) -> None:
        packer = get_packer("first_fit_shuffle", 100, seed=42)
        assert isinstance(packer, FirstFitShufflePacker)

    def test_concatenative_from_string(self) -> None:
        packer = get_packer("concatenative", 100)
        assert isinstance(packer, ConcatenativePacker)

    def test_invalid_algorithm_string(self) -> None:
        with pytest.raises(ValueError):
            get_packer("invalid_algo", 100)

    def test_enum_values(self) -> None:
        assert PackingAlgorithm.FIRST_FIT_DECREASING.value == "first_fit_decreasing"
        assert PackingAlgorithm.FIRST_FIT_SHUFFLE.value == "first_fit_shuffle"
        assert PackingAlgorithm.CONCATENATIVE.value == "concatenative"
