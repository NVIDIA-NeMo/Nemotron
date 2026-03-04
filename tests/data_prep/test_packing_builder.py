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

"""Tests for PackedSequenceBuilder."""

import pytest

from nemotron.data_prep.packing.builder import PackedSequenceBuilder


class TestPackedSequenceBuilderEmpty:
    def test_finalize_empty(self) -> None:
        builder = PackedSequenceBuilder(pack_size=100)
        packed, metadata = builder.finalize()
        assert packed == []
        assert metadata["num_sequences"] == 0
        assert metadata["num_packed_sequences"] == 0
        assert metadata["packing_factor"] == 0
        assert metadata["packing_efficiency"] == 0

    def test_get_stats_empty(self) -> None:
        builder = PackedSequenceBuilder(pack_size=100)
        stats = builder.get_stats()
        assert stats["num_sequences"] == 0
        assert stats["total_tokens"] == 0

    def test_add_empty_sequence_ignored(self) -> None:
        builder = PackedSequenceBuilder(pack_size=100)
        builder.add_sequence([])
        assert builder.get_stats()["num_sequences"] == 0


class TestPackedSequenceBuilderSingle:
    def test_single_sequence(self) -> None:
        builder = PackedSequenceBuilder(pack_size=100, seed=42)
        builder.add_sequence([1, 2, 3], loss_mask=[1, 1, 1])
        packed, metadata = builder.finalize()

        assert len(packed) == 1
        assert packed[0]["input_ids"] == [1, 2, 3]
        assert len(packed[0]["loss_mask"]) == 3
        assert packed[0]["seq_start_id"] == [0]
        assert metadata["num_sequences"] == 1
        assert metadata["num_packed_sequences"] == 1

    def test_default_loss_mask(self) -> None:
        builder = PackedSequenceBuilder(pack_size=100, seed=42)
        builder.add_sequence([10, 20, 30])
        packed, _ = builder.finalize()
        # Default mask is all 1s [1, 1, 1]
        # Per-subsequence alignment: aligned[j] = mask[j+1], aligned[L-1] = 0
        # So: [mask[1], mask[2], 0] = [1, 1, 0]
        assert packed[0]["loss_mask"] == [1, 1, 0]


class TestPackedSequenceBuilderMultiple:
    def test_two_sequences_one_bin(self) -> None:
        builder = PackedSequenceBuilder(pack_size=100, seed=42)
        builder.add_sequence([1, 2], loss_mask=[1, 1])
        builder.add_sequence([3, 4], loss_mask=[0, 1])
        packed, metadata = builder.finalize()

        assert metadata["num_sequences"] == 2
        # Both fit in one bin (4 tokens < 100)
        assert metadata["num_packed_sequences"] == 1

    def test_add_sequences_batch(self) -> None:
        builder = PackedSequenceBuilder(pack_size=100, seed=42)
        builder.add_sequences(
            [[1, 2], [3, 4, 5]],
            loss_masks=[[1, 1], [0, 1, 1]],
        )
        packed, metadata = builder.finalize()
        assert metadata["num_sequences"] == 2

    def test_add_sequences_no_masks(self) -> None:
        builder = PackedSequenceBuilder(pack_size=100, seed=42)
        builder.add_sequences([[1, 2], [3, 4, 5]])
        packed, metadata = builder.finalize()
        assert metadata["num_sequences"] == 2

    def test_get_stats_before_finalize(self) -> None:
        builder = PackedSequenceBuilder(pack_size=100)
        builder.add_sequence([1, 2, 3])
        builder.add_sequence([4, 5])
        stats = builder.get_stats()
        assert stats["num_sequences"] == 2
        assert stats["total_tokens"] == 5
        assert stats["avg_length"] == 2.5


class TestPackedSequenceBuilderPacking:
    def test_sequences_split_into_bins(self) -> None:
        builder = PackedSequenceBuilder(pack_size=5, seed=42)
        builder.add_sequence([1, 2, 3])
        builder.add_sequence([4, 5, 6])
        packed, metadata = builder.finalize()
        # 3+3=6 > 5, so should be 2 bins
        assert metadata["num_packed_sequences"] == 2

    def test_metadata_has_algorithm(self) -> None:
        builder = PackedSequenceBuilder(
            pack_size=100, algorithm="first_fit_decreasing", seed=42
        )
        builder.add_sequence([1, 2, 3])
        _, metadata = builder.finalize()
        assert "algorithm" in metadata

    def test_loss_mask_per_subsequence_aligned(self) -> None:
        builder = PackedSequenceBuilder(pack_size=100, seed=42)
        builder.add_sequence([1, 2, 3, 4], loss_mask=[1, 0, 1, 1])
        packed, _ = builder.finalize()
        # Original: [1, 0, 1, 1]
        # Per-subsequence alignment: aligned[j] = mask[j+1], aligned[L-1] = 0
        # So: [mask[1], mask[2], mask[3], 0] = [0, 1, 1, 0]
        assert packed[0]["loss_mask"] == [0, 1, 1, 0]

    def test_loss_mask_multi_subsequence_no_bleed(self) -> None:
        """Test that loss_mask alignment doesn't bleed across subsequence boundaries."""
        builder = PackedSequenceBuilder(pack_size=100, seed=42)
        # Seq1: tokens [1,2,3], mask [1,1,1] -> aligned [1,1,0]
        # Seq2: tokens [4,5], mask [0,1] -> aligned [1,0]
        builder.add_sequence([1, 2, 3], loss_mask=[1, 1, 1])
        builder.add_sequence([4, 5], loss_mask=[0, 1])
        packed, _ = builder.finalize()

        # Both fit in one bin
        packed_item = packed[0]
        assert packed_item["input_ids"] == [1, 2, 3, 4, 5]
        # Per-subsequence: [1,1,0] + [1,0] = [1,1,0,1,0]
        assert packed_item["loss_mask"] == [1, 1, 0, 1, 0]
        # seq_start_id marks boundaries
        assert packed_item["seq_start_id"] == [0, 3]

        # Verify invariant: loss_mask[end-1] == 0 for each subsequence
        boundaries = packed_item["seq_start_id"] + [len(packed_item["input_ids"])]
        for i in range(len(boundaries) - 1):
            end = boundaries[i + 1]
            assert packed_item["loss_mask"][end - 1] == 0, f"Subsequence {i} should have loss_mask[end-1]=0"

    def test_loss_mask_single_token_subsequence(self) -> None:
        """Single token subsequence should have loss_mask=0 (no label to predict)."""
        builder = PackedSequenceBuilder(pack_size=100, seed=42)
        builder.add_sequence([1], loss_mask=[1])
        packed, _ = builder.finalize()
        # Single token -> aligned mask is [0]
        assert packed[0]["loss_mask"] == [0]

    def test_seq_start_id_boundaries(self) -> None:
        builder = PackedSequenceBuilder(pack_size=100, seed=42)
        builder.add_sequence([1, 2, 3])
        builder.add_sequence([4, 5])
        packed, _ = builder.finalize()
        # Both in one bin (5 < 100)
        packed_item = packed[0]
        # seq_start_id should mark boundaries
        assert packed_item["seq_start_id"][0] == 0
        if len(packed_item["seq_start_id"]) > 1:
            assert packed_item["seq_start_id"][1] > 0
