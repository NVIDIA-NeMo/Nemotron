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

"""Tests for row-level splitting in shard assignment."""

import pytest

from nemotron.data_prep.config import FileInfo, ShardAssignment
from nemotron.data_prep.core.planning import create_size_balanced_assignments


def _make_file(path: str, size: int) -> FileInfo:
    return FileInfo(path=path, local_path=path, size=size)


class TestRowSplittingSingleFile:
    """When a single file is split across many shards."""

    def test_single_file_128_shards_all_assigned(self) -> None:
        files = [_make_file("data.jsonl", 7_000_000_000)]
        assignments = create_size_balanced_assignments(files, 128)
        assert len(assignments) == 128
        for a in assignments:
            assert len(a.files) == 1, f"shard {a.shard_index} should have exactly 1 file"

    def test_single_file_modulus_equals_num_shards(self) -> None:
        files = [_make_file("data.jsonl", 7_000_000_000)]
        assignments = create_size_balanced_assignments(files, 128)
        for a in assignments:
            fi = a.files[0]
            assert fi.row_modulus == 128
            assert fi.row_remainder == a.shard_index

    def test_single_file_remainders_cover_all(self) -> None:
        files = [_make_file("data.jsonl", 1_000_000)]
        assignments = create_size_balanced_assignments(files, 8)
        remainders = {a.files[0].row_remainder for a in assignments}
        assert remainders == set(range(8))

    def test_single_file_path_preserved(self) -> None:
        files = [_make_file("data.jsonl", 5_000)]
        assignments = create_size_balanced_assignments(files, 4)
        for a in assignments:
            assert a.files[0].path == "data.jsonl"

    def test_single_file_total_bytes_approximate(self) -> None:
        size = 8_000_000
        files = [_make_file("data.jsonl", size)]
        assignments = create_size_balanced_assignments(files, 4)
        for a in assignments:
            assert a.total_bytes == size // 4


class TestRowSplittingMultipleFiles:
    """When multiple files are split across more shards."""

    def test_three_files_128_shards_all_assigned(self) -> None:
        files = [
            _make_file("big.jsonl", 7_000_000_000),
            _make_file("med.jsonl", 3_000_000_000),
            _make_file("small.jsonl", 1_000_000_000),
        ]
        assignments = create_size_balanced_assignments(files, 128)
        assert len(assignments) == 128
        for a in assignments:
            assert len(a.files) >= 1, f"shard {a.shard_index} is empty"

    def test_proportional_distribution(self) -> None:
        files = [
            _make_file("big.jsonl", 7_000_000_000),
            _make_file("med.jsonl", 3_000_000_000),
            _make_file("small.jsonl", 1_000_000_000),
        ]
        assignments = create_size_balanced_assignments(files, 128)

        # Count shards per file
        shards_per_file: dict[str, int] = {}
        for a in assignments:
            for fi in a.files:
                shards_per_file[fi.path] = shards_per_file.get(fi.path, 0) + 1

        # big file should get the most shards
        assert shards_per_file["big.jsonl"] > shards_per_file["med.jsonl"]
        assert shards_per_file["med.jsonl"] > shards_per_file["small.jsonl"]
        # Total should equal num_shards
        assert sum(shards_per_file.values()) == 128

    def test_modulus_and_remainder_consistent(self) -> None:
        files = [
            _make_file("a.jsonl", 5_000),
            _make_file("b.jsonl", 5_000),
        ]
        assignments = create_size_balanced_assignments(files, 8)

        # Group by file path
        file_assignments: dict[str, list[FileInfo]] = {}
        for a in assignments:
            for fi in a.files:
                file_assignments.setdefault(fi.path, []).append(fi)

        for path, fis in file_assignments.items():
            # All copies of the same file should have the same modulus
            moduli = {fi.row_modulus for fi in fis}
            assert len(moduli) == 1, f"{path} has inconsistent moduli: {moduli}"
            modulus = moduli.pop()
            assert modulus == len(fis)
            # Remainders should be 0..modulus-1
            remainders = sorted(fi.row_remainder for fi in fis)
            assert remainders == list(range(modulus))

    def test_every_file_gets_at_least_one_shard(self) -> None:
        files = [
            _make_file("big.jsonl", 10_000_000),
            _make_file("tiny.jsonl", 100),
        ]
        assignments = create_size_balanced_assignments(files, 16)
        paths = set()
        for a in assignments:
            for fi in a.files:
                paths.add(fi.path)
        assert "tiny.jsonl" in paths


class TestRowSplittingNotTriggered:
    """Row splitting should NOT activate when files >= shards."""

    def test_more_files_than_shards_no_row_splitting(self) -> None:
        files = [_make_file(f"file_{i}.jsonl", 1000) for i in range(10)]
        assignments = create_size_balanced_assignments(files, 4)
        for a in assignments:
            for fi in a.files:
                assert fi.row_modulus is None
                assert fi.row_remainder is None

    def test_equal_files_and_shards_no_row_splitting(self) -> None:
        files = [_make_file(f"file_{i}.jsonl", 1000) for i in range(4)]
        assignments = create_size_balanced_assignments(files, 4)
        for a in assignments:
            for fi in a.files:
                assert fi.row_modulus is None
                assert fi.row_remainder is None

    def test_zero_size_files_no_row_splitting(self) -> None:
        """When all files have size 0, row splitting is skipped (round-robin used)."""
        files = [_make_file("a.jsonl", 0)]
        assignments = create_size_balanced_assignments(files, 4)
        for a in assignments:
            for fi in a.files:
                assert fi.row_modulus is None
                assert fi.row_remainder is None


class TestApplyRowFilter:
    """Test the _apply_row_filter helper used by shard cores."""

    def test_filter_selects_correct_rows(self) -> None:
        from nemotron.data_prep.core.chat_sft_shard_core import _apply_row_filter

        records = [{"id": i} for i in range(10)]
        result = list(_apply_row_filter(iter(records), modulus=3, remainder=0))
        assert result == [{"id": 0}, {"id": 3}, {"id": 6}, {"id": 9}]

    def test_filter_remainder_1(self) -> None:
        from nemotron.data_prep.core.chat_sft_shard_core import _apply_row_filter

        records = [{"id": i} for i in range(10)]
        result = list(_apply_row_filter(iter(records), modulus=3, remainder=1))
        assert result == [{"id": 1}, {"id": 4}, {"id": 7}]

    def test_all_remainders_cover_all_rows(self) -> None:
        from nemotron.data_prep.core.chat_sft_shard_core import _apply_row_filter

        records = [{"id": i} for i in range(20)]
        modulus = 4
        all_ids = set()
        for remainder in range(modulus):
            result = list(_apply_row_filter(iter(records), modulus=modulus, remainder=remainder))
            ids = {r["id"] for r in result}
            assert not ids & all_ids, "Overlap between remainders"
            all_ids |= ids
        assert all_ids == {i for i in range(20)}

    def test_filter_empty_input(self) -> None:
        from nemotron.data_prep.core.chat_sft_shard_core import _apply_row_filter

        result = list(_apply_row_filter(iter([]), modulus=5, remainder=0))
        assert result == []

    def test_jsonl_core_filter(self) -> None:
        from nemotron.data_prep.core.jsonl_shard_core import _apply_row_filter

        records = [{"v": i} for i in range(6)]
        result = list(_apply_row_filter(iter(records), modulus=2, remainder=1))
        assert result == [{"v": 1}, {"v": 3}, {"v": 5}]
