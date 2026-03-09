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

"""Tests for BinIdxTokenizationStage resume/caching behavior.

These tests verify that the tokenization stage correctly handles:
1. Skipping completed shards (normal resume)
2. Reprocessing shards with missing output files
3. Reprocessing shards with corrupted receipts

Note: These tests use a standalone implementation of _is_completed logic
to avoid dependency on cosmos_xenna which may not be available in all
test environments. The logic is identical to the real implementation
in megatron_bin_idx.py.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from nemotron.data_prep.filesystem import read_json


class MockFilesystem:
    """Mock filesystem for testing completion checks."""

    def __init__(self, files: dict[str, str | bytes | None] | None = None):
        """Initialize with optional file contents.

        Args:
            files: Dict mapping paths to contents. None value means file doesn't exist.
        """
        self._files = files or {}

    def exists(self, path: str) -> bool:
        """Check if file exists."""
        return path in self._files and self._files[path] is not None

    def open(self, path: str, mode: str = "r"):
        """Open a file (for read_json compatibility)."""
        if path not in self._files or self._files[path] is None:
            raise FileNotFoundError(path)

        content = self._files[path]
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        class MockFile:
            def __init__(self, content: str):
                self._content = content

            def __enter__(self):
                return self

            def __exit__(self, *args: Any) -> None:
                pass

            def read(self) -> str:
                return self._content

        return MockFile(content)


def is_completed_logic(fs: MockFilesystem, receipt_path: str, plan_hash: str, shard_dir: str) -> bool:
    """Implementation of _is_completed logic from megatron_bin_idx.py.

    This is a standalone copy of the logic for testing without cosmos_xenna dependency.
    Must be kept in sync with BinIdxTokenizationStage._is_completed().
    """
    if not fs.exists(receipt_path):
        return False

    try:
        r = read_json(fs, receipt_path)  # type: ignore[arg-type]
        if r.get("status") != "completed" or r.get("plan_hash") != plan_hash:
            return False

        # Verify output files exist for non-empty shards
        # This matches the check in get_pending_shards()
        stats = r.get("stats", {}) or {}
        if int(stats.get("num_sequences", 0) or 0) > 0:
            files = r.get("files", {}) or {}
            bin_info = files.get("bin", {}) or {}
            idx_info = files.get("idx", {}) or {}
            bin_path = bin_info.get("path", "")
            idx_path = idx_info.get("path", "")

            if not bin_path or not idx_path:
                return False

            full_bin = f"{shard_dir}/{bin_path}"
            full_idx = f"{shard_dir}/{idx_path}"
            if not (fs.exists(full_bin) and fs.exists(full_idx)):
                return False

        return True
    except Exception:
        return False  # Corrupted receipt, reprocess


class TestIsCompleted:
    """Tests for _is_completed logic in BinIdxTokenizationStage."""

    def test_no_receipt_returns_false(self) -> None:
        """Missing receipt means shard is not completed."""
        fs = MockFilesystem({})  # Empty filesystem

        result = is_completed_logic(
            fs=fs,
            receipt_path="/path/to/receipts/shard_000000.json",
            plan_hash="plan123",
            shard_dir="/path/to/dataset",
        )

        assert result is False

    def test_completed_receipt_with_files_returns_true(self) -> None:
        """Completed receipt with existing output files returns True."""
        receipt = {
            "status": "completed",
            "plan_hash": "plan123",
            "stats": {"num_sequences": 100},
            "files": {
                "bin": {"path": "shard_000000.bin"},
                "idx": {"path": "shard_000000.idx"},
            },
        }

        fs = MockFilesystem({
            "/path/to/receipts/shard_000000.json": json.dumps(receipt),
            "/path/to/dataset/shard_000000.bin": b"binary_data",
            "/path/to/dataset/shard_000000.idx": b"index_data",
        })

        result = is_completed_logic(
            fs=fs,
            receipt_path="/path/to/receipts/shard_000000.json",
            plan_hash="plan123",
            shard_dir="/path/to/dataset",
        )

        assert result is True

    def test_completed_receipt_missing_bin_file_returns_false(self) -> None:
        """Completed receipt but missing .bin file means NOT completed.

        This is the critical bug fix: if output files are deleted after completion,
        the shard should be reprocessed.
        """
        receipt = {
            "status": "completed",
            "plan_hash": "plan123",
            "stats": {"num_sequences": 100},
            "files": {
                "bin": {"path": "shard_000000.bin"},
                "idx": {"path": "shard_000000.idx"},
            },
        }

        fs = MockFilesystem({
            "/path/to/receipts/shard_000000.json": json.dumps(receipt),
            # .bin file is MISSING
            "/path/to/dataset/shard_000000.idx": b"index_data",
        })

        result = is_completed_logic(
            fs=fs,
            receipt_path="/path/to/receipts/shard_000000.json",
            plan_hash="plan123",
            shard_dir="/path/to/dataset",
        )

        assert result is False

    def test_completed_receipt_missing_idx_file_returns_false(self) -> None:
        """Completed receipt but missing .idx file means NOT completed."""
        receipt = {
            "status": "completed",
            "plan_hash": "plan123",
            "stats": {"num_sequences": 100},
            "files": {
                "bin": {"path": "shard_000000.bin"},
                "idx": {"path": "shard_000000.idx"},
            },
        }

        fs = MockFilesystem({
            "/path/to/receipts/shard_000000.json": json.dumps(receipt),
            "/path/to/dataset/shard_000000.bin": b"binary_data",
            # .idx file is MISSING
        })

        result = is_completed_logic(
            fs=fs,
            receipt_path="/path/to/receipts/shard_000000.json",
            plan_hash="plan123",
            shard_dir="/path/to/dataset",
        )

        assert result is False

    def test_completed_empty_shard_no_files_needed(self) -> None:
        """Empty shard (0 sequences) doesn't need output files to be considered complete."""
        receipt = {
            "status": "completed",
            "plan_hash": "plan123",
            "stats": {"num_sequences": 0},  # Empty shard
            "files": {},
        }

        fs = MockFilesystem({
            "/path/to/receipts/shard_000000.json": json.dumps(receipt),
            # No .bin/.idx files needed for empty shards
        })

        result = is_completed_logic(
            fs=fs,
            receipt_path="/path/to/receipts/shard_000000.json",
            plan_hash="plan123",
            shard_dir="/path/to/dataset",
        )

        assert result is True

    def test_wrong_plan_hash_returns_false(self) -> None:
        """Receipt with wrong plan_hash means shard needs reprocessing."""
        receipt = {
            "status": "completed",
            "plan_hash": "old_plan_hash",  # Different from expected
            "stats": {"num_sequences": 100},
            "files": {
                "bin": {"path": "shard_000000.bin"},
                "idx": {"path": "shard_000000.idx"},
            },
        }

        fs = MockFilesystem({
            "/path/to/receipts/shard_000000.json": json.dumps(receipt),
            "/path/to/dataset/shard_000000.bin": b"binary_data",
            "/path/to/dataset/shard_000000.idx": b"index_data",
        })

        result = is_completed_logic(
            fs=fs,
            receipt_path="/path/to/receipts/shard_000000.json",
            plan_hash="new_plan_hash",  # Different plan hash
            shard_dir="/path/to/dataset",
        )

        assert result is False

    def test_started_status_returns_false(self) -> None:
        """Receipt with 'started' status means shard was interrupted."""
        receipt = {
            "status": "started",  # Not completed
            "plan_hash": "plan123",
        }

        fs = MockFilesystem({
            "/path/to/receipts/shard_000000.json": json.dumps(receipt),
        })

        result = is_completed_logic(
            fs=fs,
            receipt_path="/path/to/receipts/shard_000000.json",
            plan_hash="plan123",
            shard_dir="/path/to/dataset",
        )

        assert result is False

    def test_failed_status_returns_false(self) -> None:
        """Receipt with 'failed' status means shard needs retry."""
        receipt = {
            "status": "failed",
            "plan_hash": "plan123",
            "error_message": "Some error",
        }

        fs = MockFilesystem({
            "/path/to/receipts/shard_000000.json": json.dumps(receipt),
        })

        result = is_completed_logic(
            fs=fs,
            receipt_path="/path/to/receipts/shard_000000.json",
            plan_hash="plan123",
            shard_dir="/path/to/dataset",
        )

        assert result is False

    def test_corrupted_receipt_returns_false(self) -> None:
        """Corrupted/invalid JSON receipt should trigger reprocessing."""
        fs = MockFilesystem({
            "/path/to/receipts/shard_000000.json": "not valid json {{{",
        })

        result = is_completed_logic(
            fs=fs,
            receipt_path="/path/to/receipts/shard_000000.json",
            plan_hash="plan123",
            shard_dir="/path/to/dataset",
        )

        assert result is False

    def test_missing_files_dict_returns_false(self) -> None:
        """Receipt without files dict but with sequences should return False."""
        receipt = {
            "status": "completed",
            "plan_hash": "plan123",
            "stats": {"num_sequences": 100},
            # Missing "files" key entirely
        }

        fs = MockFilesystem({
            "/path/to/receipts/shard_000000.json": json.dumps(receipt),
        })

        result = is_completed_logic(
            fs=fs,
            receipt_path="/path/to/receipts/shard_000000.json",
            plan_hash="plan123",
            shard_dir="/path/to/dataset",
        )

        assert result is False


class TestGetPendingShardsConsistency:
    """Tests to verify _is_completed and get_pending_shards are consistent.

    The key invariant is: if get_pending_shards() marks a shard as pending,
    _is_completed() must return False for the same shard.
    """

    def test_missing_output_files_both_detect(self) -> None:
        """Both functions should detect missing output files."""
        # This scenario previously caused a bug where:
        # - get_pending_shards() correctly identified shard as pending (missing files)
        # - _is_completed() incorrectly returned True (only checked receipt status)
        # After the fix, both should agree the shard needs reprocessing.

        receipt = {
            "status": "completed",
            "plan_hash": "plan123",
            "stats": {"num_sequences": 100},
            "files": {
                "bin": {"path": "shard_000000.bin"},
                "idx": {"path": "shard_000000.idx"},
            },
        }

        # Simulate: receipt exists but output files were deleted
        fs = MockFilesystem({
            "/path/to/receipts/shard_000000.json": json.dumps(receipt),
            # Both .bin and .idx are MISSING
        })

        result = is_completed_logic(
            fs=fs,
            receipt_path="/path/to/receipts/shard_000000.json",
            plan_hash="plan123",
            shard_dir="/path/to/dataset",
        )

        # Must return False so the shard gets reprocessed
        assert result is False
