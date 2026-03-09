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

"""Tests for the JSONL pipeline components (work items, planning, stages, recipe).

Tests cover:
1. JsonlDatasetWorkItem and JsonlShardWorkItem data structures
2. create_jsonl_shard_plan() planning without tokenizer
3. get_pending_jsonl_shards() receipt-based idempotency
4. JsonlPlanStage fan-out logic
5. JsonlShardStage transform + write logic
6. process_jsonl_shard_core() plan_hash/dataset_name in receipts
7. run_rl_resolve_pipeline() driver orchestration
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nemotron.data_prep.config import (
    DatasetConfig,
    FileInfo,
    ShardAssignment,
    ShardPlan,
)
from nemotron.data_prep.core.work_items import (
    JsonlDatasetWorkItem,
    JsonlShardWorkItem,
)


# =============================================================================
# Work Item Tests
# =============================================================================


class TestJsonlDatasetWorkItem:
    """Tests for JsonlDatasetWorkItem dataclass."""

    def test_creation_defaults(self) -> None:
        item = JsonlDatasetWorkItem(
            dataset_name="test_ds",
            path="hf://some/dataset",
            weight=1.0,
            split="train",
            subset=None,
            text_field="text",
            run_hash="abc123",
            run_dir="/output/runs/abc123",
            config_hash="def456",
            num_shards=4,
        )
        assert item.dataset_name == "test_ds"
        assert item.compression == "none"
        assert item.max_rows is None
        assert item.resolve_hf_placeholders is False

    def test_creation_with_all_fields(self) -> None:
        item = JsonlDatasetWorkItem(
            dataset_name="test_ds",
            path="hf://some/dataset",
            weight=1.0,
            split="train",
            subset="en",
            text_field="text",
            run_hash="abc123",
            run_dir="/output/runs/abc123",
            config_hash="def456",
            num_shards=4,
            compression="zstd",
            max_rows=1000,
            resolve_hf_placeholders=True,
        )
        assert item.compression == "zstd"
        assert item.max_rows == 1000
        assert item.resolve_hf_placeholders is True

    def test_none_split_and_subset(self) -> None:
        item = JsonlDatasetWorkItem(
            dataset_name="test_ds",
            path="/local/path",
            weight=1.0,
            split=None,
            subset=None,
            text_field="text",
            run_hash="abc",
            run_dir="/out/runs/abc",
            config_hash="def",
            num_shards=1,
        )
        assert item.split is None
        assert item.subset is None


class TestJsonlShardWorkItem:
    """Tests for JsonlShardWorkItem (with plan_hash field)."""

    def test_creation_with_plan_hash(self) -> None:
        item = JsonlShardWorkItem(
            dataset_name="test_ds",
            plan_hash="abcdef0123456789",
            shard_index=0,
            assignment={"files": [], "total_bytes": 0},
            output_dir="/output/datasets/test_ds/abcdef0123456789",
            receipts_dir="/output/datasets/test_ds/abcdef0123456789/receipts",
            text_field="text",
            compression="none",
            max_rows=None,
        )
        assert item.plan_hash == "abcdef0123456789"
        assert item.resolve_hf_placeholders is False

    def test_resolve_hf_placeholders_flag(self) -> None:
        item = JsonlShardWorkItem(
            dataset_name="test_ds",
            plan_hash="abc",
            shard_index=0,
            assignment={},
            output_dir="/out",
            receipts_dir="/out/receipts",
            text_field="text",
            compression="none",
            max_rows=None,
            resolve_hf_placeholders=True,
        )
        assert item.resolve_hf_placeholders is True


# =============================================================================
# Planning Tests
# =============================================================================


class TestCreateJsonlShardPlan:
    """Tests for create_jsonl_shard_plan()."""

    def test_creates_plan_with_correct_fields(self) -> None:
        """Plan should have type=none tokenizer and include transform_fingerprint."""
        from nemotron.data_prep.core.planning import create_jsonl_shard_plan

        # Mock filesystem and file discovery
        mock_fs = MagicMock()

        files = [
            FileInfo(path="/data/file1.jsonl", local_path=None, size=1000),
            FileInfo(path="/data/file2.jsonl", local_path=None, size=2000),
        ]

        dataset_cfg = DatasetConfig(
            name="test_ds",
            path="/data",
            text_field="text",
        )

        with patch("nemotron.data_prep.core.planning.discover_input_files", return_value=files):
            plan = create_jsonl_shard_plan(
                dataset_config=dataset_cfg,
                num_shards=2,
                config_hash="testhash",
                fs=mock_fs,
                transform_fingerprint="fp_abc123",
            )

        assert plan.resolved_tokenizer == {"type": "none"}
        assert plan.dataset_name == "test_ds"
        assert plan.num_shards == 2
        assert plan.config_hash == "testhash"
        assert "transform_fingerprint" in plan.determinism_constraints
        assert plan.determinism_constraints["transform_fingerprint"] == "fp_abc123"
        assert len(plan.file_assignments) == 2
        assert plan.plan_hash is not None
        assert len(plan.plan_hash) == 16  # SHA256[:16]

    def test_different_fingerprint_different_hash(self) -> None:
        """Changing transform fingerprint must produce a different plan_hash."""
        from nemotron.data_prep.core.planning import create_jsonl_shard_plan

        mock_fs = MagicMock()
        files = [FileInfo(path="/data/file1.jsonl", local_path=None, size=1000)]
        dataset_cfg = DatasetConfig(name="test_ds", path="/data", text_field="text")

        with patch("nemotron.data_prep.core.planning.discover_input_files", return_value=files):
            plan_a = create_jsonl_shard_plan(
                dataset_config=dataset_cfg,
                num_shards=1,
                config_hash="h",
                fs=mock_fs,
                transform_fingerprint="fingerprint_A",
            )
            plan_b = create_jsonl_shard_plan(
                dataset_config=dataset_cfg,
                num_shards=1,
                config_hash="h",
                fs=mock_fs,
                transform_fingerprint="fingerprint_B",
            )

        assert plan_a.plan_hash != plan_b.plan_hash

    def test_deterministic_plan_hash(self) -> None:
        """Same inputs should produce same plan_hash."""
        from nemotron.data_prep.core.planning import create_jsonl_shard_plan

        mock_fs = MagicMock()
        files = [FileInfo(path="/data/file1.jsonl", local_path=None, size=1000)]
        dataset_cfg = DatasetConfig(name="test_ds", path="/data", text_field="text")

        with patch("nemotron.data_prep.core.planning.discover_input_files", return_value=files):
            plan_1 = create_jsonl_shard_plan(
                dataset_config=dataset_cfg,
                num_shards=1,
                config_hash="h",
                fs=mock_fs,
                transform_fingerprint="fp",
            )
            plan_2 = create_jsonl_shard_plan(
                dataset_config=dataset_cfg,
                num_shards=1,
                config_hash="h",
                fs=mock_fs,
                transform_fingerprint="fp",
            )

        assert plan_1.plan_hash == plan_2.plan_hash

    def test_no_files_raises(self) -> None:
        """Should raise ValueError if no input files found."""
        from nemotron.data_prep.core.planning import create_jsonl_shard_plan

        mock_fs = MagicMock()
        dataset_cfg = DatasetConfig(name="test_ds", path="/data", text_field="text")

        with patch("nemotron.data_prep.core.planning.discover_input_files", return_value=[]):
            with pytest.raises(ValueError, match="No input files found"):
                create_jsonl_shard_plan(
                    dataset_config=dataset_cfg,
                    num_shards=1,
                    config_hash="h",
                    fs=mock_fs,
                    transform_fingerprint="fp",
                )


class TestGetPendingJsonlShards:
    """Tests for get_pending_jsonl_shards()."""

    def _make_plan(self, num_shards: int = 3, plan_hash: str = "testplanhash") -> ShardPlan:
        return ShardPlan(
            version="1.0",
            created_at="2025-01-01T00:00:00Z",
            plan_hash=plan_hash,
            dataset_name="test_ds",
            num_shards=num_shards,
            source_fingerprint="sha256:abc",
            config_hash="cfg",
            determinism_constraints={},
            resolved_tokenizer={"type": "none"},
            file_assignments=[
                ShardAssignment(shard_index=i, files=[], total_bytes=0)
                for i in range(num_shards)
            ],
        )

    def test_all_pending_when_no_receipts(self) -> None:
        """All shards pending when no receipt files exist."""
        from nemotron.data_prep.core.planning import get_pending_jsonl_shards

        mock_fs = MagicMock()
        mock_fs.glob.side_effect = FileNotFoundError

        plan = self._make_plan(num_shards=3)
        pending = get_pending_jsonl_shards(plan, "/receipts", mock_fs)
        assert pending == [0, 1, 2]

    def test_completed_shards_excluded(self) -> None:
        """Completed shards with matching plan_hash are excluded."""
        from nemotron.data_prep.core.planning import get_pending_jsonl_shards

        plan = self._make_plan(num_shards=3, plan_hash="myhash")

        receipt_0 = {
            "shard_index": 0,
            "plan_hash": "myhash",
            "status": "completed",
            "stats": {"num_records": 100},
            "output_file": "shard_000000.jsonl",
        }

        mock_fs = MagicMock()
        mock_fs.glob.return_value = ["/receipts/shard_000000.json"]
        mock_fs.exists.return_value = True  # output file exists

        with patch("nemotron.data_prep.core.planning.read_json", return_value=receipt_0):
            pending = get_pending_jsonl_shards(plan, "/receipts", mock_fs)

        assert pending == [1, 2]

    def test_wrong_plan_hash_not_excluded(self) -> None:
        """Receipts with wrong plan_hash don't count as completed."""
        from nemotron.data_prep.core.planning import get_pending_jsonl_shards

        plan = self._make_plan(num_shards=2, plan_hash="current_hash")

        old_receipt = {
            "shard_index": 0,
            "plan_hash": "old_hash",
            "status": "completed",
            "stats": {"num_records": 100},
            "output_file": "shard_000000.jsonl",
        }

        mock_fs = MagicMock()
        mock_fs.glob.return_value = ["/receipts/shard_000000.json"]

        with patch("nemotron.data_prep.core.planning.read_json", return_value=old_receipt):
            pending = get_pending_jsonl_shards(plan, "/receipts", mock_fs)

        assert pending == [0, 1]

    def test_empty_shard_receipt_accepted(self) -> None:
        """Empty shard (num_records=0) is still completed."""
        from nemotron.data_prep.core.planning import get_pending_jsonl_shards

        plan = self._make_plan(num_shards=2, plan_hash="h")

        empty_receipt = {
            "shard_index": 0,
            "plan_hash": "h",
            "status": "completed",
            "stats": {"num_records": 0},
            "output_file": None,
        }

        mock_fs = MagicMock()
        mock_fs.glob.return_value = ["/receipts/shard_000000.json"]

        with patch("nemotron.data_prep.core.planning.read_json", return_value=empty_receipt):
            pending = get_pending_jsonl_shards(plan, "/receipts", mock_fs)

        assert pending == [1]

    def test_missing_output_file_still_pending(self) -> None:
        """Non-empty shard with missing output file stays pending."""
        from nemotron.data_prep.core.planning import get_pending_jsonl_shards

        plan = self._make_plan(num_shards=1, plan_hash="h")

        receipt = {
            "shard_index": 0,
            "plan_hash": "h",
            "status": "completed",
            "stats": {"num_records": 50},
            "output_file": "shard_000000.jsonl",
        }

        mock_fs = MagicMock()
        mock_fs.glob.return_value = ["/receipts/shard_000000.json"]
        mock_fs.exists.return_value = False  # Output file missing

        with patch("nemotron.data_prep.core.planning.read_json", return_value=receipt):
            pending = get_pending_jsonl_shards(plan, "/receipts", mock_fs)

        assert pending == [0]


# =============================================================================
# Receipt Schema Tests
# =============================================================================


class TestJsonlReceiptSchema:
    """Tests for plan_hash and dataset_name in JSONL receipts."""

    def test_receipt_includes_plan_hash(self, tmp_path: Path) -> None:
        """process_jsonl_shard_core includes plan_hash in receipt."""
        from nemotron.data_prep.core.jsonl_shard_core import process_jsonl_shard_core
        from nemotron.data_prep.utils.filesystem import get_filesystem

        output_dir = str(tmp_path / "output")
        receipts_dir = str(tmp_path / "receipts")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(receipts_dir, exist_ok=True)

        output_fs, _ = get_filesystem(output_dir)

        # Create a test JSONL file
        jsonl_file = tmp_path / "input.jsonl"
        jsonl_file.write_text('{"text": "hello world"}\n{"text": "foo bar"}\n')

        file_dicts = [{"path": str(jsonl_file), "local_path": str(jsonl_file), "size": 50}]

        stats = process_jsonl_shard_core(
            shard_index=0,
            files=file_dicts,
            output_dir=output_dir,
            receipts_dir=receipts_dir,
            output_fs=output_fs,
            text_field="text",
            transform=None,
            compression="none",
            max_rows=None,
            local_files_only=True,
            plan_hash="test_plan_hash_123",
            dataset_name="my_dataset",
        )

        # Read the receipt
        receipt_path = Path(receipts_dir) / "shard_000000.json"
        assert receipt_path.exists()
        receipt = json.loads(receipt_path.read_text())

        assert receipt["plan_hash"] == "test_plan_hash_123"
        assert receipt["dataset_name"] == "my_dataset"
        assert receipt["status"] == "completed"
        assert receipt["stats"]["num_records"] == 2

    def test_receipt_without_plan_hash(self, tmp_path: Path) -> None:
        """plan_hash is optional; omitting it should not break receipts."""
        from nemotron.data_prep.core.jsonl_shard_core import process_jsonl_shard_core
        from nemotron.data_prep.utils.filesystem import get_filesystem

        output_dir = str(tmp_path / "output")
        receipts_dir = str(tmp_path / "receipts")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(receipts_dir, exist_ok=True)

        output_fs, _ = get_filesystem(output_dir)

        jsonl_file = tmp_path / "input.jsonl"
        jsonl_file.write_text('{"text": "hello"}\n')

        stats = process_jsonl_shard_core(
            shard_index=0,
            files=[{"path": str(jsonl_file), "local_path": str(jsonl_file), "size": 20}],
            output_dir=output_dir,
            receipts_dir=receipts_dir,
            output_fs=output_fs,
            text_field="text",
            transform=None,
            compression="none",
            max_rows=None,
            local_files_only=True,
            # plan_hash and dataset_name omitted (None)
        )

        receipt_path = Path(receipts_dir) / "shard_000000.json"
        receipt = json.loads(receipt_path.read_text())

        assert "plan_hash" not in receipt
        assert "dataset_name" not in receipt
        assert receipt["status"] == "completed"

    def test_empty_shard_receipt_includes_plan_hash(self, tmp_path: Path) -> None:
        """Empty shard receipts should also include plan_hash."""
        from nemotron.data_prep.core.jsonl_shard_core import process_jsonl_shard_core
        from nemotron.data_prep.utils.filesystem import get_filesystem

        output_dir = str(tmp_path / "output")
        receipts_dir = str(tmp_path / "receipts")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(receipts_dir, exist_ok=True)

        output_fs, _ = get_filesystem(output_dir)

        # Empty file list → empty shard
        stats = process_jsonl_shard_core(
            shard_index=0,
            files=[],
            output_dir=output_dir,
            receipts_dir=receipts_dir,
            output_fs=output_fs,
            text_field="text",
            transform=None,
            compression="none",
            max_rows=None,
            local_files_only=True,
            plan_hash="empty_plan_hash",
            dataset_name="empty_ds",
        )

        receipt_path = Path(receipts_dir) / "shard_000000.json"
        receipt = json.loads(receipt_path.read_text())

        assert receipt["plan_hash"] == "empty_plan_hash"
        assert receipt["dataset_name"] == "empty_ds"
        assert receipt["stats"]["num_records"] == 0


# =============================================================================
# Stage Config Tests
# =============================================================================


class TestJsonlPlanStageConfig:
    """Tests for JsonlPlanStageConfig validation."""

    def test_defaults(self) -> None:
        from nemotron.data_prep.stages.jsonl_plan import JsonlPlanStageConfig

        cfg = JsonlPlanStageConfig()
        assert cfg.planner_cpus == 0.5

    def test_custom_cpus(self) -> None:
        from nemotron.data_prep.stages.jsonl_plan import JsonlPlanStageConfig

        cfg = JsonlPlanStageConfig(planner_cpus=2.0)
        assert cfg.planner_cpus == 2.0

    def test_invalid_cpus(self) -> None:
        from nemotron.data_prep.stages.jsonl_plan import JsonlPlanStageConfig

        with pytest.raises(ValueError, match="planner_cpus must be positive"):
            JsonlPlanStageConfig(planner_cpus=0)

        with pytest.raises(ValueError, match="planner_cpus must be positive"):
            JsonlPlanStageConfig(planner_cpus=-1.0)


class TestJsonlShardStageConfig:
    """Tests for JsonlShardStageConfig validation."""

    def test_defaults(self) -> None:
        from nemotron.data_prep.stages.jsonl_write import JsonlShardStageConfig

        cfg = JsonlShardStageConfig()
        assert cfg.stage_cpus == 1.0
        assert cfg.local_files_only is True

    def test_custom_values(self) -> None:
        from nemotron.data_prep.stages.jsonl_write import JsonlShardStageConfig

        cfg = JsonlShardStageConfig(stage_cpus=2.0, local_files_only=False)
        assert cfg.stage_cpus == 2.0
        assert cfg.local_files_only is False

    def test_invalid_cpus(self) -> None:
        from nemotron.data_prep.stages.jsonl_write import JsonlShardStageConfig

        with pytest.raises(ValueError, match="stage_cpus must be positive"):
            JsonlShardStageConfig(stage_cpus=0)


# =============================================================================
# Stage Behavior Tests
# =============================================================================


class TestJsonlPlanStageBehavior:
    """Tests for JsonlPlanStage process_data logic."""

    def test_fan_out_creates_shard_work_items(self) -> None:
        """PlanStage should fan out a dataset item into shard work items."""
        from nemotron.data_prep.stages.jsonl_plan import JsonlPlanStage, JsonlPlanStageConfig
        from nemotron.data_prep.stages.context import PipelineContext

        ctx = PipelineContext(
            output_root="/output",
            run_hash="test_run",
            run_dir="/output/runs/test_run",
            config_hash="cfg_hash",
            hf_env={},
        )
        stage = JsonlPlanStage(JsonlPlanStageConfig(), ctx)

        # Mock the filesystem
        mock_fs = MagicMock()
        mock_fs.exists.return_value = False  # No existing receipts
        mock_fs.glob.side_effect = FileNotFoundError  # No receipts dir
        stage._fs = mock_fs

        input_item = JsonlDatasetWorkItem(
            dataset_name="test_ds__train",
            path="hf://test/dataset",
            weight=1.0,
            split="train",
            subset=None,
            text_field="text",
            run_hash="test_run",
            run_dir="/output/runs/test_run",
            config_hash="cfg_hash",
            num_shards=2,
            compression="none",
            max_rows=None,
            resolve_hf_placeholders=True,
        )

        files = [
            FileInfo(path="/data/f1.jsonl", local_path=None, size=1000),
            FileInfo(path="/data/f2.jsonl", local_path=None, size=2000),
        ]

        with patch("nemotron.data_prep.core.planning.discover_input_files", return_value=files):
            shard_items = stage.process_data([input_item])

        assert len(shard_items) == 2
        for item in shard_items:
            assert isinstance(item, JsonlShardWorkItem)
            assert item.dataset_name == "test_ds__train"
            assert item.text_field == "text"
            assert item.compression == "none"
            assert item.resolve_hf_placeholders is True
            assert item.plan_hash is not None
            assert len(item.plan_hash) == 16

        assert {item.shard_index for item in shard_items} == {0, 1}


class TestJsonlShardStageBehavior:
    """Tests for JsonlShardStage process_data logic."""

    def test_calls_process_jsonl_shard_core(self) -> None:
        """Stage should call process_jsonl_shard_core for each work item."""
        from nemotron.data_prep.stages.jsonl_write import JsonlShardStage, JsonlShardStageConfig
        from nemotron.data_prep.stages.context import PipelineContext

        ctx = PipelineContext(output_root="/output", hf_env={})
        stage = JsonlShardStage(JsonlShardStageConfig(), ctx)
        stage._output_fs = MagicMock()

        work_item = JsonlShardWorkItem(
            dataset_name="test_ds",
            plan_hash="abcdef",
            shard_index=0,
            assignment={"files": [{"path": "/f.jsonl", "size": 100}]},
            output_dir="/out",
            receipts_dir="/out/receipts",
            text_field="text",
            compression="none",
            max_rows=None,
            resolve_hf_placeholders=False,
        )

        with patch("nemotron.data_prep.stages.jsonl_write.process_jsonl_shard_core") as mock_core:
            mock_core.return_value = {"num_records": 10}
            result = stage.process_data([work_item])

        assert result == [work_item]
        mock_core.assert_called_once()
        call_kwargs = mock_core.call_args[1]
        assert call_kwargs["plan_hash"] == "abcdef"
        assert call_kwargs["dataset_name"] == "test_ds"
        assert call_kwargs["shard_index"] == 0

    def test_creates_resolver_when_hf_placeholders_enabled(self) -> None:
        """Stage should create transform when resolve_hf_placeholders is True."""
        from nemotron.data_prep.stages.jsonl_write import JsonlShardStage, JsonlShardStageConfig
        from nemotron.data_prep.stages.context import PipelineContext

        ctx = PipelineContext(output_root="/output", hf_env={})
        stage = JsonlShardStage(JsonlShardStageConfig(), ctx)
        stage._output_fs = MagicMock()

        work_item = JsonlShardWorkItem(
            dataset_name="test_ds",
            plan_hash="abc",
            shard_index=0,
            assignment={"files": []},
            output_dir="/out",
            receipts_dir="/out/receipts",
            text_field="text",
            compression="none",
            max_rows=None,
            resolve_hf_placeholders=True,
        )

        mock_resolver = MagicMock()
        with (
            patch("nemotron.data_prep.stages.jsonl_write.process_jsonl_shard_core") as mock_core,
            patch("nemotron.data_prep.stages.jsonl_write.HFPlaceholderResolver") as mock_cls,
            patch("nemotron.data_prep.stages.jsonl_write.resolve_hf_placeholders") as mock_transform_fn,
        ):
            mock_cls.create.return_value = mock_resolver
            mock_core.return_value = {"num_records": 0}
            mock_transform_fn.return_value = lambda x: x

            stage.process_data([work_item])

        # Resolver should have been created
        mock_cls.create.assert_called_once()
        # Transform factory should have been called with the resolver
        mock_transform_fn.assert_called_once_with(resolver=mock_resolver)

    def test_no_resolver_when_hf_placeholders_disabled(self) -> None:
        """Stage should not create resolver when resolve_hf_placeholders is False."""
        from nemotron.data_prep.stages.jsonl_write import JsonlShardStage, JsonlShardStageConfig
        from nemotron.data_prep.stages.context import PipelineContext

        ctx = PipelineContext(output_root="/output", hf_env={})
        stage = JsonlShardStage(JsonlShardStageConfig(), ctx)
        stage._output_fs = MagicMock()

        work_item = JsonlShardWorkItem(
            dataset_name="test_ds",
            plan_hash="abc",
            shard_index=0,
            assignment={"files": []},
            output_dir="/out",
            receipts_dir="/out/receipts",
            text_field="text",
            compression="none",
            max_rows=None,
            resolve_hf_placeholders=False,
        )

        with (
            patch("nemotron.data_prep.stages.jsonl_write.process_jsonl_shard_core") as mock_core,
            patch("nemotron.data_prep.stages.jsonl_write.HFPlaceholderResolver") as mock_cls,
        ):
            mock_core.return_value = {"num_records": 0}

            stage.process_data([work_item])

        # Resolver should NOT have been created
        mock_cls.create.assert_not_called()
        # Transform should be None
        call_kwargs = mock_core.call_args[1]
        assert call_kwargs["transform"] is None


# =============================================================================
# Recipe Result Tests
# =============================================================================


class TestRlResolveResult:
    """Tests for the RlResolveResult dataclass."""

    def test_creation(self) -> None:
        from nemotron.data_prep.recipes.rl import RlResolveResult

        result = RlResolveResult(
            run_hash="abc123",
            run_dir="/out/runs/abc123",
            split_paths={"train": "/out/train.jsonl", "val": "/out/val.jsonl"},
            total_records=1500,
            manifest_path="/out/manifest.json",
        )
        assert result.run_hash == "abc123"
        assert result.total_records == 1500
        assert "train" in result.split_paths
        assert "val" in result.split_paths

    def test_frozen(self) -> None:
        from nemotron.data_prep.recipes.rl import RlResolveResult

        result = RlResolveResult(
            run_hash="a",
            run_dir="/r",
            split_paths={},
            total_records=0,
            manifest_path="/m",
        )
        with pytest.raises(AttributeError):
            result.run_hash = "b"  # type: ignore[misc]


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Verify that all new symbols are properly exported."""

    def test_core_exports(self) -> None:
        from nemotron.data_prep.core import (
            JsonlDatasetWorkItem,
            JsonlShardWorkItem,
            create_jsonl_shard_plan,
            get_pending_jsonl_shards,
        )

        assert JsonlDatasetWorkItem is not None
        assert JsonlShardWorkItem is not None
        assert callable(create_jsonl_shard_plan)
        assert callable(get_pending_jsonl_shards)

    def test_stages_exports(self) -> None:
        from nemotron.data_prep.stages import (
            JsonlPlanStage,
            JsonlPlanStageConfig,
            JsonlShardStage,
            JsonlShardStageConfig,
        )

        assert JsonlPlanStage is not None
        assert JsonlPlanStageConfig is not None
        assert JsonlShardStage is not None
        assert JsonlShardStageConfig is not None

    def test_recipes_exports(self) -> None:
        from nemotron.data_prep.recipes import run_rl_resolve_pipeline

        assert callable(run_rl_resolve_pipeline)

    def test_work_items_all(self) -> None:
        import nemotron.data_prep.core.work_items as wm

        assert "JsonlDatasetWorkItem" in wm.__all__
        assert "JsonlShardWorkItem" in wm.__all__
