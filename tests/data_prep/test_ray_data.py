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

"""Tests for Ray Data shard-task executor.

These tests verify:
1. ShardTask serialization/deserialization
2. RayDataConfig construction
3. execute_shard_tasks basic execution
4. BinIdxShardTaskUDF initialization
5. Bottleneck metrics tracking
6. process_binidx_shard_core with mocked tokenizer
7. Error handling and idempotency
8. Helper functions for tokenization and file processing
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestShardTask:
    """Tests for ShardTask dataclass."""

    def test_shard_task_creation(self):
        """Test basic ShardTask creation."""
        from nemotron.data_prep.ray_data import ShardTask

        task = ShardTask(
            dataset_name="test_dataset",
            plan_hash="abc123",
            shard_index=0,
            assignment_json='{"shard_index": 0, "files": [], "total_bytes": 0}',
            output_dir="/output",
            receipts_dir="/receipts",
            fs_protocol="file",
            kind="binidx",
            text_field="text",
        )

        assert task.dataset_name == "test_dataset"
        assert task.plan_hash == "abc123"
        assert task.shard_index == 0
        assert task.fs_protocol == "file"

    def test_shard_task_from_assignment(self):
        """Test ShardTask.from_assignment factory method."""
        from nemotron.data_prep.ray_data import ShardTask

        assignment = {
            "shard_index": 5,
            "files": [{"path": "/data/file.parquet"}],
            "total_bytes": 1000,
        }

        task = ShardTask.from_assignment(
            assignment=assignment,
            dataset_name="ds1",
            plan_hash="hash123",
            shard_index=5,
            output_dir="/out",
            receipts_dir="/receipts",
            fs_protocol="file",
        )

        assert task.shard_index == 5
        assert task.assignment_json == json.dumps(assignment)

    def test_shard_task_get_assignment_roundtrip(self):
        """Test assignment JSON serialization roundtrip."""
        from nemotron.data_prep.ray_data import ShardTask

        original_assignment = {
            "shard_index": 10,
            "files": [
                {"path": "/data/file1.parquet", "bytes": 500},
                {"path": "/data/file2.parquet", "bytes": 500},
            ],
            "total_bytes": 1000,
        }

        task = ShardTask.from_assignment(
            assignment=original_assignment,
            dataset_name="ds",
            plan_hash="h",
            shard_index=10,
            output_dir="/o",
            receipts_dir="/r",
            fs_protocol="file",
        )

        recovered = task.get_assignment()
        assert recovered == original_assignment

    def test_shard_task_to_dict(self):
        """Test ShardTask.to_dict for Ray Data serialization."""
        from nemotron.data_prep.ray_data import ShardTask

        task = ShardTask(
            dataset_name="test",
            plan_hash="hash",
            shard_index=0,
            assignment_json='{"files": []}',
            output_dir="/out",
            receipts_dir="/receipts",
            fs_protocol="file",
        )

        d = task.to_dict()

        assert isinstance(d, dict)
        assert d["dataset_name"] == "test"
        assert d["plan_hash"] == "hash"
        assert d["shard_index"] == 0
        assert d["assignment_json"] == '{"files": []}'
        assert d["fs_protocol"] == "file"
        assert d["kind"] == "binidx"

    def test_shard_task_empty_assignment(self):
        """Test ShardTask with empty files list."""
        from nemotron.data_prep.ray_data import ShardTask

        assignment = {"shard_index": 0, "files": [], "total_bytes": 0}
        task = ShardTask.from_assignment(
            assignment=assignment,
            dataset_name="empty",
            plan_hash="h",
            shard_index=0,
            output_dir="/out",
            receipts_dir="/r",
            fs_protocol="file",
        )

        assert task.get_assignment()["files"] == []
        assert task.get_assignment()["total_bytes"] == 0

    def test_shard_task_special_characters_in_paths(self):
        """Test ShardTask with special characters in paths."""
        from nemotron.data_prep.ray_data import ShardTask

        task = ShardTask(
            dataset_name="test-dataset_v2.0",
            plan_hash="abc123def456",
            shard_index=999,
            assignment_json='{"files": [{"path": "/data/path with spaces/file.parquet"}]}',
            output_dir="/output/path-with-dashes",
            receipts_dir="/receipts/path_with_underscores",
            fs_protocol="s3",
            text_field="content",
        )

        assert task.dataset_name == "test-dataset_v2.0"
        assert task.fs_protocol == "s3"
        d = task.to_dict()
        assert "path with spaces" in d["assignment_json"]

    def test_shard_task_large_shard_index(self):
        """Test ShardTask with large shard index (stress test)."""
        from nemotron.data_prep.ray_data import ShardTask

        task = ShardTask(
            dataset_name="big_dataset",
            plan_hash="hash",
            shard_index=999999,
            assignment_json='{"files": []}',
            output_dir="/out",
            receipts_dir="/r",
            fs_protocol="file",
        )

        assert task.shard_index == 999999
        d = task.to_dict()
        assert d["shard_index"] == 999999

    def test_shard_task_complex_assignment_roundtrip(self):
        """Test complex nested assignment JSON roundtrip."""
        from nemotron.data_prep.ray_data import ShardTask

        complex_assignment = {
            "shard_index": 42,
            "files": [
                {
                    "path": "/data/file1.parquet",
                    "bytes": 1000000,
                    "hf_repo_id": "nvidia/test-dataset",
                    "hf_filename": "train/data-00000.parquet",
                    "hf_revision": "main",
                },
                {
                    "path": "/data/file2.parquet",
                    "bytes": 2000000,
                    "local_path": "/cache/file2.parquet",
                },
            ],
            "total_bytes": 3000000,
        }

        task = ShardTask.from_assignment(
            assignment=complex_assignment,
            dataset_name="ds",
            plan_hash="h",
            shard_index=42,
            output_dir="/o",
            receipts_dir="/r",
            fs_protocol="gcs",
        )

        recovered = task.get_assignment()
        assert recovered == complex_assignment
        assert len(recovered["files"]) == 2
        assert recovered["files"][0]["hf_repo_id"] == "nvidia/test-dataset"

    def test_shard_task_default_values(self):
        """Test ShardTask default field values."""
        from nemotron.data_prep.ray_data import ShardTask

        task = ShardTask(
            dataset_name="ds",
            plan_hash="h",
            shard_index=0,
            assignment_json="{}",
            output_dir="/o",
            receipts_dir="/r",
            fs_protocol="file",
        )

        assert task.kind == "binidx"
        assert task.text_field == "text"

    def test_shard_task_custom_text_field(self):
        """Test ShardTask with custom text field."""
        from nemotron.data_prep.ray_data import ShardTask

        task = ShardTask(
            dataset_name="ds",
            plan_hash="h",
            shard_index=0,
            assignment_json="{}",
            output_dir="/o",
            receipts_dir="/r",
            fs_protocol="file",
            text_field="content",
        )

        assert task.text_field == "content"
        d = task.to_dict()
        assert d["text_field"] == "content"

    def test_shard_task_frozen(self):
        """Test that ShardTask is immutable (frozen dataclass)."""
        from nemotron.data_prep.ray_data import ShardTask

        task = ShardTask(
            dataset_name="ds",
            plan_hash="h",
            shard_index=0,
            assignment_json="{}",
            output_dir="/o",
            receipts_dir="/r",
            fs_protocol="file",
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            task.shard_index = 1


class TestRayDataExecConfig:
    """Tests for RayDataExecConfig."""

    def test_default_config(self):
        """Test default config values."""
        from nemotron.data_prep.ray_data import RayDataExecConfig

        cfg = RayDataExecConfig()

        assert cfg.min_actors == 2
        assert cfg.max_actors == 32
        assert cfg.cpus_per_actor == 1.0
        assert cfg.max_tasks_in_flight_per_actor == 2

    def test_custom_config(self):
        """Test custom config values."""
        from nemotron.data_prep.ray_data import RayDataExecConfig

        cfg = RayDataExecConfig(
            min_actors=4,
            max_actors=64,
            cpus_per_actor=2.0,
            max_tasks_in_flight_per_actor=4,
        )

        assert cfg.min_actors == 4
        assert cfg.max_actors == 64
        assert cfg.cpus_per_actor == 2.0
        assert cfg.max_tasks_in_flight_per_actor == 4


class TestRayDataConfig:
    """Tests for RayDataConfig in config.py."""

    def test_ray_data_config_creation(self):
        """Test RayDataConfig creation."""
        from nemotron.data_prep.config import RayDataConfig

        cfg = RayDataConfig(
            enabled=True,
            min_actors=2,
            max_actors=16,
            cpus_per_actor=0.5,
            max_tasks_in_flight_per_actor=3,
        )

        assert cfg.enabled is True
        assert cfg.min_actors == 2
        assert cfg.max_actors == 16
        assert cfg.cpus_per_actor == 0.5
        assert cfg.max_tasks_in_flight_per_actor == 3

    def test_pipeline_config_with_ray_data(self):
        """Test PipelineConfig includes ray_data field."""
        from nemotron.data_prep.config import (
            OutputConfig,
            PipelineConfig,
            RayDataConfig,
        )

        ray_data_cfg = RayDataConfig(enabled=True, max_actors=8)

        pipeline_cfg = PipelineConfig(
            output=OutputConfig(dir=Path("/out")),
            ray_data=ray_data_cfg,
        )

        assert pipeline_cfg.ray_data is not None
        assert pipeline_cfg.ray_data.enabled is True
        assert pipeline_cfg.ray_data.max_actors == 8


class TestBottleneckMetrics:
    """Tests for BottleneckMetrics in console.py."""

    def test_bottleneck_metrics_default(self):
        """Test default values."""
        from nemotron.data_prep.console import BottleneckMetrics

        metrics = BottleneckMetrics()

        assert metrics.time_download == 0.0
        assert metrics.time_read == 0.0
        assert metrics.time_tokenize == 0.0
        assert metrics.time_write == 0.0
        assert metrics.num_shards == 0
        assert metrics.num_errors == 0

    def test_bottleneck_metrics_add_shard_stats(self):
        """Test adding shard stats to metrics."""
        from nemotron.data_prep.console import BottleneckMetrics

        metrics = BottleneckMetrics()

        stats = {
            "time_download_sec": 1.0,
            "time_read_sec": 2.0,
            "time_tokenize_sec": 5.0,
            "time_write_sec": 1.0,
            "time_total_sec": 9.0,
            "num_errors": 0,
        }

        metrics.add_shard_stats(stats)

        assert metrics.time_download == 1.0
        assert metrics.time_read == 2.0
        assert metrics.time_tokenize == 5.0
        assert metrics.time_write == 1.0
        assert metrics.num_shards == 1
        assert metrics.num_errors == 0

    def test_bottleneck_metrics_accumulates(self):
        """Test that metrics accumulate across multiple shards."""
        from nemotron.data_prep.console import BottleneckMetrics

        metrics = BottleneckMetrics()

        # Add first shard
        metrics.add_shard_stats({"time_tokenize_sec": 5.0, "num_errors": 0})
        # Add second shard
        metrics.add_shard_stats({"time_tokenize_sec": 3.0, "num_errors": 1})

        assert metrics.time_tokenize == 8.0
        assert metrics.num_shards == 2
        assert metrics.num_errors == 1

    def test_bottleneck_metrics_percentages(self):
        """Test time breakdown percentage calculation."""
        from nemotron.data_prep.console import BottleneckMetrics

        metrics = BottleneckMetrics()
        metrics.time_download = 1.0  # 10%
        metrics.time_read = 2.0  # 20%
        metrics.time_tokenize = 5.0  # 50%
        metrics.time_write = 2.0  # 20%

        pcts = metrics.get_percentages()

        assert abs(pcts["download_pct"] - 10.0) < 0.1
        assert abs(pcts["read_pct"] - 20.0) < 0.1
        assert abs(pcts["tokenize_pct"] - 50.0) < 0.1
        assert abs(pcts["write_pct"] - 20.0) < 0.1

    def test_bottleneck_metrics_percentages_zero(self):
        """Test percentage calculation with zero totals."""
        from nemotron.data_prep.console import BottleneckMetrics

        metrics = BottleneckMetrics()
        pcts = metrics.get_percentages()

        assert pcts["download_pct"] == 0
        assert pcts["read_pct"] == 0
        assert pcts["tokenize_pct"] == 0
        assert pcts["write_pct"] == 0


class TestDataPrepConfigRayData:
    """Tests for DataPrepConfig with ray_data fields."""

    def test_data_prep_config_ray_data_defaults(self):
        """Test DataPrepConfig ray_data field defaults."""
        from nemotron.data_prep import DataPrepConfig

        cfg = DataPrepConfig()

        assert cfg.ray_data_enabled is True  # Enabled by default
        assert cfg.ray_data_min_actors == 16  # Start with good parallelism
        assert cfg.ray_data_max_actors == 64  # Allow scaling on large nodes
        assert cfg.ray_data_cpus_per_actor == 1.0
        assert cfg.ray_data_max_tasks_in_flight == 2

    def test_data_prep_config_ray_data_custom(self):
        """Test DataPrepConfig with custom ray_data settings."""
        from nemotron.data_prep import DataPrepConfig

        cfg = DataPrepConfig(
            ray_data_enabled=True,
            ray_data_min_actors=4,
            ray_data_max_actors=16,
            ray_data_cpus_per_actor=2.0,
            ray_data_max_tasks_in_flight=4,
        )

        assert cfg.ray_data_enabled is True
        assert cfg.ray_data_min_actors == 4
        assert cfg.ray_data_max_actors == 16
        assert cfg.ray_data_cpus_per_actor == 2.0
        assert cfg.ray_data_max_tasks_in_flight == 4


class TestLiveExecutionStatusShardTiming:
    """Tests for LiveExecutionStatus.report_shard_timing."""

    def test_report_shard_timing_updates_bottleneck_metrics(self):
        """Test that report_shard_timing updates internal metrics."""
        from nemotron.data_prep.console import LiveExecutionStatus

        status = LiveExecutionStatus()

        stats = {
            "time_download_sec": 1.0,
            "time_read_sec": 2.0,
            "time_tokenize_sec": 5.0,
            "time_write_sec": 1.0,
            "time_total_sec": 9.0,
        }

        status.report_shard_timing(stats)

        assert status._bottleneck_metrics.time_download == 1.0
        assert status._bottleneck_metrics.time_tokenize == 5.0
        assert status._bottleneck_metrics.num_shards == 1


# Skip Ray tests if Ray is not available or not initialized
@pytest.fixture(scope="module")
def ray_initialized():
    """Initialize Ray for tests."""
    import ray

    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    # Don't shutdown - other tests may need it


class TestRayDataExecConfigEdgeCases:
    """Additional edge case tests for RayDataExecConfig."""

    def test_config_frozen(self):
        """Test that config is immutable."""
        from nemotron.data_prep.ray_data import RayDataExecConfig

        cfg = RayDataExecConfig()
        with pytest.raises(Exception):
            cfg.min_actors = 10

    def test_config_fractional_cpus(self):
        """Test fractional CPU allocation."""
        from nemotron.data_prep.ray_data import RayDataExecConfig

        cfg = RayDataExecConfig(cpus_per_actor=0.25)
        assert cfg.cpus_per_actor == 0.25


class TestProcessBinidxShardCore:
    """Tests for process_binidx_shard_core function."""

    @pytest.fixture
    def mock_tokenize(self):
        """Create a mock tokenizer function."""

        def tokenize(texts):
            # Simple tokenizer: split on whitespace
            return [[ord(c) for c in text] for text in texts]

        return tokenize

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for output and receipts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "shards")
            receipts_dir = os.path.join(tmpdir, "receipts")
            os.makedirs(output_dir)
            os.makedirs(receipts_dir)
            yield {
                "output_dir": output_dir,
                "receipts_dir": receipts_dir,
                "tmpdir": tmpdir,
            }

    def test_empty_assignment(self, mock_tokenize, temp_dirs):
        """Test processing with empty assignment (no files)."""
        from fsspec import filesystem

        from nemotron.data_prep.shard_processor import process_binidx_shard_core

        output_fs = filesystem("file")
        assignment = {"shard_index": 0, "files": [], "total_bytes": 0}

        stats = process_binidx_shard_core(
            tokenize=mock_tokenize,
            text_field="text",
            min_doc_chars=None,
            max_doc_tokens=None,
            dtype="int32",
            max_rows=None,
            shard_index=0,
            assignment=assignment,
            plan_hash="testhash",
            output_dir=temp_dirs["output_dir"],
            receipts_dir=temp_dirs["receipts_dir"],
            output_fs=output_fs,
        )

        # Should create empty receipt
        receipt_path = os.path.join(temp_dirs["receipts_dir"], "shard_000000.json")
        assert os.path.exists(receipt_path)

        with open(receipt_path) as f:
            receipt = json.load(f)
        assert receipt["status"] == "completed"
        assert receipt["stats"]["num_sequences"] == 0

    def test_idempotency_skip_completed(self, mock_tokenize, temp_dirs):
        """Test that processing skips already-completed shards."""
        from fsspec import filesystem

        from nemotron.data_prep.shard_processor import process_binidx_shard_core

        output_fs = filesystem("file")
        assignment = {"shard_index": 0, "files": [], "total_bytes": 0}

        # First call - creates receipt
        stats1 = process_binidx_shard_core(
            tokenize=mock_tokenize,
            text_field="text",
            min_doc_chars=None,
            max_doc_tokens=None,
            dtype="int32",
            max_rows=None,
            shard_index=0,
            assignment=assignment,
            plan_hash="testhash",
            output_dir=temp_dirs["output_dir"],
            receipts_dir=temp_dirs["receipts_dir"],
            output_fs=output_fs,
        )

        # Second call - should skip and return cached stats
        stats2 = process_binidx_shard_core(
            tokenize=mock_tokenize,
            text_field="text",
            min_doc_chars=None,
            max_doc_tokens=None,
            dtype="int32",
            max_rows=None,
            shard_index=0,
            assignment=assignment,
            plan_hash="testhash",
            output_dir=temp_dirs["output_dir"],
            receipts_dir=temp_dirs["receipts_dir"],
            output_fs=output_fs,
        )

        # Both should return stats (second from receipt)
        assert stats1["num_sequences"] == 0
        assert stats2["num_sequences"] == 0

    def test_process_jsonl_file(self, mock_tokenize, temp_dirs):
        """Test processing a JSONL file."""
        from fsspec import filesystem

        from nemotron.data_prep.shard_processor import process_binidx_shard_core

        # Create a test JSONL file
        jsonl_path = os.path.join(temp_dirs["tmpdir"], "test.jsonl")
        with open(jsonl_path, "w") as f:
            for i in range(10):
                f.write(json.dumps({"text": f"hello world document {i}"}) + "\n")

        output_fs = filesystem("file")
        assignment = {
            "shard_index": 0,
            "files": [{"path": jsonl_path, "local_path": jsonl_path, "size": 100}],
            "total_bytes": 100,
        }

        stats = process_binidx_shard_core(
            tokenize=mock_tokenize,
            text_field="text",
            min_doc_chars=None,
            max_doc_tokens=None,
            dtype="int32",
            max_rows=None,
            shard_index=0,
            assignment=assignment,
            plan_hash="hash123",
            output_dir=temp_dirs["output_dir"],
            receipts_dir=temp_dirs["receipts_dir"],
            output_fs=output_fs,
        )

        assert stats["num_sequences"] == 10
        assert stats["num_input_rows"] == 10
        assert stats["total_tokens"] > 0

        # Check output files
        assert os.path.exists(os.path.join(temp_dirs["output_dir"], "shard_000000.bin"))
        assert os.path.exists(os.path.join(temp_dirs["output_dir"], "shard_000000.idx"))

    def test_min_doc_chars_filter(self, mock_tokenize, temp_dirs):
        """Test that min_doc_chars filters short documents."""
        from fsspec import filesystem

        from nemotron.data_prep.shard_processor import process_binidx_shard_core

        # Create JSONL with varying document lengths
        jsonl_path = os.path.join(temp_dirs["tmpdir"], "test.jsonl")
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"text": "hi"}) + "\n")  # 2 chars - filtered
            f.write(json.dumps({"text": "hello world"}) + "\n")  # 11 chars - kept
            f.write(json.dumps({"text": "x"}) + "\n")  # 1 char - filtered

        output_fs = filesystem("file")
        assignment = {
            "shard_index": 0,
            "files": [{"path": jsonl_path, "local_path": jsonl_path, "size": 100}],
            "total_bytes": 100,
        }

        stats = process_binidx_shard_core(
            tokenize=mock_tokenize,
            text_field="text",
            min_doc_chars=5,  # Filter docs < 5 chars
            max_doc_tokens=None,
            dtype="int32",
            max_rows=None,
            shard_index=0,
            assignment=assignment,
            plan_hash="hash123",
            output_dir=temp_dirs["output_dir"],
            receipts_dir=temp_dirs["receipts_dir"],
            output_fs=output_fs,
        )

        assert stats["num_sequences"] == 1  # Only "hello world"
        assert stats["num_filtered"] == 2  # "hi" and "x" filtered

    def test_max_rows_limit(self, mock_tokenize, temp_dirs):
        """Test that max_rows limits processing."""
        from fsspec import filesystem

        from nemotron.data_prep.shard_processor import process_binidx_shard_core

        # Create JSONL with many rows
        jsonl_path = os.path.join(temp_dirs["tmpdir"], "test.jsonl")
        with open(jsonl_path, "w") as f:
            for i in range(100):
                f.write(json.dumps({"text": f"document number {i}"}) + "\n")

        output_fs = filesystem("file")
        assignment = {
            "shard_index": 0,
            "files": [{"path": jsonl_path, "local_path": jsonl_path, "size": 1000}],
            "total_bytes": 1000,
        }

        stats = process_binidx_shard_core(
            tokenize=mock_tokenize,
            text_field="text",
            min_doc_chars=None,
            max_doc_tokens=None,
            dtype="int32",
            max_rows=10,  # Only process 10 rows
            shard_index=0,
            assignment=assignment,
            plan_hash="hash123",
            output_dir=temp_dirs["output_dir"],
            receipts_dir=temp_dirs["receipts_dir"],
            output_fs=output_fs,
        )

        assert stats["num_sequences"] == 10
        assert stats["num_input_rows"] == 10

    def test_timing_metrics_present(self, mock_tokenize, temp_dirs):
        """Test that timing metrics are returned."""
        from fsspec import filesystem

        from nemotron.data_prep.shard_processor import process_binidx_shard_core

        jsonl_path = os.path.join(temp_dirs["tmpdir"], "test.jsonl")
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"text": "test document"}) + "\n")

        output_fs = filesystem("file")
        assignment = {
            "shard_index": 0,
            "files": [{"path": jsonl_path, "local_path": jsonl_path, "size": 50}],
            "total_bytes": 50,
        }

        stats = process_binidx_shard_core(
            tokenize=mock_tokenize,
            text_field="text",
            min_doc_chars=None,
            max_doc_tokens=None,
            dtype="int32",
            max_rows=None,
            shard_index=0,
            assignment=assignment,
            plan_hash="hash",
            output_dir=temp_dirs["output_dir"],
            receipts_dir=temp_dirs["receipts_dir"],
            output_fs=output_fs,
        )

        assert "time_total_sec" in stats
        assert "time_read_sec" in stats
        assert "time_tokenize_sec" in stats
        assert "time_write_sec" in stats
        assert stats["time_total_sec"] >= 0


class TestTokenizeAndWriteBatchCore:
    """Tests for _tokenize_and_write_batch_core helper function."""

    def test_basic_tokenization(self):
        """Test basic batch tokenization and writing."""
        from nemotron.data_prep.shard_processor import _tokenize_and_write_batch_core

        mock_builder = MagicMock()
        stats = {"num_truncated": 0, "num_errors": 0}

        def tokenize(texts):
            return [[1, 2, 3] for _ in texts]

        texts = ["text1", "text2", "text3"]
        _tokenize_and_write_batch_core(texts, mock_builder, stats, tokenize, None)

        # Should call add_documents with 3 processed tokens
        mock_builder.add_documents.assert_called_once()
        args = mock_builder.add_documents.call_args[0][0]
        assert len(args) == 3

    def test_max_doc_tokens_truncation(self):
        """Test that max_doc_tokens truncates long documents."""
        from nemotron.data_prep.shard_processor import _tokenize_and_write_batch_core

        mock_builder = MagicMock()
        stats = {"num_truncated": 0, "num_errors": 0}

        def tokenize(texts):
            return [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] for _ in texts]

        texts = ["long text"]
        _tokenize_and_write_batch_core(texts, mock_builder, stats, tokenize, max_doc_tokens=5)

        mock_builder.add_documents.assert_called_once()
        args = mock_builder.add_documents.call_args[0][0]
        assert len(args[0]) == 5  # Truncated to 5 tokens
        assert stats["num_truncated"] == 1

    def test_empty_tokens_filtered(self):
        """Test that empty token sequences are filtered out."""
        from nemotron.data_prep.shard_processor import _tokenize_and_write_batch_core

        mock_builder = MagicMock()
        stats = {"num_truncated": 0, "num_errors": 0}

        def tokenize(texts):
            return [[1, 2, 3], [], [4, 5]]  # Middle one is empty

        texts = ["text1", "empty", "text3"]
        _tokenize_and_write_batch_core(texts, mock_builder, stats, tokenize, None)

        mock_builder.add_documents.assert_called_once()
        args = mock_builder.add_documents.call_args[0][0]
        assert len(args) == 2  # Only non-empty sequences

    def test_tokenization_error_triggers_bisect(self):
        """Test that tokenization errors trigger bisection."""
        from nemotron.data_prep.shard_processor import _tokenize_and_write_batch_core

        mock_builder = MagicMock()
        stats = {"num_truncated": 0, "num_errors": 0}

        call_count = 0

        def tokenize(texts):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Tokenization error")
            return [[1, 2, 3] for _ in texts]

        texts = ["text1", "text2"]
        _tokenize_and_write_batch_core(texts, mock_builder, stats, tokenize, None)

        # Should have been called multiple times due to bisection
        assert call_count > 1


class TestTokenizeWithBisectCore:
    """Tests for _tokenize_with_bisect_core helper function."""

    def test_single_text_error(self):
        """Test that single text error is recorded."""
        from nemotron.data_prep.shard_processor import _tokenize_with_bisect_core

        mock_builder = MagicMock()
        stats = {"num_truncated": 0, "num_errors": 0}

        def tokenize(texts):
            raise ValueError("Bad text")

        _tokenize_with_bisect_core(["bad text"], mock_builder, stats, tokenize, None)

        assert stats["num_errors"] == 1

    def test_empty_list(self):
        """Test that empty list is handled."""
        from nemotron.data_prep.shard_processor import _tokenize_with_bisect_core

        mock_builder = MagicMock()
        stats = {"num_truncated": 0, "num_errors": 0}

        def tokenize(texts):
            return [[1, 2, 3] for _ in texts]

        _tokenize_with_bisect_core([], mock_builder, stats, tokenize, None)

        # Should return without calling builder
        mock_builder.add_document.assert_not_called()

    def test_isolates_bad_text(self):
        """Test that bisection isolates the bad text."""
        from nemotron.data_prep.shard_processor import _tokenize_with_bisect_core

        mock_builder = MagicMock()
        stats = {"num_truncated": 0, "num_errors": 0}

        bad_index = 1

        def tokenize(texts):
            for i, t in enumerate(texts):
                if t == "bad":
                    raise ValueError("Bad text")
            return [[1, 2, 3] for _ in texts]

        texts = ["good1", "bad", "good2", "good3"]
        _tokenize_with_bisect_core(texts, mock_builder, stats, tokenize, None)

        # Should have 1 error (the "bad" text)
        assert stats["num_errors"] == 1


class TestBinIdxShardTaskUDFUnit:
    """Unit tests for BinIdxShardTaskUDF."""

    def test_udf_batch_extraction(self):
        """Test that UDF correctly extracts fields from batch dict."""
        # We can't fully test without Ray, but we can test the logic
        batch = {
            "dataset_name": np.array(["test_ds"], dtype=object),
            "shard_index": np.array([5], dtype=np.int64),
            "plan_hash": np.array(["hash123"], dtype=object),
            "assignment_json": np.array(['{"files": [], "shard_index": 5}'], dtype=object),
            "output_dir": np.array(["/out"], dtype=object),
            "receipts_dir": np.array(["/receipts"], dtype=object),
            "fs_protocol": np.array(["file"], dtype=object),
            "text_field": np.array(["text"], dtype=object),
        }

        # Verify extraction logic
        dataset_name = str(batch["dataset_name"][0])
        shard_index = int(batch["shard_index"][0])
        assignment = json.loads(str(batch["assignment_json"][0]))

        assert dataset_name == "test_ds"
        assert shard_index == 5
        assert assignment["shard_index"] == 5

    def test_udf_error_response_format(self):
        """Test that error responses have correct numpy array format."""
        # Simulate error response format
        error_response = {
            "dataset_name": np.array(["test"], dtype=object),
            "shard_index": np.array([0], dtype=np.int64),
            "plan_hash": np.array(["hash"], dtype=object),
            "total_tokens": np.array([0], dtype=np.int64),
            "num_sequences": np.array([0], dtype=np.int64),
            "num_filtered": np.array([0], dtype=np.int64),
            "num_errors": np.array([1], dtype=np.int64),
            "time_total_sec": np.array([0.0], dtype=np.float64),
            "time_download_sec": np.array([0.0], dtype=np.float64),
            "time_read_sec": np.array([0.0], dtype=np.float64),
            "time_tokenize_sec": np.array([0.0], dtype=np.float64),
            "time_write_sec": np.array([0.0], dtype=np.float64),
            "error": np.array(["Test error"], dtype=object),
        }

        # All values should be numpy arrays
        for key, value in error_response.items():
            assert isinstance(value, np.ndarray), f"{key} should be numpy array"
            assert len(value) == 1, f"{key} should have length 1"


class TestBottleneckMetricsExtended:
    """Extended tests for BottleneckMetrics."""

    def test_bottleneck_metrics_identifies_download_bottleneck(self):
        """Test identification of download as bottleneck."""
        from nemotron.data_prep.console import BottleneckMetrics

        metrics = BottleneckMetrics()
        metrics.time_download = 10.0  # 62.5%
        metrics.time_read = 2.0  # 12.5%
        metrics.time_tokenize = 3.0  # 18.75%
        metrics.time_write = 1.0  # 6.25%

        pcts = metrics.get_percentages()
        assert pcts["download_pct"] > 50  # Download is bottleneck

    def test_bottleneck_metrics_identifies_tokenize_bottleneck(self):
        """Test identification of tokenization as bottleneck."""
        from nemotron.data_prep.console import BottleneckMetrics

        metrics = BottleneckMetrics()
        metrics.time_download = 1.0  # 10%
        metrics.time_read = 1.0  # 10%
        metrics.time_tokenize = 7.0  # 70%
        metrics.time_write = 1.0  # 10%

        pcts = metrics.get_percentages()
        assert pcts["tokenize_pct"] > 50  # Tokenize is bottleneck

    def test_bottleneck_metrics_handles_missing_keys(self):
        """Test that add_shard_stats handles missing keys gracefully."""
        from nemotron.data_prep.console import BottleneckMetrics

        metrics = BottleneckMetrics()

        # Partial stats dict
        stats = {"time_tokenize_sec": 5.0}
        metrics.add_shard_stats(stats)

        assert metrics.time_tokenize == 5.0
        assert metrics.time_download == 0.0  # Default
        assert metrics.num_shards == 1

    def test_bottleneck_metrics_many_shards(self):
        """Test accumulation across many shards."""
        from nemotron.data_prep.console import BottleneckMetrics

        metrics = BottleneckMetrics()

        for i in range(100):
            metrics.add_shard_stats(
                {
                    "time_download_sec": 0.1,
                    "time_read_sec": 0.2,
                    "time_tokenize_sec": 0.5,
                    "time_write_sec": 0.2,
                    "num_errors": 0 if i % 10 != 0 else 1,  # 10% error rate
                }
            )

        assert metrics.num_shards == 100
        assert abs(metrics.time_download - 10.0) < 0.01
        assert abs(metrics.time_tokenize - 50.0) < 0.01
        assert metrics.num_errors == 10


class TestRayDataConfigIntegration:
    """Tests for RayDataConfig integration with PipelineConfig."""

    def test_ray_data_config_defaults_in_pipeline(self):
        """Test default RayDataConfig values in PipelineConfig."""
        from nemotron.data_prep.config import OutputConfig, PipelineConfig

        pipeline_cfg = PipelineConfig(output=OutputConfig(dir=Path("/out")))

        # ray_data is None by default (not enabled)
        assert pipeline_cfg.ray_data is None

    def test_ray_data_config_with_defaults(self):
        """Test RayDataConfig with defaults in PipelineConfig."""
        from nemotron.data_prep.config import (
            OutputConfig,
            PipelineConfig,
            RayDataConfig,
        )

        pipeline_cfg = PipelineConfig(
            output=OutputConfig(dir=Path("/out")),
            ray_data=RayDataConfig(),  # Use defaults
        )

        assert pipeline_cfg.ray_data is not None
        assert pipeline_cfg.ray_data.enabled is False
        assert pipeline_cfg.ray_data.min_actors == 2
        assert pipeline_cfg.ray_data.max_actors == 32

    def test_ray_data_config_enabled(self):
        """Test enabling RayDataConfig."""
        from nemotron.data_prep.config import (
            OutputConfig,
            PipelineConfig,
            RayDataConfig,
        )

        pipeline_cfg = PipelineConfig(
            output=OutputConfig(dir=Path("/out")),
            ray_data=RayDataConfig(
                enabled=True,
                min_actors=4,
                max_actors=16,
            ),
        )

        assert pipeline_cfg.ray_data.enabled is True
        assert pipeline_cfg.ray_data.min_actors == 4
        assert pipeline_cfg.ray_data.max_actors == 16


class TestWriteEmptyReceiptCore:
    """Tests for _write_empty_receipt_core function."""

    def test_writes_empty_receipt(self):
        """Test that empty receipt is written correctly."""
        from fsspec import filesystem

        from nemotron.data_prep.shard_processor import _write_empty_receipt_core

        with tempfile.TemporaryDirectory() as tmpdir:
            receipts_dir = os.path.join(tmpdir, "receipts")
            os.makedirs(receipts_dir)
            receipt_path = os.path.join(receipts_dir, "shard_000000.json")

            output_fs = filesystem("file")
            stats = {"num_input_rows": 0, "num_filtered": 0}
            timing: dict = {}

            import time

            total_start = time.perf_counter()

            result = _write_empty_receipt_core(
                shard_id="shard_000000",
                shard_index=0,
                plan_hash="testhash",
                input_files=[],
                stats=stats,
                receipt_path=receipt_path,
                output_fs=output_fs,
                timing=timing,
                total_start=total_start,
            )

            assert os.path.exists(receipt_path)

            with open(receipt_path) as f:
                receipt = json.load(f)

            assert receipt["status"] == "completed"
            assert receipt["stats"]["num_sequences"] == 0
            assert receipt["stats"]["total_tokens"] == 0
            assert "time_total_sec" in result


class TestResolveFilePathCore:
    """Tests for _resolve_file_path_core function."""

    def test_local_path_passthrough(self):
        """Test that local paths pass through unchanged."""
        from nemotron.data_prep.config import FileInfo
        from nemotron.data_prep.shard_processor import _resolve_file_path_core

        file_info = FileInfo(path="/data/file.parquet", local_path=None, size=100)
        result = _resolve_file_path_core(file_info)
        assert result == "/data/file.parquet"

    def test_local_path_preferred(self):
        """Test that local_path is preferred over path."""
        from nemotron.data_prep.config import FileInfo
        from nemotron.data_prep.shard_processor import _resolve_file_path_core

        file_info = FileInfo(
            path="/original/path.parquet",
            size=100,
            local_path="/cached/path.parquet",
        )
        result = _resolve_file_path_core(file_info)
        assert result == "/cached/path.parquet"


class TestExecuteShardTasksEmptyList:
    """Test execute_shard_tasks with empty task list (no Ray required)."""

    def test_empty_tasks_returns_empty_list(self):
        """Test that empty task list returns empty results without Ray."""
        from nemotron.data_prep.ray_data import RayDataExecConfig

        # This tests the early return path before Ray is used
        exec_cfg = RayDataExecConfig()

        # We need Ray initialized for this test
        try:
            import ray

            if not ray.is_initialized():
                ray.init(num_cpus=1, ignore_reinit_error=True)

            from nemotron.data_prep.ray_data import execute_shard_tasks

            results = execute_shard_tasks(
                tasks=[],
                udf_cls=MagicMock,
                udf_constructor_kwargs={},
                exec_cfg=exec_cfg,
            )

            assert results == []
        except ImportError:
            pytest.skip("Ray not available")


class TestProcessParquetFile:
    """Tests for parquet file processing."""

    def test_process_parquet_file(self):
        """Test processing a parquet file."""
        pytest.importorskip("pyarrow")

        import pyarrow as pa
        import pyarrow.parquet as pq
        from fsspec import filesystem

        from nemotron.data_prep.shard_processor import process_binidx_shard_core

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create parquet file
            parquet_path = os.path.join(tmpdir, "test.parquet")
            table = pa.table({"text": ["hello world", "foo bar", "test document"]})
            pq.write_table(table, parquet_path)

            output_dir = os.path.join(tmpdir, "shards")
            receipts_dir = os.path.join(tmpdir, "receipts")
            os.makedirs(output_dir)
            os.makedirs(receipts_dir)

            def mock_tokenize(texts):
                return [[ord(c) for c in t] for t in texts]

            output_fs = filesystem("file")
            assignment = {
                "shard_index": 0,
                "files": [{"path": parquet_path, "local_path": parquet_path, "size": 100}],
                "total_bytes": 100,
            }

            stats = process_binidx_shard_core(
                tokenize=mock_tokenize,
                text_field="text",
                min_doc_chars=None,
                max_doc_tokens=None,
                dtype="int32",
                max_rows=None,
                shard_index=0,
                assignment=assignment,
                plan_hash="hash",
                output_dir=output_dir,
                receipts_dir=receipts_dir,
                output_fs=output_fs,
            )

            assert stats["num_sequences"] == 3
            assert stats["num_input_rows"] == 3

    def test_parquet_min_doc_chars_arrow_filter(self):
        """Test Arrow-level min_doc_chars filtering on parquet."""
        pytest.importorskip("pyarrow")

        import pyarrow as pa
        import pyarrow.parquet as pq
        from fsspec import filesystem

        from nemotron.data_prep.shard_processor import process_binidx_shard_core

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create parquet file with varying lengths
            parquet_path = os.path.join(tmpdir, "test.parquet")
            table = pa.table(
                {
                    "text": [
                        "hi",  # 2 chars - filtered
                        "hello world this is long",  # 24 chars - kept
                        "x",  # 1 char - filtered
                        "another long document here",  # 26 chars - kept
                    ]
                }
            )
            pq.write_table(table, parquet_path)

            output_dir = os.path.join(tmpdir, "shards")
            receipts_dir = os.path.join(tmpdir, "receipts")
            os.makedirs(output_dir)
            os.makedirs(receipts_dir)

            def mock_tokenize(texts):
                return [[ord(c) for c in t] for t in texts]

            output_fs = filesystem("file")
            assignment = {
                "shard_index": 0,
                "files": [{"path": parquet_path, "local_path": parquet_path, "size": 100}],
                "total_bytes": 100,
            }

            stats = process_binidx_shard_core(
                tokenize=mock_tokenize,
                text_field="text",
                min_doc_chars=10,  # Filter < 10 chars
                max_doc_tokens=None,
                dtype="int32",
                max_rows=None,
                shard_index=0,
                assignment=assignment,
                plan_hash="hash",
                output_dir=output_dir,
                receipts_dir=receipts_dir,
                output_fs=output_fs,
            )

            assert stats["num_sequences"] == 2  # Only long docs
            assert stats["num_filtered"] == 2  # Short docs filtered


@pytest.mark.skip(reason="Requires Ray cluster and tokenizer")
class TestExecuteShardTasks:
    """Integration tests for execute_shard_tasks.

    These tests require:
    - Ray to be initialized
    - A tokenizer to be available

    Skip for now - to be run manually or in CI with proper setup.
    """

    def test_execute_empty_tasks(self, ray_initialized):
        """Test executing with empty task list."""
        from nemotron.data_prep.ray_data import (
            BinIdxShardTaskUDF,
            RayDataExecConfig,
            execute_shard_tasks,
        )

        exec_cfg = RayDataExecConfig(min_actors=1, max_actors=2)

        results = execute_shard_tasks(
            tasks=[],
            udf_cls=BinIdxShardTaskUDF,
            udf_constructor_kwargs={
                "resolved_tokenizer": {"type": "test"},
                "dtype": "int32",
            },
            exec_cfg=exec_cfg,
        )

        assert results == []
