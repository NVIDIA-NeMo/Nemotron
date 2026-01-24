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

"""Tests for stats_callback module.

These tests verify:
1. Callback factory behavior (returns None when disabled, callable when enabled)
2. W&B logging (conditional, safe)
3. JSONL output (file creation, record structure)
4. Error handling (callback never crashes)
5. Metric extraction from PipelineStats-like objects
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemotron.data_prep.config import ObservabilityConfig
from nemotron.data_prep.observability.stats_callback import (
    _extract_jsonl_record,
    _extract_wandb_metrics,
    _sanitize_metric_name,
    make_pipeline_stats_callback,
)


# =============================================================================
# Fixtures: Mock PipelineStats objects
# =============================================================================


class MockSlotStats:
    """Mock slot stats."""

    def __init__(self, num_used: int = 5, num_empty: int = 3):
        self.num_used = num_used
        self.num_empty = num_empty


class MockActorStats:
    """Mock actor stats."""

    def __init__(
        self,
        target: int = 10,
        pending: int = 0,
        ready: int = 8,
        running: int = 8,
        idle: int = 2,
    ):
        self.target = target
        self.pending = pending
        self.ready = ready
        self.running = running
        self.idle = idle


class MockTaskStats:
    """Mock task stats."""

    def __init__(
        self,
        total_completed: int = 100,
        total_returned_none: int = 5,
        total_dynamically_spawned: int = 0,
        input_queue_size: int = 50,
        output_queue_size: int = 25,
    ):
        self.total_completed = total_completed
        self.total_returned_none = total_returned_none
        self.total_dynamically_spawned = total_dynamically_spawned
        self.input_queue_size = input_queue_size
        self.output_queue_size = output_queue_size


class MockActorPoolStats:
    """Mock actor pool stats for a stage."""

    def __init__(self, name: str = "UnnamedStage"):
        self.name = name
        self.actor_stats = MockActorStats()
        self.task_stats = MockTaskStats()
        self.slot_stats = MockSlotStats()
        self.processing_speed_tasks_per_second = 2.5


class MockResources:
    """Mock cluster resources."""

    def __init__(self, cpus: float = 64, gpus: float = 8, memory_gb: float = 256, object_store_gb: float = 128):
        self.cpus = cpus
        self.gpus = gpus
        self.memory_gb = memory_gb
        self.object_store_gb = object_store_gb


class MockClusterInfo:
    """Mock cluster info."""

    def __init__(self):
        self.total = MockResources(cpus=64, gpus=8, memory_gb=256, object_store_gb=128)
        self.available = MockResources(cpus=32, gpus=4, memory_gb=128, object_store_gb=64)


class MockResourceUsage:
    """Mock stage resource usage."""

    def __init__(self, cpu_utilization: float = 75.5, memory_usage: int = 8 * 1024**3, actor_count: int = 8):
        self.cpu_utilization = cpu_utilization
        self.memory_usage = memory_usage
        self.actor_count = actor_count


class MockPipelineStats:
    """Mock PipelineStats object.

    Note: actor_pools is a list (not a dict) to match the real PipelineStats
    from cosmos-xenna. Each ActorPoolStats has a .name attribute.
    """

    def __init__(self):
        self.inputs_processed_per_second = 10.5
        self.outputs_per_second = 8.2
        self.num_initial_input_tasks = 1000
        self.num_input_tasks_remaining = 500
        self.num_outputs = 500
        self.main_loop_rate_hz = 100.0
        self.cluster_info = MockClusterInfo()
        # actor_pools is a list of ActorPoolStats, each with a .name attribute
        self.actor_pools = [
            MockActorPoolStats(name="Stage 00 - PlanStage"),
            MockActorPoolStats(name="Stage 01 - DownloadStage"),
            MockActorPoolStats(name="Stage 02 - BinIdxTokenizationStage"),
        ]
        self.resource_usage_per_stage = {
            "Stage 02 - BinIdxTokenizationStage": MockResourceUsage(),
        }


@pytest.fixture
def mock_stats() -> MockPipelineStats:
    """Fixture providing a mock PipelineStats object."""
    return MockPipelineStats()


@pytest.fixture
def observability_disabled() -> ObservabilityConfig:
    """Fixture with all observability disabled."""
    return ObservabilityConfig(
        wandb_log_pipeline_stats=False,
        pipeline_stats_jsonl_path=None,
    )


@pytest.fixture
def observability_wandb_only() -> ObservabilityConfig:
    """Fixture with only W&B logging enabled."""
    return ObservabilityConfig(
        wandb_log_pipeline_stats=True,
        pipeline_stats_jsonl_path=None,
    )


# =============================================================================
# Tests: Sanitize metric name
# =============================================================================


class TestSanitizeMetricName:
    """Tests for stage name sanitization."""

    def test_removes_stage_prefix(self):
        """Test that 'Stage XX - ' prefix is removed."""
        assert _sanitize_metric_name("Stage 02 - BinIdxTokenizationStage") == "bin_idx_tokenization"
        assert _sanitize_metric_name("Stage 00 - PlanStage") == "plan"

    def test_removes_stage_suffix(self):
        """Test that 'Stage' suffix is removed."""
        assert _sanitize_metric_name("DownloadStage") == "download"
        assert _sanitize_metric_name("PackedSftParquetStage") == "packed_sft_parquet"

    def test_handles_simple_names(self):
        """Test simple names without Stage prefix/suffix."""
        assert _sanitize_metric_name("Tokenizer") == "tokenizer"
        assert _sanitize_metric_name("download") == "download"

    def test_returns_unknown_for_empty(self):
        """Test that empty/invalid names return 'unknown'."""
        assert _sanitize_metric_name("") == "unknown"
        assert _sanitize_metric_name("Stage") == "unknown"


# =============================================================================
# Tests: Callback factory
# =============================================================================


class TestMakeXennaPipelineStatsCallback:
    """Tests for the callback factory function."""

    def test_returns_none_when_disabled(self, observability_disabled: ObservabilityConfig):
        """Test that callback returns None when all logging is disabled."""
        callback = make_pipeline_stats_callback(
            observability=observability_disabled,
            pipeline_kind="pretrain",
        )
        assert callback is None

    def test_returns_callable_when_wandb_enabled(self, observability_wandb_only: ObservabilityConfig):
        """Test that callback returns a callable when W&B logging is enabled."""
        callback = make_pipeline_stats_callback(
            observability=observability_wandb_only,
            pipeline_kind="pretrain",
        )
        assert callback is not None
        assert callable(callback)

    def test_returns_callable_when_jsonl_enabled(self, tmp_path: Path):
        """Test that callback returns a callable when JSONL logging is enabled."""
        observability = ObservabilityConfig(
            wandb_log_pipeline_stats=False,
            pipeline_stats_jsonl_path=str(tmp_path / "stats.jsonl"),
        )
        callback = make_pipeline_stats_callback(
            observability=observability,
            pipeline_kind="pretrain",
        )
        assert callback is not None
        assert callable(callback)

    def test_callback_does_not_raise(self, mock_stats: MockPipelineStats, tmp_path: Path):
        """Test that the callback never raises exceptions."""
        observability = ObservabilityConfig(
            wandb_log_pipeline_stats=True,
            pipeline_stats_jsonl_path=str(tmp_path / "stats.jsonl"),
        )
        callback = make_pipeline_stats_callback(
            observability=observability,
            pipeline_kind="pretrain",
        )

        # Should not raise even if wandb is not initialized
        callback(mock_stats)

        # Should not raise with malformed stats
        callback(None)
        callback({})
        callback("invalid")


# =============================================================================
# Tests: W&B logging
# =============================================================================


class TestWandbLogging:
    """Tests for W&B metric extraction and logging."""

    def test_extract_wandb_metrics(self, mock_stats: MockPipelineStats):
        """Test that wandb metrics are correctly extracted."""
        import time

        start_time = time.time() - 100  # 100 seconds ago
        metrics = _extract_wandb_metrics(mock_stats, start_time)

        # Pipeline-level metrics
        assert "pipeline/pipeline_duration_s" in metrics
        assert metrics["pipeline/pipeline_duration_s"] >= 100
        assert metrics["pipeline/inputs_processed_per_s"] == 10.5
        assert metrics["pipeline/outputs_per_s"] == 8.2
        assert metrics["pipeline/num_input_remaining"] == 500
        assert metrics["pipeline/num_outputs"] == 500

        # Cluster metrics
        assert metrics["pipeline/cluster/total_cpus"] == 64
        assert metrics["pipeline/cluster/total_gpus"] == 8
        assert metrics["pipeline/cluster/avail_cpus"] == 32
        assert metrics["pipeline/cluster/avail_gpus"] == 4

        # Stage metrics (tokenization stage)
        assert metrics["pipeline/stage/bin_idx_tokenization/actors/running"] == 8
        assert metrics["pipeline/stage/bin_idx_tokenization/tasks/completed"] == 100
        assert metrics["pipeline/stage/bin_idx_tokenization/queue/input_size"] == 50
        assert metrics["pipeline/stage/bin_idx_tokenization/slots/used"] == 5
        assert metrics["pipeline/stage/bin_idx_tokenization/speed/tasks_per_s"] == 2.5

        # Resource usage
        assert metrics["pipeline/stage/bin_idx_tokenization/cpu_util_pct"] == 75.5
        assert metrics["pipeline/stage/bin_idx_tokenization/mem_gb"] == 8.0

    def test_extract_wandb_metrics_excludes_stage_metrics_when_disabled(
        self, mock_stats: MockPipelineStats
    ):
        """Test that stage metrics are excluded when include_stage_metrics=False."""
        import time

        start_time = time.time() - 100
        metrics = _extract_wandb_metrics(mock_stats, start_time, include_stage_metrics=False)

        # Pipeline-level metrics should still be present
        assert "pipeline/pipeline_duration_s" in metrics
        assert "pipeline/inputs_processed_per_s" in metrics
        assert "pipeline/cluster/total_cpus" in metrics

        # Stage metrics should NOT be present
        assert "pipeline/stage/bin_idx_tokenization/actors/running" not in metrics
        assert "pipeline/stage/bin_idx_tokenization/tasks/completed" not in metrics
        assert "pipeline/stage/bin_idx_tokenization/cpu_util_pct" not in metrics
        assert "pipeline/stage/plan/actors/running" not in metrics

    def test_consolidated_charts_only_skips_stage_metrics(self, mock_stats: MockPipelineStats):
        """Test that wandb_consolidated_charts_only=True skips per-stage metrics."""
        observability = ObservabilityConfig(
            wandb_log_pipeline_stats=True,
            pipeline_stats_jsonl_path=None,
            wandb_consolidated_charts_only=True,  # Default, but explicit for clarity
        )

        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()
        mock_wandb.log = MagicMock()

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            callback = make_pipeline_stats_callback(
                observability=observability,
                pipeline_kind="pretrain",
            )
            callback(mock_stats)

            # Verify wandb.log was called
            mock_wandb.log.assert_called_once()
            logged_metrics = mock_wandb.log.call_args[0][0]

            # Pipeline-level metrics should be present
            assert "pipeline/pipeline_duration_s" in logged_metrics

            # Stage metrics should NOT be present (consolidated charts only)
            stage_keys = [k for k in logged_metrics if "/stage/" in k]
            assert len(stage_keys) == 0, f"Found stage keys when consolidated_charts_only=True: {stage_keys}"

    def test_consolidated_charts_disabled_includes_stage_metrics(self, mock_stats: MockPipelineStats):
        """Test that wandb_consolidated_charts_only=False includes per-stage metrics."""
        observability = ObservabilityConfig(
            wandb_log_pipeline_stats=True,
            pipeline_stats_jsonl_path=None,
            wandb_consolidated_charts_only=False,  # Explicitly disable
        )

        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()
        mock_wandb.log = MagicMock()

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            callback = make_pipeline_stats_callback(
                observability=observability,
                pipeline_kind="pretrain",
            )
            callback(mock_stats)

            mock_wandb.log.assert_called_once()
            logged_metrics = mock_wandb.log.call_args[0][0]

            # Stage metrics SHOULD be present
            stage_keys = [k for k in logged_metrics if "/stage/" in k]
            assert len(stage_keys) > 0, "Expected stage keys when consolidated_charts_only=False"

    def test_wandb_log_called_when_enabled(self, mock_stats: MockPipelineStats):
        """Test that wandb.log is called when W&B logging is enabled and run exists."""
        observability = ObservabilityConfig(
            wandb_log_pipeline_stats=True,
            pipeline_stats_jsonl_path=None,
        )

        # Mock wandb module
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()  # Simulate active run
        mock_wandb.log = MagicMock()

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            callback = make_pipeline_stats_callback(
                observability=observability,
                pipeline_kind="pretrain",
            )
            callback(mock_stats)

            # Verify wandb.log was called
            mock_wandb.log.assert_called_once()
            call_args = mock_wandb.log.call_args
            assert call_args[1]["commit"] is False

    def test_wandb_not_called_when_no_run(self, mock_stats: MockPipelineStats):
        """Test that wandb.log is not called when no active run."""
        observability = ObservabilityConfig(
            wandb_log_pipeline_stats=True,
            pipeline_stats_jsonl_path=None,
        )

        # Mock wandb module with no active run
        mock_wandb = MagicMock()
        mock_wandb.run = None
        mock_wandb.log = MagicMock()

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            callback = make_pipeline_stats_callback(
                observability=observability,
                pipeline_kind="pretrain",
            )
            callback(mock_stats)

            # Verify wandb.log was NOT called
            mock_wandb.log.assert_not_called()


# =============================================================================
# Tests: JSONL output
# =============================================================================


class TestJsonlOutput:
    """Tests for JSONL file output."""

    def test_jsonl_file_created(self, mock_stats: MockPipelineStats, tmp_path: Path):
        """Test that JSONL file is created."""
        jsonl_path = tmp_path / "subdir" / "stats.jsonl"
        observability = ObservabilityConfig(
            wandb_log_pipeline_stats=False,
            pipeline_stats_jsonl_path=str(jsonl_path),
        )

        callback = make_pipeline_stats_callback(
            observability=observability,
            pipeline_kind="pretrain",
            run_hash="test123",
        )

        callback(mock_stats)

        assert jsonl_path.exists()

    def test_jsonl_appends_records(self, mock_stats: MockPipelineStats, tmp_path: Path):
        """Test that multiple callback invocations append to JSONL."""
        jsonl_path = tmp_path / "stats.jsonl"
        observability = ObservabilityConfig(
            wandb_log_pipeline_stats=False,
            pipeline_stats_jsonl_path=str(jsonl_path),
        )

        callback = make_pipeline_stats_callback(
            observability=observability,
            pipeline_kind="pretrain",
        )

        # Call multiple times
        callback(mock_stats)
        callback(mock_stats)
        callback(mock_stats)

        # Read and verify
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 3

        # Each line should be valid JSON
        for line in lines:
            record = json.loads(line)
            assert "schema_version" in record
            assert "timestamp" in record
            assert "pipeline_kind" in record

    def test_jsonl_record_structure(self, mock_stats: MockPipelineStats):
        """Test the structure of JSONL records."""
        import time

        start_time = time.time() - 100
        record = _extract_jsonl_record(
            mock_stats,
            start_time,
            pipeline_kind="pretrain",
            run_hash="abc123",
            run_dir="/output/runs/abc123",
            dataset_names=["dataset1", "dataset2"],
        )

        # Top-level structure
        assert record["schema_version"] == 1
        assert record["pipeline_kind"] == "pretrain"
        assert record["run_hash"] == "abc123"
        assert record["run_dir"] == "/output/runs/abc123"
        assert record["dataset_names"] == ["dataset1", "dataset2"]

        # Pipeline stats
        assert "pipeline" in record
        assert record["pipeline"]["pipeline_duration_s"] >= 100
        assert record["pipeline"]["inputs_processed_per_second"] == 10.5

        # Cluster info
        assert "cluster" in record
        assert record["cluster"]["total"]["cpus"] == 64
        assert record["cluster"]["available"]["gpus"] == 4

        # Stages
        assert "stages" in record
        assert len(record["stages"]) == 3

        # Find tokenization stage
        tokenization_stage = next(s for s in record["stages"] if "BinIdxTokenization" in s["name"])
        assert tokenization_stage["actors"]["running"] == 8
        assert tokenization_stage["tasks"]["total_completed"] == 100

        # Resource usage
        assert "resource_usage_per_stage" in record
        assert "Stage 02 - BinIdxTokenizationStage" in record["resource_usage_per_stage"]


# =============================================================================
# Tests: Error handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and safety."""

    def test_handles_missing_attributes(self, tmp_path: Path):
        """Test that callback handles stats objects with missing attributes."""
        observability = ObservabilityConfig(
            wandb_log_pipeline_stats=False,
            pipeline_stats_jsonl_path=str(tmp_path / "stats.jsonl"),
        )

        callback = make_pipeline_stats_callback(
            observability=observability,
            pipeline_kind="pretrain",
        )

        # Empty object
        class EmptyStats:
            pass

        # Should not raise
        callback(EmptyStats())

        # Verify file was still created
        assert (tmp_path / "stats.jsonl").exists()

    def test_handles_none_values(self, tmp_path: Path):
        """Test that callback handles None values in stats."""
        observability = ObservabilityConfig(
            wandb_log_pipeline_stats=False,
            pipeline_stats_jsonl_path=str(tmp_path / "stats.jsonl"),
        )

        callback = make_pipeline_stats_callback(
            observability=observability,
            pipeline_kind="pretrain",
        )

        # Stats with None values
        class NoneStats:
            inputs_processed_per_second = None
            outputs_per_second = None
            cluster_info = None
            actor_pools = None
            resource_usage_per_stage = None

        # Should not raise
        callback(NoneStats())

    def test_handles_wandb_import_error(self, mock_stats: MockPipelineStats):
        """Test that callback handles wandb not being installed."""
        observability = ObservabilityConfig(
            wandb_log_pipeline_stats=True,
            pipeline_stats_jsonl_path=None,
        )

        # Simulate wandb not installed
        with patch.dict(sys.modules, {"wandb": None}):
            callback = make_pipeline_stats_callback(
                observability=observability,
                pipeline_kind="pretrain",
            )

            # Should not raise
            callback(mock_stats)

    def test_handles_file_permission_error(self, mock_stats: MockPipelineStats):
        """Test that callback handles file permission errors gracefully."""
        observability = ObservabilityConfig(
            wandb_log_pipeline_stats=False,
            pipeline_stats_jsonl_path="/root/cannot_write_here.jsonl",  # Should fail
        )

        callback = make_pipeline_stats_callback(
            observability=observability,
            pipeline_kind="pretrain",
        )

        # Should not raise even with permission error
        callback(mock_stats)
