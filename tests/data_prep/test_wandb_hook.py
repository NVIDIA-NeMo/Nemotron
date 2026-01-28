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

"""Unit tests for wandb_hook module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nemotron.data_prep.config import ObservabilityConfig
from nemotron.data_prep.observability.stage_keys import canonical_stage_id
from nemotron.data_prep.observability.wandb_hook import (
    WandbStatsHook,
    _extract_stage_metrics,
    _flatten_pipeline_stats,
    make_wandb_stats_hook,
)


# =============================================================================
# Mock objects for PipelineStats (avoiding cosmos-xenna import)
# =============================================================================


class MockActorStats:
    """Mock ActorStats from cosmos-xenna."""

    def __init__(
        self,
        target: int = 4,
        pending: int = 0,
        ready: int = 4,
        running: int = 2,
        idle: int = 2,
    ) -> None:
        self.target = target
        self.pending = pending
        self.ready = ready
        self.running = running
        self.idle = idle


class MockTaskStats:
    """Mock TaskStats from cosmos-xenna."""

    def __init__(
        self,
        total_completed: int = 100,
        total_returned_none: int = 0,
        input_queue_size: int = 10,
        output_queue_size: int = 5,
    ) -> None:
        self.total_completed = total_completed
        self.total_returned_none = total_returned_none
        self.input_queue_size = input_queue_size
        self.output_queue_size = output_queue_size


class MockSlotStats:
    """Mock SlotStats from cosmos-xenna."""

    def __init__(self, num_used: int = 2, num_empty: int = 2) -> None:
        self.num_used = num_used
        self.num_empty = num_empty


class MockActorPoolStats:
    """Mock ActorPoolStats from cosmos-xenna."""

    def __init__(
        self,
        name: str = "TestStage",
        processing_speed_tasks_per_second: float | None = 10.5,
    ) -> None:
        self.name = name
        self.actor_stats = MockActorStats()
        self.task_stats = MockTaskStats()
        self.slot_stats = MockSlotStats()
        self.processing_speed_tasks_per_second = processing_speed_tasks_per_second


class MockResources:
    """Mock Resources from cosmos-xenna."""

    def __init__(
        self,
        num_cpus: float = 64.0,
        num_gpus: float = 8.0,
        memory: float = 256e9,
        object_store_memory: float = 128e9,
    ) -> None:
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.memory = memory
        self.object_store_memory = object_store_memory


class MockRayClusterInfo:
    """Mock RayClusterInfo from cosmos-xenna."""

    def __init__(self) -> None:
        self.total = MockResources()
        self.available = MockResources(num_cpus=32.0, num_gpus=4.0, memory=128e9, object_store_memory=64e9)
        self.actors = []


class MockRayStageResourceUsage:
    """Mock RayStageResourceUsage from cosmos-xenna."""

    def __init__(
        self,
        pool_name: str = "TestStage",
        cpu_utilization: float = 75.0,
        memory_usage: float = 16e9,
        actor_count: int = 4,
    ) -> None:
        self.pool_name = pool_name
        self.cpu_utilization = cpu_utilization
        self.memory_usage = memory_usage
        self.actor_count = actor_count


class MockPipelineStats:
    """Mock PipelineStats from cosmos-xenna."""

    def __init__(
        self,
        pipeline_duration_s: float = 120.5,
        main_loop_rate_hz: float = 10.0,
        num_initial_input_tasks: int = 1000,
        num_input_tasks_remaining: int = 500,
        num_outputs: int = 500,
    ) -> None:
        self.start_time = 0.0
        self.time = pipeline_duration_s
        self.pipeline_duration_s = pipeline_duration_s
        self.main_loop_rate_hz = main_loop_rate_hz
        self.num_initial_input_tasks = num_initial_input_tasks
        self.num_input_tasks_remaining = num_input_tasks_remaining
        self.num_outputs = num_outputs
        self.cluster = MockRayClusterInfo()
        self.actor_pools = [
            MockActorPoolStats("PlanStage"),
            MockActorPoolStats("DownloadStage"),
            MockActorPoolStats("BinIdxTokenizationStage"),
        ]
        self.resource_usage_per_stage = {
            "PlanStage": MockRayStageResourceUsage("PlanStage", cpu_utilization=10.0, memory_usage=1e9, actor_count=1),
            "BinIdxTokenizationStage": MockRayStageResourceUsage(
                "BinIdxTokenizationStage", cpu_utilization=300.0, memory_usage=64e9, actor_count=4
            ),
        }
        self.extra_data_per_stage = {}

    @property
    def inputs_processed_per_second(self) -> float:
        elapsed = self.time - self.start_time
        if elapsed <= 0:
            return 0.0
        return (self.num_initial_input_tasks - self.num_input_tasks_remaining) / elapsed

    @property
    def outputs_per_second(self) -> float:
        elapsed = self.time - self.start_time
        if elapsed <= 0:
            return 0.0
        return self.num_outputs / elapsed


class MockPipelineMonitor:
    """Mock PipelineMonitor class for testing patching."""

    def __init__(self) -> None:
        self._make_stats_called = False
        self._make_stats_call_count = 0

    def _make_stats(
        self,
        input_len: int,
        ext_output_lens: list[int],
        task_metadata_per_pool: list,
    ) -> MockPipelineStats:
        self._make_stats_called = True
        self._make_stats_call_count += 1
        return MockPipelineStats()


class TestCanonicalStageId:
    """Tests for canonical_stage_id from stage_keys module."""

    def test_plan_stage(self) -> None:
        assert canonical_stage_id("PlanStage") == "plan"

    def test_download_stage(self) -> None:
        assert canonical_stage_id("DownloadStage") == "download"

    def test_tokenization_stage(self) -> None:
        assert canonical_stage_id("BinIdxTokenizationStage") == "bin_idx_tokenization"

    def test_prefixed_stage(self) -> None:
        assert canonical_stage_id("Stage 00 - PlanStage") == "plan"
        assert canonical_stage_id("Stage 02 - BinIdxTokenizationStage") == "bin_idx_tokenization"

    def test_sft_stage(self) -> None:
        assert canonical_stage_id("PackedSftParquetStage") == "packed_sft_parquet"


# =============================================================================
# Test _flatten_pipeline_stats
# =============================================================================


class TestFlattenPipelineStats:
    """Tests for PipelineStats flattening (used for JSONL logging)."""

    def test_pipeline_level_metrics(self) -> None:
        """Extract pipeline-level metrics."""
        stats = MockPipelineStats()
        # Pass explicit namespace for testing (default is empty for JSONL)
        metrics = _flatten_pipeline_stats(stats, namespace="test")

        assert metrics["test/pipeline_duration_s"] == 120.5
        assert metrics["test/main_loop_rate_hz"] == 10.0
        assert metrics["test/num_input_remaining"] == 500
        assert metrics["test/num_initial_inputs"] == 1000
        assert metrics["test/num_outputs"] == 500

    def test_computed_rates(self) -> None:
        """Extract computed rate properties."""
        stats = MockPipelineStats()
        metrics = _flatten_pipeline_stats(stats, namespace="test")

        assert "test/inputs_processed_per_s" in metrics
        assert "test/outputs_per_s" in metrics
        assert metrics["test/inputs_processed_per_s"] > 0
        assert metrics["test/outputs_per_s"] > 0

    def test_progress_calculation(self) -> None:
        """Calculate progress percentage."""
        stats = MockPipelineStats(num_initial_input_tasks=1000, num_input_tasks_remaining=500)
        metrics = _flatten_pipeline_stats(stats, namespace="test")

        # (1000 - 500) / 1000 * 100 = 50%
        assert metrics["test/progress"] == 50.0

    def test_cluster_metrics(self) -> None:
        """Extract cluster resource metrics."""
        stats = MockPipelineStats()
        metrics = _flatten_pipeline_stats(stats, namespace="test")

        # Total resources
        assert metrics["test/cluster/total_cpus"] == 64.0
        assert metrics["test/cluster/total_gpus"] == 8.0
        assert metrics["test/cluster/total_mem_gb"] == 256.0
        assert metrics["test/cluster/total_obj_store_gb"] == 128.0

        # Available resources
        assert metrics["test/cluster/avail_cpus"] == 32.0
        assert metrics["test/cluster/avail_gpus"] == 4.0
        assert metrics["test/cluster/avail_mem_gb"] == 128.0
        assert metrics["test/cluster/avail_obj_store_gb"] == 64.0

    def test_per_stage_actor_metrics(self) -> None:
        """Extract per-stage actor pool metrics using consolidated format."""
        stats = MockPipelineStats()
        metrics = _flatten_pipeline_stats(stats, namespace="test")

        # Check one stage - PlanStage (uses test/stages/<metric>/<stage> format)
        assert "test/stages/actors_target/plan" in metrics
        assert "test/stages/actors_ready/plan" in metrics
        assert "test/stages/tasks_completed/plan" in metrics
        assert "test/stages/queue_in/plan" in metrics
        assert "test/stages/queue_out/plan" in metrics
        assert "test/stages/slots_used/plan" in metrics
        assert "test/stages/speed_tasks_per_s/plan" in metrics

    def test_per_stage_resource_usage(self) -> None:
        """Extract per-stage resource usage metrics using consolidated format."""
        stats = MockPipelineStats()
        metrics = _flatten_pipeline_stats(stats, namespace="test")

        # Check resource usage for BinIdxTokenizationStage (test/stages/<metric>/<stage>)
        assert "test/stages/resource_cpu_util_pct/bin_idx_tokenization" in metrics
        assert metrics["test/stages/resource_cpu_util_pct/bin_idx_tokenization"] == 300.0
        assert "test/stages/resource_mem_gb/bin_idx_tokenization" in metrics
        assert metrics["test/stages/resource_mem_gb/bin_idx_tokenization"] == 64.0

    def test_custom_namespace(self) -> None:
        """Use custom namespace prefix."""
        stats = MockPipelineStats()
        metrics = _flatten_pipeline_stats(stats, namespace="custom")

        assert "custom/pipeline_duration_s" in metrics
        assert "xenna/pipeline_duration_s" not in metrics

    def test_handles_none_speed(self) -> None:
        """Handle None processing speed gracefully."""
        stats = MockPipelineStats()
        # Set one stage's speed to None
        stats.actor_pools[0].processing_speed_tasks_per_second = None

        metrics = _flatten_pipeline_stats(stats)
        # Should not raise, and key should be absent
        assert "xenna/stages/speed_tasks_per_s/plan" not in metrics


class TestExtractStageMetrics:
    """Tests for _extract_stage_metrics helper."""

    def test_extracts_by_metric_name(self) -> None:
        """Metrics are organized by metric name, then by stage."""
        stats = MockPipelineStats()
        metrics_by_name = _extract_stage_metrics(stats)

        # Should have metric names as top-level keys
        assert "tasks_completed" in metrics_by_name
        assert "queue_in" in metrics_by_name
        assert "actors_ready" in metrics_by_name

        # Each metric should have values keyed by stage ID
        assert "plan" in metrics_by_name["tasks_completed"]
        assert "download" in metrics_by_name["tasks_completed"]
        assert "bin_idx_tokenization" in metrics_by_name["tasks_completed"]

    def test_resource_metrics(self) -> None:
        """Resource metrics are extracted correctly."""
        stats = MockPipelineStats()
        metrics_by_name = _extract_stage_metrics(stats)

        assert "resource_cpu_util_pct" in metrics_by_name
        assert "bin_idx_tokenization" in metrics_by_name["resource_cpu_util_pct"]
        assert metrics_by_name["resource_cpu_util_pct"]["bin_idx_tokenization"] == 300.0


# =============================================================================
# Test WandbStatsHook
# =============================================================================


class TestWandbStatsHook:
    """Tests for the W&B stats hook context manager."""

    def test_init(self) -> None:
        """Test initialization."""
        cfg = ObservabilityConfig(wandb_log_pipeline_stats=True)
        hook = WandbStatsHook(
            observability=cfg,
            pipeline_kind="pretrain",
            run_hash="abc123",
        )

        assert hook._pipeline_kind == "pretrain"
        assert hook._run_hash == "abc123"
        assert hook._log_count == 0

    def test_context_manager_patches(self) -> None:
        """Context manager patches _make_stats."""
        cfg = ObservabilityConfig(wandb_log_pipeline_stats=True)
        hook = WandbStatsHook(
            observability=cfg,
            pipeline_kind="pretrain",
            monitor_cls=MockPipelineMonitor,
        )

        original = MockPipelineMonitor._make_stats

        with hook:
            # Method should be patched
            assert MockPipelineMonitor._make_stats != original

        # After exit, should be restored
        assert MockPipelineMonitor._make_stats == original

    def test_wrapper_calls_original(self) -> None:
        """Wrapper calls original _make_stats and returns result."""
        cfg = ObservabilityConfig(wandb_log_pipeline_stats=True)
        hook = WandbStatsHook(
            observability=cfg,
            pipeline_kind="pretrain",
            monitor_cls=MockPipelineMonitor,
        )

        monitor = MockPipelineMonitor()

        with hook:
            result = MockPipelineMonitor._make_stats(monitor, 100, [50], [])

        assert isinstance(result, MockPipelineStats)
        assert monitor._make_stats_called

    def test_nested_contexts(self) -> None:
        """Nested contexts work with reference counting."""
        cfg = ObservabilityConfig(wandb_log_pipeline_stats=True)
        hook1 = WandbStatsHook(
            observability=cfg,
            pipeline_kind="pretrain",
            monitor_cls=MockPipelineMonitor,
        )
        hook2 = WandbStatsHook(
            observability=cfg,
            pipeline_kind="sft",
            monitor_cls=MockPipelineMonitor,
        )

        original = MockPipelineMonitor._make_stats

        with hook1:
            assert MockPipelineMonitor._make_stats != original

            with hook2:
                # Still patched
                assert MockPipelineMonitor._make_stats != original

            # Still patched (hook1 still active)
            assert MockPipelineMonitor._make_stats != original

        # Now restored
        assert MockPipelineMonitor._make_stats == original

    def test_logs_to_wandb(self) -> None:
        """Hook logs metrics to W&B using pipeline_kind as namespace."""
        cfg = ObservabilityConfig(wandb_log_pipeline_stats=True)
        hook = WandbStatsHook(
            observability=cfg,
            pipeline_kind="pretrain",
            monitor_cls=MockPipelineMonitor,
        )

        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with hook:
                # Trigger stats collection
                monitor = MockPipelineMonitor()
                MockPipelineMonitor._make_stats(monitor, 100, [50], [])

        # W&B should have been called
        mock_wandb.log.assert_called()
        logged_metrics = mock_wandb.log.call_args[0][0]
        # Now uses pipeline_kind as namespace (e.g., "pretrain/...")
        assert "pretrain/pipeline_duration_s" in logged_metrics

    def test_creates_line_series_charts(self) -> None:
        """Hook creates consolidated line_series charts after multiple steps."""
        cfg = ObservabilityConfig(wandb_log_pipeline_stats=True)
        hook = WandbStatsHook(
            observability=cfg,
            pipeline_kind="pretrain",
            monitor_cls=MockPipelineMonitor,
        )
        # Set chart update interval to 5 steps
        hook._chart_update_interval = 5

        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()
        mock_chart = MagicMock()
        mock_wandb.plot.line_series.return_value = mock_chart

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with hook:
                monitor = MockPipelineMonitor()
                # Trigger 6 stats updates to hit the chart creation (at step 5)
                for _ in range(6):
                    MockPipelineMonitor._make_stats(monitor, 100, [50], [])

        # line_series should have been called to create consolidated charts
        assert mock_wandb.plot.line_series.called, "line_series should be called after 5 steps"

        # Verify the chart was created with correct parameters
        call_args = mock_wandb.plot.line_series.call_args
        assert call_args is not None
        # Should have xs (steps), ys (values), keys (stage names)
        assert "xs" in call_args.kwargs or len(call_args.args) >= 1
        assert "ys" in call_args.kwargs or len(call_args.args) >= 2
        assert "keys" in call_args.kwargs or len(call_args.args) >= 3

    def test_logs_to_jsonl(self) -> None:
        """Hook logs to JSONL file when configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "stats.jsonl"
            cfg = ObservabilityConfig(
                wandb_log_pipeline_stats=False,
                pipeline_stats_jsonl_path=str(jsonl_path),
            )
            hook = WandbStatsHook(
                observability=cfg,
                pipeline_kind="pretrain",
                run_hash="test123",
                monitor_cls=MockPipelineMonitor,
            )

            with hook:
                monitor = MockPipelineMonitor()
                MockPipelineMonitor._make_stats(monitor, 100, [50], [])

            # Check JSONL was written
            assert jsonl_path.exists()
            with open(jsonl_path) as f:
                lines = f.readlines()

            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["pipeline_kind"] == "pretrain"
            assert record["run_hash"] == "test123"
            assert "metrics" in record

    def test_no_wandb_no_crash(self) -> None:
        """Hook handles missing wandb gracefully."""
        cfg = ObservabilityConfig(wandb_log_pipeline_stats=True)
        hook = WandbStatsHook(
            observability=cfg,
            pipeline_kind="pretrain",
            monitor_cls=MockPipelineMonitor,
        )

        # Force ImportError by removing wandb
        with patch.dict("sys.modules", {"wandb": None}):
            with hook:
                monitor = MockPipelineMonitor()
                # Should not raise
                MockPipelineMonitor._make_stats(monitor, 100, [50], [])


# =============================================================================
# Test make_wandb_stats_hook factory
# =============================================================================


class TestMakeWandbStatsHook:
    """Tests for the factory function."""

    def test_returns_hook_when_enabled(self) -> None:
        """Returns hook when W&B logging is enabled."""
        cfg = ObservabilityConfig(wandb_log_pipeline_stats=True)
        hook = make_wandb_stats_hook(
            observability=cfg,
            pipeline_kind="pretrain",
        )

        assert hook is not None
        assert isinstance(hook, WandbStatsHook)

    def test_returns_hook_when_jsonl_enabled(self) -> None:
        """Returns hook when JSONL logging is enabled."""
        cfg = ObservabilityConfig(
            wandb_log_pipeline_stats=False,
            pipeline_stats_jsonl_path="/tmp/test.jsonl",
        )
        hook = make_wandb_stats_hook(
            observability=cfg,
            pipeline_kind="pretrain",
        )

        assert hook is not None

    def test_returns_none_when_disabled(self) -> None:
        """Returns None when all logging is disabled."""
        cfg = ObservabilityConfig(
            wandb_log_pipeline_stats=False,
            wandb_log_progress_table=False,
            pipeline_stats_jsonl_path=None,
        )
        hook = make_wandb_stats_hook(
            observability=cfg,
            pipeline_kind="pretrain",
        )

        assert hook is None

    def test_passes_context_info(self) -> None:
        """Factory passes context info to hook."""
        cfg = ObservabilityConfig(wandb_log_pipeline_stats=True)
        hook = make_wandb_stats_hook(
            observability=cfg,
            pipeline_kind="sft",
            run_hash="hash123",
            run_dir="/output/run",
            dataset_names=["dataset1", "dataset2"],
            dataset_input_bytes={"dataset1": 1_500_000_000, "dataset2": 500_000_000},
        )

        assert hook is not None
        assert hook._pipeline_kind == "sft"
        assert hook._run_hash == "hash123"
        assert hook._run_dir == "/output/run"
        assert hook._dataset_names == ["dataset1", "dataset2"]
        assert hook._dataset_input_bytes == {"dataset1": 1_500_000_000, "dataset2": 500_000_000}


# =============================================================================
# Integration test
# =============================================================================


class TestIntegrationScenario:
    """Integration-style tests."""

    def test_full_pipeline_simulation(self) -> None:
        """Simulate a full pipeline with multiple stats updates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "pipeline_stats.jsonl"
            cfg = ObservabilityConfig(
                wandb_log_pipeline_stats=False,  # Avoid W&B in test
                pipeline_stats_jsonl_path=str(jsonl_path),
            )

            # Create hook directly with mock monitor class to avoid cosmos_xenna import
            hook = WandbStatsHook(
                observability=cfg,
                pipeline_kind="pretrain",
                run_hash="integration_test",
                dataset_names=["dataset_a", "dataset_b"],
                monitor_cls=MockPipelineMonitor,
            )

            # Simulate pipeline execution with multiple updates
            with hook:
                monitor = MockPipelineMonitor()
                for i in range(5):
                    MockPipelineMonitor._make_stats(monitor, 100 - i * 20, [50], [])

            # Verify all updates were logged
            assert jsonl_path.exists()
            with open(jsonl_path) as f:
                records = [json.loads(line) for line in f]

            assert len(records) == 5
            for record in records:
                assert record["pipeline_kind"] == "pretrain"
                assert record["run_hash"] == "integration_test"
                assert "timestamp" in record
                assert "metrics" in record
