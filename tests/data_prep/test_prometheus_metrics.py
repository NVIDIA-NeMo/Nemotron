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

"""Unit tests for prometheus_metrics module."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from nemotron.data_prep.prometheus_metrics import (
    PrometheusConfig,
    PrometheusMetricsLogger,
    PromSample,
    _build_wandb_key,
    _slugify_stage_name,
    enable_ray_metrics_export,
    find_free_port,
    parse_prometheus_text,
    transform_samples_to_wandb,
)


# =============================================================================
# Test parse_prometheus_text
# =============================================================================


class TestParsePrometheusText:
    """Tests for Prometheus text exposition format parsing."""

    def test_simple_metric(self) -> None:
        """Parse a simple metric without labels."""
        text = "my_metric 42.5"
        samples = parse_prometheus_text(text)

        assert len(samples) == 1
        assert samples[0].name == "my_metric"
        assert samples[0].labels == {}
        assert samples[0].value == 42.5

    def test_metric_with_labels(self) -> None:
        """Parse a metric with labels."""
        text = 'pipeline_progress{stage="Stage 01 - Download"} 0.75'
        samples = parse_prometheus_text(text)

        assert len(samples) == 1
        assert samples[0].name == "pipeline_progress"
        assert samples[0].labels == {"stage": "Stage 01 - Download"}
        assert samples[0].value == 0.75

    def test_metric_with_multiple_labels(self) -> None:
        """Parse a metric with multiple labels."""
        text = 'pipeline_actor_count{stage="Stage 02 - Tokenize",state="ready"} 8'
        samples = parse_prometheus_text(text)

        assert len(samples) == 1
        assert samples[0].name == "pipeline_actor_count"
        assert samples[0].labels == {"stage": "Stage 02 - Tokenize", "state": "ready"}
        assert samples[0].value == 8.0

    def test_skips_comments(self) -> None:
        """Comments are skipped."""
        text = """
        # HELP my_metric A description
        # TYPE my_metric gauge
        my_metric 100
        """
        samples = parse_prometheus_text(text)

        assert len(samples) == 1
        assert samples[0].name == "my_metric"
        assert samples[0].value == 100.0

    def test_skips_blank_lines(self) -> None:
        """Blank lines are skipped."""
        text = """
        metric_a 1

        metric_b 2
        """
        samples = parse_prometheus_text(text)

        assert len(samples) == 2
        assert samples[0].name == "metric_a"
        assert samples[1].name == "metric_b"

    def test_multiple_metrics(self) -> None:
        """Parse multiple metrics."""
        text = """
        pipeline_progress 0.5
        pipeline_finished_tasks{stage="Stage 01"} 10
        pipeline_total_tasks{stage="Stage 01"} 20
        """
        samples = parse_prometheus_text(text)

        assert len(samples) == 3
        names = [s.name for s in samples]
        assert "pipeline_progress" in names
        assert "pipeline_finished_tasks" in names
        assert "pipeline_total_tasks" in names

    def test_scientific_notation(self) -> None:
        """Parse scientific notation values."""
        text = "my_metric 1.5e10"
        samples = parse_prometheus_text(text)

        assert len(samples) == 1
        assert samples[0].value == 1.5e10

    def test_negative_values(self) -> None:
        """Parse negative values."""
        text = "my_metric -42.5"
        samples = parse_prometheus_text(text)

        assert len(samples) == 1
        assert samples[0].value == -42.5

    def test_integer_values(self) -> None:
        """Parse integer values."""
        text = "my_metric 42"
        samples = parse_prometheus_text(text)

        assert len(samples) == 1
        assert samples[0].value == 42.0

    def test_empty_text(self) -> None:
        """Empty text returns empty list."""
        assert parse_prometheus_text("") == []
        assert parse_prometheus_text("   \n\n  ") == []

    def test_invalid_line(self) -> None:
        """Invalid lines are skipped."""
        text = """
        valid_metric 123
        invalid line without value
        another_valid 456
        """
        samples = parse_prometheus_text(text)

        assert len(samples) == 2
        assert samples[0].name == "valid_metric"
        assert samples[1].name == "another_valid"


# =============================================================================
# Test _slugify_stage_name
# =============================================================================


class TestSlugifyStageName:
    """Tests for stage name slugification (now uses canonical_stage_id)."""

    def test_with_alias(self) -> None:
        """Use alias when available."""
        aliases = {"Stage 01 - DownloadStage": "download"}
        assert _slugify_stage_name("Stage 01 - DownloadStage", aliases) == "download"

    def test_without_alias(self) -> None:
        """Slugify without alias uses canonical_stage_id (strips prefix and suffix)."""
        result = _slugify_stage_name("Stage 02 - BinIdxTokenizationStage")
        # Now uses canonical_stage_id which strips "Stage NN - " prefix and "Stage" suffix
        assert result == "bin_idx_tokenization"

    def test_removes_stage_suffix(self) -> None:
        """Remove 'Stage' suffix."""
        result = _slugify_stage_name("MyProcessingStage")
        # canonical_stage_id removes "Stage" suffix and converts camelCase
        assert result == "my_processing"

    def test_handles_special_chars(self) -> None:
        """Handle special characters."""
        result = _slugify_stage_name("Stage 01 - My@Special#Stage!")
        assert "my" in result.lower()
        assert "@" not in result
        assert "#" not in result

    def test_empty_string(self) -> None:
        """Handle empty string."""
        assert _slugify_stage_name("") == "unknown"

    def test_no_aliases_provided(self) -> None:
        """Works when no aliases provided."""
        result = _slugify_stage_name("Stage 01 - Download", None)
        assert "download" in result.lower()


# =============================================================================
# Test _build_wandb_key
# =============================================================================


class TestBuildWandbKey:
    """Tests for W&B key construction using consolidated format."""

    def test_global_metric(self) -> None:
        """Build key for global metric (no labels)."""
        key = _build_wandb_key("pipeline_progress", {}, namespace="xenna", stage_aliases=None)
        assert key == "xenna/progress"

    def test_stage_metric(self) -> None:
        """Build key for stage metric using consolidated format: stages/<metric>/<stage>."""
        key = _build_wandb_key(
            "pipeline_finished_tasks",
            {"stage": "Stage 01 - DownloadStage"},
            namespace="xenna",
            stage_aliases={"Stage 01 - DownloadStage": "download"},
        )
        # Now uses consolidated format: stages/<metric>/<stage>
        assert key == "xenna/stages/finished_tasks/download"

    def test_stage_metric_with_state(self) -> None:
        """Build key for stage metric with state label (appended to metric name)."""
        key = _build_wandb_key(
            "pipeline_actor_count",
            {"stage": "Stage 02 - Tokenize", "state": "ready"},
            namespace="xenna",
            stage_aliases={"Stage 02 - Tokenize": "tokenize"},
        )
        # State is appended to metric name: stages/<metric>_<state>/<stage>
        assert key == "xenna/stages/actor_count_ready/tokenize"

    def test_stage_metric_with_resource(self) -> None:
        """Build key for stage metric with resource label (appended to metric name)."""
        key = _build_wandb_key(
            "pipeline_resource_usage",
            {"stage": "Stage 02 - Tokenize", "resource": "cpu"},
            namespace="xenna",
            stage_aliases={"Stage 02 - Tokenize": "tokenize"},
        )
        # Resource is appended to metric name: stages/<metric>_<resource>/<stage>
        assert key == "xenna/stages/resource_usage_cpu/tokenize"

    def test_loop_timing_metric(self) -> None:
        """Build key for loop timing metric."""
        key = _build_wandb_key(
            "pipeline_loop_time_s",
            {"step": "total", "method": "avg"},
            namespace="xenna",
            stage_aliases=None,
        )
        assert key == "xenna/loop/total/avg"

    def test_custom_namespace(self) -> None:
        """Use custom namespace."""
        key = _build_wandb_key("pipeline_progress", {}, namespace="custom", stage_aliases=None)
        assert key == "custom/progress"


# =============================================================================
# Test transform_samples_to_wandb
# =============================================================================


class TestTransformSamplesToWandb:
    """Tests for samples to W&B transformation."""

    def test_basic_transform(self) -> None:
        """Basic transformation."""
        samples = [
            PromSample("pipeline_progress", {}, 0.5),
            PromSample("pipeline_finished_tasks", {"stage": "Stage 01"}, 10),
        ]
        # Use explicit namespace for testing
        metrics = transform_samples_to_wandb(samples, namespace="test")

        assert "test/progress" in metrics
        assert metrics["test/progress"] == 0.5

    def test_filters_by_prefix(self) -> None:
        """Only include metrics with allowed prefix."""
        samples = [
            PromSample("pipeline_progress", {}, 0.5),
            PromSample("ray_gcs_storage_ops", {}, 100),
            PromSample("other_metric", {}, 999),
        ]
        metrics = transform_samples_to_wandb(samples, namespace="test", include_prefixes=("pipeline_",))

        assert "test/progress" in metrics
        assert len(metrics) == 1

    def test_skips_nan(self) -> None:
        """Skip NaN values."""
        samples = [
            PromSample("pipeline_progress", {}, float("nan")),
            PromSample("pipeline_valid", {}, 1.0),
        ]
        metrics = transform_samples_to_wandb(samples, namespace="test")

        # NaN should be skipped
        assert "test/valid" in metrics
        assert "test/progress" not in metrics

    def test_skips_inf(self) -> None:
        """Skip Inf values."""
        samples = [
            PromSample("pipeline_progress", {}, float("inf")),
            PromSample("pipeline_valid", {}, 1.0),
        ]
        metrics = transform_samples_to_wandb(samples, namespace="test")

        # Inf should be skipped
        assert "test/valid" in metrics
        assert "test/progress" not in metrics

    def test_max_for_duplicate_keys(self) -> None:
        """Take max for duplicate keys."""
        samples = [
            PromSample("pipeline_progress", {}, 0.3),
            PromSample("pipeline_progress", {}, 0.7),
        ]
        metrics = transform_samples_to_wandb(samples, namespace="test")

        assert metrics["test/progress"] == 0.7

    def test_with_stage_aliases(self) -> None:
        """Use stage aliases with consolidated format."""
        samples = [
            PromSample("pipeline_finished_tasks", {"stage": "Stage 02 - BinIdxTokenizationStage"}, 50),
        ]
        aliases = {"Stage 02 - BinIdxTokenizationStage": "tokenize"}
        metrics = transform_samples_to_wandb(samples, namespace="test", stage_aliases=aliases)

        # Now uses consolidated format: stages/<metric>/<stage>
        assert "test/stages/finished_tasks/tokenize" in metrics

    def test_empty_samples(self) -> None:
        """Empty samples returns empty dict."""
        assert transform_samples_to_wandb([]) == {}


# =============================================================================
# Test enable_ray_metrics_export
# =============================================================================


class TestEnableRayMetricsExport:
    """Tests for Ray metrics export enablement."""

    def test_sets_env_var_with_port(self) -> None:
        """Sets RAY_METRICS_PORT env var."""
        # Clean up any existing env var
        os.environ.pop("RAY_METRICS_PORT", None)

        port = enable_ray_metrics_export(port=9999)

        assert port == 9999
        assert os.environ.get("RAY_METRICS_PORT") == "9999"

        # Cleanup
        os.environ.pop("RAY_METRICS_PORT", None)

    def test_auto_selects_free_port(self) -> None:
        """Auto-selects a free port when None."""
        os.environ.pop("RAY_METRICS_PORT", None)

        port = enable_ray_metrics_export(port=None)

        assert isinstance(port, int)
        assert port > 0
        assert os.environ.get("RAY_METRICS_PORT") == str(port)

        # Cleanup
        os.environ.pop("RAY_METRICS_PORT", None)


class TestFindFreePort:
    """Tests for find_free_port."""

    def test_returns_valid_port(self) -> None:
        """Returns a valid port number."""
        port = find_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535

    def test_returns_different_ports(self) -> None:
        """Likely returns different ports on successive calls."""
        # This isn't guaranteed but should usually work
        ports = {find_free_port() for _ in range(5)}
        # At least some should be different
        assert len(ports) >= 1


# =============================================================================
# Test PrometheusConfig
# =============================================================================


class TestPrometheusConfig:
    """Tests for config dataclass."""

    def test_defaults(self) -> None:
        """Check default values."""
        cfg = PrometheusConfig()

        assert cfg.port == 8080
        assert cfg.host == "127.0.0.1"
        assert cfg.poll_interval_s == 10.0
        assert cfg.request_timeout_s == 5.0
        assert cfg.wandb_namespace == "pretrain"  # Default is now pipeline_kind
        assert cfg.include_prefixes == ("pipeline_",)
        assert isinstance(cfg.stage_aliases, dict)

    def test_custom_values(self) -> None:
        """Custom values work."""
        cfg = PrometheusConfig(
            port=9090,
            host="0.0.0.0",
            poll_interval_s=5.0,
        )

        assert cfg.port == 9090
        assert cfg.host == "0.0.0.0"
        assert cfg.poll_interval_s == 5.0

    def test_is_frozen(self) -> None:
        """Config is immutable."""
        cfg = PrometheusConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.port = 9999  # type: ignore[misc]


# =============================================================================
# Test PrometheusMetricsLogger
# =============================================================================


class TestPrometheusMetricsLogger:
    """Tests for the metrics logger class."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        logger = PrometheusMetricsLogger()

        assert logger.port == 8080
        assert logger.host == "127.0.0.1"
        assert logger.poll_interval_s == 10.0
        assert logger.metrics_url == "http://127.0.0.1:8080/metrics"

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        logger = PrometheusMetricsLogger(
            port=9090,
            host="localhost",
            poll_interval_s=5.0,
        )

        assert logger.port == 9090
        assert logger.host == "localhost"
        assert logger.metrics_url == "http://localhost:9090/metrics"

    def test_context_manager(self) -> None:
        """Test context manager protocol."""
        logger = PrometheusMetricsLogger(poll_interval_s=60.0)

        # Mock _scrape_loop to prevent actual scraping
        with patch.object(logger, "_scrape_loop"):
            with logger as ctx:
                assert ctx is logger
                assert logger._thread is not None

        # After exit, thread should be stopped
        # (may take a moment)

    def test_start_stop(self) -> None:
        """Test start and stop methods."""
        logger = PrometheusMetricsLogger(poll_interval_s=60.0)

        with patch.object(logger, "_scrape_loop"):
            logger.start()
            assert logger._thread is not None

            logger.stop(timeout_s=1.0)
            assert logger._thread is None

    def test_double_start(self) -> None:
        """Double start is a no-op."""
        logger = PrometheusMetricsLogger(poll_interval_s=60.0)

        with patch.object(logger, "_scrape_loop"):
            logger.start()
            thread1 = logger._thread

            logger.start()  # Should not create new thread
            assert logger._thread is thread1

            logger.stop(timeout_s=1.0)

    def test_stop_without_start(self) -> None:
        """Stop without start is a no-op."""
        logger = PrometheusMetricsLogger()
        logger.stop()  # Should not raise

    def test_fetch_metrics_handles_failure(self) -> None:
        """Fetch handles connection failures gracefully."""
        logger = PrometheusMetricsLogger(port=1)  # Invalid port

        result = logger._fetch_metrics_text()
        assert result is None

    def test_scrape_and_log_no_metrics(self) -> None:
        """scrape_and_log handles no metrics gracefully."""
        logger = PrometheusMetricsLogger()

        with patch.object(logger, "_fetch_metrics_text", return_value=None):
            logger._scrape_and_log()  # Should not raise

    def test_scrape_and_log_with_wandb(self) -> None:
        """scrape_and_log logs to W&B when available."""
        # Use explicit namespace for testing
        logger = PrometheusMetricsLogger(wandb_namespace="test")
        logger._start_time = 1000.0

        prometheus_text = """
        pipeline_progress 0.5
        pipeline_finished_tasks{stage="Stage 01"} 10
        """

        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()  # Active run

        with patch.object(logger, "_fetch_metrics_text", return_value=prometheus_text):
            with patch.dict("sys.modules", {"wandb": mock_wandb}):
                logger._scrape_and_log()

        # W&B log should have been called
        mock_wandb.log.assert_called_once()
        logged_metrics = mock_wandb.log.call_args[0][0]
        assert "test/progress" in logged_metrics

    def test_scrape_and_log_without_wandb(self) -> None:
        """scrape_and_log handles missing wandb gracefully."""
        logger = PrometheusMetricsLogger()

        prometheus_text = "pipeline_progress 0.5"

        with patch.object(logger, "_fetch_metrics_text", return_value=prometheus_text):
            # Force ImportError for wandb
            with patch.dict("sys.modules", {"wandb": None}):
                logger._scrape_and_log()  # Should not raise


# =============================================================================
# Test integration scenario
# =============================================================================


class TestIntegrationScenario:
    """Integration-style tests."""

    def test_full_prometheus_parsing_pipeline(self) -> None:
        """Test full pipeline from raw text to W&B metrics."""
        raw_text = """
        # HELP pipeline_progress Overall pipeline progress
        # TYPE pipeline_progress gauge
        pipeline_progress 0.75

        # HELP pipeline_finished_tasks Tasks completed per stage
        # TYPE pipeline_finished_tasks gauge
        pipeline_finished_tasks{stage="Stage 00 - PlanStage"} 5
        pipeline_finished_tasks{stage="Stage 01 - DownloadStage"} 100
        pipeline_finished_tasks{stage="Stage 02 - BinIdxTokenizationStage"} 50

        # HELP pipeline_actor_count Actor counts per stage
        # TYPE pipeline_actor_count gauge
        pipeline_actor_count{stage="Stage 02 - BinIdxTokenizationStage",state="ready"} 8
        pipeline_actor_count{stage="Stage 02 - BinIdxTokenizationStage",state="busy"} 4

        # Non-pipeline metrics should be filtered
        ray_gcs_total_ops 12345
        """

        # Parse
        samples = parse_prometheus_text(raw_text)
        assert len(samples) == 7  # 1 progress + 3 finished + 2 actor + 1 ray (filtered later)

        # Transform with explicit namespace
        aliases = {
            "Stage 00 - PlanStage": "plan",
            "Stage 01 - DownloadStage": "download",
            "Stage 02 - BinIdxTokenizationStage": "tokenize",
        }
        metrics = transform_samples_to_wandb(samples, namespace="pretrain", stage_aliases=aliases)

        # Verify expected keys using consolidated format: stages/<metric>/<stage>
        assert metrics["pretrain/progress"] == 0.75
        assert metrics["pretrain/stages/finished_tasks/plan"] == 5.0
        assert metrics["pretrain/stages/finished_tasks/download"] == 100.0
        assert metrics["pretrain/stages/finished_tasks/tokenize"] == 50.0
        assert metrics["pretrain/stages/actor_count_ready/tokenize"] == 8.0
        assert metrics["pretrain/stages/actor_count_busy/tokenize"] == 4.0

        # Verify ray metric was filtered out
        assert "ray_gcs_total_ops" not in str(metrics)
