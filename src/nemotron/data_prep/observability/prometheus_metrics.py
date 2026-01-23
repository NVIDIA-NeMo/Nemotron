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

"""
Prometheus metrics scraper for pipelines → W&B logging.

This module provides a way to log pipeline metrics to W&B
WITHOUT requiring any changes to cosmos-xenna. It works by:

1. Enabling Ray's Prometheus metrics export via RAY_METRICS_PORT
2. Running a background thread that scrapes the /metrics endpoint
3. Parsing the Prometheus text format and logging to W&B

Usage:
    from nemotron.data_prep.observability.prometheus_metrics import (
        PrometheusMetricsLogger,
        enable_ray_metrics_export,
    )

    # Before running pipeline - enable metrics export
    enable_ray_metrics_export(port=8080)

    # Use as context manager around pipeline execution
    with PrometheusMetricsLogger(port=8080, poll_interval_s=10.0):
        pipelines_v1.run_pipeline(pipeline_spec)
"""

from __future__ import annotations

import os
import re
import socket
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from nemotron.data_prep.observability.stage_keys import canonical_stage_id


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class PrometheusConfig:
    """Configuration for Prometheus metrics scraping.

    Attributes:
        port: Port for Ray metrics export. If None, auto-selects a free port.
        host: Host to scrape (default: 127.0.0.1)
        poll_interval_s: How often to scrape metrics (seconds)
        request_timeout_s: HTTP request timeout
        wandb_namespace: Prefix for W&B metric keys (e.g., "pretrain", "sft")
        include_prefixes: Only include metrics starting with these prefixes
        stage_aliases: Map stage names to shorter aliases for cleaner keys
    """

    port: int | None = 8080
    host: str = "127.0.0.1"
    poll_interval_s: float = 10.0
    request_timeout_s: float = 5.0
    wandb_namespace: str = "pretrain"  # Use pipeline_kind (e.g., "pretrain", "sft")
    include_prefixes: tuple[str, ...] = ("pipeline_",)
    stage_aliases: dict[str, str] = field(default_factory=lambda: {
        "Stage 00 - PlanStage": "plan",
        "Stage 01 - DownloadStage": "download",
        "Stage 02 - BinIdxTokenizationStage": "tokenize",
        "Stage 02 - PackedSftParquetStage": "pack_sft",
        "Stage 00 - SftPlanStage": "sft_plan",
    })


# =============================================================================
# Prometheus Parsing
# =============================================================================


@dataclass
class PromSample:
    """A single Prometheus metric sample."""

    name: str
    labels: dict[str, str]
    value: float


# Pattern: metric_name{label="value",...} value
_METRIC_LINE_RE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)'  # metric name
    r'(?:\{([^}]*)\})?'              # optional labels
    r'\s+([0-9.eE+-]+|NaN|Inf|-Inf)' # value
)

_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"')


def parse_prometheus_text(text: str) -> list[PromSample]:
    """Parse Prometheus text exposition format.

    Args:
        text: Raw Prometheus /metrics response

    Returns:
        List of PromSample objects
    """
    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        match = _METRIC_LINE_RE.match(line)
        if not match:
            continue

        name = match.group(1)
        labels_str = match.group(2) or ""
        value_str = match.group(3)

        # Parse labels
        labels = {}
        for lm in _LABEL_RE.finditer(labels_str):
            labels[lm.group(1)] = lm.group(2)

        # Parse value
        try:
            value = float(value_str)
        except ValueError:
            continue

        results.append(PromSample(name=name, labels=labels, value=value))

    return results


# =============================================================================
# Stage Name Utilities
# =============================================================================


def _slugify_stage_name(name: str, aliases: dict[str, str] | None = None) -> str:
    """Convert a stage name to a URL-safe slug.

    Uses canonical_stage_id() from stage_keys module for consistent naming
    across all observability components.

    Args:
        name: Stage name like "Stage 02 - BinIdxTokenizationStage"
        aliases: Optional mapping of full names to short aliases (checked first)

    Returns:
        Canonical stage ID like "bin_idx_tokenization" or alias if provided
    """
    if aliases and name in aliases:
        return aliases[name]

    return canonical_stage_id(name)


# =============================================================================
# Metrics Transformation
# =============================================================================


def transform_samples_to_wandb(
    samples: list[PromSample],
    *,
    namespace: str = "pretrain",  # Use pipeline_kind (e.g., "pretrain", "sft")
    include_prefixes: tuple[str, ...] = ("pipeline_",),
    stage_aliases: dict[str, str] | None = None,
) -> dict[str, float]:
    """Transform Prometheus samples to W&B metric dict.

    Args:
        samples: List of PromSample from Prometheus scrape
        namespace: W&B key prefix
        include_prefixes: Only include metrics starting with these
        stage_aliases: Optional stage name aliases

    Returns:
        Dict of W&B metric keys to values
    """
    metrics: dict[str, float] = {}

    for sample in samples:
        # Filter by prefix
        if not any(sample.name.startswith(p) for p in include_prefixes):
            continue

        # Skip NaN/Inf values
        if sample.value != sample.value or abs(sample.value) == float("inf"):  # NaN check
            continue

        # Build W&B key based on metric name and labels
        key = _build_wandb_key(
            sample.name,
            sample.labels,
            namespace=namespace,
            stage_aliases=stage_aliases,
        )

        # For metrics that can come from multiple sources, take max
        if key in metrics:
            metrics[key] = max(metrics[key], sample.value)
        else:
            metrics[key] = sample.value

    return metrics


def _build_wandb_key(
    name: str,
    labels: dict[str, str],
    *,
    namespace: str,
    stage_aliases: dict[str, str] | None,
) -> str:
    """Build a W&B metric key from Prometheus metric name and labels.

    Uses the consolidated pattern: stages/<metric>/<stage> for per-stage metrics.
    This groups all stages under each metric, producing fewer charts that are
    easier to read in W&B dashboards.

    Handles pipeline metric patterns:
    - pipeline_progress -> <namespace>/progress
    - pipeline_finished_tasks{stage=...} -> <namespace>/stages/finished_tasks/<stage_id>
    - pipeline_actor_count{stage=...,state=ready} -> <namespace>/stages/actor_count_ready/<stage_id>
    - pipeline_loop_time_s{step=total,method=avg} -> <namespace>/loop/total/avg
    """
    # Remove "pipeline_" prefix for cleaner keys
    metric_name = name.removeprefix("pipeline_")

    # Handle stage-tagged metrics - use stages/<metric>/<stage> pattern
    stage = labels.get("stage")
    if stage:
        stage_id = _slugify_stage_name(stage, stage_aliases)

        # Handle sub-labels by appending them to metric name
        state = labels.get("state")
        resource = labels.get("resource")

        if state:
            # e.g., actor_count{state=ready} -> stages/actor_count_ready/<stage>
            return f"{namespace}/stages/{metric_name}_{state}/{stage_id}"
        elif resource:
            # e.g., resource_usage{resource=cpu} -> stages/resource_usage_cpu/<stage>
            return f"{namespace}/stages/{metric_name}_{resource}/{stage_id}"
        else:
            return f"{namespace}/stages/{metric_name}/{stage_id}"

    # Handle loop timing metrics
    step = labels.get("step")
    if step:
        method = labels.get("method", "avg")
        return f"{namespace}/loop/{step}/{method}"

    # Global metrics
    return f"{namespace}/{metric_name}"


# =============================================================================
# Ray Metrics Export Setup
# =============================================================================


def find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def enable_ray_metrics_export(port: int | None = None) -> int:
    """Enable Ray Prometheus metrics export.

    Must be called BEFORE Ray is initialized (before run_pipeline).

    Args:
        port: Port to use. If None, auto-selects a free port.

    Returns:
        The port that was configured
    """
    if port is None:
        port = find_free_port()

    os.environ["RAY_METRICS_PORT"] = str(port)
    return port


# =============================================================================
# Main Logger Class
# =============================================================================


class PrometheusMetricsLogger:
    """Background logger that scrapes Prometheus metrics and logs to W&B.

    This class runs a background thread that periodically:
    1. Fetches metrics from Ray's Prometheus endpoint
    2. Parses and transforms them
    3. Logs to W&B (if an active run exists)

    Example:
        # Enable metrics export before Ray init
        enable_ray_metrics_export(port=8080)

        # Use as context manager
        with PrometheusMetricsLogger(port=8080):
            pipelines_v1.run_pipeline(pipeline_spec)
    """

    def __init__(
        self,
        port: int = 8080,
        host: str = "127.0.0.1",
        poll_interval_s: float = 10.0,
        request_timeout_s: float = 5.0,
        wandb_namespace: str = "pretrain",  # Use pipeline_kind (e.g., "pretrain", "sft")
        stage_aliases: dict[str, str] | None = None,
    ):
        """Initialize the metrics logger.

        Args:
            port: Ray metrics export port
            host: Host to scrape
            poll_interval_s: Scrape interval in seconds
            request_timeout_s: HTTP timeout
            wandb_namespace: Prefix for W&B keys (e.g., "pretrain", "sft")
            stage_aliases: Optional stage name aliases
        """
        self.port = port
        self.host = host
        self.poll_interval_s = poll_interval_s
        self.request_timeout_s = request_timeout_s
        self.wandb_namespace = wandb_namespace
        self.stage_aliases = stage_aliases or {}

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time: float | None = None
        self._step = 0

    @property
    def metrics_url(self) -> str:
        """Full URL to the metrics endpoint."""
        return f"http://{self.host}:{self.port}/metrics"

    def _fetch_metrics_text(self) -> str | None:
        """Fetch raw metrics text from Prometheus endpoint."""
        try:
            with urllib.request.urlopen(
                self.metrics_url,
                timeout=self.request_timeout_s,
            ) as resp:
                return resp.read().decode("utf-8")
        except Exception:
            return None

    def _scrape_and_log(self) -> None:
        """Single scrape and log cycle."""
        text = self._fetch_metrics_text()
        if not text:
            return

        samples = parse_prometheus_text(text)
        metrics = transform_samples_to_wandb(
            samples,
            namespace=self.wandb_namespace,
            stage_aliases=self.stage_aliases,
        )

        if not metrics:
            return

        # Add metadata
        now = time.time()
        if self._start_time:
            metrics[f"{self.wandb_namespace}/elapsed_s"] = now - self._start_time
        metrics[f"{self.wandb_namespace}/step"] = self._step

        # Log to W&B
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(metrics, commit=False)
        except ImportError:
            pass
        except Exception:
            pass

        self._step += 1

    def _scrape_loop(self) -> None:
        """Background scraping loop."""
        self._start_time = time.time()

        # Wait a bit for Ray to initialize
        time.sleep(2.0)

        while not self._stop_event.is_set():
            try:
                self._scrape_and_log()
            except Exception:
                pass  # Never crash the scraper

            self._stop_event.wait(self.poll_interval_s)

    def start(self) -> None:
        """Start the background scraping thread."""
        if self._thread is not None:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._scrape_loop, daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop the background thread."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=timeout_s)
        self._thread = None

    def __enter__(self) -> "PrometheusMetricsLogger":
        """Context manager entry - starts the scraper."""
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit - stops the scraper."""
        self.stop()


# =============================================================================
# High-Level Integration
# =============================================================================


def create_prometheus_metrics_logger(
    config: PrometheusConfig | None = None,
) -> PrometheusMetricsLogger | None:
    """Create a Prometheus metrics logger if conditions are met.

    Returns None if:
    - W&B is not available/initialized
    - Metrics export is not enabled

    Args:
        config: Optional configuration. Uses defaults if not provided.

    Returns:
        PrometheusMetricsLogger instance or None
    """
    cfg = config or PrometheusConfig()

    # Check if W&B is available and has an active run
    try:
        import wandb

        if wandb.run is None:
            return None
    except ImportError:
        return None

    # Check if metrics port is configured
    port = cfg.port
    if port is None:
        port_str = os.environ.get("RAY_METRICS_PORT")
        if not port_str:
            return None
        try:
            port = int(port_str)
        except ValueError:
            return None

    return PrometheusMetricsLogger(
        port=port,
        host=cfg.host,
        poll_interval_s=cfg.poll_interval_s,
        request_timeout_s=cfg.request_timeout_s,
        wandb_namespace=cfg.wandb_namespace,
        stage_aliases=dict(cfg.stage_aliases),
    )
