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
Pipeline stats callback for W&B and JSONL logging.

This module provides a callback factory that creates a stats_callback function
for pipelines. The callback logs pipeline metrics to W&B and/or JSONL
in real-time during pipeline execution.

Usage:
    from nemotron.data_prep.observability.stats_callback import make_pipeline_stats_callback
    from nemotron.data_prep.config import ObservabilityConfig

    observability = ObservabilityConfig(
        wandb_log_pipeline_stats=True,
        pipeline_stats_jsonl_path="/tmp/pipeline_stats.jsonl",
    )

    callback = make_pipeline_stats_callback(
        observability=observability,
        pipeline_kind="pretrain",
        run_hash="abc123",
    )

    # Pass to PipelineConfig
    config = pipelines_v1.PipelineConfig(
        stats_callback=callback,
        logging_interval_s=30,
    )
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nemotron.data_prep.config import ObservabilityConfig
from nemotron.data_prep.observability.stage_keys import canonical_stage_id


def _sanitize_metric_name(name: str) -> str:
    """Sanitize a stage name for use as a metric key component.

    Uses canonical_stage_id for consistent snake_case naming across
    all observability modules (wandb_hook, prometheus_metrics, stats_callback).

    Examples:
        "Stage 02 - BinIdxTokenizationStage" -> "bin_idx_tokenization"
        "PlanStage" -> "plan"

    Note:
        This function now uses canonical_stage_id from stage_keys module
        for consistency. Legacy code using the old format (e.g., "binidxtokenization")
        may need to be updated if it relies on specific metric key shapes.
    """
    return canonical_stage_id(name)


def _extract_wandb_metrics(
    stats: Any,
    start_time: float,
    *,
    include_stage_metrics: bool = True,
) -> dict[str, float | int]:
    """Extract flat metrics from PipelineStats for W&B logging.

    Args:
        stats: PipelineStats object from cosmos-xenna
        start_time: Pipeline start timestamp for duration calculation
        include_stage_metrics: If True, include per-stage metrics (pipeline/stage/{stage}/...).
            Set to False when using consolidated charts (WandbStatsHook) to avoid
            creating separate WandB charts for each stage.

    Returns:
        Dictionary of metric_name -> value suitable for wandb.log()
    """
    metrics: dict[str, float | int] = {}
    now = time.time()

    # Pipeline-level metrics
    metrics["pipeline/pipeline_duration_s"] = now - start_time

    if hasattr(stats, "inputs_processed_per_second") and stats.inputs_processed_per_second is not None:
        metrics["pipeline/inputs_processed_per_s"] = stats.inputs_processed_per_second

    if hasattr(stats, "outputs_per_second") and stats.outputs_per_second is not None:
        metrics["pipeline/outputs_per_s"] = stats.outputs_per_second

    if hasattr(stats, "num_input_tasks_remaining") and stats.num_input_tasks_remaining is not None:
        metrics["pipeline/num_input_remaining"] = stats.num_input_tasks_remaining

    if hasattr(stats, "num_outputs") and stats.num_outputs is not None:
        metrics["pipeline/num_outputs"] = stats.num_outputs

    if hasattr(stats, "main_loop_rate_hz") and stats.main_loop_rate_hz is not None:
        metrics["pipeline/main_loop_rate_hz"] = stats.main_loop_rate_hz

    # Cluster-level metrics
    if hasattr(stats, "cluster_info") and stats.cluster_info is not None:
        ci = stats.cluster_info
        if hasattr(ci, "total") and ci.total is not None:
            t = ci.total
            if hasattr(t, "cpus") and t.cpus is not None:
                metrics["pipeline/cluster/total_cpus"] = t.cpus
            if hasattr(t, "gpus") and t.gpus is not None:
                metrics["pipeline/cluster/total_gpus"] = t.gpus
            if hasattr(t, "memory_gb") and t.memory_gb is not None:
                metrics["pipeline/cluster/total_mem_gb"] = t.memory_gb
            if hasattr(t, "object_store_gb") and t.object_store_gb is not None:
                metrics["pipeline/cluster/total_obj_store_gb"] = t.object_store_gb

        if hasattr(ci, "available") and ci.available is not None:
            a = ci.available
            if hasattr(a, "cpus") and a.cpus is not None:
                metrics["pipeline/cluster/avail_cpus"] = a.cpus
            if hasattr(a, "gpus") and a.gpus is not None:
                metrics["pipeline/cluster/avail_gpus"] = a.gpus
            if hasattr(a, "memory_gb") and a.memory_gb is not None:
                metrics["pipeline/cluster/avail_mem_gb"] = a.memory_gb
            if hasattr(a, "object_store_gb") and a.object_store_gb is not None:
                metrics["pipeline/cluster/avail_obj_store_gb"] = a.object_store_gb

    # Per-stage metrics from actor_pools
    # Only include these when include_stage_metrics is True; otherwise WandbStatsHook
    # provides consolidated line_series charts that avoid per-stage chart proliferation
    # Note: actor_pools is a list of ActorPoolStats objects (not a dict)
    if include_stage_metrics and hasattr(stats, "actor_pools") and stats.actor_pools:
        for pool_stats in stats.actor_pools:
            stage_key = _sanitize_metric_name(pool_stats.name)
            prefix = f"pipeline/stage/{stage_key}"

            # Actor stats
            if hasattr(pool_stats, "actor_stats") and pool_stats.actor_stats is not None:
                ast = pool_stats.actor_stats
                if hasattr(ast, "target") and ast.target is not None:
                    metrics[f"{prefix}/actors/target"] = ast.target
                if hasattr(ast, "pending") and ast.pending is not None:
                    metrics[f"{prefix}/actors/pending"] = ast.pending
                if hasattr(ast, "ready") and ast.ready is not None:
                    metrics[f"{prefix}/actors/ready"] = ast.ready
                if hasattr(ast, "running") and ast.running is not None:
                    metrics[f"{prefix}/actors/running"] = ast.running
                if hasattr(ast, "idle") and ast.idle is not None:
                    metrics[f"{prefix}/actors/idle"] = ast.idle

            # Task stats
            if hasattr(pool_stats, "task_stats") and pool_stats.task_stats is not None:
                ts = pool_stats.task_stats
                if hasattr(ts, "total_completed") and ts.total_completed is not None:
                    metrics[f"{prefix}/tasks/completed"] = ts.total_completed
                if hasattr(ts, "input_queue_size") and ts.input_queue_size is not None:
                    metrics[f"{prefix}/queue/input_size"] = ts.input_queue_size
                if hasattr(ts, "output_queue_size") and ts.output_queue_size is not None:
                    metrics[f"{prefix}/queue/output_size"] = ts.output_queue_size

            # Slot stats
            if hasattr(pool_stats, "slot_stats") and pool_stats.slot_stats is not None:
                ss = pool_stats.slot_stats
                if hasattr(ss, "num_used") and ss.num_used is not None:
                    metrics[f"{prefix}/slots/used"] = ss.num_used
                if hasattr(ss, "num_empty") and ss.num_empty is not None:
                    metrics[f"{prefix}/slots/empty"] = ss.num_empty

            # Processing speed
            if (
                hasattr(pool_stats, "processing_speed_tasks_per_second")
                and pool_stats.processing_speed_tasks_per_second is not None
            ):
                metrics[f"{prefix}/speed/tasks_per_s"] = pool_stats.processing_speed_tasks_per_second

    # Per-stage resource usage (if available)
    # Only include when include_stage_metrics is True
    if include_stage_metrics and hasattr(stats, "resource_usage_per_stage") and stats.resource_usage_per_stage:
        for stage_name, usage in stats.resource_usage_per_stage.items():
            stage_key = _sanitize_metric_name(stage_name)
            prefix = f"pipeline/stage/{stage_key}"

            if hasattr(usage, "cpu_utilization") and usage.cpu_utilization is not None:
                metrics[f"{prefix}/cpu_util_pct"] = usage.cpu_utilization
            if hasattr(usage, "memory_usage") and usage.memory_usage is not None:
                # Convert bytes to GB
                metrics[f"{prefix}/mem_gb"] = usage.memory_usage / (1024**3)
            if hasattr(usage, "actor_count") and usage.actor_count is not None:
                metrics[f"{prefix}/actor_count"] = usage.actor_count

    return metrics


def _extract_jsonl_record(
    stats: Any,
    start_time: float,
    pipeline_kind: str,
    run_hash: str | None,
    run_dir: str | None,
    dataset_names: list[str] | None,
) -> dict[str, Any]:
    """Extract a structured record from PipelineStats for JSONL logging.

    Args:
        stats: PipelineStats object from cosmos-xenna
        start_time: Pipeline start timestamp
        pipeline_kind: Type of pipeline (e.g., "pretrain", "sft")
        run_hash: Unique run identifier
        run_dir: Run directory path
        dataset_names: List of dataset names being processed

    Returns:
        Dictionary suitable for JSON serialization
    """
    now = time.time()

    record: dict[str, Any] = {
        "schema_version": 1,
        "timestamp": now,
        "pipeline_kind": pipeline_kind,
        "run_hash": run_hash,
        "run_dir": run_dir,
        "dataset_names": dataset_names,
    }

    # Pipeline-level stats
    pipeline_stats: dict[str, Any] = {
        "pipeline_duration_s": now - start_time,
    }

    for attr in [
        "inputs_processed_per_second",
        "outputs_per_second",
        "num_initial_input_tasks",
        "num_input_tasks_remaining",
        "num_outputs",
        "main_loop_rate_hz",
    ]:
        if hasattr(stats, attr):
            val = getattr(stats, attr)
            if val is not None:
                pipeline_stats[attr] = val

    record["pipeline"] = pipeline_stats

    # Cluster info
    if hasattr(stats, "cluster_info") and stats.cluster_info is not None:
        ci = stats.cluster_info
        cluster: dict[str, Any] = {}

        if hasattr(ci, "total") and ci.total is not None:
            cluster["total"] = {
                "cpus": getattr(ci.total, "cpus", None),
                "gpus": getattr(ci.total, "gpus", None),
                "memory_gb": getattr(ci.total, "memory_gb", None),
                "object_store_gb": getattr(ci.total, "object_store_gb", None),
            }

        if hasattr(ci, "available") and ci.available is not None:
            cluster["available"] = {
                "cpus": getattr(ci.available, "cpus", None),
                "gpus": getattr(ci.available, "gpus", None),
                "memory_gb": getattr(ci.available, "memory_gb", None),
                "object_store_gb": getattr(ci.available, "object_store_gb", None),
            }

        record["cluster"] = cluster

    # Per-stage stats
    # Note: actor_pools is a list of ActorPoolStats objects (not a dict)
    stages: list[dict[str, Any]] = []
    if hasattr(stats, "actor_pools") and stats.actor_pools:
        for pool_stats in stats.actor_pools:
            stage_record: dict[str, Any] = {"name": pool_stats.name}

            if hasattr(pool_stats, "actor_stats") and pool_stats.actor_stats is not None:
                ast = pool_stats.actor_stats
                stage_record["actors"] = {
                    "target": getattr(ast, "target", None),
                    "pending": getattr(ast, "pending", None),
                    "ready": getattr(ast, "ready", None),
                    "running": getattr(ast, "running", None),
                    "idle": getattr(ast, "idle", None),
                }

            if hasattr(pool_stats, "task_stats") and pool_stats.task_stats is not None:
                ts = pool_stats.task_stats
                stage_record["tasks"] = {
                    "total_completed": getattr(ts, "total_completed", None),
                    "total_returned_none": getattr(ts, "total_returned_none", None),
                    "total_dynamically_spawned": getattr(ts, "total_dynamically_spawned", None),
                }

            if hasattr(pool_stats, "task_stats") and pool_stats.task_stats is not None:
                ts = pool_stats.task_stats
                stage_record["queue"] = {
                    "input_size": getattr(ts, "input_queue_size", None),
                    "output_size": getattr(ts, "output_queue_size", None),
                }

            if hasattr(pool_stats, "slot_stats") and pool_stats.slot_stats is not None:
                ss = pool_stats.slot_stats
                stage_record["slots"] = {
                    "num_used": getattr(ss, "num_used", None),
                    "num_empty": getattr(ss, "num_empty", None),
                }

            if (
                hasattr(pool_stats, "processing_speed_tasks_per_second")
                and pool_stats.processing_speed_tasks_per_second is not None
            ):
                stage_record["speed_tasks_per_second"] = pool_stats.processing_speed_tasks_per_second

            stages.append(stage_record)

    record["stages"] = stages

    # Resource usage per stage
    if hasattr(stats, "resource_usage_per_stage") and stats.resource_usage_per_stage:
        resource_usage: dict[str, Any] = {}
        for stage_name, usage in stats.resource_usage_per_stage.items():
            resource_usage[stage_name] = {
                "cpu_utilization": getattr(usage, "cpu_utilization", None),
                "memory_bytes": getattr(usage, "memory_usage", None),
                "actor_count": getattr(usage, "actor_count", None),
            }
        record["resource_usage_per_stage"] = resource_usage

    return record


def make_pipeline_stats_callback(
    *,
    observability: ObservabilityConfig,
    pipeline_kind: str,
    run_hash: str | None = None,
    run_dir: str | None = None,
    dataset_names: list[str] | None = None,
) -> Callable[[Any], None] | None:
    """Create a stats callback for pipelines.

    This factory creates a callback function that will be invoked by the
    PipelineMonitor at regular intervals (controlled by logging_interval_s).
    The callback logs pipeline metrics to W&B and/or a JSONL file.

    Args:
        observability: Observability configuration controlling what to log
        pipeline_kind: Type of pipeline ("pretrain", "sft", etc.) for labeling
        run_hash: Unique run identifier for labeling
        run_dir: Run directory path for labeling
        dataset_names: List of dataset names being processed

    Returns:
        A callback function accepting PipelineStats, or None if no logging is enabled.

    Note:
        The callback is designed to be deepcopy-safe (no captured file handles or
        non-picklable objects until first invocation). It uses lazy initialization
        to avoid issues with PipelineSpec deep-copy behavior.
    """
    # Return None if nothing to log (avoid overhead)
    if not observability.wandb_log_pipeline_stats and observability.pipeline_stats_jsonl_path is None:
        return None

    # Use a mutable container for lazy-initialized state
    # This avoids capturing file handles that can't be deep-copied
    _state: dict[str, Any] = {
        "initialized": False,
        "start_time": None,
        "jsonl_file": None,
    }

    def _ensure_initialized() -> None:
        """Lazy initialization on first callback invocation."""
        if _state["initialized"]:
            return

        _state["initialized"] = True
        _state["start_time"] = time.time()

        # Open JSONL file if configured
        jsonl_path = observability.pipeline_stats_jsonl_path
        if jsonl_path:
            try:
                path = Path(jsonl_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                _state["jsonl_file"] = open(path, "a")  # noqa: SIM115
            except Exception:
                # Don't crash pipeline on file errors
                pass

    def callback(stats: Any) -> None:
        """Stats callback invoked by PipelineMonitor.

        Args:
            stats: PipelineStats object from cosmos-xenna
        """
        try:
            _ensure_initialized()
            start_time = _state["start_time"] or time.time()

            # Log to W&B if enabled
            if observability.wandb_log_pipeline_stats:
                try:
                    import wandb

                    if wandb.run is not None:
                        # Skip per-stage metrics when consolidated charts are enabled
                        # (WandbStatsHook provides line_series charts for stage metrics)
                        include_stage = not observability.wandb_consolidated_charts_only
                        metrics = _extract_wandb_metrics(
                            stats, start_time, include_stage_metrics=include_stage
                        )
                        wandb.log(metrics, commit=False)
                except ImportError:
                    pass  # wandb not installed
                except Exception:
                    pass  # Don't crash pipeline on W&B errors

            # Write to JSONL if enabled
            jsonl_file = _state.get("jsonl_file")
            if jsonl_file is not None:
                try:
                    record = _extract_jsonl_record(
                        stats,
                        start_time,
                        pipeline_kind,
                        run_hash,
                        run_dir,
                        dataset_names,
                    )
                    jsonl_file.write(json.dumps(record) + "\n")
                    jsonl_file.flush()
                except Exception:
                    pass  # Don't crash pipeline on file errors

        except Exception:
            # Never let the callback crash the pipeline
            pass

    return callback
