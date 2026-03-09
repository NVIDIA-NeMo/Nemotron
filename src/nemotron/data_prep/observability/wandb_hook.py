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

"""Real-time W&B logging for pipelines via monkey-patching.

This module provides a context manager that patches PipelineMonitor._make_stats
to intercept PipelineStats objects and log metrics to W&B in real-time.

Usage:
    from nemotron.data_prep.observability import make_wandb_stats_hook

    hook = make_wandb_stats_hook(
        observability=observability_cfg,
        pipeline_kind="pretrain",
    )

    if hook:
        with hook:
            pipelines_v1.run_pipeline(pipeline_spec)
    else:
        pipelines_v1.run_pipeline(pipeline_spec)

Why monkey-patching?
    - PipelineMonitor.update() builds PipelineStats via _make_stats()
    - By patching _make_stats, we intercept stats at the same frequency as pipeline logging
    - No changes to cosmos-xenna required
    - Works with both streaming and batch pipelines
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from nemotron.data_prep.config import DatasetConfig, ObservabilityConfig
from nemotron.data_prep.observability.stage_keys import canonical_stage_id, get_stage_display_name
from nemotron.data_prep.utils.discovery import get_dataset_metadata
from nemotron.data_prep.utils.size import format_byte_size, format_count

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Stage-level metrics that should be consolidated into multi-line charts
# Each metric will become one chart with a line per stage
STAGE_METRICS = [
    "tasks_completed",
    "queue_in",
    "queue_out",
    "slots_used",
    "slots_empty",
    "actors_target",
    "actors_pending",
    "actors_ready",
    "actors_running",
    "actors_idle",
    "speed_tasks_per_s",
    "resource_cpu_util_pct",
    "resource_mem_gb",
    "resource_actor_count",
]

# Module-level state for safe patching
_patch_lock = threading.RLock()
_patch_depth = 0
_original_make_stats: Callable | None = None
_active_hooks: list["WandbStatsHook"] = []


def _get_dataset_size_str(item: Any) -> str:
    """Get human-readable dataset size for plan table.

    For HuggingFace datasets (hf://...), fetches size from HF API.
    For other sources, returns "-" (size will be shown in progress table
    once the plan is created).

    Args:
        item: DatasetWorkItem or SftDatasetWorkItem

    Returns:
        Human-readable size string (e.g., "1.5 GB") or "-" if unavailable.
    """
    path = getattr(item, "path", "") or ""
    if not path.startswith("hf://"):
        return "-"

    try:
        cfg = DatasetConfig(
            name=getattr(item, "dataset_name", "unknown"),
            path=path,
            weight=getattr(item, "weight", 0.0),
            text_field="text",
            split=getattr(item, "split", None),
            subset=getattr(item, "subset", None),
        )
        md = get_dataset_metadata(cfg)
        if md.size_str:
            return md.size_str
        if md.size_bytes is not None:
            return format_byte_size(int(md.size_bytes))
    except Exception:
        return "-"

    return "-"


def compute_dataset_input_bytes(dataset_items: list[Any]) -> dict[str, int]:
    """Pre-compute total input bytes per dataset from file discovery.

    Uses discover_input_files (HF Hub API / fsspec) to get actual file sizes.
    This gives accurate per-file sizes from sibling.size (HF) or fs.info (local/S3).

    When max_rows is set on a dataset item, estimates effective bytes using
    HF dataset metadata (bytes-per-row) to show realistic sizes instead of
    full dataset sizes.

    Used to seed the progress table's size column before plan.json exists.
    Once the plan stage writes plan.json, the progress table reads the exact
    file-level sizes from there instead.

    Args:
        dataset_items: List of DatasetWorkItem or SftDatasetWorkItem objects.

    Returns:
        Dict mapping dataset_name to total input bytes from discovered files.
    """
    from nemotron.data_prep.utils.discovery import discover_input_files

    result: dict[str, int] = {}
    for item in dataset_items:
        path = getattr(item, "path", "") or ""
        try:
            cfg = DatasetConfig(
                name=getattr(item, "dataset_name", "unknown"),
                path=path,
                weight=getattr(item, "weight", 0.0),
                text_field=getattr(item, "text_field", "text"),
                split=getattr(item, "split", None),
                subset=getattr(item, "subset", None),
            )

            # For HF datasets, discover_input_files calls discover_hf_files
            # which uses api.dataset_info(files_metadata=True) → sibling.size.
            # For local/S3/GCS, it uses fsspec info() for real file sizes.
            if path.startswith("hf://"):
                files = discover_input_files(cfg, fs=None)  # type: ignore[arg-type]  # HF path doesn't use fs
            else:
                from nemotron.data_prep.utils.filesystem import get_filesystem
                fs, _ = get_filesystem(path)
                files = discover_input_files(cfg, fs)

            total_bytes = sum(f.size for f in files)

            # When max_rows is set, estimate effective bytes from HF metadata
            max_rows = getattr(item, "max_rows", None)
            if max_rows and max_rows > 0 and total_bytes > 0 and path.startswith("hf://"):
                try:
                    md = get_dataset_metadata(cfg)
                    if md.num_rows and md.size_bytes and md.num_rows > 0 and md.size_bytes > 0:
                        avg_bpr = md.size_bytes / md.num_rows
                        effective = int(max_rows * avg_bpr)
                        total_bytes = min(effective, total_bytes)
                except Exception:
                    pass

            if total_bytes > 0:
                result[item.dataset_name] = total_bytes
        except Exception:
            pass
    return result


def _extract_stage_metrics(stats: Any) -> dict[str, dict[str, float | int]]:
    """Extract per-stage metrics organized by metric name.

    Returns a dict where keys are metric names (e.g., "tasks_completed")
    and values are dicts mapping stage_id -> value.

    Example return value:
        {
            "tasks_completed": {"plan": 100, "download": 50, "bin_idx_tokenization": 25},
            "queue_in": {"plan": 0, "download": 10, "bin_idx_tokenization": 5},
            ...
        }
    """
    metrics_by_name: dict[str, dict[str, float | int]] = {}

    # Per-stage metrics from actor_pools
    if hasattr(stats, "actor_pools") and stats.actor_pools is not None:
        for pool in stats.actor_pools:
            stage_id = canonical_stage_id(pool.name)

            # Actor stats
            if hasattr(pool, "actor_stats") and pool.actor_stats is not None:
                a = pool.actor_stats
                if hasattr(a, "target"):
                    metrics_by_name.setdefault("actors_target", {})[stage_id] = a.target
                if hasattr(a, "pending"):
                    metrics_by_name.setdefault("actors_pending", {})[stage_id] = a.pending
                if hasattr(a, "ready"):
                    metrics_by_name.setdefault("actors_ready", {})[stage_id] = a.ready
                if hasattr(a, "running"):
                    metrics_by_name.setdefault("actors_running", {})[stage_id] = a.running
                if hasattr(a, "idle"):
                    metrics_by_name.setdefault("actors_idle", {})[stage_id] = a.idle

            # Task stats
            if hasattr(pool, "task_stats") and pool.task_stats is not None:
                t = pool.task_stats
                if hasattr(t, "total_completed"):
                    metrics_by_name.setdefault("tasks_completed", {})[stage_id] = t.total_completed
                if hasattr(t, "input_queue_size"):
                    metrics_by_name.setdefault("queue_in", {})[stage_id] = t.input_queue_size
                if hasattr(t, "output_queue_size"):
                    metrics_by_name.setdefault("queue_out", {})[stage_id] = t.output_queue_size

            # Slot stats
            if hasattr(pool, "slot_stats") and pool.slot_stats is not None:
                s = pool.slot_stats
                if hasattr(s, "num_used"):
                    metrics_by_name.setdefault("slots_used", {})[stage_id] = s.num_used
                if hasattr(s, "num_empty"):
                    metrics_by_name.setdefault("slots_empty", {})[stage_id] = s.num_empty

            # Processing speed
            if hasattr(pool, "processing_speed_tasks_per_second"):
                speed = pool.processing_speed_tasks_per_second
                if speed is not None:
                    metrics_by_name.setdefault("speed_tasks_per_s", {})[stage_id] = speed

    # Per-stage resource usage
    if hasattr(stats, "resource_usage_per_stage") and stats.resource_usage_per_stage:
        for stage_name, usage in stats.resource_usage_per_stage.items():
            stage_id = canonical_stage_id(stage_name)

            if hasattr(usage, "cpu_utilization"):
                metrics_by_name.setdefault("resource_cpu_util_pct", {})[stage_id] = usage.cpu_utilization
            if hasattr(usage, "memory_usage"):
                metrics_by_name.setdefault("resource_mem_gb", {})[stage_id] = usage.memory_usage / 1e9
            if hasattr(usage, "actor_count"):
                metrics_by_name.setdefault("resource_actor_count", {})[stage_id] = usage.actor_count

    return metrics_by_name


def _flatten_pipeline_stats(stats: Any, *, namespace: str = "") -> dict[str, float | int]:
    """Flatten PipelineStats into a dict suitable for JSONL logging.

    Args:
        stats: A PipelineStats object from cosmos-xenna
        namespace: Prefix for all metric keys (default: empty for JSONL)

    Returns:
        Dict mapping metric names to numeric values
    """
    metrics: dict[str, float | int] = {}
    ns = namespace

    # Pipeline-level metrics
    if hasattr(stats, "pipeline_duration_s"):
        metrics[f"{ns}/pipeline_duration_s"] = stats.pipeline_duration_s
    if hasattr(stats, "main_loop_rate_hz"):
        metrics[f"{ns}/main_loop_rate_hz"] = stats.main_loop_rate_hz
    if hasattr(stats, "num_input_tasks_remaining"):
        metrics[f"{ns}/num_input_remaining"] = stats.num_input_tasks_remaining
    if hasattr(stats, "num_initial_input_tasks"):
        metrics[f"{ns}/num_initial_inputs"] = stats.num_initial_input_tasks
    if hasattr(stats, "num_outputs"):
        metrics[f"{ns}/num_outputs"] = stats.num_outputs

    # Computed rates (properties on PipelineStats)
    if hasattr(stats, "inputs_processed_per_second"):
        metrics[f"{ns}/inputs_processed_per_s"] = stats.inputs_processed_per_second
    if hasattr(stats, "outputs_per_second"):
        metrics[f"{ns}/outputs_per_s"] = stats.outputs_per_second

    # Progress (percentage of inputs processed)
    if hasattr(stats, "num_initial_input_tasks") and hasattr(stats, "num_input_tasks_remaining"):
        initial = stats.num_initial_input_tasks
        remaining = stats.num_input_tasks_remaining
        if initial > 0:
            metrics[f"{ns}/progress"] = (initial - remaining) / initial * 100.0

    # Cluster resources
    if hasattr(stats, "cluster") and stats.cluster is not None:
        cluster = stats.cluster

        if hasattr(cluster, "total") and cluster.total is not None:
            total = cluster.total
            if hasattr(total, "num_cpus"):
                metrics[f"{ns}/cluster/total_cpus"] = total.num_cpus
            if hasattr(total, "num_gpus"):
                metrics[f"{ns}/cluster/total_gpus"] = total.num_gpus
            if hasattr(total, "memory"):
                metrics[f"{ns}/cluster/total_mem_gb"] = total.memory / 1e9
            if hasattr(total, "object_store_memory"):
                metrics[f"{ns}/cluster/total_obj_store_gb"] = total.object_store_memory / 1e9

        if hasattr(cluster, "available") and cluster.available is not None:
            avail = cluster.available
            if hasattr(avail, "num_cpus"):
                metrics[f"{ns}/cluster/avail_cpus"] = avail.num_cpus
            if hasattr(avail, "num_gpus"):
                metrics[f"{ns}/cluster/avail_gpus"] = avail.num_gpus
            if hasattr(avail, "memory"):
                metrics[f"{ns}/cluster/avail_mem_gb"] = avail.memory / 1e9
            if hasattr(avail, "object_store_memory"):
                metrics[f"{ns}/cluster/avail_obj_store_gb"] = avail.object_store_memory / 1e9

    # Per-stage metrics using consolidated format: stages/<metric>/<stage>
    # This produces fewer W&B charts (one per metric) with lines for each stage
    stage_metrics = _extract_stage_metrics(stats)
    for metric_name, stage_values in stage_metrics.items():
        for stage_id, value in stage_values.items():
            metrics[f"{ns}/stages/{metric_name}/{stage_id}"] = value

    return metrics


def _make_jsonl_record(stats: Any, *, pipeline_kind: str, run_hash: str | None) -> dict[str, Any]:
    """Create a JSONL record from PipelineStats.

    Args:
        stats: A PipelineStats object
        pipeline_kind: Type of pipeline (e.g., "pretrain", "sft")
        run_hash: Unique run identifier

    Returns:
        Dict suitable for writing as a JSONL line
    """
    record: dict[str, Any] = {
        "timestamp": time.time(),
        "pipeline_kind": pipeline_kind,
        "run_hash": run_hash,
    }

    # Add flattened metrics
    metrics = _flatten_pipeline_stats(stats, namespace="")
    record["metrics"] = metrics

    # Add stage names for reference
    if hasattr(stats, "actor_pools") and stats.actor_pools:
        record["stages"] = [pool.name for pool in stats.actor_pools]

    return record


class WandbStatsHook:
    """Context manager that patches PipelineMonitor._make_stats for W&B logging.

    This hook intercepts PipelineStats objects and logs them to W&B in real-time.
    It's safe to use in nested contexts (reference counted) and is thread-safe.

    Args:
        observability: Configuration for what to log
        pipeline_kind: Type of pipeline (e.g., "pretrain", "sft")
        run_hash: Unique run identifier for JSONL records
        run_dir: Directory for JSONL output file
        dataset_names: Names of datasets being processed
        dataset_num_shards: Dict mapping dataset name to expected number of shards
        wandb_namespace: Prefix for W&B metric keys (default: pipeline_kind, e.g. "pretrain")
        monitor_cls: PipelineMonitor class to patch (for testing)

    Example:
        hook = WandbStatsHook(
            observability=observability_cfg,
            pipeline_kind="pretrain",
        )
        with hook:
            pipelines_v1.run_pipeline(pipeline_spec)
    """

    def __init__(
        self,
        *,
        observability: ObservabilityConfig,
        pipeline_kind: str,
        run_hash: str | None = None,
        run_dir: str | None = None,
        dataset_names: list[str] | None = None,
        dataset_num_shards: dict[str, int] | None = None,
        dataset_input_bytes: dict[str, int] | None = None,
        dataset_max_rows: dict[str, int] | None = None,
        wandb_namespace: str | None = None,
        monitor_cls: type | None = None,
    ) -> None:
        self._observability = observability
        self._pipeline_kind = pipeline_kind
        self._run_hash = run_hash
        self._run_dir = run_dir
        self._dataset_names = dataset_names or []
        self._dataset_num_shards = dataset_num_shards or {}
        # Use pipeline_kind as namespace by default (e.g., "pretrain", "sft")
        self._wandb_namespace = wandb_namespace if wandb_namespace is not None else pipeline_kind
        self._monitor_cls = monitor_cls
        # Per-dataset max_rows for effective size estimation
        self._dataset_max_rows = dataset_max_rows or {}

        # JSONL file handle (lazy opened)
        self._jsonl_file: Any = None
        self._jsonl_path: Path | None = None
        if observability.pipeline_stats_jsonl_path:
            self._jsonl_path = Path(observability.pipeline_stats_jsonl_path)

        # Track if we logged anything
        self._log_count = 0

        # Progress table tracking
        self._last_progress_table_time: float = 0.0
        self._last_stage_table_time: float = 0.0
        self._fs: Any = None  # Lazy-loaded filesystem
        # Cache for dataset sizes: pre-seeded from discovery, updated from plan.json
        self._dataset_input_bytes: dict[str, int] = dict(dataset_input_bytes) if dataset_input_bytes else {}

        # Step counter for W&B logging
        self._step: int = 0

        # Accumulated history for final consolidated line_series charts
        # These are logged once at the end of the pipeline run
        self._metric_history: dict[str, dict[str, list[float]]] = {}
        self._step_history: list[int] = []

    def _get_monitor_class(self) -> type:
        """Get the PipelineMonitor class to patch."""
        if self._monitor_cls is not None:
            return self._monitor_cls

        # Lazy import to avoid coupling at import time
        from cosmos_xenna.pipelines.private.monitoring import PipelineMonitor

        return PipelineMonitor

    def _create_wrapper(self, original: Callable) -> Callable:
        """Create a wrapper function that logs stats after calling original."""
        hook = self  # Capture self for the wrapper

        def wrapper(monitor_self: Any, input_len: int, ext_output_lens: list[int], task_metadata_per_pool: list) -> Any:
            # Call original _make_stats
            stats = original(monitor_self, input_len, ext_output_lens, task_metadata_per_pool)

            # Log to all active hooks
            for active_hook in _active_hooks:
                try:
                    active_hook._on_stats(stats)
                except Exception as e:
                    logger.warning(f"Error in W&B stats hook: {e}")

            return stats

        return wrapper

    def _on_stats(self, stats: Any) -> None:
        """Called when new PipelineStats are available."""
        self._log_count += 1

        # Accumulate stage metrics for final consolidated charts
        self._accumulate_stage_metrics(stats)

        # Log to JSONL
        if self._jsonl_path is not None:
            try:
                if self._jsonl_file is None:
                    self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                    self._jsonl_file = open(self._jsonl_path, "a")

                record = _make_jsonl_record(
                    stats,
                    pipeline_kind=self._pipeline_kind,
                    run_hash=self._run_hash,
                )
                self._jsonl_file.write(json.dumps(record) + "\n")
                self._jsonl_file.flush()
            except Exception as e:
                logger.debug(f"JSONL logging error: {e}")

        # Update progress table periodically
        if self._observability.wandb_log_progress_table and self._run_dir:
            now = time.time()
            interval = self._observability.wandb_progress_table_interval_s
            if now - self._last_progress_table_time >= interval:
                self._update_progress_table()
                self._last_progress_table_time = now

        # Update stage overview table periodically
        if self._observability.wandb_log_stage_table:
            now = time.time()
            interval = self._observability.wandb_stage_table_interval_s
            if now - self._last_stage_table_time >= interval:
                self._update_stage_overview_table(stats)
                self._last_stage_table_time = now

    def _accumulate_stage_metrics(self, stats: Any) -> None:
        """Accumulate stage metrics for final consolidated charts.

        This method collects per-stage metrics during pipeline execution.
        The accumulated history is used to create line_series charts at the
        end of the pipeline run (in _log_final_consolidated_charts).

        No metrics are logged to W&B during execution - only the progress table
        and stage overview table provide real-time visibility. The consolidated
        charts are logged once at the end for historical analysis.
        """
        # Extract stage metrics organized by metric name
        stage_metrics = _extract_stage_metrics(stats)

        # Accumulate per-stage metrics for final line_series charts
        self._step_history.append(self._step)
        for metric_name, stage_values in stage_metrics.items():
            if metric_name not in self._metric_history:
                self._metric_history[metric_name] = {}
            for stage_id, value in stage_values.items():
                if stage_id not in self._metric_history[metric_name]:
                    self._metric_history[metric_name][stage_id] = []
                self._metric_history[metric_name][stage_id].append(float(value))

        # Increment step
        self._step += 1

    def _get_filesystem(self) -> Any:
        """Get filesystem handle (lazy-loaded)."""
        if self._fs is None:
            from nemotron.data_prep.utils.filesystem import get_filesystem

            self._fs, _ = get_filesystem(self._run_dir)
        return self._fs

    def _estimate_effective_bytes(
        self, dataset_name: str, total_bytes: int, max_rows: int
    ) -> int:
        """Estimate effective bytes when max_rows limits processing.

        Uses HF dataset metadata to compute bytes-per-row and estimates
        how many bytes max_rows would cover. Falls back to total_bytes
        if metadata is unavailable.
        """
        try:
            # Find the dataset item's HF metadata for row count estimation
            from nemotron.data_prep.utils.discovery import fetch_hf_dataset_metadata

            # Try to get dataset metadata from plan.json assignments
            fs = self._get_filesystem()
            dataset_base = f"{self._run_dir}/datasets/{dataset_name}"
            subdirs = [p for p in fs.ls(dataset_base) if fs.isdir(p)]
            for subdir in subdirs:
                plan_path = f"{subdir}/plan.json"
                if not fs.exists(plan_path):
                    continue

                from nemotron.data_prep.utils.filesystem import read_json
                plan_data = read_json(fs, plan_path)
                file_assignments = plan_data.get("file_assignments") or []
                if not file_assignments:
                    break

                # Get hf_subset/hf_split from the first assignment (added by PlanStage)
                first_assignment = file_assignments[0]
                first_files = first_assignment.get("files", [])
                if not first_files:
                    break

                first_file = first_files[0]
                repo_id = first_file.get("hf_repo_id")
                if not repo_id:
                    break

                hf_subset = first_assignment.get("hf_subset")
                hf_split = first_assignment.get("hf_split")

                meta = fetch_hf_dataset_metadata(repo_id, subset=hf_subset, split=hf_split)
                if meta.num_rows and meta.size_bytes and meta.num_rows > 0 and meta.size_bytes > 0:
                    avg_bpr = meta.size_bytes / meta.num_rows
                    effective = int(max_rows * avg_bpr)
                    return min(effective, total_bytes)

                break
        except Exception:
            pass

        return total_bytes

    def _update_progress_table(self) -> None:
        """Scan receipts and log per-dataset progress table to W&B.

        The table shows:
        - dataset_name: Name of the dataset
        - size: Input data size from plan.json (e.g., "1.5GB")
        - total_shards: Expected number of shards for the dataset
        - downloaded: "X (Y%)" - shards downloaded and started processing
        - processed: "X (Y%)" - shards fully processed
        - total_tokens: Cumulative tokens from completed shards
        - total_sequences: Cumulative sequences from completed shards
        - status: Overall dataset status (pending/in_progress/completed)
        """
        if not self._dataset_names or not self._run_dir:
            return

        try:
            import wandb

            if wandb.run is None:
                return

            fs = self._get_filesystem()

            # Build progress data for each dataset
            columns = [
                "dataset_name",
                "size",
                "total_shards",
                "downloaded",
                "processed",
                "total_tokens",
                "total_sequences",
                "status",
            ]
            data = []

            for dataset_name in self._dataset_names:
                # Find plan_hash by scanning dataset directory
                dataset_base = f"{self._run_dir}/datasets/{dataset_name}"
                plan_hash = None
                total_shards = self._dataset_num_shards.get(dataset_name, 0)

                try:
                    if fs.exists(dataset_base):
                        subdirs = [p for p in fs.ls(dataset_base) if fs.isdir(p)]
                        for subdir in subdirs:
                            plan_path = f"{subdir}/plan.json"
                            if fs.exists(plan_path):
                                plan_hash = subdir.split("/")[-1]
                                break
                except Exception:
                    pass

                # Get dataset size: prefer plan.json (exact file-level sizes),
                # fall back to pre-seeded HF metadata (available before plan exists).
                # When max_rows is set, estimate effective size from row limit.
                dataset_size_str = "-"
                if plan_hash:
                    # Plan exists — read exact total_bytes from file assignments
                    plan_path = f"{dataset_base}/{plan_hash}/plan.json"
                    if dataset_name not in self._dataset_input_bytes or self._dataset_input_bytes[dataset_name] == 0:
                        try:
                            from nemotron.data_prep.utils.filesystem import read_json

                            plan_data = read_json(fs, plan_path)
                            file_assignments = plan_data.get("file_assignments") or []
                            plan_bytes = sum(
                                int(fa.get("total_bytes") or 0) for fa in file_assignments
                            )
                            if plan_bytes > 0:
                                self._dataset_input_bytes[dataset_name] = plan_bytes
                        except Exception:
                            pass

                # Use cached/pre-seeded bytes (from plan.json or HF metadata)
                dataset_bytes = self._dataset_input_bytes.get(dataset_name, 0)

                # When max_rows is set, estimate effective size instead of full dataset size
                max_rows = self._dataset_max_rows.get(dataset_name)
                if max_rows and max_rows > 0 and dataset_bytes > 0:
                    dataset_bytes = self._estimate_effective_bytes(
                        dataset_name, dataset_bytes, max_rows
                    )

                if dataset_bytes > 0:
                    dataset_size_str = format_byte_size(dataset_bytes)

                # Count receipts by status
                # "downloaded" = any receipt exists (started, completed, or failed)
                # "processed" = completed receipts only
                shards_downloaded = 0
                shards_processed = 0
                total_tokens = 0
                total_sequences = 0

                if plan_hash:
                    receipts_dir = f"{dataset_base}/{plan_hash}/receipts"
                    try:
                        receipt_files = fs.glob(f"{receipts_dir}/shard_*.json")
                        for receipt_path in receipt_files:
                            try:
                                from nemotron.data_prep.utils.filesystem import read_json

                                receipt = read_json(fs, receipt_path)
                                status = receipt.get("status")

                                # Any receipt means the shard was downloaded and processing started
                                if status in ("started", "completed", "failed"):
                                    shards_downloaded += 1

                                # Only count completed for processed stats
                                if status == "completed":
                                    shards_processed += 1
                                    stats = receipt.get("stats", {}) or {}
                                    total_tokens += int(stats.get("total_tokens", 0) or 0)
                                    total_sequences += int(stats.get("num_sequences", 0) or 0)
                            except Exception:
                                continue
                    except Exception:
                        pass

                # Calculate percentages
                if total_shards > 0:
                    downloaded_pct = round(shards_downloaded / total_shards * 100, 1)
                    processed_pct = round(shards_processed / total_shards * 100, 1)
                else:
                    downloaded_pct = 0.0
                    processed_pct = 0.0

                # Format as "X shards (Y%)"
                downloaded_str = f"{shards_downloaded} shards ({downloaded_pct}%)"
                processed_str = f"{shards_processed} shards ({processed_pct}%)"

                # Determine status
                if shards_processed == 0 and shards_downloaded == 0:
                    status = "pending"
                elif total_shards > 0 and shards_processed >= total_shards:
                    status = "completed"
                else:
                    status = "in_progress"

                data.append([
                    dataset_name,
                    dataset_size_str,
                    total_shards,
                    downloaded_str,
                    processed_str,
                    format_count(total_tokens),
                    format_count(total_sequences),
                    status,
                ])

            # Log table to W&B
            table = wandb.Table(columns=columns, data=data)
            wandb.log({f"{self._pipeline_kind}/progress_table": table}, commit=False)
            logger.debug(f"Updated progress table with {len(data)} datasets")

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Progress table update error: {e}")

    def _update_stage_overview_table(self, stats: Any) -> None:
        """Log stage overview table to W&B showing current stage metrics.

        The table provides a real-time view of stage-level metrics:
        - stage: Human-readable stage name
        - tasks: Total completed tasks
        - queue_in: Input queue size
        - queue_out: Output queue size
        - actors: Running actors
        - speed: Tasks per second
        - cpu_pct: CPU utilization percentage
        - mem_gb: Memory usage in GB
        """
        try:
            import wandb

            if wandb.run is None:
                return

            stage_metrics = _extract_stage_metrics(stats)
            if not stage_metrics:
                return

            # Get all stage IDs from the metrics
            stage_ids: set[str] = set()
            for metric_data in stage_metrics.values():
                stage_ids.update(metric_data.keys())

            if not stage_ids:
                return

            columns = ["stage", "tasks", "queue_in", "queue_out", "actors", "speed", "cpu_pct", "mem_gb"]
            data = []

            for stage_id in sorted(stage_ids):
                row = [
                    get_stage_display_name(stage_id),
                    stage_metrics.get("tasks_completed", {}).get(stage_id, 0),
                    stage_metrics.get("queue_in", {}).get(stage_id, 0),
                    stage_metrics.get("queue_out", {}).get(stage_id, 0),
                    stage_metrics.get("actors_running", {}).get(stage_id, 0),
                    round(stage_metrics.get("speed_tasks_per_s", {}).get(stage_id, 0.0), 2),
                    round(stage_metrics.get("resource_cpu_util_pct", {}).get(stage_id, 0.0), 1),
                    round(stage_metrics.get("resource_mem_gb", {}).get(stage_id, 0.0), 2),
                ]
                data.append(row)

            table = wandb.Table(columns=columns, data=data)
            # Use commit=True to flush all pending logs (including progress_table)
            wandb.log({f"{self._wandb_namespace}/stage_table": table}, commit=True)
            logger.debug(f"Updated stage table with {len(data)} stages")

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Stage table update error: {e}")

    def __enter__(self) -> "WandbStatsHook":
        """Install the monkey-patch."""
        global _patch_depth, _original_make_stats, _active_hooks

        with _patch_lock:
            _active_hooks.append(self)

            if _patch_depth == 0:
                # First hook - install the patch
                monitor_cls = self._get_monitor_class()
                _original_make_stats = monitor_cls._make_stats
                monitor_cls._make_stats = self._create_wrapper(_original_make_stats)
                logger.debug("Installed PipelineMonitor._make_stats patch")

            _patch_depth += 1

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Remove the monkey-patch and log final consolidated charts."""
        global _patch_depth, _original_make_stats, _active_hooks

        # Log final consolidated line_series charts before cleanup
        self._log_final_consolidated_charts()

        with _patch_lock:
            _active_hooks.remove(self)
            _patch_depth -= 1

            if _patch_depth == 0 and _original_make_stats is not None:
                # Last hook - restore original
                monitor_cls = self._get_monitor_class()
                monitor_cls._make_stats = _original_make_stats
                _original_make_stats = None
                logger.debug("Restored PipelineMonitor._make_stats")

        # Close JSONL file
        if self._jsonl_file is not None:
            try:
                self._jsonl_file.close()
            except Exception:
                pass
            self._jsonl_file = None

        logger.debug(f"W&B hook logged {self._log_count} stats updates")

    def _log_final_consolidated_charts(self) -> None:
        """Log consolidated line_series charts at the end of the pipeline run.

        This creates ONE chart per metric with multiple lines (one per stage).
        The charts are logged only once at the end to avoid chart proliferation
        that occurs when logging line_series on every step.
        """
        if not self._metric_history or not self._step_history:
            return

        try:
            import wandb

            if wandb.run is None:
                return

            ns = self._wandb_namespace
            charts: dict[str, Any] = {}

            for metric_name, stage_data in self._metric_history.items():
                # Get all stages that have data for this metric
                stages_with_data = list(stage_data.keys())
                if not stages_with_data:
                    continue

                # Build xs and ys for line_series
                xs = []
                ys = []
                keys = []

                for stage_id in stages_with_data:
                    values = stage_data[stage_id]
                    # Use only the steps that have corresponding values
                    steps = self._step_history[: len(values)]
                    if steps and values:
                        xs.append(steps)
                        ys.append(values)
                        keys.append(get_stage_display_name(stage_id))

                if xs and ys:
                    try:
                        # Create human-readable title
                        title = f"{self._pipeline_kind}: {metric_name.replace('_', ' ').title()}"
                        chart = wandb.plot.line_series(
                            xs=xs,
                            ys=ys,
                            keys=keys,
                            title=title,
                            xname="Step",
                        )
                        charts[f"{ns}/stages/{metric_name}"] = chart
                    except Exception as e:
                        logger.debug(f"Failed to create line_series chart for {metric_name}: {e}")

            # Log all consolidated charts in one call
            if charts:
                wandb.log(charts, commit=False)
                logger.debug(f"Logged {len(charts)} consolidated stage charts to W&B")

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Error logging final consolidated charts: {e}")


def make_wandb_stats_hook(
    *,
    observability: ObservabilityConfig,
    pipeline_kind: str,
    run_hash: str | None = None,
    run_dir: str | None = None,
    dataset_names: list[str] | None = None,
    dataset_num_shards: dict[str, int] | None = None,
    dataset_input_bytes: dict[str, int] | None = None,
    dataset_max_rows: dict[str, int] | None = None,
) -> WandbStatsHook | None:
    """Factory function to create a W&B stats hook if logging is enabled.

    Args:
        observability: Configuration for what to log
        pipeline_kind: Type of pipeline (e.g., "pretrain", "sft")
        run_hash: Unique run identifier
        run_dir: Directory for JSONL output
        dataset_names: Names of datasets being processed
        dataset_num_shards: Dict mapping dataset name to expected number of shards
        dataset_input_bytes: Pre-computed total input bytes per dataset (from file discovery)
        dataset_max_rows: Per-dataset max_rows for effective size estimation

    Returns:
        WandbStatsHook if logging is enabled, None otherwise
    """
    # Check if any logging is enabled
    any_logging = (
        observability.wandb_log_pipeline_stats
        or observability.wandb_log_progress_table
        or observability.pipeline_stats_jsonl_path
    )
    if not any_logging:
        return None

    return WandbStatsHook(
        observability=observability,
        pipeline_kind=pipeline_kind,
        run_hash=run_hash,
        run_dir=run_dir,
        dataset_names=dataset_names,
        dataset_num_shards=dataset_num_shards,
        dataset_input_bytes=dataset_input_bytes,
        dataset_max_rows=dataset_max_rows,
    )


def log_plan_table_to_wandb(
    *,
    observability: ObservabilityConfig,
    pipeline_kind: str,
    dataset_items: list[Any],
    run_hash: str | None = None,
) -> None:
    """Log a plan table to W&B showing datasets and their processing configuration.

    This logs a wandb.Table before the pipeline runs, showing what datasets
    will be processed and their configuration.

    Args:
        observability: Configuration for what to log
        pipeline_kind: Type of pipeline (e.g., "pretrain", "sft")
        dataset_items: List of DatasetWorkItem or SftDatasetWorkItem objects
        run_hash: Unique run identifier

    Example table columns for pretrain:
        | dataset_name | path | weight | split | num_shards | sample | status |

    Example table columns for SFT:
        | dataset_name | path | weight | split | num_shards | pack_size | sample | status |
    """
    if not observability.wandb_log_plan_table:
        return

    if not dataset_items:
        logger.debug("No dataset items to log to plan table")
        return

    try:
        import wandb

        if wandb.run is None:
            logger.debug("W&B run not active, skipping plan table logging")
            return

        # Build table based on pipeline kind
        if pipeline_kind == "pretrain":
            columns = [
                "dataset_name",
                "path",
                "size",
                "weight",
                "split",
                "subset",
                "num_shards",
                "dtype",
                "min_doc_chars",
                "max_doc_tokens",
                "max_rows",
                "sample",
                "status",
            ]
            data = []
            for item in dataset_items:
                data.append([
                    item.dataset_name,
                    item.path,
                    _get_dataset_size_str(item),
                    item.weight,
                    item.split or "-",
                    item.subset or "-",
                    item.num_shards,
                    item.dtype,
                    item.min_doc_chars if item.min_doc_chars is not None else "-",
                    item.max_doc_tokens if item.max_doc_tokens is not None else "-",
                    item.max_rows if item.max_rows is not None else "-",
                    str(item.sample) if item.sample is not None else "-",
                    "pending",
                ])

        elif pipeline_kind == "sft":
            columns = [
                "dataset_name",
                "path",
                "size",
                "weight",
                "split",
                "subset",
                "num_shards",
                "pack_size",
                "algorithm",
                "max_doc_tokens",
                "max_rows",
                "sample",
                "status",
            ]
            data = []
            for item in dataset_items:
                data.append([
                    item.dataset_name,
                    item.path,
                    _get_dataset_size_str(item),
                    item.weight,
                    item.split or "-",
                    item.subset or "-",
                    item.num_shards,
                    getattr(item, "pack_size", "-"),
                    getattr(item, "algorithm", "-"),
                    item.max_doc_tokens if item.max_doc_tokens is not None else "-",
                    item.max_rows if item.max_rows is not None else "-",
                    str(item.sample) if item.sample is not None else "-",
                    "pending",
                ])

        else:
            # Generic fallback for other pipeline types
            columns = ["dataset_name", "path", "weight", "status"]
            data = []
            for item in dataset_items:
                data.append([
                    getattr(item, "dataset_name", "unknown"),
                    getattr(item, "path", "unknown"),
                    getattr(item, "weight", 0.0),
                    "pending",
                ])

        table = wandb.Table(columns=columns, data=data)
        wandb.log({f"{pipeline_kind}/plan_table": table}, commit=False)
        logger.debug(f"Logged plan table with {len(data)} datasets to W&B")

    except ImportError:
        logger.debug("wandb not installed, skipping plan table logging")
    except Exception as e:
        logger.warning(f"Failed to log plan table to W&B: {e}")


@contextlib.contextmanager
def pipeline_wandb_hook(
    items: list[Any],
    pipeline_ctx: Any,
    pipeline_kind: str,
) -> Generator[None, None, None]:
    """W&B observability wrapper for pipeline execution.

    Logs a plan table before the pipeline runs and streams real-time
    stats to W&B during execution. Safe to use when W&B is not configured
    (degrades to a no-op).

    Usage in cookbook scripts::

        with pipeline_wandb_hook(dataset_items, pipeline_ctx, "pretrain"):
            pipelines_v1.run_pipeline(spec)

    Args:
        items: Work items (DatasetWorkItem, SftDatasetWorkItem, or JsonlDatasetWorkItem).
        pipeline_ctx: PipelineContext with observability config, run_hash, run_dir.
        pipeline_kind: Pipeline type for W&B namespacing ("pretrain", "sft", "rl").
    """
    observability_cfg = pipeline_ctx.observability

    # Log plan table before pipeline runs
    log_plan_table_to_wandb(
        observability=observability_cfg,
        pipeline_kind=pipeline_kind,
        dataset_items=items,
        run_hash=pipeline_ctx.run_hash,
    )

    # Build dataset mappings for progress tracking
    dataset_num_shards = {item.dataset_name: item.num_shards for item in items}
    dataset_input_bytes = compute_dataset_input_bytes(items)

    # Collect max_rows generically (works for all work item types)
    dataset_max_rows: dict[str, int] = {}
    for item in items:
        max_rows = getattr(item, "max_rows", None)
        if max_rows is not None and max_rows > 0:
            dataset_max_rows[item.dataset_name] = max_rows

    hook = make_wandb_stats_hook(
        observability=observability_cfg,
        pipeline_kind=pipeline_kind,
        run_hash=pipeline_ctx.run_hash,
        run_dir=pipeline_ctx.run_dir,
        dataset_names=[item.dataset_name for item in items],
        dataset_num_shards=dataset_num_shards,
        dataset_input_bytes=dataset_input_bytes,
        dataset_max_rows=dataset_max_rows or None,
    )

    if hook:
        with hook:
            yield
    else:
        yield
