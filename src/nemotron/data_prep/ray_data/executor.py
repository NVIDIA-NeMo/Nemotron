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

"""Ray Data executor for shard-task processing."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import ray
import ray.data

from nemotron.data_prep.ray_data.tasks import ShardTask

logger = logging.getLogger(__name__)


class _ProgressReporter:
    """Background thread for periodic progress reporting during Ray Data execution.

    This ensures progress is reported even when iter_rows() is blocked waiting
    for initial results (e.g., during HuggingFace downloads).
    """

    def __init__(
        self,
        on_progress: Callable[[dict[str, Any]], None],
        total_tasks: int,
        interval: float = 5.0,
    ):
        self.on_progress = on_progress
        self.total_tasks = total_tasks
        self.interval = interval

        self._start_time = time.perf_counter()
        self._tasks_completed = 0
        self._all_stats: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background progress reporter."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background progress reporter."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def update(self, tasks_completed: int, stats: list[dict[str, Any]]) -> None:
        """Update progress state from the main iteration loop."""
        with self._lock:
            self._tasks_completed = tasks_completed
            self._all_stats = stats.copy()

    def _run(self) -> None:
        """Background thread that reports progress periodically."""
        while not self._stop_event.wait(self.interval):
            with self._lock:
                tasks_completed = self._tasks_completed
                all_stats = self._all_stats.copy()

            elapsed = time.perf_counter() - self._start_time

            # Infer phase from completed stats
            phase = _infer_dominant_phase(all_stats)

            # If no tasks completed yet, show "working" with elapsed time
            if tasks_completed == 0:
                phase = "working"
                detail = f"waiting for first shard... {elapsed:.0f}s"
            else:
                detail = f"{tasks_completed}/{self.total_tasks} shards"

            try:
                self.on_progress({
                    "phase": phase,
                    "detail": detail,
                    "elapsed_sec": elapsed,
                    "tasks_completed": tasks_completed,
                    "tasks_total": self.total_tasks,
                })
            except Exception:
                pass  # Don't crash background thread on callback errors


@dataclass(frozen=True)
class RayDataExecConfig:
    """Configuration for Ray Data shard-task execution.

    These settings map directly to Ray Data's ActorPoolStrategy and
    map_batches parameters, providing explicit control over resource usage.

    Attributes:
        min_actors: Minimum actors to keep alive (warm pool)
        max_actors: Maximum actors (bounded to prevent overload)
        cpus_per_actor: CPUs allocated per actor (explicit accounting)
        max_tasks_in_flight_per_actor: Pipelining depth to reduce scheduling
            bubbles and keep actors fed. Note: does not by itself parallelize
            a single actor; true I/O latency hiding requires either more actors
            (with fractional num_cpus) or async internal concurrency.
    """

    min_actors: int = 2
    max_actors: int = 32
    cpus_per_actor: float = 1.0
    max_tasks_in_flight_per_actor: int = 2


def execute_shard_tasks(
    tasks: list[ShardTask],
    *,
    udf_cls: type,
    udf_constructor_kwargs: dict[str, Any],
    exec_cfg: RayDataExecConfig,
    on_result: Callable[[dict[str, Any]], None] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Execute shard tasks via Ray Data ActorPoolStrategy.

    Each task:
    1. Is processed by a stateful actor (tokenizer loaded once in __init__)
    2. Produces side-effects (writes shard files + receipt)
    3. Returns a small stats dict-of-numpy-arrays

    Args:
        tasks: List of ShardTask to execute
        udf_cls: UDF class (e.g., BinIdxShardTaskUDF)
        udf_constructor_kwargs: Arguments passed to UDF __init__
        exec_cfg: Execution configuration
        on_result: Optional callback for each completed task
        on_progress: Optional callback for periodic progress updates (every ~5s)
            Called with {"phase": str, "elapsed_sec": float, "tasks_completed": int}

    Returns:
        List of stats dicts from all completed tasks
    """
    if not tasks:
        logger.info("No shard tasks to execute")
        return []

    logger.info(
        f"Executing {len(tasks)} shard tasks with Ray Data "
        f"(actors: {exec_cfg.min_actors}-{exec_cfg.max_actors}, "
        f"cpus_per_actor: {exec_cfg.cpus_per_actor}, "
        f"max_in_flight_per_actor: {exec_cfg.max_tasks_in_flight_per_actor})"
    )

    execution_start = time.perf_counter()
    total_tasks = len(tasks)

    # Report initial "starting" phase
    if on_progress:
        on_progress({
            "phase": "starting",
            "detail": f"{total_tasks} shards",
            "elapsed_sec": 0.0,
            "tasks_completed": 0,
            "tasks_total": total_tasks,
        })

    # Build Ray Dataset from task dicts
    # NOTE: Serialize assignment as JSON string for stable Arrow encoding
    task_dicts = [t.to_dict() for t in tasks]
    ds = ray.data.from_items(task_dicts)

    # Configure ActorPoolStrategy
    compute = ray.data.ActorPoolStrategy(
        min_size=exec_cfg.min_actors,
        max_size=exec_cfg.max_actors,
        max_tasks_in_flight_per_actor=exec_cfg.max_tasks_in_flight_per_actor,
    )

    # Execute with explicit CPU allocation
    # batch_size=1 means one shard task per UDF call
    # Default batch_format is dict-of-numpy-arrays (NumPy is DEFAULT_BATCH_FORMAT)
    # num_cpus is a direct parameter (not via ray_remote_args)
    #
    # FAULT TOLERANCE: We rely on idempotent atomic commit in the UDF rather than
    # disabling retries. Ray Data defaults to max_restarts=-1, max_task_retries=-1.
    # The atomic write protocol (temp -> rename -> receipt) makes retries safe.
    stats_ds = ds.map_batches(
        udf_cls,
        fn_constructor_kwargs=udf_constructor_kwargs,
        batch_size=1,
        compute=compute,
        num_cpus=exec_cfg.cpus_per_actor,
    )

    # Stream results to handle callback and collect stats
    all_stats: list[dict[str, Any]] = []
    tasks_completed = 0
    last_progress_time = execution_start

    # Start background progress reporter if callback provided
    # This ensures progress is shown even when waiting for first result
    progress_reporter: _ProgressReporter | None = None
    if on_progress:
        progress_reporter = _ProgressReporter(on_progress, total_tasks, interval=5.0)
        progress_reporter.start()

    try:
        for row in stats_ds.iter_rows():
            tasks_completed += 1
            # Convert numpy scalars to Python types for easier handling
            row_dict = {k: v.item() if hasattr(v, "item") else v for k, v in row.items()}
            all_stats.append(row_dict)

            # Update background reporter with current state
            if progress_reporter:
                progress_reporter.update(tasks_completed, all_stats)

            current_time = time.perf_counter()
            elapsed = current_time - execution_start

            if on_result:
                # Add Ray Data execution metadata
                on_result(
                    {
                        **row_dict,
                        "_ray_data_tasks_completed": tasks_completed,
                        "_ray_data_elapsed_sec": elapsed,
                    }
                )

            # Immediate progress reporting on task completion
            if on_progress and (current_time - last_progress_time >= 5.0 or tasks_completed == total_tasks):
                last_progress_time = current_time
                # Infer phase from timing breakdown of completed tasks
                phase = _infer_dominant_phase(all_stats)
                on_progress({
                    "phase": phase,
                    "detail": f"{tasks_completed}/{total_tasks} shards",
                    "elapsed_sec": elapsed,
                    "tasks_completed": tasks_completed,
                    "tasks_total": total_tasks,
                })
    finally:
        # Stop background reporter
        if progress_reporter:
            progress_reporter.stop()

    execution_time = time.perf_counter() - execution_start
    logger.info(
        f"Ray Data execution complete: {tasks_completed} tasks in {execution_time:.1f}s "
        f"({tasks_completed / max(execution_time, 0.001):.1f} tasks/sec)"
    )

    # Log to W&B if active
    _log_execution_to_wandb(tasks_completed, execution_time, exec_cfg)

    return all_stats


def _infer_dominant_phase(stats: list[dict[str, Any]]) -> str:
    """Infer the dominant processing phase from completed task stats.

    Returns the phase that consumed the most time across all completed tasks.
    """
    if not stats:
        return "processing"

    # Sum up time by phase
    time_download = sum(s.get("time_download_sec", 0.0) for s in stats)
    time_read = sum(s.get("time_read_sec", 0.0) for s in stats)
    time_tokenize = sum(s.get("time_tokenize_sec", 0.0) for s in stats)
    time_write = sum(s.get("time_write_sec", 0.0) for s in stats)

    # Find dominant phase
    phases = [
        (time_download, "downloading"),
        (time_read, "reading"),
        (time_tokenize, "tokenizing"),
        (time_write, "writing"),
    ]
    dominant = max(phases, key=lambda x: x[0])

    # Only report specific phase if it's >50% of time, otherwise generic
    total_time = time_download + time_read + time_tokenize + time_write
    if total_time > 0 and dominant[0] / total_time > 0.5:
        return dominant[1]
    return "processing"


def _log_execution_to_wandb(
    tasks_completed: int,
    execution_time: float,
    exec_cfg: RayDataExecConfig,
) -> None:
    """Log Ray Data execution metrics to W&B if active."""
    try:
        import wandb

        if wandb.run is None:
            logger.debug("[W&B] No active run, skipping execution metrics log")
            return

        wandb.log(
            {
                "data_prep/ray_data/total_tasks": tasks_completed,
                "data_prep/ray_data/execution_time_sec": execution_time,
                "data_prep/ray_data/tasks_per_sec": tasks_completed / max(execution_time, 0.001),
                "data_prep/ray_data/max_actors": exec_cfg.max_actors,
                "data_prep/ray_data/max_in_flight_per_actor": exec_cfg.max_tasks_in_flight_per_actor,
            }
        )
        logger.info(f"[W&B] Logged ray_data execution: {tasks_completed} tasks in {execution_time:.1f}s")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"[W&B] Failed to log execution metrics: {e}")
