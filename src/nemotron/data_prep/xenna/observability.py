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

"""Observability helpers for Xenna pipelines: wandb logging, receipt/download polling."""

from __future__ import annotations

import json
import threading
import time


def start_receipt_poller(
    *,
    fs,
    dataset_receipt_dirs: dict[str, str],
    live_status,
    stop_event: threading.Event,
    interval_sec: float = 10.0,
    log_to_wandb: bool = True,
    total_shards: int | None = None,
) -> threading.Thread:
    """Start a daemon thread that polls receipt directories for progress updates.

    Args:
        fs: Filesystem abstraction (fsspec-compatible)
        dataset_receipt_dirs: Mapping of dataset_name -> receipts_dir path
        live_status: LiveExecutionStatus for progress UI updates
        stop_event: Event to signal thread termination
        interval_sec: Polling interval in seconds
        log_to_wandb: Whether to log metrics to wandb
        total_shards: Total number of shards across all datasets (for progress %)

    Returns:
        The started daemon thread
    """

    def _poll() -> None:
        # Check wandb availability once at start
        wandb = None
        if log_to_wandb:
            try:
                import wandb as _wandb

                if _wandb.run is not None:
                    wandb = _wandb
            except ImportError:
                pass

        last_counts: dict[str, int] = {name: 0 for name in dataset_receipt_dirs}
        seen_receipts: dict[str, set[str]] = {name: set() for name in dataset_receipt_dirs}
        tokens_by_dataset: dict[str, int] = {name: 0 for name in dataset_receipt_dirs}
        shards_by_dataset: dict[str, int] = {name: 0 for name in dataset_receipt_dirs}

        while not stop_event.is_set():
            metrics_changed = False

            for name, receipts_dir in dataset_receipt_dirs.items():
                try:
                    if not fs.exists(receipts_dir):
                        continue
                    entries = [p for p in fs.ls(receipts_dir, detail=False) if str(p).endswith(".json")]
                    count = len(entries)
                except Exception:
                    continue

                last = last_counts.get(name, 0)
                if count > last:
                    metrics_changed = True
                    # Process newly seen receipts
                    for p in entries:
                        if p not in seen_receipts[name]:
                            seen_receipts[name].add(p)
                            try:
                                with fs.open(p, "r") as f:
                                    receipt = json.load(f)
                                tokens_by_dataset[name] += int(
                                    receipt.get("stats", {}).get("total_tokens", 0)
                                )
                            except Exception:
                                pass

                    # Advance progress by delta
                    for _ in range(count - last):
                        live_status.advance_dataset(name)
                    last_counts[name] = count
                    shards_by_dataset[name] = count
                    live_status.report_tokens(name, tokens_by_dataset[name])

            # Log to wandb if metrics changed
            if wandb is not None and metrics_changed:
                total_shards_completed = sum(shards_by_dataset.values())
                total_tokens = sum(tokens_by_dataset.values())

                metrics = {
                    "data_prep/shards_completed": total_shards_completed,
                    "data_prep/tokens_total": total_tokens,
                }

                # Add progress percentage if we know the total
                if total_shards is not None and total_shards > 0:
                    metrics["data_prep/progress"] = total_shards_completed / total_shards

                # Per-dataset metrics
                for name in dataset_receipt_dirs:
                    safe_name = name.replace("-", "_").replace(" ", "_")
                    metrics[f"data_prep/datasets/{safe_name}/shards"] = shards_by_dataset[name]
                    metrics[f"data_prep/datasets/{safe_name}/tokens"] = tokens_by_dataset[name]

                wandb.log(metrics)

            stop_event.wait(interval_sec)

    thread = threading.Thread(target=_poll, daemon=True)
    thread.start()
    return thread


def start_download_poller(
    *,
    fs,
    output_root: str,
    stop_event: threading.Event,
    interval_sec: int = 30,
) -> threading.Thread:
    """Start a daemon thread that polls download progress and logs to wandb.

    Args:
        fs: Filesystem abstraction (fsspec-compatible)
        output_root: Root output directory (download progress is at {output_root}/.xenna/downloads/)
        stop_event: Event to signal thread termination
        interval_sec: Polling interval in seconds

    Returns:
        The started daemon thread
    """

    def _poll() -> None:
        try:
            import wandb
        except ImportError:
            return

        if wandb.run is None:
            return

        progress_dir = f"{output_root.rstrip('/')}/.xenna/downloads"
        last_logged = 0.0

        while not stop_event.is_set():
            now = time.time()
            if now - last_logged < interval_sec:
                stop_event.wait(1.0)
                continue
            last_logged = now

            try:
                if not fs.exists(progress_dir):
                    continue
                entries = fs.ls(progress_dir, detail=False)
            except Exception:
                continue

            total_completed = 0
            total_files = 0
            max_elapsed = 0.0
            max_rate = 0.0

            for path in entries:
                if not str(path).endswith(".json"):
                    continue
                try:
                    with fs.open(path, "r") as f:
                        data = json.load(f)
                except Exception:
                    continue
                total_completed += int(data.get("completed", 0))
                total_files += int(data.get("total", 0))
                max_elapsed = max(max_elapsed, float(data.get("elapsed_sec", 0.0)))
                max_rate = max(max_rate, float(data.get("rate", 0.0)))

            if total_files == 0:
                continue

            wandb.log(
                {
                    "data_prep/hf_download_completed": total_completed,
                    "data_prep/hf_download_total": total_files,
                    "data_prep/hf_download_rate": max_rate,
                    "data_prep/hf_download_elapsed_sec": max_elapsed,
                }
            )

    thread = threading.Thread(target=_poll, daemon=True)
    thread.start()
    return thread
