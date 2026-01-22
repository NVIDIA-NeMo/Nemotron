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

"""Xenna pipeline executor - format-agnostic.

This module provides a single run_xenna() function that executes any Xenna pipeline
with observability (receipt polling, download logging). It is format-agnostic - the
caller is responsible for building the PipelineSpec and aggregating results.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import cosmos_xenna.pipelines.v1 as pipelines_v1

from nemotron.data_prep.xenna.observability import (
    start_download_poller,
    start_receipt_poller,
)

if TYPE_CHECKING:
    from nemotron.data_prep.config import XennaConfig


def run_xenna(
    *,
    pipeline_spec: pipelines_v1.PipelineSpec,
    dataset_receipt_dirs: dict[str, str],
    output_root: str,
    fs,
    live_status,
    xenna_cfg: "XennaConfig",
) -> None:
    """Run a Xenna pipeline with observability.

    This is a format-agnostic executor. The caller is responsible for:
    1. Building the PipelineSpec (via pipeline_specs.build_*())
    2. Aggregating results after this function returns

    Args:
        pipeline_spec: Pre-built PipelineSpec from pipeline_specs module.
        dataset_receipt_dirs: Mapping of dataset_name -> receipts_dir for progress tracking
        output_root: Root output directory (used for download progress tracking)
        fs: Filesystem abstraction (fsspec-compatible)
        live_status: LiveExecutionStatus for progress UI updates
        xenna_cfg: Xenna configuration (controls wandb logging)

    Note:
        This function does NOT call live_status.start_dataset() or live_status.complete_dataset().
        The caller should handle those for consistent semantics across execution paths.
    """
    if not pipeline_spec.input_data:
        return

    total_tasks = len(pipeline_spec.input_data)
    print(f"[Xenna] Launching pipeline for {total_tasks} task(s)")

    stop_event = threading.Event()

    # Start receipt poller for progress tracking and wandb logging
    receipt_thread = start_receipt_poller(
        fs=fs,
        dataset_receipt_dirs=dataset_receipt_dirs,
        live_status=live_status,
        stop_event=stop_event,
        log_to_wandb=xenna_cfg.wandb_log_pipeline_stats,
        total_shards=total_tasks,
    )

    # Optionally start download progress poller
    download_thread = None
    if xenna_cfg.wandb_log_downloads:
        download_thread = start_download_poller(
            fs=fs,
            output_root=output_root,
            stop_event=stop_event,
            interval_sec=xenna_cfg.wandb_download_log_interval_sec,
        )

    try:
        pipelines_v1.run_pipeline(pipeline_spec)
    finally:
        stop_event.set()
        receipt_thread.join(timeout=2.0)
        if download_thread:
            download_thread.join(timeout=2.0)
