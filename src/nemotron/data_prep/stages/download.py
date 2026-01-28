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
Download Stage - Ensures files are available locally before processing.

Supports multiple source types:
- HuggingFace Hub (hf:// or datasets library paths)
- Cloud storage (s3://, gs://, az://) via fsspec
- Local files (no-op, validates existence)
- HTTP/HTTPS URLs

This stage is idempotent - already-downloaded files are skipped via caching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import cosmos_xenna.pipelines.v1 as pipelines_v1

from nemotron.data_prep.utils.filesystem import get_filesystem
from nemotron.data_prep.stages.context import PipelineContext

# Generic work item type - stage passes through unchanged
T = TypeVar("T")


@dataclass(frozen=True)
class DownloadStageConfig:
    """Configuration for DownloadStage.

    DownloadStage handles HuggingFace and cloud file downloads with
    configurable batching, retries, and hf_xet tuning.

    Attributes:
        batch_size: Number of work items to batch together for file deduplication.
            Higher values improve dedup but increase memory. Default 32.
        stage_cpus: CPU request for the download worker. Default 0.5 since
            this stage is network-bound, not CPU-bound.
        hf_xet_high_performance: Enable hf_xet high-performance mode. When True,
            saturates network bandwidth and uses all CPU cores. Default True.
        hf_xet_concurrent_range_gets: Number of concurrent chunk downloads per
            file. Default 32 (doubled from hf_xet default of 16 for faster downloads).
            Set to None to use hf_xet default.
        max_retries: Maximum number of retry attempts for failed downloads. Default 3.
        timeout_sec: Timeout in seconds for download operations. Default 300.
    """

    batch_size: int = 1
    stage_cpus: float = 0.5
    hf_xet_high_performance: bool | None = True
    hf_xet_concurrent_range_gets: int | None = 32
    max_retries: int = 3
    timeout_sec: int = 300

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.stage_cpus <= 0:
            raise ValueError(f"stage_cpus must be positive, got {self.stage_cpus}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {self.max_retries}")


class DownloadStage(pipelines_v1.Stage[T, T]):
    """
    Generic download stage: ensures all files are available locally.

    Handles multiple source types:
    - HuggingFace Hub (detected via hf_repo_id in file metadata)
    - Cloud storage (s3://, gs://, az://) via fsspec
    - Local files (no-op, validates existence)
    - HTTP/HTTPS URLs

    Batching enables efficient deduplication of downloads across work items
    that share the same source files.

    This stage is idempotent - uses HF cache and fsspec caching.

    Args:
        stage_config: Stage-specific configuration (DownloadStageConfig)
        pipeline_context: Shared runtime context (PipelineContext)
    """

    def __init__(
        self,
        stage_config: DownloadStageConfig,
        pipeline_context: PipelineContext,
    ) -> None:
        self._cfg = stage_config
        self._ctx = pipeline_context
        self._fs = None

    @property
    def stage_batch_size(self) -> int:
        """Batch size for deduplication across work items."""
        return self._cfg.batch_size

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        """Minimal CPU, network-bound stage."""
        return pipelines_v1.Resources(cpus=self._cfg.stage_cpus, gpus=0)

    def _hf_extra_env_vars(self) -> dict[str, str]:
        """Build extra environment variables for hf_xet tuning."""
        env: dict[str, str] = {}
        if self._cfg.hf_xet_high_performance is not None:
            env["HF_XET_HIGH_PERFORMANCE"] = "1" if self._cfg.hf_xet_high_performance else "0"
        if self._cfg.hf_xet_concurrent_range_gets is not None:
            env["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = str(self._cfg.hf_xet_concurrent_range_gets)
        return env

    @property
    def env_info(self) -> pipelines_v1.RuntimeEnv:
        """Runtime environment with HF credentials and hf_xet tuning."""
        return self._ctx.hf_runtime_env(extra_env_vars=self._hf_extra_env_vars())

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        """Initialize filesystem on worker."""
        self._fs, _ = get_filesystem(self._ctx.output_root)

    def process_data(self, tasks: list[T]) -> list[T]:
        """Download files for all tasks, then pass through unchanged."""
        # Collect unique files across all tasks in batch
        hf_files: list[dict[str, str]] = []
        cloud_files: set[str] = set()

        for task in tasks:
            assignment = getattr(task, "assignment", None)
            if assignment is None:
                continue

            # Select which HF files to download (may be limited by max_rows)
            selected_files = self._select_files_for_task(task, assignment)

            for file_info in selected_files:
                # Prefer explicit HF metadata over path heuristics
                if file_info.get("hf_repo_id"):
                    hf_files.append({
                        "repo_id": file_info["hf_repo_id"],
                        "filename": file_info.get("hf_filename", ""),
                        "revision": file_info.get("hf_revision"),
                    })
                else:
                    path = file_info.get("path", "")
                    if self._is_cloud_path(path):
                        cloud_files.add(path)
                    # Local files need no download

        # Download HuggingFace files (deduplicated by repo_id + filename)
        if hf_files:
            self._download_hf_files(hf_files)

        # Validate/download cloud files (deduplicated)
        if cloud_files:
            self._download_cloud_files(cloud_files)

        # Pass through all tasks unchanged
        return tasks

    def _select_files_for_task(self, task: T, assignment: dict) -> list[dict]:
        """Select which files to download for a task.

        When max_rows is set, estimates how many files are needed to cover
        the requested row count and only returns those files. This avoids
        downloading entire datasets when only a small sample is needed.

        Falls back to all files if row estimation is not possible.
        """
        files = assignment.get("files", [])
        if not files:
            return files

        max_rows = getattr(task, "max_rows", None)
        if max_rows is None or max_rows <= 0:
            return files

        # Only apply limiting for HF files where we can estimate rows
        hf_files = [f for f in files if f.get("hf_repo_id")]
        non_hf_files = [f for f in files if not f.get("hf_repo_id")]

        if not hf_files:
            return files

        # Estimate bytes-per-row from HF dataset metadata
        avg_bpr = self._estimate_bytes_per_row(hf_files, assignment)
        if avg_bpr is None:
            return files  # Can't estimate, download all for correctness

        import logging
        _logger = logging.getLogger(__name__)

        # Select files in assignment order until we cover max_rows
        # Use 2x safety margin to avoid under-downloading
        target_rows = max_rows * 2
        selected_hf: list[dict] = []
        est_rows = 0

        for f in hf_files:
            size_bytes = int(f.get("size", 0) or 0)
            file_est_rows = max(1, int(size_bytes / avg_bpr)) if size_bytes > 0 else 1

            selected_hf.append(f)
            est_rows += file_est_rows

            if est_rows >= target_rows:
                break

        _logger.info(
            f"DownloadStage: max_rows={max_rows}, selected {len(selected_hf)}/{len(hf_files)} "
            f"HF files (est ~{est_rows} rows, avg {avg_bpr:.0f} bytes/row)"
        )

        return selected_hf + non_hf_files

    def _estimate_bytes_per_row(self, hf_files: list[dict], assignment: dict) -> float | None:
        """Estimate average bytes-per-row using HF dataset metadata API.

        Uses the first file's repo_id and assignment-level subset/split
        to query dataset metadata. Returns None if metadata is unavailable.
        """
        from nemotron.data_prep.utils.discovery import fetch_hf_dataset_metadata

        first = hf_files[0]
        repo_id = first.get("hf_repo_id")
        if not repo_id:
            return None

        # Use subset/split from assignment (set by PlanStage)
        subset = assignment.get("hf_subset")
        split = assignment.get("hf_split")

        meta = fetch_hf_dataset_metadata(repo_id, subset=subset, split=split)
        if meta.num_rows and meta.size_bytes and meta.num_rows > 0 and meta.size_bytes > 0:
            return meta.size_bytes / meta.num_rows

        return None

    def _is_cloud_path(self, path: str) -> bool:
        """Check if path is cloud storage."""
        return any(path.startswith(p) for p in ["s3://", "gs://", "gcs://", "az://", "abfs://"])

    def _download_hf_files(self, hf_files: list[dict[str, str]]) -> None:
        """
        Download HuggingFace files using huggingface_hub.

        hf_xet (the new HF transfer backend) handles parallelism internally
        via concurrent range GETs in Rust, so we iterate serially over files
        and let hf_xet maximize bandwidth per file.

        The HF_XET_HIGH_PERFORMANCE and HF_XET_NUM_CONCURRENT_RANGE_GETS
        environment variables are configured via env_info to tune performance.
        """
        import logging
        import time

        from huggingface_hub import hf_hub_download

        logger = logging.getLogger(__name__)

        # Deduplicate by (repo_id, filename, revision)
        unique_files: dict[tuple, dict] = {}
        for f in hf_files:
            key = (f["repo_id"], f["filename"], f.get("revision"))
            if key not in unique_files:
                unique_files[key] = f

        if not unique_files:
            return

        logger.info(f"Downloading {len(unique_files)} unique HuggingFace files")

        # Download files serially - hf_xet handles parallelism within each file
        max_retries = self._cfg.max_retries
        failed: list[tuple[str, str]] = []
        for file_info in unique_files.values():
            repo_id = file_info["repo_id"]
            filename = file_info["filename"]
            revision = file_info.get("revision")

            success = False
            last_error = ""
            for attempt in range(max_retries):
                try:
                    # hf_hub_download handles caching automatically
                    # hf_xet (if installed) accelerates via concurrent range GETs
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        revision=revision,
                        repo_type="dataset",
                        local_files_only=False,
                    )
                    success = True
                    break
                except Exception as e:
                    last_error = str(e)
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(
                            f"Download attempt {attempt + 1} failed for {filename}: {e}. "
                            f"Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)

            if not success:
                failed.append((filename, last_error))
                logger.warning(f"Failed to download {filename}: {last_error}")

        if failed:
            logger.error(f"Failed to download {len(failed)} HuggingFace files")
            # Don't raise - let downstream stages handle missing files

    def _download_cloud_files(self, paths: set[str]) -> None:
        """
        Validate cloud files exist (and optionally prefetch).

        By default, just validates existence - actual reading happens
        in downstream stages via fsspec streaming.
        """
        import logging

        logger = logging.getLogger(__name__)

        missing = []
        for url in paths:
            try:
                fs, normalized = get_filesystem(url)
                if not fs.exists(normalized):
                    missing.append(url)
            except Exception as e:
                logger.warning(f"Could not validate cloud file {url}: {e}")
                missing.append(url)

        if missing:
            logger.warning(f"Missing {len(missing)} cloud files: {missing[:5]}...")
            # Don't raise - let downstream stages handle missing files
