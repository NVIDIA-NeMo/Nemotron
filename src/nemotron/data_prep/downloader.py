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

"""Parallel HuggingFace file downloader with overlapped shard processing.

This module provides parallel downloading of HuggingFace files that overlaps
with shard processing. Instead of download-all-then-process, shards are
yielded for processing as soon as all their required files are downloaded.

Architecture:
    - Downloads are I/O-bound (network) → use lightweight Ray tasks
    - Processing is CPU/memory-bound → use Ray Data actors
    - Overlap them: start processing shards while downloads continue

Usage:
    from nemotron.data_prep.downloader import OverlappedDownloader

    downloader = OverlappedDownloader(shard_tasks, max_concurrent_downloads=64)

    # Yields shards ready for processing while downloads continue
    for ready_tasks in downloader.iter_ready_batches():
        process_shards(ready_tasks)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import ray

logger = logging.getLogger(__name__)


@dataclass
class DownloadStats:
    """Statistics from parallel download phase."""

    total_files: int
    downloaded_files: int
    cached_files: int
    failed_files: int
    elapsed_sec: float

    @property
    def success_rate(self) -> float:
        """Percentage of successful downloads."""
        if self.total_files == 0:
            return 100.0
        return (self.downloaded_files + self.cached_files) / self.total_files * 100


@ray.remote(num_cpus=0.1)  # Light on CPU, mostly I/O bound
def _download_hf_file(
    repo_id: str,
    filename: str,
    revision: str | None,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Download a single HuggingFace file.

    Returns dict with status and path information.

    Args:
        repo_id: HuggingFace repository ID (e.g., "nvidia/Nemotron-CC-v2.1")
        filename: Path within the repository (e.g., "data/part_00001.parquet")
        revision: Git revision/commit SHA for determinism
        cache_dir: HuggingFace cache directory (defaults to HF_HOME/hub)

    Returns:
        Dict with keys: status ("downloaded", "cached", "failed"), path, error
    """
    try:
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type="dataset",
            local_files_only=False,
            cache_dir=cache_dir,
        )

        # Return file key for tracking
        return {
            "status": "downloaded",
            "repo_id": repo_id,
            "filename": filename,
            "revision": revision,
            "local_path": local_path,
            "error": None,
        }
    except Exception as e:
        logger.warning(f"Failed to download {repo_id}/{filename}: {e}")
        return {
            "status": "failed",
            "repo_id": repo_id,
            "filename": filename,
            "revision": revision,
            "local_path": None,
            "error": str(e),
        }


def _make_file_key(repo_id: str, filename: str, revision: str | None) -> str:
    """Create a unique key for a HuggingFace file."""
    return f"{repo_id}:{filename}:{revision or ''}"


@dataclass
class OverlappedDownloader:
    """Downloads files and yields shards ready for processing.

    This class manages parallel file downloads while tracking which shards
    are ready for processing. Shards are yielded as soon as all their
    required files have been downloaded.

    Attributes:
        shard_tasks: List of ShardTask objects to process
        max_concurrent_downloads: Maximum parallel downloads (rate limiting)
        on_progress: Optional callback for progress updates
        cache_dir: HuggingFace cache directory (defaults to HF_HOME/hub)
    """

    shard_tasks: list
    max_concurrent_downloads: int = 64
    on_progress: Callable[[dict[str, Any]], None] | None = None
    cache_dir: str | None = None

    # Internal state (initialized in __post_init__)
    _file_to_shards: dict[str, set[int]] = field(default_factory=dict, init=False)
    _shard_to_files: dict[int, set[str]] = field(default_factory=dict, init=False)
    _shard_pending_files: dict[int, set[str]] = field(default_factory=dict, init=False)
    _downloaded_files: set[str] = field(default_factory=set, init=False)
    _failed_files: set[str] = field(default_factory=set, init=False)
    _ready_shards: list[int] = field(default_factory=list, init=False)
    _all_unique_files: list[dict[str, str]] = field(default_factory=list, init=False)
    _shard_index_to_task: dict[int, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Build file→shard dependency mappings."""
        import os

        # Compute cache_dir from HF_HOME if not specified
        if self.cache_dir is None:
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                # Use object.__setattr__ because dataclass is frozen-like
                object.__setattr__(self, "cache_dir", os.path.join(hf_home, "hub"))

        self._file_to_shards = defaultdict(set)
        self._shard_to_files = defaultdict(set)
        self._shard_pending_files = {}
        self._downloaded_files = set()
        self._failed_files = set()
        self._ready_shards = []
        self._shard_index_to_task = {}

        seen_files = set()
        unique_files = []

        for idx, task in enumerate(self.shard_tasks):
            self._shard_index_to_task[idx] = task
            assignment = json.loads(task.assignment_json)
            files = assignment.get("files", [])

            shard_file_keys = set()
            for file_info in files:
                repo_id = file_info.get("hf_repo_id")
                if repo_id is None:
                    continue  # Skip non-HF files

                filename = file_info.get("hf_filename")
                revision = file_info.get("hf_revision")
                file_key = _make_file_key(repo_id, filename, revision)

                self._file_to_shards[file_key].add(idx)
                shard_file_keys.add(file_key)

                if file_key not in seen_files:
                    seen_files.add(file_key)
                    unique_files.append({
                        "repo_id": repo_id,
                        "filename": filename,
                        "revision": revision,
                    })

            self._shard_to_files[idx] = shard_file_keys
            self._shard_pending_files[idx] = shard_file_keys.copy()

            # If shard has no HF files, it's ready immediately
            if not shard_file_keys:
                self._ready_shards.append(idx)

        self._all_unique_files = unique_files

    @property
    def total_files(self) -> int:
        """Total unique files to download."""
        return len(self._all_unique_files)

    @property
    def total_shards(self) -> int:
        """Total shards to process."""
        return len(self.shard_tasks)

    def _mark_file_downloaded(self, file_key: str) -> list[int]:
        """Mark a file as downloaded and return newly-ready shard indices."""
        if file_key in self._downloaded_files:
            return []

        self._downloaded_files.add(file_key)
        newly_ready = []

        # Update all shards that needed this file
        for shard_idx in self._file_to_shards.get(file_key, []):
            pending = self._shard_pending_files.get(shard_idx)
            if pending is not None:
                pending.discard(file_key)
                if not pending:
                    # All files for this shard are downloaded!
                    newly_ready.append(shard_idx)
                    del self._shard_pending_files[shard_idx]

        return newly_ready

    def _mark_file_failed(self, file_key: str) -> None:
        """Mark a file as failed."""
        self._failed_files.add(file_key)
        # Note: shards with failed files will never become ready
        # This is intentional - they'll be handled during processing

    def iter_ready_batches(
        self,
        min_batch_size: int = 1,
        max_wait_sec: float = 1.0,
    ) -> Iterator[list]:
        """Yield batches of shard tasks ready for processing.

        Downloads files in parallel and yields shard tasks as soon as
        all their required files are downloaded. Processing can start
        while downloads continue.

        Args:
            min_batch_size: Minimum shards to accumulate before yielding
            max_wait_sec: Maximum time to wait for batch to fill

        Yields:
            Lists of ShardTask objects ready for processing
        """
        start_time = time.perf_counter()
        total_files = self.total_files

        if total_files == 0:
            print("[Download+Process] No HuggingFace files - all shards ready immediately")
            sys.stdout.flush()
            # All shards are ready immediately
            yield self.shard_tasks
            return

        print(f"[Download+Process] Starting overlapped download of {total_files} files "
              f"for {self.total_shards} shards (max_concurrent={self.max_concurrent_downloads})")
        sys.stdout.flush()

        # Track statistics
        downloaded = 0
        failed = 0
        shards_yielded = 0

        # Submit initial batch of downloads
        futures: list = []
        future_to_file: dict = {}
        pending_files = list(self._all_unique_files)

        while pending_files and len(futures) < self.max_concurrent_downloads:
            file_info = pending_files.pop(0)
            future = _download_hf_file.remote(
                file_info["repo_id"],
                file_info["filename"],
                file_info["revision"],
                self.cache_dir,
            )
            futures.append(future)
            file_key = _make_file_key(
                file_info["repo_id"],
                file_info["filename"],
                file_info["revision"],
            )
            future_to_file[future] = file_key

        # Yield any shards that were ready immediately (no HF files)
        if self._ready_shards:
            ready_tasks = [self._shard_index_to_task[i] for i in self._ready_shards]
            shards_yielded += len(ready_tasks)
            self._ready_shards.clear()
            yield ready_tasks

        # Process downloads and yield ready shards
        last_progress_time = start_time
        batch_start_time = time.perf_counter()
        ready_batch: list[int] = []

        while futures or ready_batch:
            # Wait for downloads (non-blocking if we have ready shards)
            if futures:
                timeout = 0.1 if ready_batch else max_wait_sec
                done, futures = ray.wait(futures, num_returns=1, timeout=timeout)

                for future in done:
                    file_key = future_to_file.pop(future, None)
                    try:
                        result = ray.get(future)
                        if result["status"] in ("downloaded", "cached"):
                            downloaded += 1
                            if file_key:
                                newly_ready = self._mark_file_downloaded(file_key)
                                ready_batch.extend(newly_ready)
                        else:
                            failed += 1
                            if file_key:
                                self._mark_file_failed(file_key)
                    except Exception as e:
                        failed += 1
                        logger.warning(f"Download task failed: {e}")
                        if file_key:
                            self._mark_file_failed(file_key)

                # Submit more downloads
                while pending_files and len(futures) < self.max_concurrent_downloads:
                    file_info = pending_files.pop(0)
                    future = _download_hf_file.remote(
                        file_info["repo_id"],
                        file_info["filename"],
                        file_info["revision"],
                        self.cache_dir,
                    )
                    futures.append(future)
                    file_key = _make_file_key(
                        file_info["repo_id"],
                        file_info["filename"],
                        file_info["revision"],
                    )
                    future_to_file[future] = file_key

            # Yield ready batch if we have enough or waited long enough
            batch_wait = time.perf_counter() - batch_start_time
            if ready_batch and (len(ready_batch) >= min_batch_size or batch_wait >= max_wait_sec or not futures):
                ready_tasks = [self._shard_index_to_task[i] for i in ready_batch]
                shards_yielded += len(ready_tasks)
                ready_batch.clear()
                batch_start_time = time.perf_counter()

                # Progress update before yielding
                current_time = time.perf_counter()
                elapsed = current_time - start_time
                rate = downloaded / max(elapsed, 0.001)
                print(f"[Download+Process] {downloaded}/{total_files} files, "
                      f"{shards_yielded}/{self.total_shards} shards ready ({rate:.1f} files/s)")
                sys.stdout.flush()

                yield ready_tasks

            # Progress reporting (every 5 seconds)
            current_time = time.perf_counter()
            if current_time - last_progress_time >= 5.0:
                last_progress_time = current_time
                elapsed = current_time - start_time
                rate = downloaded / max(elapsed, 0.001)
                print(f"[Download+Process] {downloaded}/{total_files} files, "
                      f"{shards_yielded}/{self.total_shards} shards ready ({rate:.1f} files/s)")
                sys.stdout.flush()

                if self.on_progress:
                    self.on_progress({
                        "phase": "downloading",
                        "detail": f"{downloaded}/{total_files} files, {shards_yielded} shards ready",
                        "elapsed_sec": elapsed,
                        "files_completed": downloaded,
                        "files_total": total_files,
                        "shards_ready": shards_yielded,
                    })

        elapsed = time.perf_counter() - start_time
        print(f"[Download+Process] Complete: {downloaded}/{total_files} files, "
              f"{shards_yielded}/{self.total_shards} shards in {elapsed:.1f}s "
              f"({failed} failed)")
        sys.stdout.flush()

    def get_stats(self) -> DownloadStats:
        """Get current download statistics."""
        return DownloadStats(
            total_files=self.total_files,
            downloaded_files=len(self._downloaded_files),
            cached_files=0,  # Can't distinguish from downloaded
            failed_files=len(self._failed_files),
            elapsed_sec=0.0,  # Updated during iteration
        )


def _collect_unique_hf_files(shard_tasks: list) -> list[dict[str, str]]:
    """Extract unique HuggingFace files from shard tasks.

    Deduplicates files across all shard tasks since the same file
    may appear in multiple shards.

    Args:
        shard_tasks: List of ShardTask objects with assignment_json

    Returns:
        List of unique file dicts with repo_id, filename, revision
    """
    seen = set()
    unique_files = []

    for task in shard_tasks:
        # Parse the assignment JSON
        assignment = json.loads(task.assignment_json)
        files = assignment.get("files", [])

        for file_info in files:
            # Only include HuggingFace files (have hf_repo_id)
            repo_id = file_info.get("hf_repo_id")
            if repo_id is None:
                continue

            filename = file_info.get("hf_filename")
            revision = file_info.get("hf_revision")

            # Create unique key for deduplication
            key = (repo_id, filename, revision or "")
            if key in seen:
                continue
            seen.add(key)

            unique_files.append({
                "repo_id": repo_id,
                "filename": filename,
                "revision": revision,
            })

    return unique_files


def parallel_predownload(
    shard_tasks: list,
    *,
    max_concurrent: int = 64,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    cache_dir: str | None = None,
) -> DownloadStats:
    """Pre-download all HuggingFace files in parallel before shard processing.

    This function collects all unique HuggingFace files from the shard tasks
    and downloads them in parallel using Ray tasks. The files are stored in
    the HuggingFace cache (controlled by cache_dir or HF_HOME), so subsequent
    access during shard processing will be instant cache hits.

    Rate limiting is implemented to avoid overwhelming the HuggingFace servers
    or the local network. The max_concurrent parameter controls how many
    downloads can be in flight simultaneously.

    Args:
        shard_tasks: List of ShardTask objects with assignment_json
        max_concurrent: Maximum concurrent downloads (rate limiting)
        on_progress: Optional callback for progress updates
        cache_dir: HuggingFace cache directory (defaults to HF_HOME/hub or ~/.cache/huggingface/hub)

    Returns:
        DownloadStats with counts and timing information
    """
    import os
    start_time = time.perf_counter()

    # Use HF_HOME if cache_dir not specified
    if cache_dir is None:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            cache_dir = os.path.join(hf_home, "hub")

    # Collect unique files
    unique_files = _collect_unique_hf_files(shard_tasks)
    total_files = len(unique_files)

    if total_files == 0:
        print("[Pre-download] No HuggingFace files found - skipping download phase")
        sys.stdout.flush()
        return DownloadStats(
            total_files=0,
            downloaded_files=0,
            cached_files=0,
            failed_files=0,
            elapsed_sec=0.0,
        )

    print(f"[Pre-download] Starting download of {total_files} unique files (max_concurrent={max_concurrent})")
    sys.stdout.flush()

    if on_progress:
        on_progress({
            "phase": "downloading",
            "detail": f"0/{total_files} files",
            "elapsed_sec": 0.0,
            "files_completed": 0,
            "files_total": total_files,
        })

    # Track statistics
    downloaded = 0
    cached = 0
    failed = 0
    completed = 0

    # Submit downloads with rate limiting
    futures = []
    pending_files = list(unique_files)  # Copy to avoid modifying original

    # Initial batch
    while pending_files and len(futures) < max_concurrent:
        file_info = pending_files.pop(0)
        future = _download_hf_file.remote(
            file_info["repo_id"],
            file_info["filename"],
            file_info["revision"],
            cache_dir,
        )
        futures.append(future)

    # Process completions and submit new downloads
    last_progress_time = start_time
    while futures:
        # Wait for at least one to complete
        done, futures = ray.wait(futures, num_returns=1, timeout=1.0)

        for future in done:
            try:
                result = ray.get(future)
                completed += 1

                if result["status"] == "downloaded":
                    downloaded += 1
                elif result["status"] == "cached":
                    cached += 1
                else:
                    failed += 1
                    if result["error"]:
                        logger.warning(f"Download failed: {result['repo_id']}/{result['filename']}: {result['error']}")
            except Exception as e:
                completed += 1
                failed += 1
                logger.warning(f"Download task failed: {e}")

        # Submit more downloads if we have capacity
        while pending_files and len(futures) < max_concurrent:
            file_info = pending_files.pop(0)
            future = _download_hf_file.remote(
                file_info["repo_id"],
                file_info["filename"],
                file_info["revision"],
                cache_dir,
            )
            futures.append(future)

        # Progress reporting (every 5 seconds or on completion)
        current_time = time.perf_counter()
        if current_time - last_progress_time >= 5.0 or completed == total_files:
            last_progress_time = current_time
            elapsed = current_time - start_time
            rate = completed / max(elapsed, 0.001)
            print(f"[Pre-download] {completed}/{total_files} files ({rate:.1f}/s, {elapsed:.0f}s elapsed)")
            sys.stdout.flush()
            if on_progress:
                on_progress({
                    "phase": "downloading",
                    "detail": f"{completed}/{total_files} files ({rate:.1f}/s)",
                    "elapsed_sec": elapsed,
                    "files_completed": completed,
                    "files_total": total_files,
                })

    elapsed = time.perf_counter() - start_time

    stats = DownloadStats(
        total_files=total_files,
        downloaded_files=downloaded,
        cached_files=cached,
        failed_files=failed,
        elapsed_sec=elapsed,
    )

    print(
        f"[Pre-download] Complete: {downloaded + cached}/{total_files} files in {elapsed:.1f}s "
        f"({(downloaded + cached) / max(elapsed, 0.001):.1f} files/s, {failed} failed)"
    )
    sys.stdout.flush()

    return stats


__all__ = [
    "parallel_predownload",
    "OverlappedDownloader",
    "DownloadStats",
]
