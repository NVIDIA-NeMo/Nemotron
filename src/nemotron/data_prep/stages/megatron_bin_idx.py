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
Megatron BinIdx Tokenization Stage - Tokenizes text to Megatron .bin/.idx format.

This stage processes ShardWorkItems and:
1. Reads input files (JSONL, Parquet, etc.)
2. Tokenizes text using the configured tokenizer
3. Writes Megatron-compatible .bin/.idx files
4. Writes receipts for idempotency and progress tracking

The stage owns all receipt writing - it's the single source of truth for
checkpoint/resume semantics.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import cosmos_xenna.pipelines.v1 as pipelines_v1

from nemotron.data_prep.utils.filesystem import get_filesystem, read_json
from nemotron.data_prep.stages.context import PipelineContext
from nemotron.data_prep.core.work_items import ShardWorkItem


@dataclass(frozen=True)
class BinIdxTokenizationStageConfig:
    """Configuration for BinIdxTokenizationStage.

    This stage tokenizes text to Megatron .bin/.idx format. It's memory-intensive
    so cpus_per_worker is used as a proxy for memory allocation.

    Attributes:
        cpus_per_worker: CPU request for tokenization workers. This acts as a
            memory proxy - 4 CPUs ~ 32GB on typical nodes. Adjust based on
            cluster node shapes. Default 4.
    """

    cpus_per_worker: int = 4

    def __post_init__(self) -> None:
        if self.cpus_per_worker <= 0:
            raise ValueError(f"cpus_per_worker must be positive, got {self.cpus_per_worker}")


class BinIdxTokenizationStage(pipelines_v1.Stage[ShardWorkItem, ShardWorkItem]):
    """
    Tokenization stage: process shards and write Megatron bin/idx files.

    This stage is the SINGLE receipt writer - it handles:
    - Checking if shard is already completed (idempotency)
    - Writing "started" receipt before processing
    - Calling tokenization core
    - Writing "completed" receipt with stats

    Memory management:
    - Uses Resources(cpus=K) as memory proxy for cluster autoscaling
    - StageSpec should set slots_per_actor=1 to prevent concurrent tasks

    Args:
        stage_config: Stage-specific configuration (BinIdxTokenizationStageConfig)
        pipeline_context: Shared runtime context (PipelineContext). Must have
            resolved_tokenizer and run_hash set.
    """

    def __init__(
        self,
        stage_config: BinIdxTokenizationStageConfig,
        pipeline_context: PipelineContext,
    ) -> None:
        # Validate required context fields
        if pipeline_context.resolved_tokenizer is None:
            raise ValueError("BinIdxTokenizationStage requires resolved_tokenizer in PipelineContext")
        if pipeline_context.run_hash is None:
            raise ValueError("BinIdxTokenizationStage requires run_hash in PipelineContext")

        self._cfg = stage_config
        self._ctx = pipeline_context
        self._tokenize = None
        self._fs = None

    @property
    def stage_batch_size(self) -> int:
        """Process one shard at a time."""
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        """Memory proxy: request CPUs to limit concurrency based on memory."""
        return pipelines_v1.Resources(cpus=self._cfg.cpus_per_worker, gpus=0)

    @property
    def env_info(self) -> pipelines_v1.RuntimeEnv:
        """Runtime environment with HF credentials for tokenizer loading."""
        return self._ctx.hf_runtime_env()

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        """Initialize tokenizer and filesystem on worker."""
        from nemotron.data_prep.core.providers import create_tokenizer

        self._tokenize = create_tokenizer(self._ctx.resolved_tokenizer)
        self._fs, _ = get_filesystem(self._ctx.output_root)

    def process_data(self, tasks: list[ShardWorkItem]) -> list[None]:
        """Process shards: check cache, tokenize, write receipts.

        Returns empty list since this is the terminal stage (no downstream output).
        cosmos-xenna requires process_data to return a list, not None.
        """
        for task in tasks:
            self._process_shard(task)
        return []  # Terminal stage - no output, but must return list for xenna

    def _process_shard(self, task: ShardWorkItem) -> None:
        """Process a single shard with idempotency and receipt handling."""
        receipt_path = f"{task.receipts_dir.rstrip('/')}/shard_{task.shard_index:06d}.json"
        shard_dir = task.output_dir

        # Check if already completed (idempotency)
        if self._is_completed(receipt_path, task.plan_hash, shard_dir):
            return

        # Write "started" receipt
        self._write_started_receipt(receipt_path, task)

        try:
            # Process the shard
            stats, files = self._tokenize_shard(task)

            # Write "completed" receipt
            self._write_completed_receipt(receipt_path, task, stats, files)

        except Exception as e:
            # Write "failed" receipt and re-raise
            self._write_failed_receipt(receipt_path, task, e)
            raise

    def _is_completed(self, receipt_path: str, plan_hash: str, shard_dir: str) -> bool:
        """Check if shard is already completed with valid outputs.

        This check must match get_pending_shards() in planning.py to avoid
        skipping shards that were correctly identified as pending due to
        missing output files.
        """
        if not self._fs.exists(receipt_path):
            return False

        try:
            r = read_json(self._fs, receipt_path)
            if r.get("status") != "completed" or r.get("plan_hash") != plan_hash:
                return False

            # Verify output files exist for non-empty shards
            # This matches the check in get_pending_shards()
            stats = r.get("stats", {}) or {}
            if int(stats.get("num_sequences", 0) or 0) > 0:
                files = r.get("files", {}) or {}
                bin_info = files.get("bin", {}) or {}
                idx_info = files.get("idx", {}) or {}
                bin_path = bin_info.get("path", "")
                idx_path = idx_info.get("path", "")

                if not bin_path or not idx_path:
                    return False

                full_bin = f"{shard_dir}/{bin_path}"
                full_idx = f"{shard_dir}/{idx_path}"
                if not (self._fs.exists(full_bin) and self._fs.exists(full_idx)):
                    return False

            return True
        except Exception:
            return False  # Corrupted receipt, reprocess

    def _write_started_receipt(self, receipt_path: str, task: ShardWorkItem) -> None:
        """Write receipt indicating shard processing has started."""
        receipt = {
            "status": "started",
            "run_hash": self._ctx.run_hash,
            "dataset_name": task.dataset_name,
            "plan_hash": task.plan_hash,
            "shard_index": task.shard_index,
            "started_at": time.time(),
        }
        self._write_json_atomic(receipt_path, receipt)

    def _write_completed_receipt(
        self,
        receipt_path: str,
        task: ShardWorkItem,
        stats: dict[str, Any],
        files: dict[str, Any],
    ) -> None:
        """Write receipt indicating successful completion."""
        receipt = {
            "status": "completed",
            "run_hash": self._ctx.run_hash,
            "dataset_name": task.dataset_name,
            "plan_hash": task.plan_hash,
            "shard_index": task.shard_index,
            "stats": stats,
            "files": files,
            "completed_at": time.time(),
        }
        self._write_json_atomic(receipt_path, receipt)

    def _write_failed_receipt(
        self,
        receipt_path: str,
        task: ShardWorkItem,
        error: Exception,
    ) -> None:
        """Write receipt indicating failure."""
        import traceback

        receipt = {
            "status": "failed",
            "run_hash": self._ctx.run_hash,
            "dataset_name": task.dataset_name,
            "plan_hash": task.plan_hash,
            "shard_index": task.shard_index,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "failed_at": time.time(),
        }
        self._write_json_atomic(receipt_path, receipt)

    def _write_json_atomic(self, path: str, payload: dict[str, Any]) -> None:
        """Write JSON atomically using tmp + rename pattern."""
        tmp = f"{path}.tmp"
        with self._fs.open(tmp, "w") as f:
            json.dump(payload, f)
        try:
            self._fs.rm(path)
        except Exception:
            pass
        self._fs.mv(tmp, path)

    def _tokenize_shard(self, task: ShardWorkItem) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Tokenize a shard and write bin/idx files.

        Returns:
            (stats, files) tuple where:
            - stats: {"num_sequences": N, "total_tokens": N, ...}
            - files: {"bin": {"path": ..., "bytes": ...}, "idx": {...}}
        """
        from nemotron.data_prep.core.shard_processor import process_binidx_shard_files_core

        stats, files = process_binidx_shard_files_core(
            tokenize=self._tokenize,
            text_field=task.text_field,
            min_doc_chars=task.min_doc_chars,
            max_doc_tokens=task.max_doc_tokens,
            dtype=task.dtype,
            max_rows=task.max_rows,
            shard_index=task.shard_index,
            assignment=task.assignment,
            output_dir=task.output_dir,
            output_fs=self._fs,
        )

        return stats, files
