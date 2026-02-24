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

from dataclasses import dataclass
from typing import Any

import cosmos_xenna.pipelines.v1 as pipelines_v1

from nemotron.data_prep.utils.filesystem import get_filesystem
from nemotron.data_prep.core.receipt import ReceiptManager
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
        self._receipts: ReceiptManager | None = None

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
        self._receipts = ReceiptManager(self._fs, self._ctx.run_hash)

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
        receipts = self._get_receipts()
        rpath = receipts.receipt_path(task.receipts_dir, task.shard_index)

        if receipts.is_completed(
            rpath, task.plan_hash,
            verify_outputs=lambda: self._outputs_exist(task),
        ):
            return

        meta = dict(
            plan_hash=task.plan_hash,
            shard_index=task.shard_index,
            dataset_name=task.dataset_name,
        )
        receipts.write_started(rpath, **meta)

        try:
            stats, files = self._tokenize_shard(task)
            receipts.write_completed(rpath, stats=stats, files=files, **meta)
        except Exception as e:
            receipts.write_failed(rpath, error=e, **meta)
            raise

    def _outputs_exist(self, task: ShardWorkItem) -> bool:
        """Verify output files exist for a completed receipt."""
        rpath = self._get_receipts().receipt_path(task.receipts_dir, task.shard_index)
        from nemotron.data_prep.utils.filesystem import read_json
        try:
            r = read_json(self._fs, rpath)
            stats = r.get("stats", {}) or {}
            if int(stats.get("num_sequences", 0) or 0) == 0:
                return True
            files = r.get("files", {}) or {}
            bin_info = files.get("bin", {}) or {}
            idx_info = files.get("idx", {}) or {}
            bin_path = bin_info.get("path", "")
            idx_path = idx_info.get("path", "")
            if not bin_path or not idx_path:
                return False
            return (
                self._fs.exists(f"{task.output_dir}/{bin_path}")
                and self._fs.exists(f"{task.output_dir}/{idx_path}")
            )
        except Exception:
            return False

    def _get_receipts(self) -> ReceiptManager:
        if self._receipts is None:
            self._receipts = ReceiptManager(self._fs, self._ctx.run_hash)
        return self._receipts

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
