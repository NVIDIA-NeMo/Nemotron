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
Packed SFT Parquet Stage - Tokenize+mask to SequenceSpool, then pack to Parquet.

This stage processes SftShardWorkItems and:
1) Generates/updates SequenceSpool for the shard (idempotent via manifest.json)
2) Computes packing assignment and writes shard_*.parquet (packed SFT Parquet spec)
3) Writes receipts for checkpoint/resume and progress tracking

The stage owns all receipt writing (single source of truth).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import cosmos_xenna.pipelines.v1 as pipelines_v1
import numpy as np

from nemotron.data_prep.core.chat_sft_shard_core import (
    process_chat_sft_parquet_from_spool_core,
    process_chat_sft_spool_core,
)
from nemotron.data_prep.utils.filesystem import get_filesystem, read_json
from nemotron.data_prep.stages.context import PipelineContext
from nemotron.data_prep.core.work_items import SftShardWorkItem


@dataclass(frozen=True)
class PackedSftParquetStageConfig:
    """Configuration for PackedSftParquetStage.

    This stage tokenizes, masks, and packs SFT data to Parquet format.
    Like BinIdxTokenizationStage, it's memory-intensive.

    Attributes:
        cpus_per_worker: CPU request for tokenization workers. This acts as a
            memory proxy - 4 CPUs ~ 32GB on typical nodes. Adjust based on
            cluster node shapes. Default 4.
    """

    cpus_per_worker: int = 4

    def __post_init__(self) -> None:
        if self.cpus_per_worker <= 0:
            raise ValueError(f"cpus_per_worker must be positive, got {self.cpus_per_worker}")


class PackedSftParquetStage(pipelines_v1.Stage[SftShardWorkItem, SftShardWorkItem]):
    """Shard processing stage for xenna-native packed SFT Parquet output.

    Args:
        stage_config: Stage-specific configuration (PackedSftParquetStageConfig)
        pipeline_context: Shared runtime context (PipelineContext). Must have
            resolved_tokenizer and run_hash set.
    """

    def __init__(
        self,
        stage_config: PackedSftParquetStageConfig,
        pipeline_context: PipelineContext,
    ) -> None:
        # Validate required context fields
        if pipeline_context.resolved_tokenizer is None:
            raise ValueError("PackedSftParquetStage requires resolved_tokenizer in PipelineContext")
        if pipeline_context.run_hash is None:
            raise ValueError("PackedSftParquetStage requires run_hash in PipelineContext")

        self._cfg = stage_config
        self._ctx = pipeline_context
        self._tokenizer = None
        self._fs = None

    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=self._cfg.cpus_per_worker, gpus=0)

    @property
    def env_info(self) -> pipelines_v1.RuntimeEnv:
        return self._ctx.hf_runtime_env()

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._ctx.resolved_tokenizer["model"],
            revision=self._ctx.resolved_tokenizer.get("resolved_revision"),
            trust_remote_code=self._ctx.resolved_tokenizer.get("trust_remote_code", False),
            local_files_only=True,
        )
        self._fs, _ = get_filesystem(self._ctx.output_root)

    def process_data(self, tasks: list[SftShardWorkItem]) -> list[None]:
        """Process shards: tokenize, mask, spool, pack to Parquet.

        Returns empty list since this is the terminal stage (no downstream output).
        cosmos-xenna requires process_data to return a list, not None.
        """
        for task in tasks:
            self._process_shard(task)
        return []  # Terminal stage - no output, but must return list for xenna

    def _process_shard(self, task: SftShardWorkItem) -> None:
        receipt_path = f"{task.receipts_dir.rstrip('/')}/shard_{task.shard_index:06d}.json"

        if self._is_completed(receipt_path, task.plan_hash):
            return

        self._write_started_receipt(receipt_path, task)

        try:
            self._run_spool_and_pack(task)
            self._write_completed_receipt(receipt_path, task)
        except Exception as e:
            self._write_failed_receipt(receipt_path, task, e)
            raise

    def _is_completed(self, receipt_path: str, plan_hash: str) -> bool:
        if not self._fs.exists(receipt_path):
            return False
        try:
            r = read_json(self._fs, receipt_path)
            return r.get("status") == "completed" and r.get("plan_hash") == plan_hash
        except Exception:
            return False

    def _write_started_receipt(self, receipt_path: str, task: SftShardWorkItem) -> None:
        payload = {
            "status": "started",
            "run_hash": self._ctx.run_hash,
            "dataset_name": task.dataset_name,
            "plan_hash": task.plan_hash,
            "shard_index": task.shard_index,
            "started_at": time.time(),
        }
        self._write_json_atomic(receipt_path, payload)

    def _write_completed_receipt(self, receipt_path: str, task: SftShardWorkItem) -> None:
        shard_id = f"shard_{task.shard_index:06d}"
        parquet_rel = f"{shard_id}.parquet"
        parquet_path = f"{task.output_dir.rstrip('/')}/{parquet_rel}"

        try:
            parquet_bytes = int(self._fs.size(parquet_path))
        except Exception:
            parquet_bytes = 0

        stats, files = process_chat_sft_parquet_from_spool_core(
            shard_index=task.shard_index,
            output_dir=task.output_dir,
            spool_dir=self._resolve_spool_dir(task),
            output_fs=self._fs,
            pack_size=int(task.pack_size),
            algorithm=str(task.algorithm),
            dtype=np.dtype(task.dtype),
            seed=task.seed,
            parquet_row_group_size=int(task.parquet_row_group_size),
            parquet_compression=str(task.parquet_compression),
        )

        # Ensure receipt has a stable files entry for progress/aggregation.
        if not isinstance(files, dict):
            files = {}
        if "parquet" not in files:
            files["parquet"] = {"path": parquet_rel, "bytes": parquet_bytes, "checksum": "xxh64:unknown"}
        else:
            try:
                files["parquet"].setdefault("bytes", parquet_bytes)
                files["parquet"].setdefault("checksum", "xxh64:unknown")
                files["parquet"].setdefault("path", parquet_rel)
            except Exception:
                files["parquet"] = {"path": parquet_rel, "bytes": parquet_bytes, "checksum": "xxh64:unknown"}

        payload = {
            "status": "completed",
            "run_hash": self._ctx.run_hash,
            "dataset_name": task.dataset_name,
            "plan_hash": task.plan_hash,
            "shard_index": task.shard_index,
            "stats": stats,
            "files": files,
            "completed_at": time.time(),
        }
        self._write_json_atomic(receipt_path, payload)

    def _write_failed_receipt(self, receipt_path: str, task: SftShardWorkItem, error: Exception) -> None:
        import traceback

        payload = {
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
        self._write_json_atomic(receipt_path, payload)

    def _write_json_atomic(self, path: str, payload: dict[str, Any]) -> None:
        tmp = f"{path}.tmp"
        with self._fs.open(tmp, "w") as f:
            json.dump(payload, f)
        try:
            self._fs.rm(path)
        except Exception:
            pass
        self._fs.mv(tmp, path)

    def _resolve_spool_dir(self, task: SftShardWorkItem) -> str:
        if task.spool_dir:
            return task.spool_dir
        shard_id = f"shard_{task.shard_index:06d}"
        return f"{task.output_dir.rstrip('/')}/spool/{shard_id}"

    def _run_spool_and_pack(self, task: SftShardWorkItem) -> None:
        spool_dir = self._resolve_spool_dir(task)

        process_chat_sft_spool_core(
            shard_index=task.shard_index,
            files=task.assignment.get("files", []),
            output_dir=task.output_dir,
            receipts_dir=task.receipts_dir,
            spool_dir=spool_dir,
            output_fs=self._fs,
            tokenizer=self._tokenizer,
            messages_field=task.messages_field,
            tools_field=task.tools_field,
            pack_size=int(task.pack_size),
            algorithm=str(task.algorithm),
            dtype=np.dtype(task.dtype),
            chat_template=task.chat_template,
            max_doc_tokens=task.max_doc_tokens,
            max_rows=task.max_rows,
            seed=task.seed,
            used_in_filter=task.used_in_filter,
            used_in_field=task.used_in_field,
        )

        # Packing to Parquet is executed during _write_completed_receipt() to ensure stats/files are captured there.


__all__ = ["PackedSftParquetStage"]