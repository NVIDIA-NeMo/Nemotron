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

"""Shared receipt lifecycle for shard processing.

All terminal stages (BinIdxTokenizationStage, PackedSftParquetStage,
JsonlShardStage) use ReceiptManager for atomic receipt writes,
started-race detection, and pluggable output verification.

Pipeline-specific data flows through stats, files, and **extra kwargs.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from typing import Any

from fsspec import AbstractFileSystem

from nemotron.data_prep.utils.filesystem import read_json


class ReceiptManager:
    """Atomic receipt lifecycle for shard processing.

    Shared mechanism: atomic JSON write, started-race detection,
    pluggable output verification. Pipeline-specific data goes
    through stats/files/extra kwargs.
    """

    def __init__(self, fs: AbstractFileSystem, run_hash: str) -> None:
        self._fs = fs
        self._run_hash = run_hash

    def receipt_path(self, receipts_dir: str, shard_index: int) -> str:
        return f"{receipts_dir.rstrip('/')}/shard_{shard_index:06d}.json"

    def is_completed(
        self,
        path: str,
        plan_hash: str,
        verify_outputs: Callable[[], bool] | None = None,
        started_timeout_min: float = 30,
    ) -> bool:
        """Check if shard is done.

        Handles three cases beyond simple status check:
        - Race detection: respects another worker's "started" claim
          for up to started_timeout_min before assuming crash.
        - Output verification: stage-provided callback confirms outputs
          exist on disk. Pretrain checks bin/idx, JSONL checks output_file,
          SFT checks parquet. If None, receipt alone is sufficient.
        - Corrupted receipts: treated as incomplete (reprocess).
        """
        if not self._fs.exists(path):
            return False
        try:
            r = read_json(self._fs, path)
            status = r.get("status")

            if status == "started" and r.get("plan_hash") == plan_hash:
                elapsed = (time.time() - r.get("started_at", 0)) / 60
                if elapsed < started_timeout_min:
                    return True  # another worker owns it

            if status != "completed" or r.get("plan_hash") != plan_hash:
                return False

            if verify_outputs and not verify_outputs():
                return False

            return True
        except Exception:
            return False

    def write_started(self, path: str, *, plan_hash: str, shard_index: int, **extra: Any) -> None:
        self._write_atomic(path, {
            "status": "started",
            "plan_hash": plan_hash,
            "shard_index": shard_index,
            "started_at": time.time(),
            **extra,
        })

    def write_completed(
        self,
        path: str,
        *,
        plan_hash: str,
        shard_index: int,
        stats: dict[str, Any],
        files: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        self._write_atomic(path, {
            "status": "completed",
            "plan_hash": plan_hash,
            "shard_index": shard_index,
            "stats": stats,
            "files": files or {},
            "completed_at": time.time(),
            **extra,
        })

    def write_failed(
        self,
        path: str,
        *,
        plan_hash: str,
        shard_index: int,
        error: Exception,
        **extra: Any,
    ) -> None:
        self._write_atomic(path, {
            "status": "failed",
            "plan_hash": plan_hash,
            "shard_index": shard_index,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "failed_at": time.time(),
            **extra,
        })

    def _write_atomic(self, path: str, payload: dict[str, Any]) -> None:
        payload["run_hash"] = self._run_hash
        tmp = f"{path}.tmp"
        with self._fs.open(tmp, "w") as f:
            json.dump(payload, f)
        try:
            self._fs.rm(path)
        except Exception:
            pass
        self._fs.mv(tmp, path)


__all__ = ["ReceiptManager"]
