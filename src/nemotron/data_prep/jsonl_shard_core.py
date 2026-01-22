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

"""Core JSONL shard processing (retry-safe, engine-agnostic)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from typing import Any, Literal

import pyarrow.parquet as pq
from fsspec import AbstractFileSystem

from nemotron.data_prep.config import FileInfo
from nemotron.data_prep.filesystem import ensure_dir, get_filesystem, read_json, write_json
from nemotron.data_prep.formats.jsonl_dataset import JsonlDatasetBuilder

Transform = Callable[[dict], dict | None]


def process_jsonl_shard_core(
    *,
    shard_index: int,
    files: list[dict] | list[FileInfo],
    output_dir: str,
    receipts_dir: str,
    output_fs: AbstractFileSystem,
    text_field: str,
    transform: Transform | None,
    compression: Literal["none", "zstd"],
    max_rows: int | None,
) -> dict[str, Any]:
    """Process a JSONL shard with retry-safe atomic commits."""
    shard_id = f"shard_{shard_index:06d}"
    ext = ".jsonl.zst" if compression == "zstd" else ".jsonl"
    jsonl_path = f"{output_dir}/{shard_id}{ext}"
    jsonl_tmp = f"{jsonl_path}.tmp"
    receipt_path = f"{receipts_dir}/{shard_id}.json"

    if output_fs.exists(receipt_path):
        try:
            receipt = read_json(output_fs, receipt_path)
            if receipt.get("status") == "completed":
                return receipt.get("stats", {})
        except Exception:
            pass

    ensure_dir(output_fs, output_dir)
    ensure_dir(output_fs, receipts_dir)

    file_infos = [FileInfo(**f) if isinstance(f, dict) else f for f in files]
    input_file_paths = [f.path for f in file_infos]

    if not file_infos:
        return _write_empty_receipt(
            shard_id=shard_id,
            shard_index=shard_index,
            input_files=input_file_paths,
            receipt_path=receipt_path,
            output_fs=output_fs,
        )

    rows_processed = 0
    with output_fs.open(jsonl_tmp, "wb") as f:
        builder = JsonlDatasetBuilder(
            file=f,
            transform=transform,
            compression=compression,
        )

        for file_info in file_infos:
            if max_rows and rows_processed >= max_rows:
                break
            rows_processed = _process_file(
                file_info=file_info,
                builder=builder,
                rows_processed=rows_processed,
                max_rows=max_rows,
            )

        builder.finalize()
        total_bytes, checksum = builder.get_info()
        stats = builder.get_stats()

    if stats.get("num_records", 0) == 0:
        try:
            output_fs.rm(jsonl_tmp)
        except Exception:
            pass
        return _write_empty_receipt(
            shard_id=shard_id,
            shard_index=shard_index,
            input_files=input_file_paths,
            receipt_path=receipt_path,
            output_fs=output_fs,
        )

    output_fs.rename(jsonl_tmp, jsonl_path)

    receipt = {
        "shard_id": shard_id,
        "shard_index": shard_index,
        "status": "completed",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_files": input_file_paths,
        "output_file": f"{shard_id}{ext}",
        "total_bytes": total_bytes,
        "checksum": checksum,
        "stats": {
            "total_tokens": 0,
            "num_records": stats.get("num_records", 0),
            "num_skipped": stats.get("num_skipped", 0),
            "total_bytes": stats.get("total_bytes", 0),
        },
    }

    write_json(output_fs, receipt_path, receipt)
    return receipt["stats"]


def _process_file(
    *,
    file_info: FileInfo,
    builder: JsonlDatasetBuilder,
    rows_processed: int,
    max_rows: int | None,
) -> int:
    local_path = _resolve_file_path(file_info)
    input_path = local_path if file_info.hf_repo_id is not None else (file_info.local_path or file_info.path)
    input_fs, normalized = get_filesystem(input_path)

    # Use original filename for format detection (hf_hub_download returns blob path without extension)
    format_check_path = (file_info.hf_filename or normalized) if file_info.hf_repo_id else normalized
    is_parquet = format_check_path.endswith(".parquet") or not (
        format_check_path.endswith(".jsonl") or format_check_path.endswith(".json")
    )

    if is_parquet:
        record_iter = _iter_parquet_records(normalized, input_fs)
    else:
        record_iter = _iter_jsonl_records(normalized, input_fs)

    for record in record_iter:
        if max_rows and rows_processed >= max_rows:
            break
        builder.add_record(record)
        rows_processed += 1

    return rows_processed


def _resolve_file_path(file_info: FileInfo) -> str:
    if file_info.hf_repo_id is not None:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=file_info.hf_repo_id,
            filename=file_info.hf_filename,
            revision=file_info.hf_revision,
            repo_type="dataset",
            local_files_only=True,  # Files should be pre-downloaded by HfPredownloadStage
        )

    return file_info.local_path or file_info.path


def _iter_parquet_records(path: str, fs: AbstractFileSystem) -> Iterator[dict]:
    with fs.open(path, "rb") as f:
        parquet_file = pq.ParquetFile(f)
        for batch in parquet_file.iter_batches(batch_size=10000):
            table = batch.to_pydict()
            num_rows = len(next(iter(table.values()))) if table else 0
            for i in range(num_rows):
                yield {k: v[i] for k, v in table.items()}


def _iter_jsonl_records(path: str, fs: AbstractFileSystem) -> Iterator[dict]:
    with fs.open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _write_empty_receipt(
    *,
    shard_id: str,
    shard_index: int,
    input_files: list[str],
    receipt_path: str,
    output_fs: AbstractFileSystem,
) -> dict[str, Any]:
    receipt = {
        "shard_id": shard_id,
        "shard_index": shard_index,
        "status": "completed",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_files": input_files,
        "output_file": None,
        "total_bytes": 0,
        "checksum": "xxh64:empty",
        "stats": {
            "total_tokens": 0,
            "num_records": 0,
            "num_skipped": 0,
            "total_bytes": 0,
        },
    }

    write_json(output_fs, receipt_path, receipt)
    return receipt["stats"]

