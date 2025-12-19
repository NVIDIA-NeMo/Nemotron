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

"""ShardProcessor Ray actor for parallel shard processing."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import ray
from fsspec import filesystem

from nemotron.data_prep.config import FileInfo
from nemotron.data_prep.filesystem import ensure_dir, get_filesystem, write_json
from nemotron.data_prep.formats.indexed_dataset import IndexedDatasetBuilder
from nemotron.data_prep.providers import create_tokenizer

logger = logging.getLogger(__name__)


# =============================================================================
# Core Processing Function (Shared by Legacy Actor and Ray Data UDF)
# =============================================================================


def process_binidx_shard_core(
    *,
    # Tokenizer (pre-initialized callable, NOT config dict)
    tokenize: Callable[[list[str]], list[list[int]]],
    text_field: str,
    min_doc_chars: int | None,
    max_doc_tokens: int | None,
    dtype: str,
    max_rows: int | None,
    # Shard identity
    shard_index: int,
    assignment: dict[str, Any],
    plan_hash: str,
    # Output locations
    output_dir: str,
    receipts_dir: str,
    output_fs: Any,  # fsspec filesystem for OUTPUT
) -> dict[str, Any]:
    """Core implementation of binidx shard processing.

    This function contains the actual processing logic, shared by:
    - Legacy ShardProcessor actor (process_shard method)
    - Ray Data BinIdxShardTaskUDF

    IMPORTANT: tokenize is a pre-initialized callable, not a config dict.
    The caller (actor or UDF) is responsible for creating the tokenizer
    once during initialization to amortize the cost across shards.

    Flow:
    0. Check receipt (idempotency for Ray retries)
    1. Read assigned files (parquet/jsonl) - using per-file filesystems
    2. Filter by min_doc_chars
    3. Tokenize documents
    4. Write .bin/.idx files using ATOMIC COMMIT PROTOCOL
    5. Write receipt JSON
    6. Return stats dict

    Args:
        tokenize: Pre-initialized tokenizer callable (texts -> token lists)
        text_field: Column name containing text
        min_doc_chars: Minimum document length filter
        max_doc_tokens: Maximum tokens per document (truncation)
        dtype: Numpy dtype for tokens (e.g., "int32")
        max_rows: Maximum rows to process (for testing)
        shard_index: Index of this shard
        assignment: ShardAssignment as dict
        plan_hash: Hash of the shard plan
        output_dir: Directory for output files
        receipts_dir: Directory for receipt files
        output_fs: fsspec filesystem for OUTPUT (input files use own protocols)

    Returns:
        Stats dict with keys: total_tokens, num_sequences, timing, etc.
    """
    total_start = time.perf_counter()
    timing: dict[str, float] = {}

    shard_id = f"shard_{shard_index:06d}"
    bin_path = f"{output_dir}/{shard_id}.bin"
    idx_path = f"{output_dir}/{shard_id}.idx"
    receipt_path = f"{receipts_dir}/{shard_id}.json"

    # IDEMPOTENCY: If receipt exists and is completed, skip processing
    # This makes "at-least-once" execution safe for Ray retries
    if output_fs.exists(receipt_path):
        try:
            with output_fs.open(receipt_path, "r") as f:
                existing = json.load(f)
            if existing.get("status") == "completed":
                logger.debug(f"Shard {shard_id} already completed, skipping")
                return existing.get("stats", {})
        except Exception:
            pass  # Receipt corrupted, reprocess

    np_dtype = np.dtype(dtype)

    # Ensure output directories
    ensure_dir(output_fs, output_dir)
    ensure_dir(output_fs, receipts_dir)

    # Stats tracking
    stats: dict[str, Any] = {
        "num_input_rows": 0,
        "num_filtered": 0,
        "num_truncated": 0,
        "num_errors": 0,
    }

    files = [FileInfo(**f) for f in assignment["files"]]
    input_file_paths = [f.path for f in files]

    # Handle empty assignment
    if not files:
        return _write_empty_receipt_core(
            shard_id,
            shard_index,
            plan_hash,
            input_file_paths,
            stats,
            receipt_path,
            output_fs,
            timing,
            total_start,
        )

    # ATOMIC COMMIT PROTOCOL for retry safety:
    # 1. Write to temp paths (.tmp suffix)
    # 2. Rename temp -> final (atomic on most filesystems)
    # 3. Write receipt LAST (signals completion)
    # This ensures partial writes from retries don't corrupt output.

    bin_tmp = f"{bin_path}.tmp"
    idx_tmp = f"{idx_path}.tmp"

    # Track download time separately
    download_start = time.perf_counter()

    # Process files and write shard to TEMP path
    read_start = time.perf_counter()
    timing["time_download_sec"] = read_start - download_start

    with output_fs.open(bin_tmp, "wb") as bin_file:
        builder = IndexedDatasetBuilder(bin_file, dtype=np_dtype)

        # Track rows processed across files for max_rows limit
        rows_processed = 0
        tokenize_time_total = 0.0

        # Process files SEQUENTIALLY for determinism
        for file_info in files:
            # Use per-file filesystem (input may be HF/S3/local, differs from output)
            input_fs, _ = get_filesystem(file_info.path)

            rows_processed, file_tokenize_time = _process_file_core(
                file_info=file_info,
                builder=builder,
                stats=stats,
                input_fs=input_fs,
                text_field=text_field,
                min_doc_chars=min_doc_chars,
                max_doc_tokens=max_doc_tokens,
                tokenize=tokenize,
                rows_processed=rows_processed,
                max_rows=max_rows,
            )
            tokenize_time_total += file_tokenize_time

            # Stop if we've hit max_rows
            if max_rows and rows_processed >= max_rows:
                break

        bin_bytes, bin_checksum = builder.get_bin_info()
        builder_stats = builder.get_stats()

    read_end = time.perf_counter()
    timing["time_read_sec"] = read_end - read_start - tokenize_time_total
    timing["time_tokenize_sec"] = tokenize_time_total

    # Handle empty result (all rows filtered)
    if builder_stats["num_sequences"] == 0:
        # Remove empty temp file
        try:
            output_fs.rm(bin_tmp)
        except Exception:
            pass
        return _write_empty_receipt_core(
            shard_id,
            shard_index,
            plan_hash,
            input_file_paths,
            stats,
            receipt_path,
            output_fs,
            timing,
            total_start,
        )

    # Write index to TEMP path
    write_start = time.perf_counter()
    with output_fs.open(idx_tmp, "wb") as idx_file:
        idx_bytes, idx_checksum = builder.write_index(idx_file)

    # ATOMIC RENAME: temp -> final
    # On S3/GCS this is a copy+delete, but still prevents partial reads
    output_fs.rename(bin_tmp, bin_path)
    output_fs.rename(idx_tmp, idx_path)

    write_end = time.perf_counter()
    timing["time_write_sec"] = write_end - write_start

    # Write receipt LAST (signals successful completion)
    receipt = {
        "shard_id": shard_id,
        "shard_index": shard_index,
        "plan_hash": plan_hash,
        "status": "completed",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_files": input_file_paths,
        "files": {
            "bin": {
                "path": f"{shard_id}.bin",
                "bytes": bin_bytes,
                "checksum": bin_checksum,
            },
            "idx": {
                "path": f"{shard_id}.idx",
                "bytes": idx_bytes,
                "checksum": idx_checksum,
            },
        },
        "stats": {**builder_stats, **stats},
    }

    write_json(output_fs, receipt_path, receipt)

    # Add timing to stats
    timing["time_total_sec"] = time.perf_counter() - total_start
    return {**receipt["stats"], **timing}


def _write_empty_receipt_core(
    shard_id: str,
    shard_index: int,
    plan_hash: str,
    input_files: list[str],
    stats: dict[str, Any],
    receipt_path: str,
    output_fs: Any,
    timing: dict[str, float],
    total_start: float,
) -> dict[str, Any]:
    """Write receipt for empty shard (CRITICAL for resume correctness)."""
    receipt = {
        "shard_id": shard_id,
        "shard_index": shard_index,
        "plan_hash": plan_hash,
        "status": "completed",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_files": input_files,
        "files": {
            "bin": {
                "path": f"{shard_id}.bin",
                "bytes": 0,
                "checksum": "xxh64:empty",
            },
            "idx": {
                "path": f"{shard_id}.idx",
                "bytes": 0,
                "checksum": "xxh64:empty",
            },
        },
        "stats": {
            "num_sequences": 0,
            "total_tokens": 0,
            "min_length": 0,
            "max_length": 0,
            **stats,
        },
    }

    write_json(output_fs, receipt_path, receipt)

    # Add timing to stats
    timing["time_total_sec"] = time.perf_counter() - total_start
    timing.setdefault("time_download_sec", 0.0)
    timing.setdefault("time_read_sec", 0.0)
    timing.setdefault("time_tokenize_sec", 0.0)
    timing.setdefault("time_write_sec", 0.0)
    return {**receipt["stats"], **timing}


def _process_file_core(
    *,
    file_info: FileInfo,
    builder: IndexedDatasetBuilder,
    stats: dict[str, Any],
    input_fs: Any,
    text_field: str,
    min_doc_chars: int | None,
    max_doc_tokens: int | None,
    tokenize: Callable[[list[str]], list[list[int]]],
    rows_processed: int,
    max_rows: int | None,
) -> tuple[int, float]:
    """Process a single file, writing documents to builder.

    Returns:
        Tuple of (rows_processed, tokenize_time_seconds)
    """
    tokenize_time = 0.0

    # Resolve file path - handle HF deferred download
    local_path = _resolve_file_path_core(file_info)

    # Determine file type and iterate records
    is_parquet = local_path.endswith(".parquet") or not (
        local_path.endswith(".jsonl") or local_path.endswith(".json")
    )

    if is_parquet:
        rows_processed, tokenize_time = _process_parquet_file_core(
            local_path=local_path,
            builder=builder,
            stats=stats,
            input_fs=input_fs,
            text_field=text_field,
            min_doc_chars=min_doc_chars,
            max_doc_tokens=max_doc_tokens,
            tokenize=tokenize,
            rows_processed=rows_processed,
            max_rows=max_rows,
        )
    else:
        rows_processed, tokenize_time = _process_jsonl_file_core(
            local_path=local_path,
            builder=builder,
            stats=stats,
            input_fs=input_fs,
            text_field=text_field,
            min_doc_chars=min_doc_chars,
            max_doc_tokens=max_doc_tokens,
            tokenize=tokenize,
            rows_processed=rows_processed,
            max_rows=max_rows,
        )

    return rows_processed, tokenize_time


def _resolve_file_path_core(file_info: FileInfo) -> str:
    """Resolve file to a local path, downloading from HF if needed."""
    # HF files need deferred download
    if file_info.hf_repo_id is not None:
        from huggingface_hub import hf_hub_download

        local_path = hf_hub_download(
            repo_id=file_info.hf_repo_id,
            filename=file_info.hf_filename,
            revision=file_info.hf_revision,
            repo_type="dataset",
            local_files_only=False,
        )
        return local_path

    # For non-HF files, use local_path if available
    return file_info.local_path or file_info.path


def _process_parquet_file_core(
    *,
    local_path: str,
    builder: IndexedDatasetBuilder,
    stats: dict[str, Any],
    input_fs: Any,
    text_field: str,
    min_doc_chars: int | None,
    max_doc_tokens: int | None,
    tokenize: Callable[[list[str]], list[list[int]]],
    rows_processed: int,
    max_rows: int | None,
) -> tuple[int, float]:
    """Process parquet file with optimized Arrow-level filtering.

    Returns:
        Tuple of (rows_processed, tokenize_time_seconds)
    """
    import pyarrow.compute as pc

    batch_texts: list[str] = []
    tokenize_batch_size = 1000
    hit_max_rows = False
    tokenize_time = 0.0

    # Determine if remote path
    is_remote = local_path.startswith(("s3://", "gs://", "gcs://", "az://", "abfs://"))

    def iter_parquet_batches():
        if is_remote:
            with input_fs.open(local_path, "rb") as f:
                parquet_file = pq.ParquetFile(f)
                yield from _iter_parquet_batches_internal(parquet_file, text_field, min_doc_chars)
        else:
            parquet_file = pq.ParquetFile(local_path)
            yield from _iter_parquet_batches_internal(parquet_file, text_field, min_doc_chars)

    for texts, num_filtered_by_length in iter_parquet_batches():
        if hit_max_rows:
            break

        # Account for rows filtered by min_doc_chars at Arrow level
        stats["num_filtered"] += num_filtered_by_length
        stats["num_input_rows"] += num_filtered_by_length

        for text in texts:
            # Check max_rows limit
            if max_rows and rows_processed >= max_rows:
                hit_max_rows = True
                break

            stats["num_input_rows"] += 1
            rows_processed += 1

            # Filter None values
            if text is None:
                stats["num_filtered"] += 1
                continue

            batch_texts.append(str(text))

            # Process batch
            if len(batch_texts) >= tokenize_batch_size:
                t0 = time.perf_counter()
                _tokenize_and_write_batch_core(
                    batch_texts, builder, stats, tokenize, max_doc_tokens
                )
                tokenize_time += time.perf_counter() - t0
                batch_texts = []

    # Process remaining
    if batch_texts:
        t0 = time.perf_counter()
        _tokenize_and_write_batch_core(batch_texts, builder, stats, tokenize, max_doc_tokens)
        tokenize_time += time.perf_counter() - t0

    return rows_processed, tokenize_time


def _iter_parquet_batches_internal(
    parquet_file: pq.ParquetFile,
    text_field: str,
    min_doc_chars: int | None,
) -> Iterator[tuple[list[str | None], int]]:
    """Iterate batches from parquet file efficiently."""
    import pyarrow.compute as pc

    for batch in parquet_file.iter_batches(
        columns=[text_field],
        batch_size=10000,
    ):
        column = batch.column(text_field)
        original_len = len(column)
        num_filtered = 0

        # Apply min_doc_chars filter at Arrow level if configured
        if min_doc_chars:
            lengths = pc.utf8_length(column)
            mask = pc.greater_equal(lengths, min_doc_chars)
            column = pc.filter(column, mask)
            num_filtered = original_len - len(column)

        # Use to_pylist() for bulk conversion
        yield column.to_pylist(), num_filtered


def _process_jsonl_file_core(
    *,
    local_path: str,
    builder: IndexedDatasetBuilder,
    stats: dict[str, Any],
    input_fs: Any,
    text_field: str,
    min_doc_chars: int | None,
    max_doc_tokens: int | None,
    tokenize: Callable[[list[str]], list[list[int]]],
    rows_processed: int,
    max_rows: int | None,
) -> tuple[int, float]:
    """Process JSONL file.

    Returns:
        Tuple of (rows_processed, tokenize_time_seconds)
    """
    batch_size = 1000
    batch_texts: list[str] = []
    tokenize_time = 0.0

    is_remote = local_path.startswith(("s3://", "gs://", "gcs://", "az://", "abfs://"))

    def iter_records():
        if is_remote:
            with input_fs.open(local_path, "r") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        else:
            with open(local_path) as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)

    for record in iter_records():
        # Check max_rows limit
        if max_rows and rows_processed >= max_rows:
            break

        stats["num_input_rows"] += 1
        rows_processed += 1

        # Extract text
        text = record.get(text_field)
        if text is None:
            stats["num_filtered"] += 1
            continue

        text = str(text)

        # Filter short docs
        if min_doc_chars and len(text) < min_doc_chars:
            stats["num_filtered"] += 1
            continue

        batch_texts.append(text)

        # Process batch
        if len(batch_texts) >= batch_size:
            t0 = time.perf_counter()
            _tokenize_and_write_batch_core(batch_texts, builder, stats, tokenize, max_doc_tokens)
            tokenize_time += time.perf_counter() - t0
            batch_texts = []

    # Process remaining
    if batch_texts:
        t0 = time.perf_counter()
        _tokenize_and_write_batch_core(batch_texts, builder, stats, tokenize, max_doc_tokens)
        tokenize_time += time.perf_counter() - t0

    return rows_processed, tokenize_time


def _tokenize_and_write_batch_core(
    texts: list[str],
    builder: IndexedDatasetBuilder,
    stats: dict[str, Any],
    tokenize: Callable[[list[str]], list[list[int]]],
    max_doc_tokens: int | None,
) -> None:
    """Tokenize a batch and write documents."""
    try:
        all_tokens = tokenize(texts)

        # Pre-filter and truncate
        processed: list[list[int]] = []
        for tokens in all_tokens:
            # Truncate if needed
            if max_doc_tokens and len(tokens) > max_doc_tokens:
                tokens = tokens[:max_doc_tokens]
                stats["num_truncated"] += 1

            if tokens:
                processed.append(tokens)

        # Batch add - reduces numpy allocation overhead
        builder.add_documents(processed)

    except Exception as e:
        # Bisect to isolate bad rows instead of dropping entire batch
        if len(texts) > 1:
            _tokenize_with_bisect_core(texts, builder, stats, tokenize, max_doc_tokens)
        else:
            # Single text failed, count as error
            stats["num_errors"] += 1
            logger.warning(f"Tokenization error for single text: {e}")


def _tokenize_with_bisect_core(
    texts: list[str],
    builder: IndexedDatasetBuilder,
    stats: dict[str, Any],
    tokenize: Callable[[list[str]], list[list[int]]],
    max_doc_tokens: int | None,
) -> None:
    """Bisect a batch to isolate problematic rows."""
    if len(texts) == 0:
        return

    if len(texts) == 1:
        # Single text - try it alone
        try:
            all_tokens = tokenize(texts)
            for tokens in all_tokens:
                if max_doc_tokens and len(tokens) > max_doc_tokens:
                    tokens = tokens[:max_doc_tokens]
                    stats["num_truncated"] += 1
                if tokens:
                    builder.add_document(tokens)
        except Exception as e:
            stats["num_errors"] += 1
            logger.debug(f"Skipping problematic text: {e}")
        return

    # Try first half
    mid = len(texts) // 2
    first_half = texts[:mid]
    second_half = texts[mid:]

    try:
        all_tokens = tokenize(first_half)
        for tokens in all_tokens:
            if max_doc_tokens and len(tokens) > max_doc_tokens:
                tokens = tokens[:max_doc_tokens]
                stats["num_truncated"] += 1
            if tokens:
                builder.add_document(tokens)
    except Exception:
        # First half has issues, recurse
        _tokenize_with_bisect_core(first_half, builder, stats, tokenize, max_doc_tokens)

    try:
        all_tokens = tokenize(second_half)
        for tokens in all_tokens:
            if max_doc_tokens and len(tokens) > max_doc_tokens:
                tokens = tokens[:max_doc_tokens]
                stats["num_truncated"] += 1
            if tokens:
                builder.add_document(tokens)
    except Exception:
        # Second half has issues, recurse
        _tokenize_with_bisect_core(second_half, builder, stats, tokenize, max_doc_tokens)


# =============================================================================
# Legacy ShardProcessor Ray Actor
# =============================================================================


@ray.remote
class ShardProcessor:
    """
    Long-lived actor that processes multiple shards.

    Loads tokenizer once, then processes assigned shards sequentially.
    Ensures deterministic output through sequential file/row processing.
    """

    def __init__(
        self,
        resolved_tokenizer: dict,
        text_field: str,
        min_doc_chars: int | None,
        max_doc_tokens: int | None,
        dtype: str,
        max_rows: int | None = None,
    ):
        self.text_field = text_field
        self.min_doc_chars = min_doc_chars
        self.max_doc_tokens = max_doc_tokens
        self.max_rows = max_rows
        self.dtype = np.dtype(dtype)

        # Load tokenizer ONCE
        self._tokenize = create_tokenizer(resolved_tokenizer)
        self.vocab_size = self._tokenize.vocab_size

    def process_shard(
        self,
        shard_index: int,
        assignment: dict,  # ShardAssignment as dict
        plan_hash: str,
        output_dir: str,
        receipts_dir: str,
        fs_protocol: str,
    ) -> dict:
        """
        Process a single shard: read files -> tokenize -> write .bin/.idx -> receipt.

        Returns shard statistics.
        """
        fs = filesystem(fs_protocol)

        shard_id = f"shard_{shard_index:06d}"
        bin_path = f"{output_dir}/{shard_id}.bin"
        idx_path = f"{output_dir}/{shard_id}.idx"
        receipt_path = f"{receipts_dir}/{shard_id}.json"

        # Ensure directories
        ensure_dir(fs, output_dir)
        ensure_dir(fs, receipts_dir)

        # Stats tracking
        stats = {
            "num_input_rows": 0,
            "num_filtered": 0,
            "num_truncated": 0,
            "num_errors": 0,
        }

        files = [FileInfo(**f) for f in assignment["files"]]
        input_file_paths = [f.path for f in files]

        # Handle empty assignment
        if not files:
            return self._write_empty_receipt(
                shard_id,
                shard_index,
                plan_hash,
                input_file_paths,
                stats,
                receipt_path,
                fs,
            )

        # Process files and write shard
        with fs.open(bin_path, "wb") as bin_file:
            builder = IndexedDatasetBuilder(bin_file, dtype=self.dtype)

            # Track rows processed across files for max_rows limit
            rows_processed = 0

            # Process files SEQUENTIALLY for determinism
            for file_info in files:
                rows_processed = self._process_file(file_info, builder, stats, fs, rows_processed)
                # Stop if we've hit max_rows
                if self.max_rows and rows_processed >= self.max_rows:
                    break

            bin_bytes, bin_checksum = builder.get_bin_info()
            builder_stats = builder.get_stats()

        # Handle empty result (all rows filtered)
        if builder_stats["num_sequences"] == 0:
            # Remove empty bin file
            try:
                fs.rm(bin_path)
            except Exception:
                pass
            return self._write_empty_receipt(
                shard_id,
                shard_index,
                plan_hash,
                input_file_paths,
                stats,
                receipt_path,
                fs,
            )

        # Write index
        with fs.open(idx_path, "wb") as idx_file:
            idx_bytes, idx_checksum = builder.write_index(idx_file)

        # Write receipt (commits the shard)
        receipt = {
            "shard_id": shard_id,
            "shard_index": shard_index,
            "plan_hash": plan_hash,
            "status": "completed",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_files": input_file_paths,
            "files": {
                "bin": {
                    "path": f"{shard_id}.bin",
                    "bytes": bin_bytes,
                    "checksum": bin_checksum,
                },
                "idx": {
                    "path": f"{shard_id}.idx",
                    "bytes": idx_bytes,
                    "checksum": idx_checksum,
                },
            },
            "stats": {**builder_stats, **stats},
        }

        write_json(fs, receipt_path, receipt)
        return receipt["stats"]

    def _process_file(
        self,
        file_info: FileInfo,
        builder: IndexedDatasetBuilder,
        stats: dict,
        fs,
        rows_processed: int = 0,
    ) -> int:
        """Process a single file, writing documents to builder.

        Returns the total number of rows processed (for max_rows tracking).
        """
        # Resolve file path - handle HF deferred download
        local_path = self._resolve_file_path(file_info)

        # Determine file type and iterate records
        is_parquet = local_path.endswith(".parquet") or not (
            local_path.endswith(".jsonl") or local_path.endswith(".json")
        )

        if is_parquet:
            # Parquet: yields raw text values (pre-filtered by min_doc_chars at Arrow level)
            rows_processed = self._process_parquet_file(
                local_path, builder, stats, fs, rows_processed
            )
        else:
            # JSONL: yields dicts, needs text extraction
            rows_processed = self._process_jsonl_file(
                local_path, builder, stats, fs, rows_processed
            )

        return rows_processed

    def _process_parquet_file(
        self,
        local_path: str,
        builder: IndexedDatasetBuilder,
        stats: dict,
        fs,
        rows_processed: int,
    ) -> int:
        """Process parquet file with optimized Arrow-level filtering."""
        batch_texts: list[str] = []
        tokenize_batch_size = 1000
        hit_max_rows = False

        # Parquet iterator yields (texts, num_filtered) tuples
        for texts, num_filtered_by_length in self._iter_parquet_batches_from_path(local_path, fs):
            if hit_max_rows:
                break

            # Account for rows filtered by min_doc_chars at Arrow level
            stats["num_filtered"] += num_filtered_by_length
            stats["num_input_rows"] += num_filtered_by_length

            for text in texts:
                # Check max_rows limit
                if self.max_rows and rows_processed >= self.max_rows:
                    hit_max_rows = True
                    break

                stats["num_input_rows"] += 1
                rows_processed += 1

                # Filter None values
                if text is None:
                    stats["num_filtered"] += 1
                    continue

                batch_texts.append(str(text))

                # Process batch
                if len(batch_texts) >= tokenize_batch_size:
                    self._tokenize_and_write_batch(batch_texts, builder, stats)
                    batch_texts = []

        # Process remaining
        if batch_texts:
            self._tokenize_and_write_batch(batch_texts, builder, stats)

        return rows_processed

    def _process_jsonl_file(
        self,
        local_path: str,
        builder: IndexedDatasetBuilder,
        stats: dict,
        fs,
        rows_processed: int,
    ) -> int:
        """Process JSONL file."""
        batch_size = 1000
        batch_texts: list[str] = []

        for record in self._iter_jsonl_records(local_path, fs):
            # Check max_rows limit
            if self.max_rows and rows_processed >= self.max_rows:
                break

            stats["num_input_rows"] += 1
            rows_processed += 1

            # Extract text
            text = record.get(self.text_field)
            if text is None:
                stats["num_filtered"] += 1
                continue

            text = str(text)

            # Filter short docs
            if self.min_doc_chars and len(text) < self.min_doc_chars:
                stats["num_filtered"] += 1
                continue

            batch_texts.append(text)

            # Process batch
            if len(batch_texts) >= batch_size:
                self._tokenize_and_write_batch(batch_texts, builder, stats)
                batch_texts = []

        # Process remaining
        if batch_texts:
            self._tokenize_and_write_batch(batch_texts, builder, stats)

        return rows_processed

    def _resolve_file_path(self, file_info: FileInfo) -> str:
        """
        Resolve file to a local path, downloading from HF if needed.

        For HF files, downloads to local cache (node-local).
        For other files, returns local_path or path.
        """
        # HF files need deferred download
        if file_info.hf_repo_id is not None:
            from huggingface_hub import hf_hub_download

            local_path = hf_hub_download(
                repo_id=file_info.hf_repo_id,
                filename=file_info.hf_filename,
                revision=file_info.hf_revision,
                repo_type="dataset",
                local_files_only=False,
            )
            return local_path

        # For non-HF files, use local_path if available
        return file_info.local_path or file_info.path

    def _tokenize_and_write_batch(
        self,
        texts: list[str],
        builder: IndexedDatasetBuilder,
        stats: dict,
    ) -> None:
        """Tokenize a batch and write documents."""
        try:
            all_tokens = self._tokenize(texts)

            # Pre-filter and truncate
            processed: list[list[int]] = []
            for tokens in all_tokens:
                # Truncate if needed
                if self.max_doc_tokens and len(tokens) > self.max_doc_tokens:
                    tokens = tokens[: self.max_doc_tokens]
                    stats["num_truncated"] += 1

                if tokens:
                    processed.append(tokens)

            # Batch add - reduces numpy allocation overhead
            builder.add_documents(processed)

        except Exception as e:
            # Bisect to isolate bad rows instead of dropping entire batch
            if len(texts) > 1:
                self._tokenize_with_bisect(texts, builder, stats)
            else:
                # Single text failed, count as error
                stats["num_errors"] += 1
                logger.warning(f"Tokenization error for single text: {e}")

    def _tokenize_with_bisect(
        self,
        texts: list[str],
        builder: IndexedDatasetBuilder,
        stats: dict,
    ) -> None:
        """
        Bisect a batch to isolate problematic rows.

        When a batch fails, recursively split in half to find and skip
        only the problematic texts while processing valid ones.
        """
        if len(texts) == 0:
            return

        if len(texts) == 1:
            # Single text - try it alone
            try:
                all_tokens = self._tokenize(texts)
                for tokens in all_tokens:
                    if self.max_doc_tokens and len(tokens) > self.max_doc_tokens:
                        tokens = tokens[: self.max_doc_tokens]
                        stats["num_truncated"] += 1
                    if tokens:
                        builder.add_document(tokens)
            except Exception as e:
                stats["num_errors"] += 1
                logger.debug(f"Skipping problematic text: {e}")
            return

        # Try first half
        mid = len(texts) // 2
        first_half = texts[:mid]
        second_half = texts[mid:]

        try:
            all_tokens = self._tokenize(first_half)
            for tokens in all_tokens:
                if self.max_doc_tokens and len(tokens) > self.max_doc_tokens:
                    tokens = tokens[: self.max_doc_tokens]
                    stats["num_truncated"] += 1
                if tokens:
                    builder.add_document(tokens)
        except Exception:
            # First half has issues, recurse
            self._tokenize_with_bisect(first_half, builder, stats)

        try:
            all_tokens = self._tokenize(second_half)
            for tokens in all_tokens:
                if self.max_doc_tokens and len(tokens) > self.max_doc_tokens:
                    tokens = tokens[: self.max_doc_tokens]
                    stats["num_truncated"] += 1
                if tokens:
                    builder.add_document(tokens)
        except Exception:
            # Second half has issues, recurse
            self._tokenize_with_bisect(second_half, builder, stats)

    def _iter_parquet_batches_from_path(
        self, path: str, fs
    ) -> Iterator[tuple[list[str | None], int]]:
        """
        Iterate (texts, num_filtered) batches from parquet file.

        Uses iter_batches for memory efficiency.
        Handles remote paths via fsspec.
        """
        # For remote paths, need to open via fsspec
        # For local paths after HF download, can read directly
        if self._is_remote_path(path):
            with fs.open(path, "rb") as f:
                parquet_file = pq.ParquetFile(f)
                yield from self._iter_parquet_batches(parquet_file)
        else:
            parquet_file = pq.ParquetFile(path)
            yield from self._iter_parquet_batches(parquet_file)

    def _iter_parquet_batches(
        self, parquet_file: pq.ParquetFile
    ) -> Iterator[tuple[list[str | None], int]]:
        """Iterate batches from parquet file efficiently.

        Yields (texts, num_filtered) tuples where:
        - texts: list of text values (already filtered by min_doc_chars)
        - num_filtered: count of rows filtered by min_doc_chars

        Uses to_pylist() for bulk conversion instead of per-element as_py().
        """
        import pyarrow.compute as pc

        for batch in parquet_file.iter_batches(
            columns=[self.text_field],
            batch_size=10000,
        ):
            column = batch.column(self.text_field)
            original_len = len(column)
            num_filtered = 0

            # Apply min_doc_chars filter at Arrow level if configured
            if self.min_doc_chars:
                lengths = pc.utf8_length(column)
                mask = pc.greater_equal(lengths, self.min_doc_chars)
                column = pc.filter(column, mask)
                num_filtered = original_len - len(column)

            # Use to_pylist() for bulk conversion - much faster than per-element as_py()
            yield column.to_pylist(), num_filtered

    def _iter_jsonl_records(self, path: str, fs) -> Iterator[dict]:
        """
        Iterate records from JSONL file.

        Handles remote paths via fsspec.
        """
        if self._is_remote_path(path):
            with fs.open(path, "r") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        else:
            with open(path) as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)

    def _is_remote_path(self, path: str) -> bool:
        """Check if path is a remote path (S3/GCS/etc)."""
        return path.startswith(("s3://", "gs://", "gcs://", "az://", "abfs://"))

    def _write_empty_receipt(
        self,
        shard_id: str,
        shard_index: int,
        plan_hash: str,
        input_files: list[str],
        stats: dict,
        receipt_path: str,
        fs,
    ) -> dict:
        """Write receipt for empty shard (CRITICAL for resume correctness)."""
        receipt = {
            "shard_id": shard_id,
            "shard_index": shard_index,
            "plan_hash": plan_hash,
            "status": "completed",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_files": input_files,
            "files": {
                "bin": {
                    "path": f"{shard_id}.bin",
                    "bytes": 0,
                    "checksum": "xxh64:empty",
                },
                "idx": {
                    "path": f"{shard_id}.idx",
                    "bytes": 0,
                    "checksum": "xxh64:empty",
                },
            },
            "stats": {
                "num_sequences": 0,
                "total_tokens": 0,
                "min_length": 0,
                "max_length": 0,
                **stats,
            },
        }

        write_json(fs, receipt_path, receipt)
        return receipt["stats"]
