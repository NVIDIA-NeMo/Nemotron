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

"""Core ChatSFT shard processing (retry-safe, engine-agnostic)."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from fsspec import AbstractFileSystem
from transformers import PreTrainedTokenizerBase

from nemotron.data_prep.core.chat_template import (
    create_masked_messages,
    replace_json_args,
    split_system_user_chunks,
    validate_conversation,
)
from nemotron.data_prep.config import FileInfo
from nemotron.data_prep.utils.filesystem import ensure_dir, get_filesystem, read_json
from nemotron.data_prep.packing.algorithms import get_packer
from nemotron.data_prep.packing.bin_assignment import BinAssignment
from nemotron.data_prep.packing.materialize import materialize_bin_arrays
from nemotron.data_prep.packing.writers import ParquetShardWriter
from nemotron.data_prep.packing.spool import (
    SequenceSpoolPaths,
    SequenceSpoolReader,
    SequenceSpoolWriter,
)


def _apply_chat_template(tokenizer: PreTrainedTokenizerBase, chat_template: str) -> None:
    if chat_template == "nano3":
        template_path = Path(__file__).parent.parent / "templates" / "nano3.jinja"
        with open(template_path) as f:
            tokenizer.chat_template = f.read()
    elif Path(chat_template).exists():
        with open(chat_template) as f:
            tokenizer.chat_template = f.read()
    else:
        tokenizer.chat_template = chat_template


def _tokenize_chunks_with_mask(
    tokenizer: PreTrainedTokenizerBase,
    chunks: list[dict],
) -> tuple[list[int], list[int]]:
    all_input_ids: list[int] = []
    all_loss_mask: list[int] = []

    for chunk in chunks:
        tokens = tokenizer.encode(chunk["content"], add_special_tokens=False)
        mask_value = 1 if chunk["role"] == "assistant" else 0
        mask = [mask_value] * len(tokens)
        all_input_ids.extend(tokens)
        all_loss_mask.extend(mask)

    return all_input_ids, all_loss_mask


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
    try:
        with fs.open(path, "rb") as f:
            parquet_file = pq.ParquetFile(f)
            for batch in parquet_file.iter_batches(batch_size=1000):
                table = batch.to_pydict()
                keys = list(table.keys())
                num_rows = len(table[keys[0]]) if keys else 0
                for i in range(num_rows):
                    yield {k: table[k][i] for k in keys}
    except Exception as e:
        raise RuntimeError(f"Failed to read parquet file: {path}") from e


def _iter_jsonl_records(path: str, fs: AbstractFileSystem) -> Iterator[dict]:
    with fs.open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _matches_used_in_filter(used_in: str | list | None, used_in_filter: str) -> bool:
    if used_in is None:
        return False

    if isinstance(used_in, list):
        return used_in_filter in used_in

    if isinstance(used_in, str):
        if used_in == used_in_filter:
            return True
        values = [v.strip() for v in used_in.split(",")]
        return used_in_filter in values

    return False


def process_chat_sft_spool_core(
    *,
    shard_index: int,
    files: list[dict] | list[FileInfo],
    output_dir: str,
    receipts_dir: str,
    spool_dir: str | None,
    output_fs: AbstractFileSystem,
    tokenizer: PreTrainedTokenizerBase,
    messages_field: str,
    tools_field: str,
    pack_size: int,
    algorithm: str,
    dtype: np.dtype,
    chat_template: str | None,
    max_doc_tokens: int | None,
    max_rows: int | None,
    seed: int | None,
    used_in_filter: str | None,
    used_in_field: str,
) -> dict[str, Any]:
    """Tokenize+mask a ChatSFT shard into a SequenceSpool intermediate.

    Retry safety:
    - The spool is considered committed when manifest.json exists.
    - SequenceSpoolWriter writes data to *.tmp then renames + writes manifest last.
    """
    shard_id = f"shard_{shard_index:06d}"
    spool_root = spool_dir or f"{output_dir.rstrip('/')}/spool/{shard_id}"
    paths = SequenceSpoolPaths.for_root(spool_root)

    # If the spool manifest exists, treat it as completed.
    if output_fs.exists(paths.manifest_path):
        try:
            manifest = read_json(output_fs, paths.manifest_path)
            tokenization_stats = manifest.get("tokenization_stats", {})
            return tokenization_stats if isinstance(tokenization_stats, dict) else {}
        except Exception:
            # Fall through to regenerate spool if manifest is unreadable.
            pass

    ensure_dir(output_fs, output_dir)
    ensure_dir(output_fs, receipts_dir)
    ensure_dir(output_fs, spool_root)

    file_infos = [FileInfo(**f) if isinstance(f, dict) else f for f in files]
    input_file_paths = [f.path for f in file_infos]

    if chat_template:
        _apply_chat_template(tokenizer, chat_template)

    stats: dict[str, Any] = {
        "num_input_rows": 0,
        "num_output_sequences": 0,
        "num_filtered": 0,
        "num_validation_errors": 0,
        "num_truncated": 0,  # truncation due to max_doc_tokens
        "num_errors": 0,
    }

    writer = SequenceSpoolWriter(fs=output_fs, paths=paths)

    rows_processed = 0

    def _process_record_to_spool(record: dict) -> None:
        if used_in_filter:
            used_in = record.get(used_in_field)
            if not _matches_used_in_filter(used_in, used_in_filter):
                stats["num_filtered"] += 1
                return

        messages = record.get(messages_field)
        tools = record.get(tools_field)

        if not messages:
            stats["num_filtered"] += 1
            return

        is_valid, _ = validate_conversation(messages, tools)
        if not is_valid:
            stats["num_filtered"] += 1
            stats["num_validation_errors"] += 1
            return

        try:
            messages_local = replace_json_args(messages)
        except (json.JSONDecodeError, KeyError, TypeError):
            stats["num_filtered"] += 1
            stats["num_errors"] += 1
            return

        try:
            masked_results = create_masked_messages(messages_local, tokenizer, tools)
        except Exception:
            stats["num_filtered"] += 1
            stats["num_errors"] += 1
            return

        for chunks, _ in masked_results:
            processed_chunks = split_system_user_chunks(chunks)
            try:
                input_ids, loss_mask = _tokenize_chunks_with_mask(tokenizer, processed_chunks)
            except Exception:
                stats["num_errors"] += 1
                continue

            if not input_ids:
                continue

            if max_doc_tokens and len(input_ids) > max_doc_tokens:
                input_ids = input_ids[:max_doc_tokens]
                loss_mask = loss_mask[:max_doc_tokens]
                stats["num_truncated"] += 1

            writer.append(input_ids, loss_mask)
            stats["num_output_sequences"] += 1

    for file_info in file_infos:
        if max_rows and rows_processed >= max_rows:
            break

        local_path = _resolve_file_path(file_info)
        input_path = (
            local_path
            if file_info.hf_repo_id is not None
            else (file_info.local_path or file_info.path)
        )
        input_fs, normalized = get_filesystem(input_path)

        # Use original filename for format detection (hf_hub_download returns blob path without extension)
        format_check_path = (file_info.hf_filename or normalized) if file_info.hf_repo_id else normalized
        is_parquet = format_check_path.endswith(".parquet") or not (
            format_check_path.endswith(".jsonl") or format_check_path.endswith(".json")
        )

        record_iter = _iter_parquet_records(normalized, input_fs) if is_parquet else _iter_jsonl_records(normalized, input_fs)

        for record in record_iter:
            if max_rows and rows_processed >= max_rows:
                break
            stats["num_input_rows"] += 1
            rows_processed += 1
            _process_record_to_spool(record)

    writer.finalize(
        extra_manifest={
            "shard_id": shard_id,
            "shard_index": shard_index,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_files": input_file_paths,
            "messages_field": messages_field,
            "tools_field": tools_field,
            "chat_template": chat_template,
            "max_doc_tokens": max_doc_tokens,
            "max_rows": max_rows,
            "seed": seed,
            "used_in_filter": used_in_filter,
            "used_in_field": used_in_field,
            "pack_size": pack_size,
            "algorithm": algorithm,
            "dtype": str(dtype),
            "tokenization_stats": stats,
        }
    )

    return stats


def process_chat_sft_parquet_from_spool_core(
    *,
    shard_index: int,
    output_dir: str,
    spool_dir: str,
    output_fs: AbstractFileSystem,
    pack_size: int,
    algorithm: str,
    dtype: np.dtype,
    seed: int | None,
    parquet_row_group_size: int = 1000,
    parquet_compression: str = "zstd",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Two-pass pack from a SequenceSpool and write packed Parquet (receipt-free).

    This function:
      1) reads SequenceSpool lengths
      2) computes a packing assignment (bin_id -> seq indices)
      3) materializes each bin into numpy arrays
      4) writes shard_{i:06d}.parquet using ParquetShardWriter

    It does NOT write receipts; the calling stage owns all receipt/checkpoint logic.

    Returns:
        (stats, files_metadata)
    """
    shard_id = f"shard_{shard_index:06d}"
    parquet_path = f"{output_dir.rstrip('/')}/{shard_id}.parquet"

    if pack_size <= 0:
        raise ValueError(f"pack_size must be positive, got {pack_size}")
    if parquet_row_group_size <= 0:
        raise ValueError(f"parquet_row_group_size must be positive, got {parquet_row_group_size}")

    ensure_dir(output_fs, output_dir)

    paths = SequenceSpoolPaths.for_root(spool_dir)
    if not output_fs.exists(paths.manifest_path):
        raise RuntimeError(f"Missing spool manifest for shard {shard_id}: {paths.manifest_path}")

    try:
        manifest = read_json(output_fs, paths.manifest_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read spool manifest for shard {shard_id}: {paths.manifest_path}") from e

    tokenization_stats = manifest.get("tokenization_stats", {})
    if not isinstance(tokenization_stats, dict):
        tokenization_stats = {}

    input_files = manifest.get("input_files", [])
    if not isinstance(input_files, list):
        input_files = []

    reader = SequenceSpoolReader(fs=output_fs, paths=paths)

    try:
        _, lengths = reader.load_offsets_and_lengths()
        num_sequences = int(lengths.shape[0])

        lengths_i64 = lengths.astype(np.int64, copy=False)
        lengths_clamped = np.minimum(lengths_i64, int(pack_size))
        num_truncated_to_pack_size = int((lengths_i64 > int(pack_size)).sum())
        total_tokens = int(lengths_clamped.sum())

        if num_sequences > 0:
            packer = get_packer(algorithm, pack_size, seed=seed)
            bins, _ = packer.pack([int(x) for x in lengths_clamped.tolist()])
        else:
            bins = []

        assignment = BinAssignment.from_bins(bins=bins, num_sequences=num_sequences)
        num_bins = int(assignment.num_bins)

        packing_factor = round(num_sequences / num_bins, 2) if num_bins else 0.0
        packing_efficiency = (
            round((total_tokens / (num_bins * pack_size)) * 100, 1) if num_bins else 0.0
        )

        # Best-effort cleanup of leftovers from previous failed attempts (no receipts here).
        try:
            if output_fs.exists(parquet_path):
                output_fs.rm(parquet_path)
        except Exception:
            pass
        try:
            tmp_path = parquet_path + ".tmp"
            if output_fs.exists(tmp_path):
                output_fs.rm(tmp_path)
        except Exception:
            pass

        # Use PyArrow filesystem wrapper for non-local fsspec outputs.
        pa_filesystem = None
        try:
            import pyarrow as pa

            protocol = getattr(output_fs, "protocol", None)
            if isinstance(protocol, (tuple, list)):
                protocol = protocol[0] if protocol else None
            if protocol not in (None, "", "file", "local"):
                pa_filesystem = pa.fs.PyFileSystem(pa.fs.FSSpecHandler(output_fs))
        except Exception:
            pa_filesystem = None

        writer = ParquetShardWriter(
            output_path=parquet_path,
            row_group_size=int(parquet_row_group_size),
            compression=str(parquet_compression),
            filesystem=pa_filesystem,
        )

        scratch_input_ids = np.zeros((int(pack_size),), dtype=np.int32)
        scratch_loss_mask = np.zeros((int(pack_size),), dtype=np.uint8)

        for bin_id in range(num_bins):
            packed_len, seq_start_id = materialize_bin_arrays(
                spool_reader=reader,
                assignment=assignment,
                bin_id=bin_id,
                pack_size=int(pack_size),
                scratch_input_ids=scratch_input_ids,
                scratch_loss_mask=scratch_loss_mask,
            )

            writer.write_bin(
                bin_id=bin_id,
                input_ids=scratch_input_ids[:packed_len].copy(),
                loss_mask=scratch_loss_mask[:packed_len].copy(),
                seq_start_id=seq_start_id,
            )

        writer_result = writer.finalize()

        try:
            parquet_bytes = int(output_fs.size(parquet_path))
        except Exception:
            parquet_bytes = 0

        stats: dict[str, Any] = {
            "num_sequences": num_sequences,
            "num_packed_sequences": num_bins,
            "total_tokens": total_tokens,
            "num_truncated_to_pack_size": num_truncated_to_pack_size,
            "packing": {
                "pack_size": int(pack_size),
                "algorithm": str(algorithm),
                "seed": seed,
                "packing_factor": packing_factor,
                "packing_efficiency": packing_efficiency,
                "parquet_row_group_size": int(parquet_row_group_size),
                "parquet_compression": str(parquet_compression),
                "writer": writer_result,
            },
            **tokenization_stats,
        }

        files_metadata: dict[str, Any] = {
            "parquet": {
                "path": f"{shard_id}.parquet",
                "bytes": parquet_bytes,
                "checksum": "xxh64:unknown",
            },
            "input_files": [str(x) for x in input_files],
        }

        _ = dtype  # kept for API symmetry; Parquet spec is int32 tokens

        return stats, files_metadata

    finally:
        try:
            reader.close()
        except Exception:
            pass


# =============================================================================
# Standardized Alias
# =============================================================================

# Alias following the naming convention: process_<format>_<operation>_core
# Provides a consistent name pattern across all core processing functions
process_chat_sft_parquet_core = process_chat_sft_parquet_from_spool_core
