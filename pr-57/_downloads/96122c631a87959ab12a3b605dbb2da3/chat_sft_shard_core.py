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

from nemotron.data_prep.chat_template import (
    create_masked_messages,
    replace_json_args,
    split_system_user_chunks,
    validate_conversation,
)
from nemotron.data_prep.config import FileInfo
from nemotron.data_prep.filesystem import ensure_dir, get_filesystem, read_json, write_json
from nemotron.data_prep.packing.algorithms import get_packer
from nemotron.data_prep.packing.bin_assignment import BinAssignment
from nemotron.data_prep.packing.builder import PackedSequenceBuilder
from nemotron.data_prep.packing.materialize import materialize_packed_samples
from nemotron.data_prep.packing.spool import (
    SequenceSpoolPaths,
    SequenceSpoolReader,
    SequenceSpoolWriter,
)


def process_chat_sft_shard_core(
    *,
    shard_index: int,
    files: list[dict] | list[FileInfo],
    output_dir: str,
    receipts_dir: str,
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
    """Process a ChatSFT shard with retry-safe atomic commits."""
    shard_id = f"shard_{shard_index:06d}"
    npy_path = f"{output_dir}/{shard_id}.npy"
    npy_tmp = f"{npy_path}.tmp"
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

    stats: dict[str, Any] = {
        "num_input_rows": 0,
        "num_output_sequences": 0,
        "num_filtered": 0,
        "num_validation_errors": 0,
        "num_truncated": 0,
        "num_errors": 0,
    }

    if not file_infos:
        return _write_empty_receipt(
            shard_id=shard_id,
            shard_index=shard_index,
            input_files=input_file_paths,
            stats=stats,
            receipt_path=receipt_path,
            output_fs=output_fs,
            pack_size=pack_size,
            algorithm=algorithm,
        )

    if chat_template:
        _apply_chat_template(tokenizer, chat_template)

    builder = PackedSequenceBuilder(
        pack_size=pack_size,
        algorithm=algorithm,
        seed=seed,
        dtype=str(dtype),
    )

    rows_processed = 0
    for file_info in file_infos:
        rows_processed = _process_file(
            file_info=file_info,
            builder=builder,
            stats=stats,
            tokenizer=tokenizer,
            messages_field=messages_field,
            tools_field=tools_field,
            max_doc_tokens=max_doc_tokens,
            max_rows=max_rows,
            rows_processed=rows_processed,
            used_in_filter=used_in_filter,
            used_in_field=used_in_field,
        )
        if max_rows and rows_processed >= max_rows:
            break

    packed_data, packing_metadata = builder.finalize()

    if not packed_data:
        return _write_empty_receipt(
            shard_id=shard_id,
            shard_index=shard_index,
            input_files=input_file_paths,
            stats=stats,
            receipt_path=receipt_path,
            output_fs=output_fs,
            pack_size=pack_size,
            algorithm=algorithm,
        )

    with output_fs.open(npy_tmp, "wb") as f:
        np.save(f, packed_data, allow_pickle=True)

    output_fs.rename(npy_tmp, npy_path)
    npy_bytes = output_fs.size(npy_path)

    receipt = {
        "shard_id": shard_id,
        "shard_index": shard_index,
        "status": "completed",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_files": input_file_paths,
        "output_file": f"{shard_id}.npy",
        "npy_bytes": npy_bytes,
        "packing": packing_metadata,
        "stats": {
            "num_sequences": packing_metadata.get("num_sequences", 0),
            "num_packed_sequences": packing_metadata.get("num_packed_sequences", 0),
            "total_tokens": packing_metadata.get("total_tokens", 0),
            **stats,
        },
    }

    write_json(output_fs, receipt_path, receipt)
    return receipt["stats"]


def _apply_chat_template(tokenizer: PreTrainedTokenizerBase, chat_template: str) -> None:
    if chat_template == "nano3":
        template_path = Path(__file__).parent / "templates" / "nano3.jinja"
        with open(template_path) as f:
            tokenizer.chat_template = f.read()
    elif Path(chat_template).exists():
        with open(chat_template) as f:
            tokenizer.chat_template = f.read()
    else:
        tokenizer.chat_template = chat_template


def _process_file(
    *,
    file_info: FileInfo,
    builder: PackedSequenceBuilder,
    stats: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    messages_field: str,
    tools_field: str,
    max_doc_tokens: int | None,
    max_rows: int | None,
    rows_processed: int,
    used_in_filter: str | None,
    used_in_field: str,
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

        stats["num_input_rows"] += 1
        rows_processed += 1

        _process_record(
            record=record,
            builder=builder,
            stats=stats,
            tokenizer=tokenizer,
            messages_field=messages_field,
            tools_field=tools_field,
            max_doc_tokens=max_doc_tokens,
            used_in_filter=used_in_filter,
            used_in_field=used_in_field,
        )

    return rows_processed


def _process_record(
    *,
    record: dict,
    builder: PackedSequenceBuilder,
    stats: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    messages_field: str,
    tools_field: str,
    max_doc_tokens: int | None,
    used_in_filter: str | None,
    used_in_field: str,
) -> None:
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
        messages = replace_json_args(messages)
    except (json.JSONDecodeError, KeyError, TypeError):
        stats["num_filtered"] += 1
        stats["num_errors"] += 1
        return

    try:
        masked_results = create_masked_messages(messages, tokenizer, tools)
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

        builder.add_sequence(input_ids, loss_mask=loss_mask)
        stats["num_output_sequences"] += 1


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


def _write_empty_receipt(
    *,
    shard_id: str,
    shard_index: int,
    input_files: list[str],
    stats: dict[str, Any],
    receipt_path: str,
    output_fs: AbstractFileSystem,
    pack_size: int,
    algorithm: str,
) -> dict[str, Any]:
    receipt = {
        "shard_id": shard_id,
        "shard_index": shard_index,
        "status": "completed",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_files": input_files,
        "output_file": None,
        "npy_bytes": 0,
        "packing": {
            "pack_size": pack_size,
            "algorithm": algorithm,
            "num_sequences": 0,
            "num_packed_sequences": 0,
            "packing_factor": 0,
            "packing_efficiency": 0,
            "total_tokens": 0,
        },
        "stats": {
            "num_sequences": 0,
            "num_packed_sequences": 0,
            "total_tokens": 0,
            **stats,
        },
    }

    write_json(output_fs, receipt_path, receipt)
    return receipt["stats"]


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


def process_chat_sft_pack_from_spool_core(
    *,
    shard_index: int,
    output_dir: str,
    receipts_dir: str,
    spool_dir: str | None,
    output_fs: AbstractFileSystem,
    pack_size: int,
    algorithm: str,
    dtype: np.dtype,
    seed: int | None,
) -> dict[str, Any]:
    """Two-pass pack from a SequenceSpool and write packed .npy + standard receipt."""
    shard_id = f"shard_{shard_index:06d}"
    npy_path = f"{output_dir}/{shard_id}.npy"
    npy_tmp = f"{npy_path}.tmp"
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

    spool_root = spool_dir or f"{output_dir.rstrip('/')}/spool/{shard_id}"
    paths = SequenceSpoolPaths.for_root(spool_root)

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

        if num_sequences == 0:
            return _write_empty_receipt(
                shard_id=shard_id,
                shard_index=shard_index,
                input_files=[str(x) for x in input_files],
                stats=tokenization_stats,
                receipt_path=receipt_path,
                output_fs=output_fs,
                pack_size=pack_size,
                algorithm=algorithm,
            )

        lengths_clamped = np.minimum(lengths.astype(np.int64), int(pack_size))
        num_truncated_to_pack_size = int((lengths.astype(np.int64) > int(pack_size)).sum())

        packer = get_packer(algorithm, pack_size, seed=seed)
        bins, _ = packer.pack([int(x) for x in lengths_clamped.tolist()])

        assignment = BinAssignment.from_bins(bins=bins, num_sequences=num_sequences)

        packed_data: list[dict] = []
        for item in materialize_packed_samples(
            spool_reader=reader,
            assignment=assignment,
            pack_size=pack_size,
        ):
            packed_data.append(item)

    finally:
        try:
            reader.close()
        except Exception:
            pass

    if not packed_data:
        return _write_empty_receipt(
            shard_id=shard_id,
            shard_index=shard_index,
            input_files=[str(x) for x in input_files],
            stats=tokenization_stats,
            receipt_path=receipt_path,
            output_fs=output_fs,
            pack_size=pack_size,
            algorithm=algorithm,
        )

    with output_fs.open(npy_tmp, "wb") as f:
        np.save(f, packed_data, allow_pickle=True)

    # Explicitly free memory from large data structures before computing stats
    # This prevents memory accumulation when processing multiple shards sequentially
    num_bins = int(assignment.num_bins)
    total_tokens = int(lengths_clamped.sum())

    del packed_data
    del bins
    del assignment
    del lengths_clamped

    import gc
    gc.collect()

    output_fs.rename(npy_tmp, npy_path)
    npy_bytes = output_fs.size(npy_path)

    packing_factor = round(num_sequences / num_bins, 2) if num_bins else 0.0
    packing_efficiency = (
        round((total_tokens / (num_bins * pack_size)) * 100, 1) if num_bins else 0.0
    )

    packing_metadata = {
        "pack_size": pack_size,
        "algorithm": str(algorithm),
        "num_sequences": num_sequences,
        "num_packed_sequences": num_bins,
        "packing_factor": packing_factor,
        "packing_efficiency": packing_efficiency,
        "num_truncated": num_truncated_to_pack_size,
        "total_tokens": total_tokens,
    }

    receipt = {
        "shard_id": shard_id,
        "shard_index": shard_index,
        "status": "completed",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_files": [str(x) for x in input_files],
        "output_file": f"{shard_id}.npy",
        "npy_bytes": npy_bytes,
        "packing": packing_metadata,
        "stats": {
            "num_sequences": packing_metadata.get("num_sequences", 0),
            "num_packed_sequences": packing_metadata.get("num_packed_sequences", 0),
            "total_tokens": packing_metadata.get("total_tokens", 0),
            "num_truncated_to_pack_size": num_truncated_to_pack_size,
            **tokenization_stats,
        },
    }

    write_json(output_fs, receipt_path, receipt)
    return receipt["stats"]
