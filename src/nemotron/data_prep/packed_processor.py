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

"""PackedShardProcessor Ray actor for parallel packed sequence output processing."""

import json
import logging
import time
from collections.abc import Iterator

import numpy as np
import pyarrow.parquet as pq
import ray
from fsspec import filesystem

from nemotron.data_prep.config import FileInfo
from nemotron.data_prep.filesystem import ensure_dir, write_json
from nemotron.data_prep.packing.builder import PackedSequenceBuilder
from nemotron.data_prep.providers import create_tokenizer

logger = logging.getLogger(__name__)


@ray.remote
class PackedShardProcessor:
    """Ray actor for processing data files to packed sequence output.

    Reads input files (parquet or jsonl), tokenizes text, packs sequences
    using the specified algorithm, and writes to .npy files compatible
    with Megatron-Bridge's GPTSFTPackedDataset.
    """

    def __init__(
        self,
        resolved_tokenizer: dict,
        text_field: str,
        pack_size: int,
        algorithm: str,
        dtype: str,
        min_doc_chars: int | None = None,
        max_doc_tokens: int | None = None,
        max_rows: int | None = None,
        seed: int | None = None,
    ):
        """Initialize packed processor.

        Args:
            resolved_tokenizer: Tokenizer configuration dict with resolved SHA.
            text_field: Field name for text in input records.
            pack_size: Maximum tokens per packed sequence.
            algorithm: Packing algorithm ("first_fit_decreasing", "first_fit_shuffle", etc).
            dtype: Token dtype for output.
            min_doc_chars: Skip documents shorter than this.
            max_doc_tokens: Truncate documents longer than this.
            max_rows: Maximum rows to process per shard.
            seed: Random seed for shuffle-based algorithms.
        """
        self.text_field = text_field
        self.pack_size = pack_size
        self.algorithm = algorithm
        self.dtype = np.dtype(dtype)
        self.min_doc_chars = min_doc_chars
        self.max_doc_tokens = max_doc_tokens
        self.max_rows = max_rows
        self.seed = seed

        # Load tokenizer ONCE
        self._tokenize = create_tokenizer(resolved_tokenizer)
        self.vocab_size = self._tokenize.vocab_size

    def process_shard(
        self,
        shard_index: int,
        files: list[dict],  # FileInfo as dicts for Ray serialization
        output_dir: str,
        receipts_dir: str,
        fs_protocol: str,
    ) -> dict:
        """Process files to a single packed shard.

        Args:
            shard_index: Index of this shard.
            files: List of FileInfo dicts to process.
            output_dir: Output directory for .npy files.
            receipts_dir: Directory for receipt files.
            fs_protocol: Filesystem protocol (e.g., "file", "s3").

        Returns:
            Shard statistics dict.
        """
        fs = filesystem(fs_protocol)

        shard_id = f"shard_{shard_index:06d}"
        npy_path = f"{output_dir}/{shard_id}.npy"
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

        # Convert file dicts back to FileInfo
        file_infos = [FileInfo(**f) for f in files]
        input_file_paths = [f.path for f in file_infos]

        # Handle empty assignment
        if not file_infos:
            return self._write_empty_receipt(
                shard_id,
                shard_index,
                input_file_paths,
                stats,
                receipt_path,
                fs,
            )

        # Create packing builder
        builder = PackedSequenceBuilder(
            pack_size=self.pack_size,
            algorithm=self.algorithm,
            seed=self.seed,
            dtype=str(self.dtype),
        )

        # Track rows processed across files for max_rows limit
        rows_processed = 0

        # Process files SEQUENTIALLY for determinism
        for file_info in file_infos:
            rows_processed = self._process_file(file_info, builder, stats, fs, rows_processed)
            # Stop if we've hit max_rows
            if self.max_rows and rows_processed >= self.max_rows:
                break

        # Finalize packing
        packed_data, packing_metadata = builder.finalize()

        # Handle empty result (all rows filtered)
        if not packed_data:
            return self._write_empty_receipt(
                shard_id,
                shard_index,
                input_file_paths,
                stats,
                receipt_path,
                fs,
            )

        # Save packed data as .npy
        # Use allow_pickle=True for list of dicts format
        with fs.open(npy_path, "wb") as f:
            np.save(f, packed_data, allow_pickle=True)

        # Get file size
        npy_bytes = fs.size(npy_path)

        # Write receipt (commits the shard)
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
                "num_sequences": packing_metadata["num_sequences"],
                "num_packed_sequences": packing_metadata["num_packed_sequences"],
                "total_tokens": packing_metadata["total_tokens"],
                **stats,
            },
        }

        write_json(fs, receipt_path, receipt)
        return receipt["stats"]

    def _process_file(
        self,
        file_info: FileInfo,
        builder: PackedSequenceBuilder,
        stats: dict,
        fs,
        rows_processed: int = 0,
    ) -> int:
        """Process a single file, adding sequences to builder.

        Returns the total number of rows processed (for max_rows tracking).
        """
        # Resolve file path - handle HF deferred download
        local_path = self._resolve_file_path(file_info)

        # Determine file type and iterate records
        is_parquet = local_path.endswith(".parquet") or not (
            local_path.endswith(".jsonl") or local_path.endswith(".json")
        )

        if is_parquet:
            rows_processed = self._process_parquet_file(
                local_path, builder, stats, fs, rows_processed
            )
        else:
            rows_processed = self._process_jsonl_file(
                local_path, builder, stats, fs, rows_processed
            )

        return rows_processed

    def _process_parquet_file(
        self,
        local_path: str,
        builder: PackedSequenceBuilder,
        stats: dict,
        fs,
        rows_processed: int,
    ) -> int:
        """Process parquet file with optimized Arrow-level filtering."""
        batch_texts: list[str] = []
        tokenize_batch_size = 1000
        hit_max_rows = False

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
                    self._tokenize_and_add_batch(batch_texts, builder, stats)
                    batch_texts = []

        # Process remaining
        if batch_texts:
            self._tokenize_and_add_batch(batch_texts, builder, stats)

        return rows_processed

    def _process_jsonl_file(
        self,
        local_path: str,
        builder: PackedSequenceBuilder,
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
                self._tokenize_and_add_batch(batch_texts, builder, stats)
                batch_texts = []

        # Process remaining
        if batch_texts:
            self._tokenize_and_add_batch(batch_texts, builder, stats)

        return rows_processed

    def _resolve_file_path(self, file_info: FileInfo) -> str:
        """Resolve file to a local path, downloading from HF if needed."""
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

        return file_info.local_path or file_info.path

    def _tokenize_and_add_batch(
        self,
        texts: list[str],
        builder: PackedSequenceBuilder,
        stats: dict,
    ) -> None:
        """Tokenize a batch and add sequences to builder."""
        try:
            all_tokens = self._tokenize(texts)

            for tokens in all_tokens:
                # Truncate if needed
                if self.max_doc_tokens and len(tokens) > self.max_doc_tokens:
                    tokens = tokens[: self.max_doc_tokens]
                    stats["num_truncated"] += 1

                if tokens:
                    # For pretraining, loss_mask is all 1s (compute loss on all tokens)
                    builder.add_sequence(tokens, loss_mask=None)

        except Exception as e:
            # Bisect to isolate bad rows
            if len(texts) > 1:
                self._tokenize_with_bisect(texts, builder, stats)
            else:
                stats["num_errors"] += 1
                logger.warning(f"Tokenization error for single text: {e}")

    def _tokenize_with_bisect(
        self,
        texts: list[str],
        builder: PackedSequenceBuilder,
        stats: dict,
    ) -> None:
        """Bisect a batch to isolate problematic rows."""
        if len(texts) == 0:
            return

        if len(texts) == 1:
            try:
                all_tokens = self._tokenize(texts)
                for tokens in all_tokens:
                    if self.max_doc_tokens and len(tokens) > self.max_doc_tokens:
                        tokens = tokens[: self.max_doc_tokens]
                        stats["num_truncated"] += 1
                    if tokens:
                        builder.add_sequence(tokens, loss_mask=None)
            except Exception as e:
                stats["num_errors"] += 1
                logger.debug(f"Skipping problematic text: {e}")
            return

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
                    builder.add_sequence(tokens, loss_mask=None)
        except Exception:
            self._tokenize_with_bisect(first_half, builder, stats)

        try:
            all_tokens = self._tokenize(second_half)
            for tokens in all_tokens:
                if self.max_doc_tokens and len(tokens) > self.max_doc_tokens:
                    tokens = tokens[: self.max_doc_tokens]
                    stats["num_truncated"] += 1
                if tokens:
                    builder.add_sequence(tokens, loss_mask=None)
        except Exception:
            self._tokenize_with_bisect(second_half, builder, stats)

    def _iter_parquet_batches_from_path(
        self, path: str, fs
    ) -> Iterator[tuple[list[str | None], int]]:
        """Iterate (texts, num_filtered) batches from parquet file."""
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
        """Iterate batches from parquet file efficiently."""
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

            yield column.to_pylist(), num_filtered

    def _iter_jsonl_records(self, path: str, fs) -> Iterator[dict]:
        """Iterate records from JSONL file."""
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
        input_files: list[str],
        stats: dict,
        receipt_path: str,
        fs,
    ) -> dict:
        """Write receipt for empty shard."""
        receipt = {
            "shard_id": shard_id,
            "shard_index": shard_index,
            "status": "completed",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_files": input_files,
            "output_file": None,
            "npy_bytes": 0,
            "packing": {
                "pack_size": self.pack_size,
                "algorithm": self.algorithm,
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

        write_json(fs, receipt_path, receipt)
        return receipt["stats"]
