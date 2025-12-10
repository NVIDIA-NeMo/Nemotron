"""ChatSftShardProcessor Ray actor for parallel chat-templated SFT output processing.

Applies chat templates to OpenAI-format messages, tokenizes with role-based
loss masking, and outputs packed .npy files compatible with GPTSFTPackedDataset.

Pipeline:
1. Apply materialize.py chat template logic -> role-labeled chunks
2. Tokenize chunks -> input_ids
3. Build loss_mask based on role (0=system/user, 1=assistant)
4. Pack sequences -> .npy output
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import ray
from fsspec import filesystem

from nemotron.data_prep.chat_template import (
    create_masked_messages,
    replace_json_args,
    split_system_user_chunks,
    validate_conversation,
)
from nemotron.data_prep.config import FileInfo
from nemotron.data_prep.filesystem import ensure_dir, write_json
from nemotron.data_prep.packing.builder import PackedSequenceBuilder

logger = logging.getLogger(__name__)


@ray.remote
class ChatSftShardProcessor:
    """Ray actor for chat-templated SFT output with loss masking.

    Reads input files with OpenAI-format messages, applies chat templates
    to generate loss-masked sequences, packs them, and writes to .npy files
    compatible with Megatron-Bridge's GPTSFTPackedDataset.
    """

    def __init__(
        self,
        resolved_tokenizer: dict,
        messages_field: str,
        tools_field: str,
        pack_size: int,
        algorithm: str,
        dtype: str,
        chat_template: str | None = None,
        max_doc_tokens: int | None = None,
        max_rows: int | None = None,
        seed: int | None = None,
    ):
        """Initialize chat SFT processor.

        Args:
            resolved_tokenizer: Tokenizer configuration dict with resolved SHA.
            messages_field: Field name for messages in input records.
            tools_field: Field name for tools in input records.
            pack_size: Maximum tokens per packed sequence.
            algorithm: Packing algorithm.
            dtype: Token dtype for output.
            chat_template: "nano3", path to .jinja file, or inline template string.
            max_doc_tokens: Truncate sequences longer than this.
            max_rows: Maximum rows to process per shard.
            seed: Random seed for shuffle-based algorithms.
        """
        from transformers import AutoTokenizer

        self.messages_field = messages_field
        self.tools_field = tools_field
        self.pack_size = pack_size
        self.algorithm = algorithm
        self.dtype = np.dtype(dtype)
        self.max_doc_tokens = max_doc_tokens
        self.max_rows = max_rows
        self.seed = seed

        # Load HuggingFace tokenizer with full chat template support
        self._tokenizer = AutoTokenizer.from_pretrained(
            resolved_tokenizer["model"],
            revision=resolved_tokenizer.get("resolved_revision"),
            trust_remote_code=resolved_tokenizer.get("trust_remote_code", False),
        )

        # Load chat template
        if chat_template:
            if chat_template == "nano3":
                # Load bundled template
                template_path = Path(__file__).parent / "templates" / "nano3.jinja"
                with open(template_path) as f:
                    self._tokenizer.chat_template = f.read()
            elif Path(chat_template).exists():
                # Load from file path
                with open(chat_template) as f:
                    self._tokenizer.chat_template = f.read()
            else:
                # Assume inline template string
                self._tokenizer.chat_template = chat_template

    def process_shard(
        self,
        shard_index: int,
        files: list[dict],  # FileInfo as dicts for Ray serialization
        output_dir: str,
        receipts_dir: str,
        fs_protocol: str,
    ) -> dict:
        """Process files to a single packed shard with loss masks.

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
            "num_output_sequences": 0,
            "num_filtered": 0,
            "num_validation_errors": 0,
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
            rows_processed = self._process_file(
                file_info, builder, stats, fs, rows_processed
            )
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
            record_iter = self._iter_parquet_records(local_path, fs)
        else:
            record_iter = self._iter_jsonl_records(local_path, fs)

        for record in record_iter:
            # Check max_rows limit
            if self.max_rows and rows_processed >= self.max_rows:
                break

            stats["num_input_rows"] += 1
            rows_processed += 1

            self._process_record(record, builder, stats)

        return rows_processed

    def _process_record(
        self,
        record: dict,
        builder: PackedSequenceBuilder,
        stats: dict,
    ) -> None:
        """Process a single record using materialize.py logic."""
        messages = record.get(self.messages_field)
        tools = record.get(self.tools_field)

        # Skip if no messages
        if not messages:
            stats["num_filtered"] += 1
            return

        # Step 1: Validate (materialize_fast.py checks)
        is_valid, error = validate_conversation(messages, tools)
        if not is_valid:
            stats["num_filtered"] += 1
            stats["num_validation_errors"] += 1
            return

        # Step 2: Pre-process (materialize.py)
        try:
            messages = replace_json_args(messages)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            stats["num_filtered"] += 1
            stats["num_errors"] += 1
            logger.debug(f"Error in replace_json_args: {e}")
            return

        # Step 3: Apply chat template, get role-labeled chunks (materialize.py)
        try:
            masked_results = create_masked_messages(messages, self._tokenizer, tools)
        except Exception as e:
            stats["num_filtered"] += 1
            stats["num_errors"] += 1
            logger.debug(f"Error in create_masked_messages: {e}")
            return

        # Step 4: For each output sequence (may be multiple due to multi-turn splitting)
        for chunks, _ in masked_results:
            # Post-process: split system/user (materialize_fast.py)
            processed_chunks = split_system_user_chunks(chunks)

            # Step 5: Tokenize and build loss_mask
            try:
                input_ids, loss_mask = self._tokenize_chunks_with_mask(processed_chunks)
            except Exception as e:
                stats["num_errors"] += 1
                logger.debug(f"Error tokenizing chunks: {e}")
                continue

            # Skip empty sequences
            if not input_ids:
                continue

            # Truncate if needed
            if self.max_doc_tokens and len(input_ids) > self.max_doc_tokens:
                input_ids = input_ids[: self.max_doc_tokens]
                loss_mask = loss_mask[: self.max_doc_tokens]
                stats["num_truncated"] += 1

            # Step 6: Add to packer
            builder.add_sequence(input_ids, loss_mask=loss_mask)
            stats["num_output_sequences"] += 1

    def _tokenize_chunks_with_mask(
        self, chunks: list[dict]
    ) -> tuple[list[int], list[int]]:
        """Tokenize chunks and generate loss_mask based on role.

        Loss mask: 0 for system/user chunks, 1 for assistant chunks.

        Args:
            chunks: List of chunks with 'role' and 'content' fields.

        Returns:
            Tuple of (input_ids, loss_mask).
        """
        all_input_ids: list[int] = []
        all_loss_mask: list[int] = []

        for chunk in chunks:
            # Tokenize the pre-rendered content (no special tokens - already in template)
            tokens = self._tokenizer.encode(chunk["content"], add_special_tokens=False)

            # Build mask based on role: assistant = 1, others = 0
            mask_value = 1 if chunk["role"] == "assistant" else 0
            mask = [mask_value] * len(tokens)

            all_input_ids.extend(tokens)
            all_loss_mask.extend(mask)

        return all_input_ids, all_loss_mask

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

    def _iter_parquet_records(self, path: str, fs) -> Iterator[dict]:
        """Iterate records from parquet file."""
        if self._is_remote_path(path):
            with fs.open(path, "rb") as f:
                parquet_file = pq.ParquetFile(f)
                yield from self._iter_parquet_batches_as_dicts(parquet_file)
        else:
            parquet_file = pq.ParquetFile(path)
            yield from self._iter_parquet_batches_as_dicts(parquet_file)

    def _iter_parquet_batches_as_dicts(
        self, parquet_file: pq.ParquetFile
    ) -> Iterator[dict]:
        """Iterate batches from parquet file as dicts."""
        for batch in parquet_file.iter_batches(batch_size=1000):
            table = batch.to_pydict()
            # Transpose from column-oriented to row-oriented
            keys = list(table.keys())
            num_rows = len(table[keys[0]]) if keys else 0
            for i in range(num_rows):
                yield {k: table[k][i] for k in keys}

    def _iter_jsonl_records(self, path: str, fs) -> Iterator[dict]:
        """Iterate records from JSONL file."""
        if self._is_remote_path(path):
            with fs.open(path, "r") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        else:
            with open(path, "r") as f:
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
