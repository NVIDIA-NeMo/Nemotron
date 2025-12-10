"""JsonlShardProcessor Ray actor for parallel JSONL output processing."""

import json
import logging
import time
from dataclasses import asdict
from typing import Callable, Iterator

import pyarrow.parquet as pq
import ray
from fsspec import filesystem

from nemotron.data_prep.config import FileInfo
from nemotron.data_prep.filesystem import ensure_dir, write_json
from nemotron.data_prep.formats.jsonl_dataset import JsonlDatasetBuilder

logger = logging.getLogger(__name__)


@ray.remote
class JsonlShardProcessor:
    """Ray actor for processing data files to JSONL output.

    Reads input files (parquet or jsonl), applies optional transform,
    and writes to JSONL output (optionally compressed).
    """

    def __init__(
        self,
        text_field: str,
        transform: Callable[[dict], dict | None] | None = None,
        compression: str = "none",
        max_rows: int | None = None,
    ):
        """Initialize JSONL processor.

        Args:
            text_field: Field name for text in input records.
            transform: Optional callable to transform records.
            compression: Output compression ("none" or "zstd").
            max_rows: Maximum rows to process per shard.
        """
        self.text_field = text_field
        self.transform = transform
        self.compression = compression
        self.max_rows = max_rows

    def process_shard(
        self,
        shard_index: int,
        files: list[dict],  # FileInfo as dicts for Ray serialization
        output_dir: str,
        fs_protocol: str,
    ) -> dict:
        """Process files to a single JSONL shard.

        Args:
            shard_index: Index of this shard.
            files: List of FileInfo dicts to process.
            output_dir: Output directory for JSONL files.
            fs_protocol: Filesystem protocol (e.g., "file", "s3").

        Returns:
            Shard statistics dict.
        """
        fs = filesystem(fs_protocol)

        shard_id = f"shard_{shard_index:06d}"
        ext = ".jsonl.zst" if self.compression == "zstd" else ".jsonl"
        jsonl_path = f"{output_dir}/{shard_id}{ext}"
        receipt_path = f"{output_dir}/{shard_id}.receipt.json"

        # Ensure directory exists
        ensure_dir(fs, output_dir)

        # Convert file dicts back to FileInfo
        file_infos = [FileInfo(**f) for f in files]

        # Handle empty assignment
        if not file_infos:
            return self._write_empty_receipt(shard_id, shard_index, receipt_path, fs)

        # Process files and write JSONL
        with fs.open(jsonl_path, "wb") as f:
            builder = JsonlDatasetBuilder(
                file=f,
                transform=self.transform,
                compression=self.compression,
            )

            rows_processed = 0
            for file_info in file_infos:
                if self.max_rows and rows_processed >= self.max_rows:
                    break
                rows_processed = self._process_file(file_info, builder, fs, rows_processed)

            builder.finalize()
            total_bytes, checksum = builder.get_info()
            stats = builder.get_stats()

        # Write receipt
        receipt = {
            "shard_id": shard_id,
            "shard_index": shard_index,
            "status": "completed",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_files": [f.path for f in file_infos],
            "output_file": f"{shard_id}{ext}",
            "total_bytes": total_bytes,
            "checksum": checksum,
            **stats,
        }

        write_json(fs, receipt_path, receipt)
        return stats

    def _process_file(
        self,
        file_info: FileInfo,
        builder: JsonlDatasetBuilder,
        fs,
        rows_processed: int,
    ) -> int:
        """Process a single file, writing records to builder.

        Returns the total number of rows processed.
        """
        local_path = self._resolve_file_path(file_info)

        # Determine file type
        is_parquet = local_path.endswith(".parquet") or not (
            local_path.endswith(".jsonl") or local_path.endswith(".json")
        )

        if is_parquet:
            for record in self._iter_parquet_records(local_path, fs):
                if self.max_rows and rows_processed >= self.max_rows:
                    break
                builder.add_record(record)
                rows_processed += 1
        else:
            for record in self._iter_jsonl_records(local_path, fs):
                if self.max_rows and rows_processed >= self.max_rows:
                    break
                builder.add_record(record)
                rows_processed += 1

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

    def _iter_parquet_records(self, path: str, fs) -> Iterator[dict]:
        """Iterate records from parquet file."""
        if self._is_remote_path(path):
            with fs.open(path, "rb") as f:
                parquet_file = pq.ParquetFile(f)
                yield from self._iter_parquet_batches_as_dicts(parquet_file)
        else:
            parquet_file = pq.ParquetFile(path)
            yield from self._iter_parquet_batches_as_dicts(parquet_file)

    def _iter_parquet_batches_as_dicts(self, parquet_file: pq.ParquetFile) -> Iterator[dict]:
        """Iterate records from parquet file as dicts."""
        for batch in parquet_file.iter_batches(batch_size=10000):
            # Convert batch to list of dicts
            table = batch.to_pydict()
            num_rows = len(next(iter(table.values())))
            for i in range(num_rows):
                yield {k: v[i] for k, v in table.items()}

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
        receipt_path: str,
        fs,
    ) -> dict:
        """Write receipt for empty shard."""
        receipt = {
            "shard_id": shard_id,
            "shard_index": shard_index,
            "status": "completed",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input_files": [],
            "output_file": None,
            "total_bytes": 0,
            "checksum": "xxh64:empty",
            "num_records": 0,
            "num_skipped": 0,
        }

        write_json(fs, receipt_path, receipt)
        return {"num_records": 0, "num_skipped": 0, "total_bytes": 0}
