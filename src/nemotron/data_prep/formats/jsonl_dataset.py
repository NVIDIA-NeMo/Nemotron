"""Streaming JSONL writer with optional compression.

Uses orjson for fast serialization and zstandard for compression.
"""

from typing import BinaryIO, Callable, Literal

import xxhash

try:
    import orjson

    def _dumps(obj: dict) -> bytes:
        return orjson.dumps(obj)

except ImportError:
    import json

    def _dumps(obj: dict) -> bytes:
        return json.dumps(obj, ensure_ascii=False).encode("utf-8")


try:
    import zstandard as zstd

    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False


class JsonlDatasetBuilder:
    """Streaming JSONL writer with checksums and optional zstd compression.

    Uses orjson for fast serialization (following nvdata pattern).
    Writes incrementally with buffering to reduce syscall overhead.
    """

    def __init__(
        self,
        file: BinaryIO,
        transform: Callable[[dict], dict | None] | None = None,
        compression: Literal["none", "zstd"] = "none",
        buffer_size: int = 64 * 1024,
    ):
        """Initialize JSONL writer.

        Args:
            file: Binary file handle to write to.
            transform: Optional callable to transform records. Returns dict or None to skip.
            compression: Output compression ("none" or "zstd").
            buffer_size: Write buffer size in bytes.
        """
        self._file = file
        self._transform = transform or (lambda x: x)
        self._compression = compression
        self._buffer_size = buffer_size

        # Stats
        self._num_records = 0
        self._num_skipped = 0
        self._total_bytes = 0

        # Checksum
        self._hasher = xxhash.xxh64()

        # Write buffer
        self._write_buffer = bytearray()

        # Setup compression
        if compression == "zstd":
            if not _HAS_ZSTD:
                raise ImportError(
                    "zstandard package required for zstd compression. "
                    "Install with: pip install zstandard"
                )
            self._compressor = zstd.ZstdCompressor()
            self._writer = self._compressor.stream_writer(file)
        else:
            self._writer = file
            self._compressor = None

    def add_record(self, record: dict) -> bool:
        """Add record, apply transform, write.

        Args:
            record: Input record dict.

        Returns:
            True if record was written, False if filtered by transform.
        """
        transformed = self._transform(record)
        if transformed is None:
            self._num_skipped += 1
            return False

        # orjson returns bytes directly
        line = _dumps(transformed) + b"\n"
        self._hasher.update(line)
        self._total_bytes += len(line)

        self._write_buffer.extend(line)
        self._num_records += 1

        # Flush when buffer is full
        if len(self._write_buffer) >= self._buffer_size:
            self._flush_buffer()

        return True

    def add_records(self, records: list[dict]) -> int:
        """Batch add records.

        Args:
            records: List of input record dicts.

        Returns:
            Number of records written (not skipped).
        """
        written = 0
        for record in records:
            if self.add_record(record):
                written += 1
        return written

    def _flush_buffer(self) -> None:
        """Flush write buffer to file."""
        if self._write_buffer:
            self._writer.write(self._write_buffer)
            self._write_buffer.clear()

    def finalize(self) -> None:
        """Flush buffers and close compressor if used."""
        self._flush_buffer()
        if self._compressor is not None:
            self._writer.close()

    def get_info(self) -> tuple[int, str]:
        """Get bytes written and checksum.

        Returns:
            Tuple of (bytes_written, checksum_string).
        """
        return self._total_bytes, f"xxh64:{self._hasher.hexdigest()}"

    def get_stats(self) -> dict:
        """Get writer statistics.

        Returns:
            Dict with num_records, num_skipped, total_bytes.
        """
        return {
            "num_records": self._num_records,
            "num_skipped": self._num_skipped,
            "total_bytes": self._total_bytes,
        }


__all__ = ["JsonlDatasetBuilder"]
