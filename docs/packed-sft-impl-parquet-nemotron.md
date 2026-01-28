# Implementation Plan: Parquet Writer (Nemotron)

This document details the Nemotron data_prep changes for Option B (Parquet) from the [Packed SFT Format Design](./packed-sft-format-design.md).

For the Megatron-Bridge reader implementation, see [packed-sft-impl-parquet-megatron-bridge.md](./packed-sft-impl-parquet-megatron-bridge.md).

## Format Specification

```
shard_000000.parquet
  Schema:
    - input_ids: list<int32>      # Variable-length token ids
    - loss_mask: list<uint8>      # Variable-length loss mask
    - seq_start_id: list<int32>   # Variable-length sequence start positions

  Compression: zstd (default)
  Row groups: ~1000 rows each (tunable)
```

**Key advantages:**
- Single file per shard
- Native variable-length support (no padding waste)
- 2-3x compression with zstd
- Direct cloud storage support (S3, GCS, Azure)

## Output Structure

Nano3 SFT data-prep produces a canonical output structure:

```
output_dir/
├── blend.json                  # Per-split blend for provenance
├── splits/                     # Canonical split directories
│   ├── train/                  # Symlinks to training shards
│   │   ├── shard_000000.parquet -> ../runs/.../shard_000000.parquet
│   │   └── ...
│   ├── valid/                  # Symlinks to validation shards
│   │   └── ...
│   └── test/                   # Symlinks to test shards
│       └── ...
└── runs/{run_hash}/            # Run directory with actual shard files
    └── datasets/{name}/{hash}/
        ├── shard_000000.parquet
        └── ...
```

**Training inputs:** Use split directories or globs:
- Directory: `packed_train_data_path: /path/to/output_dir/splits/train/`
- Glob: `packed_train_data_path: /path/to/output_dir/splits/train/*.parquet`

**Seamless config:** Use `nano3_packed_sft_dir` to auto-resolve split paths:
```yaml
dataset:
  nano3_packed_sft_dir: /path/to/output_dir
  packed_sequence_specs:
    packed_sequence_size: 4096
    # packed_train_data_path auto-resolves to /path/to/output_dir/splits/train/
    # packed_val_data_path auto-resolves to /path/to/output_dir/splits/valid/
```

---

## Files to Modify

| File | Change |
|------|--------|
| `src/nemotron/data_prep/packing/writers.py` | New file with `ParquetShardWriter` class |
| `src/nemotron/data_prep/packing/materialize.py` | Add `materialize_bin_arrays()` function |
| `src/nemotron/data_prep/chat_sft_shard_core.py` | Integrate writer selection |
| `src/nemotron/data_prep/config.py` | Add `packed_storage` config option |

---

## Step 1: Writer Module

Create `src/nemotron/data_prep/packing/writers.py`:

```python
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class ParquetShardWriter:
    """Memory-efficient writer using Parquet with streaming row groups.

    Writes bins incrementally, flushing to disk every `row_group_size` bins.
    Supports direct cloud writes via PyArrow filesystem integration.

    Peak memory: O(row_group_size * pack_size) for the row group buffer.
    """

    SCHEMA = pa.schema([
        ('input_ids', pa.list_(pa.int32())),
        ('loss_mask', pa.list_(pa.uint8())),
        ('seq_start_id', pa.list_(pa.int32())),
    ])

    def __init__(
        self,
        output_path: str,
        row_group_size: int = 1000,
        compression: str = 'zstd',
        filesystem: pa.fs.FileSystem | None = None,
    ):
        """
        Args:
            output_path: Path to output .parquet file (local or cloud URI)
            row_group_size: Number of bins per row group (tune for access patterns)
            compression: Compression codec ('zstd', 'snappy', 'gzip', 'none')
            filesystem: Optional PyArrow filesystem for cloud storage
        """
        self.output_path = output_path
        self.tmp_path = output_path + '.tmp'
        self.row_group_size = row_group_size
        self.compression = compression
        self.filesystem = filesystem

        # Accumulate numpy arrays for current row group
        self._input_ids_values: list[np.ndarray] = []
        self._loss_mask_values: list[np.ndarray] = []
        self._seq_start_values: list[np.ndarray] = []
        self._count = 0
        self._total_bins = 0

        # Open writer
        self._writer = pq.ParquetWriter(
            self.tmp_path,
            self.SCHEMA,
            compression=compression,
            filesystem=filesystem,
        )

    def write_bin(
        self,
        bin_id: int,
        input_ids: np.ndarray,
        loss_mask: np.ndarray,
        seq_start_id: np.ndarray,
    ) -> None:
        """Buffer a single bin, flushing to disk when row group is full."""
        # Store numpy arrays directly (no .tolist() conversion!)
        self._input_ids_values.append(input_ids.astype(np.int32, copy=False))
        self._loss_mask_values.append(loss_mask.astype(np.uint8, copy=False))
        self._seq_start_values.append(seq_start_id.astype(np.int32, copy=False))
        self._count += 1
        self._total_bins += 1

        # Flush row group when buffer is full
        if self._count >= self.row_group_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Write buffered bins as a row group."""
        if self._count == 0:
            return

        # Build Arrow arrays from numpy (efficient, minimal copying)
        input_ids_arr = pa.array(self._input_ids_values, type=pa.list_(pa.int32()))
        loss_mask_arr = pa.array(self._loss_mask_values, type=pa.list_(pa.uint8()))
        seq_start_arr = pa.array(self._seq_start_values, type=pa.list_(pa.int32()))

        table = pa.Table.from_arrays(
            [input_ids_arr, loss_mask_arr, seq_start_arr],
            schema=self.SCHEMA,
        )
        self._writer.write_table(table)

        # Clear buffers
        self._input_ids_values.clear()
        self._loss_mask_values.clear()
        self._seq_start_values.clear()
        self._count = 0

    def finalize(self) -> dict[str, Any]:
        """Flush remaining data, close writer, rename to final path."""
        self._flush_buffer()
        self._writer.close()

        # Atomic rename
        if self.filesystem:
            self.filesystem.move(self.tmp_path, self.output_path)
        else:
            os.rename(self.tmp_path, self.output_path)

        return {
            'format': 'parquet',
            'compression': self.compression,
            'num_bins': self._total_bins,
            'row_group_size': self.row_group_size,
        }


class LegacyPickleWriter:
    """Backward-compatible writer using pickle-of-dicts (current format)."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.packed_data: list[dict] = []

    def write_bin(
        self,
        bin_id: int,
        input_ids: np.ndarray,
        loss_mask: np.ndarray,
        seq_start_id: np.ndarray,
    ) -> None:
        self.packed_data.append({
            "input_ids": input_ids.tolist(),
            "loss_mask": loss_mask.tolist(),
            "seq_start_id": seq_start_id.tolist(),
        })

    def finalize(self) -> dict[str, Any]:
        tmp_path = self.output_path + ".tmp"
        with open(tmp_path, "wb") as f:
            np.save(f, self.packed_data, allow_pickle=True)
        os.rename(tmp_path, self.output_path)
        return {"format": "legacy_pickle", "num_bins": len(self.packed_data)}
```

---

## Step 2: Materialize Function

Add to `src/nemotron/data_prep/packing/materialize.py`:

```python
def materialize_bin_arrays(
    spool_reader: SequenceSpoolReader,
    assignment: BinAssignment,
    bin_id: int,
    pack_size: int,
    scratch_input_ids: np.ndarray,
    scratch_loss_mask: np.ndarray,
) -> tuple[int, np.ndarray]:
    """
    Materialize a single bin directly to numpy arrays.

    Avoids Python list conversion - writes directly to preallocated buffers.

    Args:
        spool_reader: Reader for tokenized sequence spool
        assignment: Bin assignment from packing algorithm
        bin_id: Which bin to materialize
        pack_size: Maximum packed sequence length
        scratch_input_ids: Preallocated buffer of shape (pack_size,)
        scratch_loss_mask: Preallocated buffer of shape (pack_size,)

    Returns:
        packed_len: Actual length of packed tokens (excluding padding)
        seq_start_id: Array of sequence START positions within the bin.
                      Invariant: seq_start_id[0] == 0, strictly increasing,
                      seq_start_id[-1] < packed_len.
                      To get boundaries: list(seq_start_id) + [packed_len]
    """
    seq_indices = assignment.bin_indices(bin_id)

    # Zero the scratch buffers (for padding)
    scratch_input_ids[:] = 0
    scratch_loss_mask[:] = 0

    pos = 0
    seq_start_ids = []  # Collect START positions

    for seq_index in seq_indices:
        input_ids_arr, loss_mask_arr = spool_reader.read_sequence(int(seq_index))

        # Truncate if needed
        seq_len = min(len(input_ids_arr), pack_size)
        if pos + seq_len > pack_size:
            seq_len = pack_size - pos

        if seq_len <= 0:
            break

        # Record start position BEFORE writing
        seq_start_ids.append(pos)

        # Write directly to scratch buffers (no Python list!)
        scratch_input_ids[pos:pos + seq_len] = input_ids_arr[:seq_len]
        scratch_loss_mask[pos:pos + seq_len] = loss_mask_arr[:seq_len]
        pos += seq_len

    # Apply loss_mask roll (shift right by 1, first position is 0)
    # This ensures loss is computed on predicting token[i+1] from token[i]
    if pos > 0:
        scratch_loss_mask[1:pos] = scratch_loss_mask[:pos-1].copy()
        scratch_loss_mask[0] = 0

    return pos, np.array(seq_start_ids, dtype=np.uint32)
```

---

## Step 3: Update Central Pack Function

Update `src/nemotron/data_prep/chat_sft_shard_core.py`:

```python
from nemotron.data_prep.packing.writers import (
    ParquetShardWriter,
    LegacyPickleWriter,
)
from nemotron.data_prep.packing.materialize import materialize_bin_arrays


def process_chat_sft_pack_from_spool_core(
    *,
    spool_dir: str,
    output_dir: str,
    shard_id: str,
    pack_size: int,
    packer: Packer,
    output_fs: AbstractFileSystem,
    dtype: np.dtype = np.int32,
    packed_storage: str = "legacy_npy_pickle",  # NEW PARAM
    parquet_row_group_size: int = 1000,
    parquet_compression: str = "zstd",
) -> dict[str, Any]:
    """Process spool files into packed output."""

    # ... existing setup code (load spool, run packer) ...

    # Choose writer based on config
    if packed_storage == "parquet":
        parquet_path = f"{output_dir}/{shard_id}.parquet"

        # Get PyArrow filesystem for cloud support
        pa_filesystem = _get_pyarrow_filesystem(output_fs)

        writer = ParquetShardWriter(
            output_path=parquet_path,
            row_group_size=parquet_row_group_size,
            compression=parquet_compression,
            filesystem=pa_filesystem,
        )
        output_path = parquet_path
    else:
        npy_path = f"{output_dir}/{shard_id}.npy"
        writer = LegacyPickleWriter(npy_path)
        output_path = npy_path

    # Preallocate scratch buffers
    scratch_input_ids = np.zeros(pack_size, dtype=dtype)
    scratch_loss_mask = np.zeros(pack_size, dtype=np.uint8)

    # Stream bins
    for bin_id in range(num_bins):
        packed_len, seq_start_id = materialize_bin_arrays(
            spool_reader=reader,
            assignment=assignment,
            bin_id=bin_id,
            pack_size=pack_size,
            scratch_input_ids=scratch_input_ids,
            scratch_loss_mask=scratch_loss_mask,
        )

        writer.write_bin(
            bin_id=bin_id,
            input_ids=scratch_input_ids[:packed_len].copy(),
            loss_mask=scratch_loss_mask[:packed_len].copy(),
            seq_start_id=seq_start_id,
        )

    result = writer.finalize()

    # Build receipt
    receipt = {
        "shard_id": shard_id,
        "output_path": output_path,
        "format": packed_storage,
        "num_bins": result["num_bins"],
        # ... other receipt fields ...
    }

    return receipt


def _get_pyarrow_filesystem(fsspec_fs: AbstractFileSystem) -> pa.fs.FileSystem | None:
    """Convert fsspec filesystem to PyArrow filesystem."""
    if fsspec_fs.protocol == "file":
        return None  # Use default local filesystem

    if fsspec_fs.protocol == "s3":
        return pa.fs.S3FileSystem()
    elif fsspec_fs.protocol == "gs" or fsspec_fs.protocol == "gcs":
        return pa.fs.GcsFileSystem()
    elif fsspec_fs.protocol == "az" or fsspec_fs.protocol == "abfs":
        return pa.fs.AzureFileSystem()
    else:
        # Fallback: use PyArrow's fsspec wrapper
        from pyarrow.fs import PyFileSystem, FSSpecHandler
        return PyFileSystem(FSSpecHandler(fsspec_fs))
```

---

## Step 4: Config Changes

Update `src/nemotron/data_prep/config.py`:

```python
from typing import Literal


class ChatSftOutputConfig(BaseModel):
    # ... existing fields ...
    packed_storage: Literal["legacy_npy_pickle", "parquet"] = "legacy_npy_pickle"
    parquet_row_group_size: int = 1000
    parquet_compression: Literal["zstd", "snappy", "gzip", "none"] = "zstd"
```

---

## Cloud Storage Support

Parquet has native cloud storage support via PyArrow:

### Direct Cloud Writes

```python
import pyarrow.fs as pafs


def get_filesystem_for_uri(uri: str) -> tuple[pa.fs.FileSystem, str]:
    """Parse URI and return appropriate filesystem + path."""
    if uri.startswith("s3://"):
        return pafs.S3FileSystem(), uri[5:]
    elif uri.startswith("gs://"):
        return pafs.GcsFileSystem(), uri[5:]
    elif uri.startswith("az://") or uri.startswith("abfs://"):
        return pafs.AzureFileSystem(), uri.split("://", 1)[1]
    else:
        return None, uri  # Local filesystem


# Usage in writer
def write_to_cloud(output_uri: str, ...):
    filesystem, path = get_filesystem_for_uri(output_uri)

    writer = ParquetShardWriter(
        output_path=path,
        filesystem=filesystem,
        compression='zstd',
    )
    # ... write bins ...
    writer.finalize()
```

---

## Performance Tuning

### Row Group Size

The `row_group_size` parameter affects both write and read performance:

| Row Group Size | Write Memory | Read Latency | Best For |
|----------------|--------------|--------------|----------|
| 100 | ~2 MB | Lower | Random access heavy |
| 1000 (default) | ~20 MB | Medium | Balanced workloads |
| 10000 | ~200 MB | Higher | Sequential scans |

```python
# For training (sequential access): larger row groups
writer = ParquetShardWriter(output_path, row_group_size=5000)

# For inference (random access): smaller row groups
writer = ParquetShardWriter(output_path, row_group_size=100)
```

### Compression

| Codec | Ratio | Write Speed | Read Speed |
|-------|-------|-------------|------------|
| zstd | ~2.5x | Medium | Fast |
| snappy | ~1.5x | Fast | Fast |
| gzip | ~3x | Slow | Medium |
| none | 1x | Fastest | Fastest |

---

## Conversion Tool

Convert existing legacy `.npy` files to Parquet:

```python
def convert_legacy_to_parquet(
    legacy_npy_path: str,
    output_path: str,
    row_group_size: int = 1000,
    compression: str = 'zstd',
) -> dict[str, Any]:
    """Convert existing pickle .npy to Parquet format.

    WARNING: Only use with trusted .npy files - pickle loading is unsafe
    with untrusted inputs.
    """
    import os

    data = np.load(legacy_npy_path, allow_pickle=True)

    writer = ParquetShardWriter(
        output_path=output_path,
        row_group_size=row_group_size,
        compression=compression,
    )

    for i, d in enumerate(data):
        writer.write_bin(
            bin_id=i,
            input_ids=np.array(d["input_ids"], dtype=np.int32),
            loss_mask=np.array(d["loss_mask"], dtype=np.uint8),
            seq_start_id=np.array(d["seq_start_id"], dtype=np.int32),
        )

    result = writer.finalize()

    # Print size comparison
    original_size = os.path.getsize(legacy_npy_path)
    new_size = os.path.getsize(output_path)
    print(f"Converted {result['num_bins']} bins to {output_path}")
    print(f"Size: {original_size / 1e6:.1f} MB -> {new_size / 1e6:.1f} MB "
          f"({new_size / original_size * 100:.1f}%)")

    return result
```

---

## Testing

### Unit Tests

```python
import tempfile
import numpy as np
import pytest

from nemotron.data_prep.packing.writers import ParquetShardWriter


def test_parquet_writer_roundtrip():
    """Test write and read produce identical data."""
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        output_path = f.name

    try:
        # Write
        writer = ParquetShardWriter(
            output_path=output_path,
            row_group_size=2,
            compression='zstd',
        )

        test_data = [
            (np.array([1, 2, 3]), np.array([0, 1, 1]), np.array([0])),
            (np.array([4, 5, 6, 7]), np.array([0, 0, 1, 1]), np.array([0, 2])),
            (np.array([8, 9]), np.array([0, 1]), np.array([0, 1, 2])),
        ]

        for i, (ids, mask, starts) in enumerate(test_data):
            writer.write_bin(i, ids, mask, starts)

        writer.finalize()

        # Read back
        import pyarrow.parquet as pq
        table = pq.read_table(output_path)

        for i, (expected_ids, expected_mask, expected_starts) in enumerate(test_data):
            actual_ids = table['input_ids'][i].as_py()
            actual_mask = table['loss_mask'][i].as_py()
            actual_starts = table['seq_start_id'][i].as_py()

            np.testing.assert_array_equal(actual_ids, expected_ids)
            np.testing.assert_array_equal(actual_mask, expected_mask)
            np.testing.assert_array_equal(actual_starts, expected_starts)

    finally:
        import os
        os.unlink(output_path)


def test_parquet_memory_efficiency():
    """Verify peak memory stays reasonable."""
    import tracemalloc

    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        output_path = f.name

    try:
        tracemalloc.start()

        writer = ParquetShardWriter(
            output_path=output_path,
            row_group_size=100,  # Small row groups
            compression='zstd',
        )

        for i in range(10000):
            writer.write_bin(
                i,
                np.random.randint(0, 50000, size=2000, dtype=np.int32),
                np.random.randint(0, 2, size=2000, dtype=np.uint8),
                np.array([0, 500, 1000, 1500], dtype=np.int32),
            )

        writer.finalize()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"Peak memory: {peak / 1e6:.1f} MB")
        # With row_group_size=100, peak should be well under 50MB
        assert peak < 50 * 1024 * 1024

    finally:
        import os
        os.unlink(output_path)


def test_parquet_compression_ratio():
    """Verify compression achieves expected ratio."""
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        output_path = f.name

    try:
        writer = ParquetShardWriter(
            output_path=output_path,
            row_group_size=1000,
            compression='zstd',
        )

        for i in range(1000):
            writer.write_bin(
                i,
                np.random.randint(0, 50000, size=2000, dtype=np.int32),
                np.random.randint(0, 2, size=2000, dtype=np.uint8),
                np.array([0, 500, 1000, 1500], dtype=np.int32),
            )

        writer.finalize()

        import os
        file_size = os.path.getsize(output_path)
        raw_size = 1000 * (2000 * 4 + 2000 * 1 + 4 * 4)  # Uncompressed estimate

        compression_ratio = raw_size / file_size
        print(f"Compression ratio: {compression_ratio:.1f}x")
        assert compression_ratio > 1.5

    finally:
        os.unlink(output_path)
```
