# Implementation Plan: Memmap Packed SFT Format

This document details the implementation for Option A (Padded Fixed-Shape Memmap Arrays) from the [Packed SFT Format Design](./packed-sft-format-design.md).

## Format Specification

```
shard_000000/
├── input_ids.npy      # int32[num_bins, pack_size]  - padded to pack_size
├── loss_mask.npy      # uint8[num_bins, pack_size]  - padded to pack_size
├── packed_len.npy     # uint32[num_bins]            - actual length per bin
├── seq_offsets.npy    # uint32[num_bins + 1]        - CSR pointers
├── seq_starts.npy     # uint32[total_seq_starts]    - seq_start_id values
└── manifest.json      # schema version, metadata
```

---

## Phase 1: Nemotron Writer Changes

### 1.1 Writer Module

Create `src/nemotron/data_prep/packing/writers.py`:

```python
from __future__ import annotations

import json
import os
from typing import Any, Protocol

import numpy as np


class PackedShardWriter(Protocol):
    """Abstract interface for packed shard output."""

    def write_bin(
        self,
        bin_id: int,
        input_ids: np.ndarray,
        loss_mask: np.ndarray,
        seq_start_id: np.ndarray,
    ) -> None: ...

    def finalize(self) -> dict[str, Any]: ...


class MemmapShardWriter:
    """Memory-efficient writer using numpy memmap.

    Uses np.lib.format.open_memmap to pre-allocate arrays on disk,
    then writes bins incrementally without accumulating in memory.

    Peak memory: O(pack_size) for scratch buffers only.
    """

    def __init__(
        self,
        output_dir: str,
        num_bins: int,
        pack_size: int,
        total_seq_starts: int,
        dtype: np.dtype = np.int32,
    ):
        self.output_dir = output_dir
        self.num_bins = num_bins
        self.pack_size = pack_size
        self.dtype = dtype

        os.makedirs(output_dir, exist_ok=True)

        # Pre-allocate memmap arrays with known sizes
        self.input_ids = np.lib.format.open_memmap(
            f"{output_dir}/input_ids.npy.tmp",
            mode='w+',
            dtype=dtype,
            shape=(num_bins, pack_size),
        )
        self.loss_mask = np.lib.format.open_memmap(
            f"{output_dir}/loss_mask.npy.tmp",
            mode='w+',
            dtype=np.uint8,
            shape=(num_bins, pack_size),
        )
        self.packed_len = np.lib.format.open_memmap(
            f"{output_dir}/packed_len.npy.tmp",
            mode='w+',
            dtype=np.uint32,
            shape=(num_bins,),
        )
        # CSR format for variable-length seq_start_id
        self.seq_offsets = np.lib.format.open_memmap(
            f"{output_dir}/seq_offsets.npy.tmp",
            mode='w+',
            dtype=np.uint32,
            shape=(num_bins + 1,),
        )
        self.seq_starts = np.lib.format.open_memmap(
            f"{output_dir}/seq_starts.npy.tmp",
            mode='w+',
            dtype=np.uint32,
            shape=(total_seq_starts,),
        )

        self._seq_write_pos = 0
        self.seq_offsets[0] = 0
        self._bins_written = 0

    def write_bin(
        self,
        bin_id: int,
        input_ids: np.ndarray,
        loss_mask: np.ndarray,
        seq_start_id: np.ndarray,
    ) -> None:
        """Write a single packed bin to the memmap arrays."""
        L = len(input_ids)

        # Write padded input_ids and loss_mask (rest stays zero)
        self.input_ids[bin_id, :L] = input_ids
        self.loss_mask[bin_id, :L] = loss_mask
        self.packed_len[bin_id] = L

        # Write seq_start_id in CSR format
        S = len(seq_start_id)
        self.seq_starts[self._seq_write_pos:self._seq_write_pos + S] = seq_start_id
        self._seq_write_pos += S
        self.seq_offsets[bin_id + 1] = self._seq_write_pos

        self._bins_written += 1

    def finalize(self) -> dict[str, Any]:
        """Flush memmaps, rename to final paths, write manifest."""
        # Flush and close memmaps
        del self.input_ids
        del self.loss_mask
        del self.packed_len
        del self.seq_offsets
        del self.seq_starts

        # Atomic rename from .tmp to final
        for name in ['input_ids', 'loss_mask', 'packed_len', 'seq_offsets', 'seq_starts']:
            tmp_path = f"{self.output_dir}/{name}.npy.tmp"
            final_path = f"{self.output_dir}/{name}.npy"
            os.rename(tmp_path, final_path)

        # Write manifest with explicit dtype including endianness
        manifest = {
            "version": "1.0",
            "format": "memmap_padded_v1",
            "num_bins": self.num_bins,
            "pack_size": self.pack_size,
            "dtype": self.dtype.str,  # e.g., "<i4" for little-endian int32
            "loss_mask_dtype": "<u1",
            "index_dtype": "<u4",
            "bins_written": self._bins_written,
        }
        with open(f"{self.output_dir}/manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest


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

### 1.2 Materialize Function

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

### 1.3 Update Central Pack Function

Update `src/nemotron/data_prep/chat_sft_shard_core.py`:

```python
from nemotron.data_prep.packing.writers import (
    MemmapShardWriter,
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
) -> dict[str, Any]:
    """Process spool files into packed output."""

    # ... existing setup code (load spool, run packer) ...

    # After packing, we know all sizes
    num_bins = assignment.num_bins
    num_sequences = int(lengths.shape[0])
    total_seq_starts = num_sequences  # Each sequence contributes one entry

    # Choose writer based on config
    if packed_storage == "memmap_v1":
        shard_output_dir = f"{output_dir}/{shard_id}"
        writer = MemmapShardWriter(
            output_dir=shard_output_dir,
            num_bins=num_bins,
            pack_size=pack_size,
            total_seq_starts=total_seq_starts,
            dtype=dtype,
        )
        output_path = shard_output_dir
    else:
        npy_path = f"{output_dir}/{shard_id}.npy"
        writer = LegacyPickleWriter(npy_path)
        output_path = npy_path

    # Preallocate scratch buffers (reused for every bin!)
    scratch_input_ids = np.zeros(pack_size, dtype=dtype)
    scratch_loss_mask = np.zeros(pack_size, dtype=np.uint8)

    # Stream bins without accumulating
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

    writer.finalize()

    # Build receipt
    receipt = {
        "shard_id": shard_id,
        "output_path": output_path,
        "format": packed_storage,
        "num_bins": num_bins,
        "num_sequences": num_sequences,
        # ... other receipt fields ...
    }

    return receipt
```

### 1.4 Config Changes

Update `src/nemotron/data_prep/config.py`:

```python
from typing import Literal

class ChatSftOutputConfig(BaseModel):
    # ... existing fields ...
    packed_storage: Literal["legacy_npy_pickle", "memmap_v1"] = "legacy_npy_pickle"
```

---

## Phase 2: Megatron-Bridge Reader Changes

### 2.1 Memmap Dataset Class

Add to `src/megatron/bridge/data/datasets/sft.py`:

```python
import json
from pathlib import Path

import numpy as np


class GPTSFTPackedMemmapDataset(GPTSFTDataset):
    """Memory-efficient packed dataset using memmap arrays.

    Reads from the memmap_padded_v1 format:
    - input_ids.npy: int32[num_bins, pack_size]
    - loss_mask.npy: uint8[num_bins, pack_size]
    - packed_len.npy: uint32[num_bins]
    - seq_offsets.npy: uint32[num_bins + 1]
    - seq_starts.npy: uint32[total_seq_starts]
    - manifest.json: metadata

    Memory usage: O(metadata) at load time, O(pack_size) per sample access.

    DataLoader compatibility: Uses lazy-open pattern. Memmaps are opened on
    first access in each worker process, avoiding pickling issues with
    num_workers > 0.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: MegatronTokenizer,
        **kwargs,
    ):
        self.shard_dir = file_path
        self._loaded = False
        self._arrays = None
        super().__init__(file_path, tokenizer, **kwargs)

    def _load_dataset(self):
        """Load manifest only. Memmaps opened lazily on first access."""
        manifest_path = Path(self.shard_dir) / "manifest.json"
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        self._num_bins = self.manifest["num_bins"]
        self._pack_size = self.manifest["pack_size"]

    def _ensure_memmaps_open(self):
        """Open memmaps on first access (per-worker). Thread-safe."""
        if self._loaded:
            return

        # Memory-map arrays (NOT loaded into RAM!)
        self.input_ids = np.load(
            f"{self.shard_dir}/input_ids.npy", mmap_mode='r'
        )
        self.loss_mask = np.load(
            f"{self.shard_dir}/loss_mask.npy", mmap_mode='r'
        )
        self.packed_len = np.load(
            f"{self.shard_dir}/packed_len.npy", mmap_mode='r'
        )
        self.seq_offsets = np.load(
            f"{self.shard_dir}/seq_offsets.npy", mmap_mode='r'
        )
        self.seq_starts = np.load(
            f"{self.shard_dir}/seq_starts.npy", mmap_mode='r'
        )
        self._loaded = True

    def __len__(self):
        return self._num_bins

    def __getitem__(self, idx):
        """Read a single packed sample with O(1) access."""
        self._ensure_memmaps_open()  # Lazy open per worker

        if self.samples_mapping is not None:
            idx = self.samples_mapping[idx]

        # Read only the data we need
        L = int(self.packed_len[idx])
        input_ids = self.input_ids[idx, :L]
        loss_mask = self.loss_mask[idx, :L]

        # Reconstruct seq_start_id from CSR format
        start = int(self.seq_offsets[idx])
        end = int(self.seq_offsets[idx + 1])
        seq_start_id = self.seq_starts[start:end].tolist()

        # Boundaries = start positions + final length
        # Invariant: seq_start_id contains starts, we add packed_len as final boundary
        seq_boundaries = seq_start_id + [L]

        if idx < 0:
            loss_mask = np.zeros_like(loss_mask)

        return {
            "input_ids": input_ids,
            "seq_boundaries": seq_boundaries,
            "loss_mask": loss_mask,
        }

    def __getstate__(self):
        """For pickling across DataLoader workers - exclude memmaps."""
        state = self.__dict__.copy()
        # Remove unpicklable memmaps
        state['_loaded'] = False
        for key in ['input_ids', 'loss_mask', 'packed_len', 'seq_offsets', 'seq_starts']:
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        """Restore from pickle - memmaps will be reopened on first access."""
        self.__dict__.update(state)
```

### 2.2 Multi-Shard Support

```python
from torch.utils.data import Dataset


class GPTSFTPackedMultiShardDataset(Dataset):
    """Combines multiple packed memmap shards into one dataset."""

    def __init__(
        self,
        shard_dirs: list[str],
        tokenizer: MegatronTokenizer,
        **kwargs,
    ):
        self.shards = [
            GPTSFTPackedMemmapDataset(d, tokenizer, **kwargs)
            for d in shard_dirs
        ]

        # Build cumulative index for O(1) shard lookup
        self.shard_offsets = np.cumsum([0] + [len(s) for s in self.shards])

    def __len__(self):
        return int(self.shard_offsets[-1])

    def __getitem__(self, idx):
        # Find which shard contains this index
        shard_id = int(np.searchsorted(self.shard_offsets[1:], idx, side='right'))
        local_idx = idx - int(self.shard_offsets[shard_id])
        return self.shards[shard_id][local_idx]
```

### 2.3 Update Factory Function

Update `create_sft_dataset` in `sft.py`:

```python
def create_sft_dataset(path: Path, tokenizer, ...) -> GPTSFTDataset:
    """Factory function to create appropriate dataset based on format."""

    if path.suffix == ".npy":
        # Legacy pickle format
        return GPTSFTPackedDataset(
            file_path=str(path),
            tokenizer=tokenizer,
            **gpt_sft_dataset_kwargs,
        )
    elif path.is_dir() and (path / "manifest.json").exists():
        # New memmap format
        return GPTSFTPackedMemmapDataset(
            file_path=str(path),
            tokenizer=tokenizer,
            **gpt_sft_dataset_kwargs,
        )
    # ... rest of existing logic ...
```

---

## Phase 3: Cloud Storage Support

Memmap requires local POSIX paths. For cloud outputs, use write-local-then-upload:

```python
import tempfile


def write_memmap_to_cloud(
    output_fs: AbstractFileSystem,
    output_dir: str,
    num_bins: int,
    pack_size: int,
    total_seq_starts: int,
    write_fn: callable,
) -> dict[str, Any]:
    """Write memmap format to cloud storage."""

    if output_fs.protocol == "file":
        # Local filesystem - write directly
        writer = MemmapShardWriter(output_dir, num_bins, pack_size, total_seq_starts)
        write_fn(writer)
        return writer.finalize()

    # Cloud storage - write to temp, then upload
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = MemmapShardWriter(tmpdir, num_bins, pack_size, total_seq_starts)
        write_fn(writer)
        manifest = writer.finalize()

        # Upload all files
        output_fs.makedirs(output_dir, exist_ok=True)
        for filename in os.listdir(tmpdir):
            local_path = f"{tmpdir}/{filename}"
            remote_path = f"{output_dir}/{filename}"
            output_fs.put(local_path, remote_path)

        return manifest
```

---

## Conversion Tool

```python
def convert_legacy_to_memmap(legacy_npy_path: str, output_dir: str) -> dict[str, Any]:
    """Convert existing pickle .npy to memmap format."""
    data = np.load(legacy_npy_path, allow_pickle=True)

    num_bins = len(data)
    pack_size = max(len(d["input_ids"]) for d in data)
    total_seq_starts = sum(len(d["seq_start_id"]) for d in data)

    writer = MemmapShardWriter(output_dir, num_bins, pack_size, total_seq_starts)

    for i, d in enumerate(data):
        writer.write_bin(
            bin_id=i,
            input_ids=np.array(d["input_ids"], dtype=np.int32),
            loss_mask=np.array(d["loss_mask"], dtype=np.uint8),
            seq_start_id=np.array(d["seq_start_id"], dtype=np.uint32),
        )

    return writer.finalize()
```

---

## Testing

### Unit Tests

```python
import tempfile
import numpy as np
import pytest

from nemotron.data_prep.packing.writers import MemmapShardWriter


def test_memmap_writer_roundtrip():
    """Test write and read produce identical data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write
        writer = MemmapShardWriter(
            output_dir=tmpdir,
            num_bins=3,
            pack_size=10,
            total_seq_starts=6,
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
        input_ids = np.load(f"{tmpdir}/input_ids.npy", mmap_mode='r')
        packed_len = np.load(f"{tmpdir}/packed_len.npy", mmap_mode='r')

        for i, (expected_ids, _, _) in enumerate(test_data):
            L = packed_len[i]
            actual_ids = input_ids[i, :L]
            np.testing.assert_array_equal(actual_ids, expected_ids)


def test_memmap_writer_memory_efficiency():
    """Verify peak memory stays constant regardless of data size."""
    import tracemalloc

    with tempfile.TemporaryDirectory() as tmpdir:
        tracemalloc.start()

        writer = MemmapShardWriter(
            output_dir=tmpdir,
            num_bins=10000,
            pack_size=2048,
            total_seq_starts=50000,
        )

        # Write many bins
        for i in range(10000):
            writer.write_bin(
                i,
                np.random.randint(0, 50000, size=2000, dtype=np.int32),
                np.random.randint(0, 2, size=2000, dtype=np.uint8),
                np.array([0, 500, 1000, 1500], dtype=np.uint32),
            )

        writer.finalize()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Peak should be well under 100MB (actual data would be ~400MB)
        assert peak < 100 * 1024 * 1024, f"Peak memory {peak / 1e6:.1f} MB too high"


def test_dataloader_multiworker():
    """Verify dataset works with DataLoader num_workers > 0."""
    from torch.utils.data import DataLoader

    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup: write test data
        writer = MemmapShardWriter(tmpdir, num_bins=100, pack_size=64, total_seq_starts=200)
        for i in range(100):
            writer.write_bin(
                i,
                np.arange(50, dtype=np.int32),
                np.ones(50, dtype=np.uint8),
                np.array([0, 25], dtype=np.uint32),
            )
        writer.finalize()

        # Create dataset and dataloader with workers
        dataset = GPTSFTPackedMemmapDataset(tmpdir, tokenizer=None)
        loader = DataLoader(
            dataset,
            batch_size=8,
            num_workers=4,
            persistent_workers=True,
        )

        # Iterate through entire dataset
        total_samples = 0
        for batch in loader:
            total_samples += len(batch["input_ids"])

        assert total_samples == 100, f"Expected 100 samples, got {total_samples}"
```
