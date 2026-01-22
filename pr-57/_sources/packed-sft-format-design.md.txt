# RFC: Scalable Packed SFT Data Format

**Status:** Proposed
**Authors:** Data Infrastructure Team
**Created:** 2025-01-20
**Target:** Megatron-Bridge + Nemotron Data Prep

---

## Summary

This RFC proposes adding new memory-efficient formats for packed SFT data in the Megatron-Bridge training pipeline.

**The problem:** The current pickle-based `.npy` format requires loading the entire dataset into memory—both when writing (Nemotron data_prep) and when reading (Megatron-Bridge training). For a typical 50,000-sample packed dataset, this consumes ~4 GB of RAM. This limits the size of datasets we can process and creates memory pressure during multi-node training where every node loads the full dataset.

**Modern scale requirements:** State-of-the-art SFT pipelines operate at significantly larger scale. For example, [Nemotron-3 Nano](https://huggingface.co/blog/nvidia/nemotron-3-nano-efficient-open-intelligent-models) uses a 13-million-sample post-training corpus spanning code, math, multi-turn conversations, and tool use. At this scale, the current format would require **~1 TB of RAM** just to load the dataset—clearly infeasible.

**The proposal:** Add new format options (Parquet, Memmap) that support streaming writes and lazy reads, reducing memory usage by 200-500x for large datasets. The current `.npy` format will remain supported and continue to be the default, ensuring full backward compatibility. Users can opt into new formats when working with larger datasets or cloud storage.

---

## Background

### Current Packed Sequence Format

Megatron-Bridge's `GPTSFTPackedDataset` reads packed SFT data from `.npy` files produced by Nemotron's data preparation pipeline. The format stores a Python list of dictionaries:

```python
# Current format: list of dicts with Python lists
[
    {
        "input_ids": [101, 2054, 2003, ...],      # Variable-length token ids
        "loss_mask": [0, 0, 1, 1, ...],           # Variable-length loss mask
        "seq_start_id": [0, 45, 128, ...]         # Sequence boundaries within the pack
    },
    {
        "input_ids": [...],
        "loss_mask": [...],
        "seq_start_id": [...]
    },
    # ... thousands more packed samples
]
```

This is serialized using `np.save(..., allow_pickle=True)`.

This format works well for small to medium datasets and will remain supported. However, it has scalability limitations for larger workloads:

### Scalability Limitations

#### Problem 1: Write-Side Memory Explosion

In Nemotron's `process_chat_sft_pack_from_spool_core()`:

```python
packed_data: list[dict] = []
for item in materialize_packed_samples(...):
    packed_data.append(item)  # Accumulates ALL samples in memory

np.save(f, packed_data, allow_pickle=True)
```

The Python object overhead is severe:
- Each token (4-byte int32) becomes a ~28-byte Python int object
- Each list adds 8 bytes per element for pointers + 56 bytes header
- **Result:** 50,000 packed samples with pack_size=2048 → **~4 GB peak memory**

#### Problem 2: Read-Side Full Load

In Megatron-Bridge's `GPTSFTPackedDataset._load_dataset()`:

```python
self.indexed_dataset = np.load(self.file_path, allow_pickle=True)
```

This deserializes the entire pickle into RAM before any training iteration. There is no lazy loading or memory mapping—the "random access" only works because everything is already in memory.

| Dataset Size | Memory Required |
|--------------|-----------------|
| 10k samples | ~800 MB |
| 50k samples | ~4 GB |
| 200k samples | ~16 GB |

This becomes prohibitive for multi-node training where each node loads the full dataset.

For reference, modern SFT pipelines like [Nemotron-3 Nano's post-training](https://developer.nvidia.com/blog/inside-nvidia-nemotron-3-techniques-tools-and-data-that-make-it-efficient-and-accurate/) use **13+ million samples** across diverse domains (code, math, multi-turn conversations, tool use). At this scale, the current format is not viable.

#### Problem 3: No Cloud Storage Support

The pickle-based format cannot be streamed from cloud storage (S3, GCS). The entire file must be downloaded and deserialized, adding startup latency to training jobs.

### Contrast: Megatron Pretrain Format

The Megatron pretrain format (`.bin` + `.idx`) demonstrates the right approach:

```python
# Data: memory-mapped binary file
mdata = np.memmap(data_path, dtype=np.uint8, mode="r")

# Index: offsets for O(1) random access
midx = np.load(idx_path, mmap_mode="r")

# Access: read only what you need
sample = mdata[midx[i]:midx[i+1]]
```

**Properties:**
- O(1) memory at load time (just mmap headers)
- O(sample_size) memory per access
- True random access without loading everything

We need similar properties for packed SFT data.

---

## Goals

1. **Streaming writes:** New formats should support writing without accumulating in memory
2. **Lazy reads:** New formats should load only the samples needed, not the entire dataset
3. **Cloud-native:** New formats should support direct read/write to S3, GCS, Azure
4. **Backward compatible:** Existing `.npy` files must continue to work unchanged
5. **Opt-in:** New formats are opt-in via configuration; existing pipelines unaffected
6. **Minimal training code changes:** Same `__getitem__` interface regardless of format

---

## New Format Options

The following formats are proposed as additional options alongside the existing legacy format.

**Which option is closest to the current format?** Parquet (Option B) is the most similar to the current pickle-based format:
- Both store variable-length lists without padding
- Both use a single file per shard
- Both preserve the same data model (`input_ids`, `loss_mask`, `seq_start_id` as lists)

The main difference is the serialization format (Parquet columnar vs Python pickle) and that Parquet adds compression. Migration from legacy to Parquet requires no changes to the logical data structure.

Memmap (Option A) is a larger departure: it pads sequences to fixed length and splits data across multiple files. This trades some disk space (~10%) for true O(1) random access without row group boundaries.

---

### Option A: Padded Fixed-Shape Memmap Arrays

Use `np.lib.format.open_memmap()` to write fixed-shape numpy arrays that can be memory-mapped for reading.

**Format:**
```
shard_000000/
├── input_ids.npy      # int32[num_bins, pack_size]  - padded
├── loss_mask.npy      # uint8[num_bins, pack_size]  - padded
├── packed_len.npy     # uint32[num_bins]            - actual lengths
├── seq_offsets.npy    # uint32[num_bins + 1]        - CSR pointers
├── seq_starts.npy     # uint32[total_seq_starts]    - boundary values
└── manifest.json
```

**Pros:**
- True O(1) random access via memmap
- Simplest implementation (direct numpy)
- Zero external dependencies

**Cons:**
- ~10% padding waste
- Multiple files per shard
- Local filesystem required for writes (cloud needs temp + upload)

**Implementation:** [packed-sft-impl-memmap.md](./packed-sft-impl-memmap.md)

---

### Option B: Parquet (Recommended)

Use Apache Parquet with PyArrow for columnar storage with native variable-length support.

**Format:**
```
shard_000000.parquet
  Schema:
    - input_ids: list<int32>
    - loss_mask: list<uint8>
    - seq_start_id: list<int32>
  Compression: zstd
  Row groups: ~1000 rows
```

**Pros:**
- Industry standard with excellent tooling
- Default format for Hugging Face datasets
- Native variable-length lists (no padding)
- 2-3x compression with zstd
- Direct cloud storage support (S3, GCS, Azure)
- Single file per shard
- Queryable with DuckDB, Polars, pandas

**Cons:**
- Adds PyArrow dependency (already used in pipeline)
- Row-group granularity for access (configurable, typically fine)

**Implementation:**
- Writer (Nemotron): [packed-sft-impl-parquet-nemotron.md](./packed-sft-impl-parquet-nemotron.md)
- Reader (Megatron-Bridge): [packed-sft-impl-parquet-megatron-bridge.md](./packed-sft-impl-parquet-megatron-bridge.md)

---

### Option C: Megatron bin/idx Style

Mimic the proven Megatron pretrain format with a flat binary data file and separate index.

**Current Megatron pretrain format:**
```
dataset.bin    # Concatenated tokens as raw bytes (uint16/uint32)
dataset.idx    # Header + document offsets + sequence lengths
```

The pretrain format stores only `input_ids` with document boundaries. For packed SFT, we would need to extend this to also store `loss_mask` and `seq_start_id`.

**Proposed extended format:**
```
shard_000000.bin         # Concatenated input_ids (int32)
shard_000000.loss.bin    # Concatenated loss_mask (uint8)  [NEW]
shard_000000.idx         # Bin offsets + lengths
shard_000000.seq.idx     # Sequence boundary offsets (CSR)  [NEW]
shard_000000.seq.bin     # Sequence start positions  [NEW]
```

**What needs to be built:**
1. **Extended index format** — Add fields for `loss_mask` offsets and `seq_start_id` CSR pointers
2. **Loss mask storage** — Separate `.loss.bin` file or interleaved with tokens
3. **Sequence boundary storage** — CSR-style index similar to Option A's `seq_offsets.npy` + `seq_starts.npy`
4. **Writer changes** — Extend `IndexedDatasetBuilder` to write additional arrays
5. **Reader changes** — Extend `MMapIndexedDataset` to read loss mask and boundaries

**Pros:**
- Proven memory-mapping approach in Megatron ecosystem
- Consistent with existing pretrain data loading patterns
- True O(1) random access

**Cons:**
- Requires extending Megatron's indexed dataset format (non-trivial)
- 5 files per shard (more than Parquet's 1, similar to Option A's 6)
- No compression (larger on disk than Parquet)
- Less ecosystem tooling compared to Parquet

**Recommendation:** Consider this option if strict consistency with Megatron pretrain tooling is required. Otherwise, Parquet (Option B) provides similar benefits with less implementation effort and better compression.

---

## Comparison

| Aspect | Legacy (.npy pickle) | Memmap (Option A) | Parquet (Option B) |
|--------|----------------------|-------------------|---------------------|
| Write memory (Python heap) | O(dataset) | O(pack_size)† | O(row_group) |
| Read memory at load | O(dataset) | O(metadata) | O(metadata) |
| Read memory per sample | O(1)* | O(pack_size) | O(row_group) |
| Random access | O(1)* | O(1) true | O(row_group) |
| Disk size | 1x | 1.1x (padding) | 0.4x (compressed) |
| Files per shard | 1 | 6 | 1 |
| Cloud support | None | Local only | Native (local) / Ranged reads (cloud) |
| Variable-length | Via pickle | Padding | Native |
| Tooling | None | numpy | DuckDB, Polars |
| Status | Supported (default) | New option | New option |

*After full dataset load
†OS page cache may grow with output size during writes; Python heap stays small

---

## When to Use Each Format

### Legacy (.npy pickle) — Keep using when:
- Existing pipelines that work well at current scale
- Small to medium datasets (< 50k packed samples, ~4 GB memory budget)
- No need to change what already works

### Parquet (Option B) — Use when:
- Large datasets where memory is a constraint (100k+ samples, millions for production SFT)
- Cloud storage (S3, GCS, Azure) is involved
- You want compression to reduce disk/network I/O
- You need to inspect data with standard tools (DuckDB, pandas)

**Parquet is the recommended new format** because:
1. **Already a dependency** — PyArrow is used for reading input data
2. **Ecosystem standard** — Default format for Hugging Face datasets; widely adopted for ML data
3. **Single file** — Simpler than multi-file memmap directory
4. **Compression** — 2-3x smaller on disk reduces I/O
5. **Cloud-native** — Direct S3/GCS support without temp files
6. **No padding waste** — Native variable-length arrays

### Memmap (Option A) — Use when:
- Maximum raw access speed is critical
- You want zero dependencies beyond numpy
- All storage is local POSIX filesystem
- You need true O(1) random access without row group boundaries

---

## Megatron-Bridge Changes Required

### New Dataset Classes

```python
# For Parquet format
class GPTSFTPackedParquetDataset(GPTSFTDataset):
    def _load_dataset(self):
        self._pf = pq.ParquetFile(self.path, memory_map=True)
        # No data loaded yet

    def __getitem__(self, idx):
        # Read only the row group containing idx
        rg = self._pf.read_row_group(self._get_rg(idx))
        return rg[idx % rg_size]

# For Memmap format
class GPTSFTPackedMemmapDataset(GPTSFTDataset):
    def _load_dataset(self):
        self.input_ids = np.load(..., mmap_mode='r')
        # No data loaded yet

    def __getitem__(self, idx):
        L = self.packed_len[idx]
        return self.input_ids[idx, :L]
```

### Factory Function Update

```python
def create_sft_dataset(path: Path, tokenizer, ...) -> GPTSFTDataset:
    if path.suffix == ".npy":
        return GPTSFTPackedDataset(...)        # Legacy
    elif path.suffix == ".parquet":
        return GPTSFTPackedParquetDataset(...)  # New
    elif path.is_dir() and (path / "manifest.json").exists():
        return GPTSFTPackedMemmapDataset(...)   # New
```

### Backward Compatibility

- Existing `.npy` files continue to work unchanged
- Format detected automatically by file extension/structure
- Training code uses same `__getitem__` interface

---

## Format Invariants

All formats must maintain these invariants for each packed sample. Implementations should assert these during writes and verify during tests:

```python
# For each packed bin:
assert 0 < packed_len <= pack_size
assert len(input_ids) == packed_len
assert len(loss_mask) == packed_len

# seq_start_id contains START positions of each sequence in the pack
# First sequence always starts at 0
assert seq_start_id[0] == 0
assert all(seq_start_id[i] < seq_start_id[i+1] for i in range(len(seq_start_id)-1))
assert seq_start_id[-1] < packed_len  # Last start is before end

# To reconstruct sequence boundaries for attention masking:
# boundaries = list(seq_start_id) + [packed_len]
# seq_i spans input_ids[boundaries[i]:boundaries[i+1]]
```

**Key invariant:** `seq_start_id` stores **start positions** `[0, start_1, start_2, ...]`, not end positions. The final boundary (`packed_len`) is reconstructed at read time.

---

## DataLoader Multi-Worker Considerations

PyTorch DataLoader with `num_workers > 0` spawns worker processes. File handles and memory-mapped arrays do not survive pickling across process boundaries.

**Requirements for new dataset classes:**

1. **Lazy open pattern:** Store paths in `__init__`, open files in a separate `_load_dataset()` method
2. **Per-worker initialization:** Call `_load_dataset()` inside each worker, not in the main process
3. **Use `worker_init_fn`:** Reopen mmaps/parquet files per worker

```python
class GPTSFTPackedParquetDataset(GPTSFTDataset):
    def __init__(self, path: Path, ...):
        self.path = path
        self._pf = None  # Opened lazily

    def _ensure_loaded(self):
        if self._pf is None:
            self._pf = pq.ParquetFile(self.path, memory_map=True)

    def __getitem__(self, idx):
        self._ensure_loaded()  # Opens on first access in each worker
        ...
```

**Testing requirement:** Verify dataset works with `num_workers=4` and `persistent_workers=True`.

---

## Cloud Storage Behavior

The new formats behave differently for local vs cloud storage:

### Parquet
- **Local:** Uses true memory-mapped reads via `memory_map=True`
- **Cloud (S3/GCS):** Uses ranged HTTP reads; `memory_map=True` is ignored. PyArrow's filesystem layer handles buffering. Row group size affects read efficiency—smaller groups = more requests but finer granularity.

### Memmap
- **Local:** True O(1) random access via OS virtual memory
- **Cloud:** Not directly supported. Requires download-to-local or write-local-then-upload pattern. For cloud outputs, write to local temp directory, then upload atomically.

**Recommendation:** Use Parquet for cloud workloads; use Memmap only for local high-performance scenarios.

---

## Migration Path

### Phase 1: Add New Formats

1. Implement Parquet writer in Nemotron data_prep
2. Implement Parquet reader in Megatron-Bridge
3. Add `packed_storage` config option (default: `legacy_npy_pickle`)
4. Existing pipelines continue to work without changes

### Phase 2: Validation

1. Run training with new format, verify identical loss curves
2. Benchmark memory usage and throughput
3. Test cloud storage integration

### Phase 3: Adoption

1. Document when to use new formats vs legacy
2. Provide conversion tool for users who want to migrate existing datasets
3. Legacy format remains fully supported for backward compatibility

---

## Memory Impact (New Formats)

For workloads that benefit from the new formats:

| Metric | Legacy | Parquet | Memmap |
|--------|--------|---------|--------|
| Write peak (Python heap, 50k bins) | ~4 GB | ~20 MB | ~16 KB† |
| Read at load (RSS) | ~4 GB | ~few MB (metadata) | ~few KB (metadata) |
| Read per batch (bs=8) | ~4 GB* | ~8 MB (row group) | ~128 KB |

*Already loaded
†OS page cache grows with output size but Python heap stays minimal

**Note on memory estimates:** These are back-of-envelope calculations based on Python object overhead (~28 bytes per int vs 4 bytes in numpy). Actual memory depends on CPython version, allocator behavior, and workload. The relative improvements (200-500x) are consistent across configurations.

**New formats provide ~200x reduction in write memory and ~500x reduction in read memory at load**, enabling larger datasets and more efficient multi-node training.

---

## Open Questions

1. **Row group size tuning:** What's the optimal row group size for training access patterns? (Proposed: 1000 rows, ~2-10 MB per group)

2. **Multi-shard handling:** Should we use a single Parquet dataset with partitions or multiple files with a wrapper dataset?

3. **Conversion tooling:** Should we provide a conversion tool for users who want to migrate existing datasets to new formats?

4. **Default format:** Should the default remain `legacy_npy_pickle` indefinitely, or switch to Parquet after validation?

---

## References

### Implementation Plans
- [Parquet Writer (Nemotron)](./packed-sft-impl-parquet-nemotron.md)
- [Parquet Reader (Megatron-Bridge)](./packed-sft-impl-parquet-megatron-bridge.md)
- [Memmap Format](./packed-sft-impl-memmap.md)

### Existing Code
- [Megatron-Bridge GPTSFTPackedDataset](../../../Megatron-Bridge/src/megatron/bridge/data/datasets/sft.py)
- [Nemotron chat_sft_shard_core.py](../src/nemotron/data_prep/chat_sft_shard_core.py)

### External Documentation
- [Apache Parquet Format](https://parquet.apache.org/docs/file-format/)
- [NumPy Memory-Mapped Files](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)
- [Hugging Face Datasets (Parquet backend)](https://huggingface.co/docs/datasets/about_arrow)
