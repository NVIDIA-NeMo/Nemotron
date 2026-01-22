# Implementation Plan: Parquet Reader (Megatron-Bridge)

This document details the Megatron-Bridge changes for Option B (Parquet) from the [Packed SFT Format Design](./packed-sft-format-design.md).

For the Nemotron writer implementation, see [packed-sft-impl-parquet-nemotron.md](./packed-sft-impl-parquet-nemotron.md).

## Format Specification

```
shard_000000.parquet
  Schema:
    - input_ids: list<int32>      # Variable-length token ids
    - loss_mask: list<uint8>      # Variable-length loss mask
    - seq_start_id: list<int32>   # Variable-length sequence start positions

  Compression: zstd (default)
  Row groups: ~1000 rows each
```

**Invariant:** `seq_start_id` contains START positions `[0, start_1, start_2, ...]`. To reconstruct boundaries for attention masking: `boundaries = list(seq_start_id) + [len(input_ids)]`.

---

## Files to Modify

| File | Change |
|------|--------|
| `src/megatron/bridge/data/datasets/sft.py` | Add `GPTSFTPackedParquetDataset` class (after line 1003) |
| `src/megatron/bridge/data/datasets/sft.py` | Update `create_sft_dataset()` factory (lines 173-179) |
| `tests/functional_tests/data/datasets/test_sft.py` | Add `TestDataGPTSFTPackedParquetDataset` class |

---

## Existing Class Structure

The new class must follow the existing `GPTSFTPackedDataset` pattern (lines 738-1003 in `sft.py`):

```
GPTSFTDataset (base class, lines 194-736)
├── __init__(file_path, tokenizer, **kwargs)
├── __len__()
├── __getitem__(idx) → dict
├── _load_dataset()
├── _build_samples_mapping()
├── _build_loss_mask(processed_example)
├── collate_fn(batch) → dict of tensors
└── _collate_item(item, max_length, pad_id)

GPTSFTPackedDataset (current .npy reader, lines 738-1003)
├── Inherits all base methods
├── Overrides _load_dataset() to use np.load(..., allow_pickle=True)
├── Overrides __getitem__() to return {input_ids, seq_boundaries, loss_mask}
├── Overrides collate_fn() for packed sequence handling with cu_seqlens
└── Adds return_cu_seqlen, pad_cu_seqlens parameters
```

---

## Step 1: Add Parquet Dataset Class

Add after `GPTSFTPackedDataset` class (after line 1003) in `sft.py`:

```python
import pyarrow.parquet as pq


class GPTSFTPackedParquetDataset(GPTSFTPackedDataset):
    """Memory-efficient packed dataset using Parquet format.

    Identical interface to GPTSFTPackedDataset but reads from .parquet files
    instead of .npy files. Uses lazy loading with row group caching for
    memory efficiency.

    Memory usage: O(metadata) at load, O(row_group_size * pack_size) during access.

    DataLoader compatibility: Uses lazy-open pattern. ParquetFile is opened on
    first access in each worker process, avoiding pickling issues with
    num_workers > 0.

    Cloud behavior: For local files, uses true memory-mapping. For cloud URIs
    (s3://, gs://), falls back to ranged HTTP reads.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: MegatronTokenizer,
        return_cu_seqlen: bool = True,
        pad_cu_seqlens: bool = False,
        pack_metadata_file_path: str | None = None,
        **kwargs,
    ):
        # Store path for lazy loading
        self._parquet_path = file_path
        self._pq_loaded = False

        # Call parent init (will call _load_dataset)
        super().__init__(
            file_path=file_path,
            tokenizer=tokenizer,
            return_cu_seqlen=return_cu_seqlen,
            pad_cu_seqlens=pad_cu_seqlens,
            pack_metadata_file_path=pack_metadata_file_path,
            **kwargs,
        )

    def _load_dataset(self):
        """Override: Load only metadata, defer full load to first access.

        This enables pickling across DataLoader workers - the ParquetFile
        handle is not picklable, so we open it lazily per-worker.
        """
        # Don't load yet - will be opened in _ensure_parquet_open()
        self._pq_loaded = False

        # Create a minimal indexed_dataset proxy for __len__ to work
        # We need to peek at the parquet metadata to get row count
        try:
            if MultiStorageClientFeature.is_enabled():
                # For cloud storage, we still need to read metadata
                msc = MultiStorageClientFeature.import_package()
                pf = pq.ParquetFile(self._parquet_path, filesystem=msc.get_filesystem(self._parquet_path))
            else:
                pf = pq.ParquetFile(self._parquet_path)

            self._num_rows = pf.metadata.num_rows
            self._num_row_groups = pf.metadata.num_row_groups

            # Build row group offset index
            self._row_group_offsets = [0]
            for i in range(self._num_row_groups):
                rg_rows = pf.metadata.row_group(i).num_rows
                self._row_group_offsets.append(self._row_group_offsets[-1] + rg_rows)
            self._row_group_offsets = np.array(self._row_group_offsets)

            # Close the file - will reopen lazily
            pf = None

        except Exception as e:
            logger.error(
                f"Failed to load packed Parquet dataset. The dataset should be a `.parquet` file. "
                f"Please check if the packed dataset was prepared correctly. The original error was:\n {e}",
            )
            exit(1)

        # Create a proxy object that supports len() for _build_samples_mapping
        class _ParquetProxy:
            def __init__(self, num_rows):
                self._num_rows = num_rows
            def __len__(self):
                return self._num_rows

        self.indexed_dataset = _ParquetProxy(self._num_rows)

    def _ensure_parquet_open(self):
        """Open ParquetFile on first access (per-worker)."""
        if self._pq_loaded:
            return

        if MultiStorageClientFeature.is_enabled():
            msc = MultiStorageClientFeature.import_package()
            self._parquet_file = pq.ParquetFile(
                self._parquet_path,
                filesystem=msc.get_filesystem(self._parquet_path),
                memory_map=True,
            )
        else:
            self._parquet_file = pq.ParquetFile(
                self._parquet_path,
                memory_map=True,  # Only effective for local files
            )

        # Row group cache
        self._cached_rg_id = -1
        self._cached_data = None

        self._pq_loaded = True

    def _get_row_group_for_idx(self, idx: int) -> int:
        """Find which row group contains the given index."""
        return int(np.searchsorted(self._row_group_offsets[1:], idx, side='right'))

    def __len__(self):
        return self._num_rows

    def __getitem__(self, idx):
        """Read a single packed sample.

        Returns same format as GPTSFTPackedDataset:
            {
                "input_ids": list[int],
                "seq_boundaries": list[int],  # seq_start_id + [len(input_ids)]
                "loss_mask": list[int],
            }
        """
        self._ensure_parquet_open()  # Lazy open per worker

        if self.samples_mapping is not None:
            idx = self.samples_mapping[idx]

        # Find row group and local index within row group
        rg_id = self._get_row_group_for_idx(idx)
        local_idx = idx - self._row_group_offsets[rg_id]

        # Load row group if not cached
        if rg_id != self._cached_rg_id:
            table = self._parquet_file.read_row_group(
                rg_id,
                columns=['input_ids', 'loss_mask', 'seq_start_id'],
            )
            # Convert to Python lists once per row group
            self._cached_data = {
                'input_ids': table['input_ids'].to_pylist(),
                'loss_mask': table['loss_mask'].to_pylist(),
                'seq_start_id': table['seq_start_id'].to_pylist(),
            }
            self._cached_rg_id = rg_id

        input_ids = self._cached_data['input_ids'][local_idx]
        loss_mask = self._cached_data['loss_mask'][local_idx]
        seq_start_id = self._cached_data['seq_start_id'][local_idx]

        # Reconstruct seq_boundaries from seq_start_id (same as GPTSFTPackedDataset)
        seq_boundaries = seq_start_id + [len(input_ids)]

        if idx < 0:
            loss_mask = [0] * len(loss_mask)

        return {
            "input_ids": input_ids,
            "seq_boundaries": seq_boundaries,
            "loss_mask": loss_mask,
        }

    def __getstate__(self):
        """For pickling across DataLoader workers - exclude file handle."""
        state = self.__dict__.copy()
        state['_pq_loaded'] = False
        for key in ['_parquet_file', '_cached_rg_id', '_cached_data']:
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        """Restore from pickle - file will be reopened on first access."""
        self.__dict__.update(state)

    # Note: collate_fn is inherited from GPTSFTPackedDataset - no changes needed
    # since __getitem__ returns the same format
```

---

## Step 2: Update Factory Function

Modify `create_sft_dataset()` (around line 173) to detect `.parquet` files:

```python
def create_sft_dataset(
    path: Path,
    tokenizer: "MegatronTokenizer",
    # ... existing parameters ...
) -> "GPTSFTDataset":
    # ... existing docstring and gpt_sft_dataset_kwargs setup ...

    if path.suffix == ".npy":
        return GPTSFTPackedDataset(
            pack_metadata_file_path=pack_metadata_file_path,
            pad_cu_seqlens=pad_cu_seqlens,
            **gpt_sft_dataset_kwargs,
            **kwargs,
        )
    elif path.suffix == ".parquet":
        # NEW: Parquet packed format
        return GPTSFTPackedParquetDataset(
            pack_metadata_file_path=pack_metadata_file_path,
            pad_cu_seqlens=pad_cu_seqlens,
            **gpt_sft_dataset_kwargs,
            **kwargs,
        )
    elif chat:
        return GPTSFTChatDataset(
            # ... existing code ...
        )
    else:
        return GPTSFTDataset(
            # ... existing code ...
        )
```

---

## Step 3: Add Import

Add PyArrow import at top of `sft.py` (around line 20):

```python
try:
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    pq = None
```

And add a check in `GPTSFTPackedParquetDataset.__init__`:

```python
if not PYARROW_AVAILABLE:
    raise ImportError(
        "PyArrow is required for Parquet dataset support. "
        "Install with: pip install pyarrow"
    )
```

---

## Step 4: Multi-Shard Support (Optional)

For datasets split across multiple parquet files:

```python
class GPTSFTPackedParquetMultiShardDataset(Dataset):
    """Combines multiple Parquet shards into one dataset.

    Usage:
        paths = sorted(glob.glob("data/shard_*.parquet"))
        dataset = GPTSFTPackedParquetMultiShardDataset(paths, tokenizer, **kwargs)
    """

    def __init__(
        self,
        parquet_paths: list[str],
        tokenizer: MegatronTokenizer,
        **kwargs,
    ):
        self.shards = [
            GPTSFTPackedParquetDataset(p, tokenizer, **kwargs)
            for p in parquet_paths
        ]

        # Build cumulative index for O(log n) shard lookup
        self.shard_offsets = np.cumsum([0] + [len(s) for s in self.shards])

        # Store collate_fn from first shard
        self.collate_fn = self.shards[0].collate_fn

    def __len__(self):
        return int(self.shard_offsets[-1])

    def __getitem__(self, idx):
        shard_id = int(np.searchsorted(self.shard_offsets[1:], idx, side='right'))
        local_idx = idx - int(self.shard_offsets[shard_id])
        return self.shards[shard_id][local_idx]
```

---

## Testing

Add tests to `tests/functional_tests/data/datasets/test_sft.py`:

```python
class TestDataGPTSFTPackedParquetDataset:
    """Tests for GPTSFTPackedParquetDataset."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown_parallel_state(self):
        """Initialize distributed state for tests."""
        if not dist.is_initialized():
            dist.init_process_group("gloo", rank=0, world_size=1)
        parallel_state.initialize_model_parallel()
        yield
        parallel_state.destroy_model_parallel()

    def test_parquet_dataset_basic(self, tmp_path, get_tokenizer):
        """Test basic read functionality."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Create test parquet file
        parquet_path = tmp_path / "test.parquet"
        schema = pa.schema([
            ('input_ids', pa.list_(pa.int32())),
            ('loss_mask', pa.list_(pa.uint8())),
            ('seq_start_id', pa.list_(pa.int32())),
        ])

        table = pa.Table.from_pydict({
            'input_ids': [[1, 2, 3, 4, 5], [10, 20, 30, 40]],
            'loss_mask': [[0, 0, 1, 1, 1], [0, 1, 1, 1]],
            'seq_start_id': [[0, 2], [0]],
        }, schema=schema)

        pq.write_table(table, str(parquet_path), compression='zstd')

        # Load dataset
        tokenizer = get_tokenizer()
        dataset = GPTSFTPackedParquetDataset(
            file_path=str(parquet_path),
            tokenizer=tokenizer,
            max_seq_length=2048,
        )

        assert len(dataset) == 2

        sample = dataset[0]
        assert sample['input_ids'] == [1, 2, 3, 4, 5]
        assert sample['seq_boundaries'] == [0, 2, 5]
        assert sample['loss_mask'] == [0, 0, 1, 1, 1]

    def test_parquet_dataset_collate_fn(self, tmp_path, get_tokenizer):
        """Test that collate_fn works correctly (inherited from parent)."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet_path = tmp_path / "test.parquet"
        schema = pa.schema([
            ('input_ids', pa.list_(pa.int32())),
            ('loss_mask', pa.list_(pa.uint8())),
            ('seq_start_id', pa.list_(pa.int32())),
        ])

        # Create batch of samples
        table = pa.Table.from_pydict({
            'input_ids': [[1, 2, 3, 4], [5, 6, 7, 8]],
            'loss_mask': [[0, 1, 1, 1], [0, 0, 1, 1]],
            'seq_start_id': [[0], [0]],
        }, schema=schema)

        pq.write_table(table, str(parquet_path))

        tokenizer = get_tokenizer()
        dataset = GPTSFTPackedParquetDataset(
            file_path=str(parquet_path),
            tokenizer=tokenizer,
            max_seq_length=2048,
        )

        batch = [dataset[0], dataset[1]]
        collated = dataset.collate_fn(batch)

        assert 'tokens' in collated
        assert 'labels' in collated
        assert 'loss_mask' in collated
        assert 'position_ids' in collated

    def test_parquet_dataset_with_samples_mapping(self, tmp_path, get_tokenizer):
        """Test that max_num_samples and shuffling work."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet_path = tmp_path / "test.parquet"
        schema = pa.schema([
            ('input_ids', pa.list_(pa.int32())),
            ('loss_mask', pa.list_(pa.uint8())),
            ('seq_start_id', pa.list_(pa.int32())),
        ])

        # Create 10 samples
        rows = [{'input_ids': [i], 'loss_mask': [1], 'seq_start_id': [0]} for i in range(10)]
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, str(parquet_path))

        tokenizer = get_tokenizer()
        dataset = GPTSFTPackedParquetDataset(
            file_path=str(parquet_path),
            tokenizer=tokenizer,
            max_seq_length=2048,
            max_num_samples=5,  # Limit to 5 samples
            seed=42,
        )

        assert len(dataset) == 5
```

---

## Cloud Storage Support

The implementation automatically supports cloud storage through `MultiStorageClientFeature` (existing Megatron-Bridge pattern):

```python
# Local file
dataset = GPTSFTPackedParquetDataset("data/shard.parquet", tokenizer)

# S3 (if MultiStorageClient is configured)
dataset = GPTSFTPackedParquetDataset("s3://bucket/data/shard.parquet", tokenizer)

# GCS
dataset = GPTSFTPackedParquetDataset("gs://bucket/data/shard.parquet", tokenizer)
```

---

## Memory Characteristics

| Metric | Value |
|--------|-------|
| Load time memory | O(metadata) - parquet footer + row group index |
| Per-sample access | O(row_group) - cached row group converted to Python lists |
| Row group cache | 1 row group per dataset instance per worker |

**Performance notes:**
- Row group cache uses `to_pylist()` which reintroduces Python object overhead
- For training with sequential access, this is efficient (cache hit rate ~99%)
- For random access patterns, consider smaller row groups (100-500 rows)

---

## Debugging & Inspection

```python
# Quick inspection with PyArrow
import pyarrow.parquet as pq

pf = pq.ParquetFile('shard.parquet')
print(f"Rows: {pf.metadata.num_rows}")
print(f"Row groups: {pf.metadata.num_row_groups}")
print(f"Schema: {pf.schema_arrow}")

# Read first row group
df = pf.read_row_group(0).to_pandas()
print(df.head())
```

```bash
# Query with DuckDB
duckdb -c "SELECT COUNT(*) FROM 'shard.parquet'"
duckdb -c "SELECT * FROM 'shard.parquet' LIMIT 5"
```
