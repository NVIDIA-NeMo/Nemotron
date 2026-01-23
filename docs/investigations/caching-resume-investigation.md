# Investigation: Data Prep Caching/Resume Behavior

## Summary
The data prep pipeline has a solid caching mechanism based on filesystem receipts, but there's a **critical bug** where the tokenization stage's completion check doesn't verify output file existence, which can cause shards to be incorrectly skipped when outputs are missing.

## Symptoms/Requirements
- Running `uv run nemotron nano3 data prep pretrain --run prep-cpu` twice should resume from where the first run left off
- Critical for SLURM environments with 4-hour job limits
- Need to ensure no work is duplicated and no work is lost

## Investigation Log

### Phase 1 - Architecture Understanding

**Hypothesis:** The caching mechanism is based on filesystem receipts and run hashes.

**Findings:** The caching system has three key components:

1. **`run_hash`** (determines run directory `runs/<run_hash>/`)
   - Computed in `src/nemotron/data_prep/recipes/pretrain.py:_setup_run()` (line ~135)
   - Hash of: datasets config, resolved tokenizer, output settings, sample spec
   - `force=True` appends timestamp to disable resume

2. **`plan_hash`** (determines dataset directory `datasets/<name>/<plan_hash>/`)
   - Computed in `src/nemotron/data_prep/planning.py:create_shard_plan()` (line ~225)
   - Hash of: dataset identity, num_shards, source fingerprint, resolved tokenizer, determinism constraints, file paths

3. **Receipts** (track shard completion)
   - Written by `BinIdxTokenizationStage` in `src/nemotron/data_prep/stages/megatron_bin_idx.py`
   - Location: `<run_dir>/datasets/<dataset>/<plan_hash>/receipts/shard_NNNNNN.json`
   - States: `started`, `completed`, `failed`

**Evidence:** 
- `planning.py:265-290` - `get_pending_shards()` checks receipts and verifies output files
- `megatron_bin_idx.py:145-155` - `_is_completed()` only checks receipt status, NOT output files

**Conclusion:** Architecture is sound, but there's an inconsistency in completion checking.

---

### Phase 2 - Resume Logic Analysis

**Hypothesis:** Shards are correctly skipped when already completed.

**Findings:** There are TWO places where completion is checked:

#### Layer 1: PlanStage (fan-out filtering)
Location: `src/nemotron/data_prep/stages/plan.py` → calls `get_pending_shards()`

`get_pending_shards()` in `planning.py:265-290` marks a shard as complete IFF:
1. Receipt exists and parses
2. `receipt["plan_hash"] == plan.plan_hash`
3. `receipt["status"] == "completed"`
4. **AND** if `num_sequences > 0`, verifies `.bin` and `.idx` files exist

#### Layer 2: BinIdxTokenizationStage (per-shard check)
Location: `megatron_bin_idx.py:145-155`

```python
def _is_completed(self, receipt_path: str, plan_hash: str) -> bool:
    if not self._fs.exists(receipt_path):
        return False
    try:
        r = read_json(self._fs, receipt_path)
        return r.get("status") == "completed" and r.get("plan_hash") == plan_hash
    except Exception:
        return False
```

**BUG IDENTIFIED:** `_is_completed()` does NOT verify output files exist!

**Scenario where this causes data loss:**
1. Run 1: Shard 42 tokenizes, writes `.bin/.idx` and `completed` receipt
2. Something deletes the `.bin/.idx` files (disk issue, manual cleanup, etc.)
3. Run 2: `get_pending_shards()` correctly marks shard 42 as pending (files missing)
4. BUT: `BinIdxTokenizationStage._is_completed()` sees `completed` receipt and SKIPS
5. Result: Shard 42 never gets regenerated!

**Evidence:** 
- `planning.py:280-286` - verifies files exist
- `megatron_bin_idx.py:145-155` - does NOT verify files

**Conclusion:** CONFIRMED BUG - tokenization stage can incorrectly skip shards with missing outputs.

---

### Phase 3 - Stability Concerns

**Hypothesis:** Hash stability may cause unexpected cache invalidation.

**Findings:**

#### Tokenizer Resolution Instability
In `planning.py:107-133` (`resolve_tokenizer`):
- Calls `HfApi().model_info()` to get SHA
- If API call fails, falls back to `config.revision` (often `None`)
- Different network conditions → different `resolved_revision` → different hashes

Additionally in `recipes/pretrain.py:141-148`:
```python
tokenizer_cfg = InternalTokenizerConfig(
    type=tok_cfg.type,
    model=tok_cfg.model,
    revision=None,  # ← HARDCODED TO NONE!
    ...
)
```
The user's specified revision is ignored!

**Evidence:**
- `recipes/pretrain.py:144` - `revision=None` is hardcoded
- `planning.py:118-127` - network-dependent resolution

**Conclusion:** Hash stability depends on network conditions and ignores user-specified revision.

---

## Root Cause

**Primary Bug:** `BinIdxTokenizationStage._is_completed()` doesn't verify output files exist, causing it to skip shards that `get_pending_shards()` correctly identified as needing reprocessing.

**Secondary Issue:** Tokenizer revision handling can cause unexpected hash changes across runs.

## Recommendations

### Fix 1: Align completion checks (CRITICAL)

In `src/nemotron/data_prep/stages/megatron_bin_idx.py`, update `_is_completed()`:

```python
def _is_completed(self, receipt_path: str, plan_hash: str, shard_dir: str) -> bool:
    """Check if shard is already completed with valid outputs."""
    if not self._fs.exists(receipt_path):
        return False

    try:
        r = read_json(self._fs, receipt_path)
        if r.get("status") != "completed" or r.get("plan_hash") != plan_hash:
            return False
        
        # Verify output files exist for non-empty shards
        stats = r.get("stats", {}) or {}
        if int(stats.get("num_sequences", 0) or 0) > 0:
            files = r.get("files", {}) or {}
            bin_info = files.get("bin", {}) or {}
            idx_info = files.get("idx", {}) or {}
            bin_path = bin_info.get("path", "")
            idx_path = idx_info.get("path", "")
            
            if not bin_path or not idx_path:
                return False
            
            full_bin = f"{shard_dir}/{bin_path}"
            full_idx = f"{shard_dir}/{idx_path}"
            if not (self._fs.exists(full_bin) and self._fs.exists(full_idx)):
                return False
        
        return True
    except Exception:
        return False  # Corrupted receipt, reprocess
```

Update `_process_shard()` to pass `shard_dir`:
```python
def _process_shard(self, task: ShardWorkItem) -> None:
    receipt_path = f"{task.receipts_dir.rstrip('/')}/shard_{task.shard_index:06d}.json"
    shard_dir = str(Path(task.receipts_dir).parent)  # Add this
    
    if self._is_completed(receipt_path, task.plan_hash, shard_dir):  # Pass shard_dir
        return
    # ... rest unchanged
```

### Fix 2: Respect tokenizer revision (RECOMMENDED)

In `src/nemotron/data_prep/recipes/pretrain.py:_setup_run()`:

```python
tokenizer_cfg = InternalTokenizerConfig(
    type=tok_cfg.type,
    model=tok_cfg.model,
    revision=tok_cfg.revision,  # Was: None
    add_eos=tok_cfg.add_eos,
    add_bos=tok_cfg.add_bos,
    trust_remote_code=tok_cfg.trust_remote_code,
)
```

### Fix 3: Document SLURM requirements (IMPORTANT)

Users need to know:
1. `output_dir` must be on persistent shared storage (same path across jobs)
2. Don't use `force=true` if you want resume
3. Keep config stable across job submissions
4. Set `HF_HOME` to shared cache if workers lack internet

## Preventive Measures

1. **Add integration test** that:
   - Runs pipeline, deletes some output files, runs again
   - Verifies deleted files are regenerated

2. **Add logging** when shards are skipped due to completion:
   ```python
   logger.info(f"Skipping completed shard {task.shard_index} (receipts valid, outputs exist)")
   ```

3. **Consider receipt versioning** - add a version field to detect schema changes

## Files Changed (IMPLEMENTED)

All fixes have been implemented and tested:

1. **`src/nemotron/data_prep/stages/megatron_bin_idx.py`** - Fixed `_is_completed()` to verify output files exist
2. **`src/nemotron/data_prep/recipes/pretrain.py`** - Respects tokenizer revision from config
3. **`src/nemotron/data_prep/recipes/sft.py`** - Respects tokenizer revision from config
4. **`src/nemotron/data_prep/config.py`** - Added `revision` field to public `TokenizerConfig`
5. **`tests/data_prep/test_binidx_stage_resume.py`** - Added 11 tests for resume behavior

## Test Results

All 11 tests pass:
```
tests/data_prep/test_binidx_stage_resume.py::TestIsCompleted::test_no_receipt_returns_false PASSED
tests/data_prep/test_binidx_stage_resume.py::TestIsCompleted::test_completed_receipt_with_files_returns_true PASSED
tests/data_prep/test_binidx_stage_resume.py::TestIsCompleted::test_completed_receipt_missing_bin_file_returns_false PASSED
tests/data_prep/test_binidx_stage_resume.py::TestIsCompleted::test_completed_receipt_missing_idx_file_returns_false PASSED
tests/data_prep/test_binidx_stage_resume.py::TestIsCompleted::test_completed_empty_shard_no_files_needed PASSED
tests/data_prep/test_binidx_stage_resume.py::TestIsCompleted::test_wrong_plan_hash_returns_false PASSED
tests/data_prep/test_binidx_stage_resume.py::TestIsCompleted::test_started_status_returns_false PASSED
tests/data_prep/test_binidx_stage_resume.py::TestIsCompleted::test_failed_status_returns_false PASSED
tests/data_prep/test_binidx_stage_resume.py::TestIsCompleted::test_corrupted_receipt_returns_false PASSED
tests/data_prep/test_binidx_stage_resume.py::TestIsCompleted::test_missing_files_dict_returns_false PASSED
tests/data_prep/test_binidx_stage_resume.py::TestGetPendingShardsConsistency::test_missing_output_files_both_detect PASSED
```
