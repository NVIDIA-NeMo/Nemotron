# Nemotron Data Prep Xenna Port

Status: Draft
Owner: Nemotron team
Last updated: TBD

## Summary

Port the Stage 0 (pretrain) data preparation execution engine from the current
Ray actor pool to Cosmos-Xenna pipelines. Keep the existing data prep planning,
shard formats, and output artifacts, while introducing a thin Xenna-based
execution layer that accepts a dict dataset spec (local paths or hf:// sources),
performs optimized HuggingFace pre-downloads, and emits Megatron-compatible
.bin/.idx shards plus blend.json.

## Goals

- Use Cosmos-Xenna pipeline execution for shard processing with minimal new
  abstraction.
- Introduce a dict-based dataset input for the pretrain entrypoint.
- Keep HuggingFace Hub as a first-class source with parallel pre-downloads.
- Preserve current outputs (bin/idx + blend.json) and receipts for resuming.
- Keep the existing planning model (ShardPlan, determinism, caching).

## Non-Goals

- Rework SFT/RL data prep formats (JSONL, packed, chat SFT) in this effort.
- Change the on-disk artifact layout or receipt schema.
- Replace fsspec or the current discovery/planning logic.
- Add new distributed download backends beyond HuggingFace and existing fsspec.

## Current State (Stage 0)

- `run_data_prep(DataPrepConfig)` loads a blend.json, builds a `PipelineConfig`,
  then calls `last_mile_process()` in `data_prep/pipeline.py`.
- `_process_split()` creates a ShardPlan per dataset, applies caching, and
  runs pending shards through a Ray actor pool.
- Shard execution uses `process_binidx_shard_core()` in
  `data_prep/shard_processor.py`, writing .bin/.idx plus receipt JSON.
- HuggingFace datasets use discovery + optional parallel predownload
  (`downloader.parallel_predownload()`).

## Proposed Design

### 1) Dict Dataset Input

Add a new entrypoint that accepts a dict structure describing datasets and
converts it to `DataBlend` in-memory (no required JSON file).

Proposed schema (aligned with `DataBlend`/`Dataset`):

```json
{
  "datasets": [
    {
      "name": "pile",
      "path": "hf://EleutherAI/pile",
      "weight": 1.0,
      "split": "train",
      "subset": null,
      "text_field": "text"
    }
  ]
}
```

Per-split mode remains supported:

```json
{
  "train": [ ... ],
  "valid": [ ... ],
  "test": [ ... ]
}
```

Notes:
- `path` supports local paths/globs, `s3://`, `gs://`, and `hf://`.
- `split` is required for `hf://` paths and optional otherwise.
- Missing fields use current defaults (weight=1.0, text_field="text").

This entrypoint will live alongside `run_data_prep()` to preserve backward
compatibility. Stage 0 pretrain will switch to the dict-based entrypoint.

### 2) Xenna Execution Layer (Thin Wrapper)

Introduce a new Xenna-based shard runner that reuses the existing planning and
processing logic:

- Planning: keep `create_shard_plan()` and `get_pending_shards()`.
- Execution: replace the manual Ray actor pool with a Xenna `Stage` that calls
  `process_binidx_shard_core()`.

New flow (per split):

1. Convert dict input -> `DataBlend`
2. Create/validate shard plans (existing logic)
3. Build a list of `ShardTask` items (one per shard index)
4. Optionally predownload HF files (see below)
5. Run `cosmos_xenna.pipelines.run_pipeline()` with one Stage
6. Aggregate receipts -> stats -> blend.json (existing logic)

### 3) Xenna Stage Definition (nemotron.data_prep.xenna)

All Xenna-facing wrappers live under `nemotron.data_prep.xenna` to keep the
integration boundary clean. This includes stage classes, work item types, and
the pipeline runner.

Define a single stage class (CPU only):

- `required_resources`: `Resources(cpus=1.0, gpus=0.0)` (configurable).
- `setup()`: initialize tokenizer using `create_tokenizer()` and store
  resolved config; set output fs once (fsspec `url_to_fs`).
- `process_data()`: accept one `ShardTask` dict, invoke
  `process_binidx_shard_core()` with the cached tokenizer and output fs.

`ShardTask` fields (minimal):

```
{
  "dataset_name": str,
  "shard_index": int,
  "assignment_json": str,
  "plan_hash": str,
  "output_dir": str,
  "receipts_dir": str,
  "text_field": str,
  "dtype": "int32|int64|uint16",
  "min_doc_chars": int|null,
  "max_doc_tokens": int|null,
  "max_rows": int|null
}
```

The stage is intentionally thin: no new shard logic or I/O semantics, just a
bridge from Xenna tasks to existing core processing.

### 3.1) Proposed Xenna Stages (Nemotron Data Prep)

This effort defines a minimal set of Xenna stages. Stage 0 (pretrain) is the
first target; the other formats remain future work.

#### Stage: PretrainShardStage (bin/idx)

Responsibilities:
- Process a `ShardTask` (one shard) using `process_binidx_shard_core()`
- Write `.bin/.idx` and receipt JSON

Sketch:

```py
import cosmos_xenna.pipelines.v1 as pipelines_v1
from nemotron.data_prep.shard_processor import process_binidx_shard_core
from nemotron.data_prep.providers import create_tokenizer
from nemotron.data_prep.filesystem import get_filesystem

class PretrainShardStage(pipelines_v1.Stage):
    @property
    def stage_batch_size(self) -> int:
        return 1  # one shard per task

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(gpus=0, cpus=1.0)

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        self._tokenize = create_tokenizer(self._resolved_tokenizer)
        self._output_fs, _ = get_filesystem(self._output_dir)

    def process_data(self, tasks: list[dict]) -> list[dict]:
        results = []
        for task in tasks:
            stats = process_binidx_shard_core(
                tokenize=self._tokenize,
                text_field=task["text_field"],
                min_doc_chars=task["min_doc_chars"],
                max_doc_tokens=task["max_doc_tokens"],
                dtype=task["dtype"],
                max_rows=task["max_rows"],
                shard_index=task["shard_index"],
                assignment=task["assignment"],
                plan_hash=task["plan_hash"],
                output_dir=task["output_dir"],
                receipts_dir=task["receipts_dir"],
                output_fs=self._output_fs,
            )
            results.append(stats)
        return results
```

Notes:
- All Xenna scaffolding (stage class, work item types, runner) is located in
  `nemotron.data_prep.xenna` (e.g., `xenna/stages.py`, `xenna/work_items.py`,
  `xenna/runner.py`).
- `assignment` is parsed from `assignment_json` before pipeline submission.
- `_resolved_tokenizer` and `_output_dir` are injected at stage construction time.
- The stage is intentionally thin and delegates all shard semantics to existing code.

#### Stage (future): JsonlStage

Not part of this effort. Placeholder if JSONL output migrates later:
- Wrap `jsonl_processor` logic into a Xenna stage.
- Preserve existing transform semantics.

#### Stage (future): PackedStage / ChatSftStage

Not part of this effort. Placeholders for packed SFT and chat SFT formats.

### 4) HuggingFace Optimized Downloading

Continue using the existing HF downloader, but move it earlier in the flow:

- Collect all HF files from shard assignments.
- Run `parallel_predownload()` with `max_concurrent_downloads`.
- Store in HF cache (HF_HOME/hub).
- Shard processing uses `hf_hub_download(..., local_files_only=True)` to
  avoid network I/O and ensure determinism.

This keeps HF as a first-class source while minimizing new download machinery.
Cosmos-Xenna distributed download is not used for HF (it targets object stores).

### 5) Output and Artifacts

Outputs remain unchanged:

- `.bin`/`.idx` shards under `runs/<hash>/datasets/<name>/<plan_hash>/`
- `blend.json` with `train`/`valid`/`test` or `data_paths` + split
- Receipts per shard for resuming and cache hits

`PretrainBlendsArtifact` creation stays in `run_data_prep()` (or new entrypoint).

## API and Configuration Changes

### New entrypoint

```
run_data_prep_from_dict(
    datasets: dict,
    *,
    output_dir: Path,
    tokenizer_model: str,
    ...  # existing DataPrepConfig fields
)
```

### Execution engine flag

Add a config flag (default to Xenna for stage0):

```
execution_engine: Literal["ray", "xenna"] = "xenna"
```

This allows phased rollout and easy fallback.

## Integration Points

- `nemotron/data_prep/pipeline.py`:
  - add a Xenna execution path (replacing `_process_all_shards_parallel`)
  - keep planning, receipts, and manifest generation.
- `nemotron/data_prep/xenna/`:
  - new Xenna integration module (stages, work items, runner).
- `nemotron/data_prep/shard_processor.py`:
  - reuse `process_binidx_shard_core()` as-is.
- `nemotron/data_prep/downloader.py`:
  - reuse `parallel_predownload()` before Xenna execution.
- `recipes/nano3/stage0_pretrain/data_prep.py`:
  - accept dict dataset spec and call new entrypoint.
- `recipes/nano3/stage0_pretrain/prep_xenna.py`:
  - new Xenna-specific entrypoint for staged validation.
- `cli/nano3/data/prep/pretrain_xenna.py`:
  - new recipe command module (mirrors `pretrain.py` style).
- `cli/nano3/data/prep/app.py`:
  - register a new `prep xenna` or `prep pretrain-xenna` command via `make_recipe_command`.

## Rollout Plan

1. Add `prep_xenna.py` for pretrain and wire it into the CLI as an opt-in path
   using the existing recipe command pattern.
2. Land the Xenna execution path behind `execution_engine` flag (default stays Ray).
3. Validate Xenna path in recipes and tests.
4. Switch Stage 0 pretrain to Xenna by default once stable.
5. Keep legacy Ray path for regression fallback until removed.
6. Add docs update describing dict input and Xenna engine selection.

## Testing Strategy

- Unit tests:
  - dict schema -> DataBlend conversion
  - ShardTask mapping (assignment_json, plan_hash, output dirs)
- Integration tests (existing):
  - `tests/recipes/nano3/stage0_pretrain/test_data_prep_train_integration.py`
    should pass for both engines.
- HF download tests:
  - Mock HF cache predownload; verify local_files_only path works.

## Risks and Mitigations

- **Xenna pipeline expectations**: Stage `process_data()` must be pure and
  pickle-friendly. Mitigate by keeping ShardTask data primitive types only.
- **HF cache not warmed**: Ensure predownload step runs before Xenna pipeline,
  or fallback to allowing direct download if cache misses occur.
- **Resource sizing**: Xenna defaults may oversubscribe CPUs. Provide explicit
  config mapping to `PipelineConfig` and `Resources`.
- **Behavioral drift**: Keep shard planning + core processor unchanged and
  preserve receipts to avoid output changes.

## Open Questions

- Should dict input be accepted by `run_data_prep()` directly (optional), or
  only by a new explicit entrypoint?
- Should Xenna execution be enabled for non-pretrain formats later, or keep
  scope limited to bin/idx until validated?
