<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Resume or Skip Generation Stages

MCQ **generate** and **translate** honor `skip_until`, a string that names an
enum entry on the internal stage list. Stages whose enum value is **less than**
the named stage are skipped as long as the expected Parquet already exists.

## Generation enum names

From `McqGenerationStage` in `runtime/benchmark_families/mcq/pipeline.py`, valid names include:

`GENERATION`, `JUDGEMENT`, `SEMANTIC_DEDUPLICATION`, `DISTRACTOR_EXPANSION`, `COVERAGE_CHECK`, `DISTRACTOR_VALIDITY_CHECK`, `SEMANTIC_OUTLIER_DETECTION`, `HALLUCINATION_EASINESS_DETECTION`, `FINAL_OUTPUT`

## Translation enum names

From `McqTranslationStage`:

`TRANSLATION`, `BACKTRANSLATION`, `QUALITY_METRICS`, `FINAL_OUTPUT`

## CLI usage

Pass the resume point as a dotlist override:

```console
uv run nemotron steps run byob/mcq -c /path/to/generate.yaml skip_until=JUDGEMENT
```

```console
uv run nemotron steps run byob/mcq -c translate stage=translate skip_until=BACKTRANSLATION
```

## Verified BFCL generation resume

BFCL generation uses lowercase canonical names:

`reference_profile`, `expand`, `state_machine`, `render`, `expected_trace`,
`schema_validation`, `executable_replay`, `surface_quality`,
`dedup_balancing`, `final_output`

The named stage and every later enabled stage run. For example:

```console
uv run nemotron steps run byob/bfcl -c /path/to/generate.yaml stage=generate skip_until=expected_trace
```

Unlike MCQ's file-presence shortcut, BFCL recursively verifies immutable
checkpoint snapshots and their parent identities. It also revalidates config,
Oracle pack, endpoint identity, pipeline source, schemas, hashes, counts, task
IDs, and ordering. Optional stages are valid targets only when enabled.

## Preconditions

MCQ skipping requires the Parquet file produced by the previous stage under
`output_dir/expt_name/stage_cache/`. BFCL resume requires the complete verified
predecessor chain under `stage_cache/checkpoints/`; missing, stale, or tampered
evidence fails closed and requires a clean generation run. Restoration keeps the
append-only model I/O caches so a re-run stage replays recorded model responses
rather than generating new ones.

For other common failure modes, see {doc}`../reference/troubleshooting`.
