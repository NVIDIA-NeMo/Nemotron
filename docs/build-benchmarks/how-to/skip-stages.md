<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Skip Stages When Iterating

Both **generate** and **translate** honor `skip_until`, a string that names an enum entry on the internal stage list.
Stages whose enum value is **less than** the named stage are skipped as long as the expected Parquet already exists.

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

## Preconditions

Skipping only works when the Parquet file produced by the previous stage is already on disk under `output_dir/expt_name/stage_cache/`.
Otherwise the next stage reads missing input and fails.

For other common failure modes, see {doc}`../reference/troubleshooting`.
