<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Pipeline Overview

The `mcq` family is registered in `runtime/benchmark_families/registry.py` and exposes three entrypoints: `prepare_data`, `generate`, and `translate`.

`nemotron.steps.byob.scripts.runtime.run_byob` dispatches by `stage`:

| CLI / YAML `stage` | Calls |
| --- | --- |
| `prepare` | `prepare_mcq_data` |
| `generate` | `generate_mcq` |
| `translate` | `translate_mcq` |
| `all` | `prepare_mcq_data` then `generate_mcq` |

`nemotron steps run byob/mcq` executes `mcq/step.py`, which forwards to the BYOB argparse dispatcher in `src/nemotron/steps/byob/scripts/run.py`.

## Generate stage order

`generate_mcq` in `runtime/benchmark_families/mcq/pipeline.py` writes Parquet files under `output_dir/expt_name/stage_cache/` in this order:

1. **GENERATION** — `generated_questions.parquet`
2. **JUDGEMENT** — `judged_questions.parquet`
3. **SEMANTIC_DEDUPLICATION** — `semantic_deduplicated_questions.parquet` (skipped body when `semantic_deduplication_config.enabled` is false, but the file still materializes with duplicate flags)
4. **DISTRACTOR_EXPANSION** — optional when `do_distractor_expansion` is true
5. **COVERAGE_CHECK** — optional when `do_coverage_check` is true
6. **DISTRACTOR_VALIDITY_CHECK** — `valid_distractors.parquet`
7. **SEMANTIC_OUTLIER_DETECTION** — `semantic_outlier_detection.parquet`
8. **HALLUCINATION_EASINESS_DETECTION** — `filtered_questions.parquet`
9. **FINAL_OUTPUT** — copies into `benchmark_raw.parquet`, applies `remove_hallucinated` / `remove_easy`, renames columns, and writes `benchmark.parquet`

## Translate stage order

`translate_mcq` writes:

1. **TRANSLATION** — `stage_cache/translated_questions.parquet`
2. **BACKTRANSLATION** — `stage_cache/backtranslated_questions.parquet`
3. **QUALITY_METRICS** — `stage_cache/quality_metrics.parquet`
4. **FINAL_OUTPUT** — `benchmark_raw.parquet`, optional `remove_low_quality` filter, column rename to the MCQ schema, `benchmark.parquet`

See {doc}`../reference/output-files` for the exact filenames.
