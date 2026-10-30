<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

<!-- Reference: symptom-to-remedy tables for bring your own benchmark MCQ runs and translation YAML checks. -->

# Troubleshooting

This page lists common symptoms when you run `nemotron steps run byob/mcq` with the bring your own benchmark (BYOB) multiple choice question (MCQ) family and when you tune BYOB translation settings.
Each table row pairs a symptom with the files, fields, or flags you should inspect first.
For stage flow and design rationale, see the explanation pages linked from {doc}`../explanation/index`.

## Configuration Rejected Before the Pipeline Runs

| Symptom | What to do |
| --- | --- |
| Assertion that `tags` should not be specified when `metadata_file` is absent | Remove `tags` from every `target_source_mapping` entry, or set top-level `metadata_file` to the comma-separated values (CSV) file that supplies tag metadata. See {doc}`../how-to/prepare-data`. |
| Assertion that BYOB translation does not support FAITH evaluation | Under `translation_model_config.stage`, set `enable_faith_eval` to false or omit the field. BYOB translation expects `backtranslation_quality_metrics` instead of FAITH. See {doc}`translation-config` and {doc}`../explanation/translation`. |
| Message that a prompt override key is missing or is not a string, or validation fails on `prompt_config` | Open your `prompt_config` YAML and match the structure in {doc}`../how-to/prompt-tuning`. Easiness and hallucination overrides must include the expected blocks. |
| Schema validation reports that `distractor_validity_model_config` is missing | The current generation schema requires this block even when distractor expansion is off. Add the block as described in {doc}`../how-to/custom-model-endpoints`. |

## Skipped Stages and Missing Parquet Inputs

| Symptom | What to do |
| --- | --- |
| A stage fails because an expected Parquet file is missing under `output_dir/<expt_name>/stage_cache/` | When you use `skip_until`, every stage before the resume point must have written its output file to disk. Rerun from an earlier stage without skipping, or copy valid caches from a prior run. See {doc}`../how-to/skip-stages`. |

## Generation Ends With No Final Benchmark Rows

| Symptom | What to do |
| --- | --- |
| Log line `No questions left after filtering`, or `benchmark.parquet` is missing after a run that exited early | Open `stage_cache/filtered_questions.parquet` and inspect `is_easy`, `is_hallucination`, and score columns. Loosen `easiness_threshold` or `hallucination_threshold`, set `remove_easy` and `remove_hallucinated` to false for a diagnostic run, or adjust upstream judgement and deduplication settings. See {doc}`../explanation/filtering`. |

## Translation Export Drops Every Row

| Symptom | What to do |
| --- | --- |
| `benchmark.parquet` exists but has zero rows after translation | When `remove_low_quality` is true, only rows with `is_quality_metric_passed` are exported. Inspect `stage_cache/quality_metrics.parquet`, relax metric thresholds in `backtranslation_quality_metrics`, or set `remove_low_quality` to false while you tune. See {doc}`../explanation/translation`. |

## Few-Shot Sampling Finds No Rows

| Symptom | What to do |
| --- | --- |
| Runtime error stating there are no samples for a given source subject and tag combination | Tag filters must match comma-separated tag strings that appear in the metadata CSV for that Hugging Face subject. Widen `tags`, correct the CSV, or confirm `source_subjects` and `target_source_mapping` names align with {doc}`../how-to/prepare-data` and {doc}`../explanation/get-right-questions`. |

## Hosted Models Throttle or Stall

| Symptom | What to do |
| --- | --- |
| Timeouts, HTTP responses in the 429 range, or bursty failures when calling remote endpoints | Reduce `max_parallel_requests` and related batch settings in your YAML, rerun on a smaller slice of data, and confirm API keys and quotas. See {doc}`../explanation/question-generation` and {doc}`../how-to/custom-model-endpoints`. |

## Related Reference

- Parquet layout and final column list: {doc}`output-files`
- Generation YAML fields: {doc}`generate-config`
- Translation YAML fields: {doc}`translation-config`
