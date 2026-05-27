<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

<!-- Reference: BYOB translate YAML keys, metric semantics, and parse-time validation. -->

# Translation Configuration Reference

This page describes the YAML configuration file that you provide for the translate stage: the keys you set or override, how backtranslation quality scores show up in Parquet output columns, and which rows are omitted from the final `benchmark.parquet` when quality filtering is enabled.

## Required Keys

| Key | Notes |
| --- | --- |
| `expt_name` | Directory name under `output_dir` for caches and final artifacts. |
| `dataset_path` | Existing `benchmark.parquet` from a generation run. |
| `output_dir` | Parent directory for `expt_name`. |
| `source_language` | BCP-47 style tag (for example `en-US`). |
| `target_language` | Target locale tag (for example `hi-IN`). |
| `translation_model_config` | Dictionary with `backend_type`, `params`, and optional `stage` and `segment_stage`. |
| `backtranslation_quality_metrics` | Non-empty list; each element is a dictionary with `type` and `threshold`. |

## Quality Metrics

Each `type` must be `sacrebleu`, `chrf`, or `ter`.
Each `threshold` must be nonnegative.

NeMo Curator writes one numeric score column and one boolean pass column per configured metric, for example `score_chrf` and `score_chrf_passed`.
The column `is_quality_metric_passed` is true on a row when every per-metric pass column is true for that row.

Each score compares the original benchmark text with the round-trip backtranslation from the target locale, using sentence-level APIs from the *sacrebleu* library.

| `type` | Measures | Scale and direction | How to interpret scores | Row passes when |
| --- | --- | --- | --- | --- |
| `sacrebleu` | Sentence bilingual evaluation understudy (BLEU) | 0 through 100 after *sacrebleu* tokenization; higher values track closer matches to the reference. | High scores mean the backtranslation preserved wording and order; scores near zero mean little *n*-gram overlap. | `score_sacrebleu` ≥ `threshold` |
| `chrf` | Sentence character n-gram F-score (chrF) | 0 through 100 in typical sentence outputs; higher values mean closer character-level match. | High scores track spelling and phrasing fidelity; low scores mean the backtranslation diverged on the surface string. | `score_chrf` ≥ `threshold` |
| `ter` | Sentence translation error rate (TER) | Zero means no edits; larger values report more insertions, deletions, or substitutions relative to the reference. | Values close to zero mean the backtranslation needed minimal editing to match the original; large values signal heavy rewrites or mismatch. | `score_ter` ≤ `threshold` |

Inspect `stage_cache/quality_metrics.parquet` under your experiment directory to pick thresholds from the score spread you see in data.

## Optional Keys

| Key | Default | Notes |
| --- | --- | --- |
| `remove_low_quality` | `True` unless YAML overrides it | When true, the pipeline omits rows where `is_quality_metric_passed` is false before export. |

## FAITH evaluation

`translation_model_config.stage.enable_faith_eval` must not be true.
The benchmark translation relies on backtranslation metrics instead of FAITH filtering
