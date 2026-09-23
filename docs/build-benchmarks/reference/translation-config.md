<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

<!-- Reference: BYOB translate YAML keys, metric semantics, and parse-time validation. -->

# Translation Configuration Reference

This page describes the YAML configuration file that you provide for the translate stage: the keys you set or override, how backtranslation quality scores show up in Parquet output columns, and the stricter truth-preserving rules used by BFCL.

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

For MCQ, `dataset_path` names `benchmark.parquet`. BFCL instead requires
`family: bfcl`, `stage: translate`, `config_status: resolved`, and
`source_run_manifest` naming the immutable generation `run_manifest.json`. BFCL
refuses `dataset_path` because a bare Parquet file cannot prove publication
lineage or artifact hashes.

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

BFCL requires `remove_low_quality: false`: localization cannot add, drop, or
reorder task IDs. Its optional `translate_tool_descriptions` key defaults to
false. When true, only `tools[].function.description` may change; function names
and the complete parameter schema remain immutable.

BFCL model parameters must name `provider`, `model`, and a stable
`canonical_id` (or `alias`). The translation manifest records these fields,
optional weight source/revision/digest, and a hash of the complete model config
for contamination analysis.

BFCL also accepts a `localization` mapping. Its
`deterministic_fixes.normalize_unicode` switch defaults to `true`; output is
normalized to NFC, LF line endings, and no trailing line whitespace before
backtranslation. `validation.minimum_changed_fraction` defaults to `0.01` and
rejects an identity/no-op translator. `validation.required_script` is inferred
for Arabic, Cyrillic, Devanagari, Greek, Han, Hangul, Hebrew, Japanese, and Thai
targets, and may be set explicitly or to `null`. Case-insensitive regular
expressions under `response_guards.forbidden_patterns` reject model refusals,
source-language leakage, or other deployment-specific phrases.

Locale comparisons use normalized primary BCP-47 subtags, so source metadata
`en` is compatible with configured `en-US`. Full canonical tags remain in the
artifacts, while the normalized lowercase tag is used in the output filename.
The translator is the sole model-facing localization stage; deterministic
fixes do not introduce a second, untracked model identity.

## FAITH evaluation

`translation_model_config.stage.enable_faith_eval` must not be true.
The benchmark translation relies on backtranslation metrics instead of FAITH filtering
