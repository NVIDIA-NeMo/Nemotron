<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Quality Validation

This page summarizes optional and mandatory checks between generation and the final Parquet export.

## Judgement

`judge_questions` scores each candidate against the `judge_model_config` prompt template, producing `judged_questions.parquet`.

## Semantic deduplication

When `semantic_deduplication_config.enabled` is true, `TextSemanticDeduplicationMCQ` runs inside `runtime/benchmark_families/mcq/deduplication.py`.
If the flag is false, the stage copies the input and marks `is_duplicate` as false for every row.

## Distractor expansion and validity

If `do_distractor_expansion` is true, the pipeline expands four-choice rows toward ten choices, then runs `check_distractor_validity` with `distractor_validity_model_config`.

## Coverage

When `do_coverage_check` is true, `TextCoverageMCQ` analyzes whether generated text still reflects the source chunk windows.

## Semantic outliers

`semantic_outlier_detection_config.enabled` toggles `TextSemanticOutlierDetectionMCQ`.
When disabled, the stage writes `is_outlier = False` and null neighbour metadata while still emitting the Parquet file expected by later stages.

Each stage writes its own Parquet under `stage_cache/`; see {doc}`../reference/output-files`.
