<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Generation Configuration Reference

This page lists the YAML keys validated inside `ByobConfig.from_yaml` in `runtime/config.py`.
Optional keys show their defaults in the dataclass or in the `get` calls inside `from_yaml`.

## Experiment layout

| Key | Notes |
| --- | --- |
| `expt_name` | Required string; builds `output_dir/expt_name/`. |
| `output_dir` | Required writable directory. |
| `input_dir` | Required path containing each `target_source_mapping` corpus. |
| `random_seed` | Optional; seeds Python and NumPy when present. |
| `ndd_batch_size` | Optional positive int; defaults to `1000` in the dataclass but sample configs use smaller values. |

## Dataset selection

| Key | Notes |
| --- | --- |
| `hf_dataset` | Must appear in `ALLOWED_HF_DATASETS`. |
| `subset` | Optional; defaults per {doc}`benchmarks`. |
| `split` | Optional; defaults to `"test"`. |
| `language` | Required string (for example `en-US`). |
| `metadata_file` | Optional CSV path enabling tag-aware sampling. |

## Subjects and sampling

| Key | Notes |
| --- | --- |
| `source_subjects` | Required non-empty list of benchmark subject strings. |
| `target_source_mapping` | Required mapping; see {doc}`../how-to/prepare-data`. |
| `few_shot_samples_per_query` | Required positive int. |
| `queries_per_target_subject_document` | Required positive int. |
| `num_questions_per_query` | Required positive int. |

## Prompts and models

| Key | Notes |
| --- | --- |
| `prompt_config` | `null` loads packaged defaults; a string path loads YAML with all required stages. |
| `generation_model_config` | Required mapping for Data Designer calls. |
| `judge_model_config` | Required mapping. |
| `do_distractor_expansion` | Required bool. |
| `distractor_expansion_model_config` | Required when expansion is true. |
| `distractor_validity_model_config` | Required mapping. |
| `filtering_model_configs` | Required dict with `hallucination` and `easiness` lists. |
| `easiness_threshold` | Required float in `[0, 1]`. |
| `hallucination_threshold` | Required float in `[0, 1]`. |
| `remove_hallucinated` | Optional bool; defaults to `True`. |
| `remove_easy` | Optional bool; defaults to `False`. |

## Optional quality stages

| Key | Notes |
| --- | --- |
| `semantic_deduplication_config` | Dict with `enabled`, embedding model id, clustering parameters, and `remove_duplicates`. |
| `semantic_outlier_detection_config` | Dict with `enabled`, embedding model id, and neighbour thresholds. |
| `chunking_config` | Dict; supports `window_size` (`null` keeps whole documents). |
| `do_coverage_check` | Bool; defaults to `False` when omitted. |
| `coverage_check_config` | Dict with `window_size` and `model_identifier`. |

## Curator mount

Sample configs include a `run.env.mounts` entry that uses `${auto_mount:...}` to place NeMo Curator on `/opt/Curator`.
Remote profiles must preserve the same mount contract your environment expects.

## Stage dispatch for `nemotron steps run`

The generic steps CLI accepts `family`, `stage`, and `skip_until` as dotlist overrides, and you can also place those keys directly in YAML.
`family` defaults to `mcq` when omitted.
