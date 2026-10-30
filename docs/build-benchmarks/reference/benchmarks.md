<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Supported Hugging Face Benchmarks

`hf_dataset` in your generation YAML must be one of the strings in `ALLOWED_HF_DATASETS` inside `src/nemotron/steps/byob/runtime/constants.py`.

If you omit `subset` in YAML, `ByobConfig.from_yaml` substitutes the default from `HF_DATASET_TO_SUBSET`.

| `hf_dataset` | Default `subset` when omitted |
| --- | --- |
| `cais/mmlu` | `all` |
| `TIGER-Lab/MMLU-Pro` | `default` |
| `ai4bharat/MILU` | `English` |
| `CohereLabs/Global-MMLU` | `en` |
| `CohereLabs/Global-MMLU-Lite` | `en` |
| `LinguaLift/IndicMMLU-Pro` | `hindi` |
| `openai/MMMLU` | `default` |
| `sarvamai/mmlu-indic` | `en` |
| `Idavidrein/gpqa` | `gpqa_main` |

Each identifier maps to the shared MCQ dataset implementation through `HF_DATASET_TO_MODULE`, which currently resolves every row to `nemotron.steps.byob.runtime.benchmark_families.mcq`.
