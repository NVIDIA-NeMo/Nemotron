<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

<!-- Explanation: easiness and hallucination scoring, YAML controls, and how rows reach benchmark.parquet. -->

# Easiness and Hallucination Filtering

Decide which generated multiple-choice question (MCQ) rows look too trivial or unreliable before you export `benchmark.parquet`, using only configuration and prompts.

The `HALLUCINATION_EASINESS_DETECTION` stage runs two prompt families in parallel, each backed by one or more models listed in YAML.
Every row receives numeric scores and boolean flags.
Optional removal rules decide what survives into the final export.

Artifacts and filenames are listed in {doc}`../reference/output-files`.
Field-level YAML is documented in {doc}`../reference/generate-config`.

## What You Control

| Control | Purpose |
| --- | --- |
| `filtering_model_configs` | Which models answer the easiness and hallucination judge prompts. |
| `easiness_threshold` | When the easiness vote fraction is above this value, the row is marked `is_easy`. |
| `hallucination_threshold` | When the hallucination vote fraction is below this value, the row is marked `is_hallucination`. |
| `remove_easy` | If true, rows with `is_easy` drop out of `benchmark.parquet` during final assembly. |
| `remove_hallucinated` | If true, rows with `is_hallucination` drop out of `benchmark.parquet` during final assembly. |
| `prompt_config` | `null` keeps packaged prompts; a path must supply `easiness_filter` and `hallucination_filter` blocks. Refer to {doc}`../how-to/prompt-tuning`. |
| `ndd_batch_size` | Batch size for the Data Designer calls that run this stage. |

Even when removals are false, `stage_cache/filtered_questions.parquet` keeps `correct_ratio_easiness`, `correct_ratio_hallucination`, `is_easy`, and `is_hallucination` so you can audit scores before tightening thresholds.

## How the Scores Work

For each filter family, every configured model returns a parsed letter answer.
The pipeline compares those letters to `answer_generated` on the row.

- Easiness builds `correct_ratio_easiness` as the fraction of easiness models whose answer matches `answer_generated`.
  When that fraction is greater than `easiness_threshold`, `is_easy` is set to true to indicate the item was too easy for the swarm to disagree with.
- Hallucination builds `correct_ratio_hallucination` the same way for the hallucination model list.
  When that fraction is less than `hallucination_threshold`, `is_hallucination` is set to true to indicate that too few models agreed with the generated label. Typically, these need human-in-the-loop review.

Adding more models under a list increases the denominator of the ratio, so revisit thresholds whenever you add or remove swarm members.

Each model alias produces its own response column before ratios are computed.
This approach helps to compare votes row by row.

## Sample Configuration

The following pattern matches the sample `tiny` configuration: one model per family with aliases that differ from other stages and from each other.

```yaml
easiness_threshold: 0.8
hallucination_threshold: 0.5
remove_hallucinated: false
remove_easy: false
filtering_model_configs:
  hallucination:
    - alias: hal_gpt-oss-120b
      model: openai/gpt-oss-120b
      provider: nvidia
      inference_parameters:
        max_tokens: 1024
        max_parallel_requests: 1
        temperature: 0.0
        top_p: 1.0
  easiness:
    - alias: eas_gpt-oss-120b
      model: openai/gpt-oss-120b
      provider: nvidia
      inference_parameters:
        max_tokens: 1024
        max_parallel_requests: 1
        temperature: 0.0
        top_p: 1.0
```

### Optional Second Opinion

Add another list entry when you want independent votes without changing code:

```yaml
filtering_model_configs:
  hallucination:
    - alias: hal_primary
      model: openai/gpt-oss-120b
      provider: nvidia
      inference_parameters:
        max_tokens: 1024
        temperature: 0.0
        top_p: 1.0
    - alias: hal_backup
      model: nvidia/nemotron-3-nano-omni-30b-a3b-reasoning
      provider: nvidia
      inference_parameters:
        max_tokens: 65536
        temperature: 0.2
        top_p: 0.95
  easiness:
    - alias: eas_primary
      model: openai/gpt-oss-120b
      provider: nvidia
      inference_parameters:
        max_tokens: 1024
        temperature: 0.0
        top_p: 1.0
```

Use a model from a different model family when the generator model is also the primary filter model.
Set each entry to the production models you want in the vote.

## Removal Defaults

The `remove_hallucinated` field defaults to true when omitted from YAML, while `remove_easy` defaults to false.
Set both explicitly when you want reproducible behavior across environments.

```yaml
remove_hallucinated: true
remove_easy: true
```

Turn removals off while you tune thresholds, inspect `filtered_questions.parquet`, then re-enable them for production runs.

## Prompts

If you specify a custom `prompt_config` file, include the `easiness_filter` and `hallucination_filter` sections expected by validation.
The default configuration files wire those prompts when `prompt_config` is `null`.

## Related Information

- {doc}`quality-validation` for upstream judgement, deduplication, distractors, and outlier stages that feed this step.
- {doc}`question-generation` for how rows enter the generate pipeline before filtering.
- {doc}`../how-to/skip-stages` if you need to rerun filtering after tweaking thresholds.
