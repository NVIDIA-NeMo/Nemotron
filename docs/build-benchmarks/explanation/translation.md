<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

<!-- Explanation: translation YAML, quality gates, and operator-visible artifacts for BYOB MCQ translate. -->

# Translation

Learn how to take an existing `benchmark.parquet` from generation, translate it to a target locale, score quality with backtranslation metrics, and export another `benchmark.parquet`.

The field names, defaults, and validation rules are listed in {doc}`../reference/translation-config`.
Artifact paths are summarized in {doc}`../reference/output-files`.

## What You Configure

| Control | What you set |
| --- | --- |
| `dataset_path` | Absolute or workspace path to the source `benchmark.parquet` you want translated. |
| `output_dir` / `expt_name` | Where caches and the translated `benchmark.parquet` are written. |
| `source_language` / `target_language` | BCP-47 style tags, for example `en-US` and `hi-IN`. |
| `translation_model_config` | Curator experimental translation block: `backend_type`, `params` (model, provider, credentials, `inference_parameters`), plus optional `stage` and `segment_stage` maps. |
| `backtranslation_quality_metrics` | List of `{type, threshold}` entries. Each `type` must be `sacrebleu`, `chrf`, or `ter`; each `threshold` must be nonnegative. Keep at least one entry so scoring runs; outputs land in `quality_metrics.parquet` with per-metric scores and `is_quality_metric_passed`. |
| `remove_low_quality` | When `true` (the default if you omit the key), rows that fail the aggregate quality gate are dropped before the final export. When `false`, every row is kept so you can inspect scores first. |

Do not set `translation_model_config.stage.enable_faith_eval` to true.
Translation relies on backtranslation metrics instead of FAITH.

## Running the Translate Stage

Pass `stage=translate` unless your YAML sets a top-level `stage` key.
The CLI requires an explicit stage when that key is absent.

```console
uv run nemotron steps run byob/mcq -c translate stage=translate
```

## Tune Quality Gates

The `backtranslation_quality_metrics` field is the only place you define automatic pass or fail rules for backtranslation checks.
Add or remove list entries to change which scores are computed, and adjust `threshold` values when you want stricter or looser gates.

After a run, open `quality_metrics.parquet` under `output_dir`/`expt_name`/`stage_cache/` to read per-metric score columns and `is_quality_metric_passed` before you change YAML again.

## Final Filtering Control

- `remove_low_quality` decides whether failing rows disappear from the exported `benchmark.parquet`.

   ```yaml
   remove_low_quality: true   # omit to get the same default
   ```

   ```yaml
   remove_low_quality: false  # keep failing rows; filter manually using Parquet columns
   ```

## Reference Layout

The sample `src/nemotron/steps/byob/mcq/config/translate.yaml` file shows a complete `translation_model_config` with `backend_type: llm`, NVIDIA provider parameters, and `stage` / `segment_stage` tuning.
Copy that structure, then swap model IDs, concurrency, and language tags for your workload.

The YAML below mirrors the sample configu, including `remove_low_quality: false` so rows that fail the aggregate quality gate remain in `benchmark.parquet` and you can inspect `stage_cache/quality_metrics.parquet` while you tune thresholds.

When you omit `remove_low_quality` or set it to `true`, failing rows are dropped before export.

```yaml
expt_name: byob_mcq_translation
dataset_path: /path/to/benchmark.parquet
output_dir: /path/to/outputs
source_language: en-US
target_language: hi-IN

translation_model_config:
  backend_type: llm
  params:
    alias: gpt-oss-120b
    model: openai/gpt-oss-120b
    provider: nvidia
    api_key_env: NGC_API_KEY
    inference_parameters:
      max_tokens: 16000
      max_parallel_requests: 8
      temperature: 0.0
      top_p: 0.95
  stage:
    segmentation_mode: coarse
    min_segment_chars: 0
    output_mode: both
  segment_stage:
    health_check: true
    max_concurrent_requests: 8

backtranslation_quality_metrics:
  - type: sacrebleu
    threshold: 25
  - type: chrf
    threshold: 50
  - type: ter
    threshold: 50

remove_low_quality: false
```

## Directory Structure

The translation stage writes intermediate Parquet files to `<output_dir>/<expt_name>/<stage_cache>` as `translated_questions.parquet`, `backtranslated_questions.parquet`, and `quality_metrics.parquet`, followed by `benchmark_raw.parquet` and the renamed `benchmark.parquet` in the experiment root.
Use the intermediate files to debug language mix-ups, threshold misses, or model refusals before you change configuration again.

## Related Information

- {doc}`../reference/translation-config` for every required and optional YAML key.
- {doc}`../how-to/custom-model-endpoints` for credentials and base URLs on your model blocks.
- {doc}`../how-to/skip-stages` when you need to rerun only translation or downstream stages after a config tweak.
- {doc}`pipeline-overview` to see where translate sits after generate.
