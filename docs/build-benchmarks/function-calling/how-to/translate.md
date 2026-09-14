<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Translate a Published Benchmark

Use this guide to localize a benchmark that has already been published. Translation is
a separate run with its own configuration, so localizing text cannot change the source
release's identity. It does not invent tasks, rewrite oracle truth, or drop rows.

## Before You Start

- A completed, verified generation run: `run_manifest.json` beside
  `benchmark.parquet` and `benchmark_raw.parquet`. Translation verifies the
  publication contract and hashes before contacting a model.
- A pinned translator identity and the environment variable that holds its credential.
  A literal key in the configuration is refused.
- A distinct empty `output_dir`. Translation refuses to reuse the generation
  publication tree.
- The source language of the published rows and the target language you want.

Translation does not support `skip_until`. A failed run is restarted from the beginning.

## Step 1: Copy And Resolve The Translation Config

```bash
mkdir -p /srv/bfcl/translate && \
  cp src/nemotron/steps/byob/bfcl/config/translate.yaml \
    /srv/bfcl/translate/en-vi.yaml
```

Every `REPLACE_ME_*` value must be resolved and `config_status` must become
`resolved`. Relative paths resolve from `src/nemotron/steps/byob/`, not from the
config file's directory; use an absolute `source_run_manifest` for an external
publication tree.

```yaml
family: bfcl
stage: translate
config_status: resolved
source_run_manifest: /srv/bfcl/runs/warehouse-gold-output/bfcl_warehouse_gold/run_manifest.json
output_dir: /srv/bfcl/translate/en-vi-out
source_language: en
target_language: vi
translate_tool_descriptions: false
remove_low_quality: false
```

Keep `remove_low_quality: false`. Translation never filters rows: the localized
release has exactly the task set and publication order of its source. Enabling the
filter is refused as `bfcl_translation_invalid`.

`translate_tool_descriptions` may be turned on when function descriptions should be
localized. Function names, parameter schemas, slot values, expected calls,
assertions, held-out state, and lineage stay exact, because those are the fields a
score compares.

## Step 2: Pin The Translator

Fill `translation_model_config` with a pinned identity. A branch-style revision such
as `main` is refused. The translator is recorded on `translation_manifest.json` and
enters the contamination inventory like any other model that read published rows.

## Step 3: Run Translation

```bash
uv run nemotron steps run byob/bfcl \
  -c /srv/bfcl/translate/en-vi.yaml \
  stage=translate \
  family=bfcl
```

The adapter verifies both source tables through the manifest, replaces protected
tokens with placeholders, translates and backtranslates the approved text fields,
restores every token, and compares each localized row with its source before writing
the new artifacts atomically.

## Step 4: Read The Artifacts

A completed run writes a content-addressed localized release under
`output_dir/expt_name/`:

| Artifact | Holds |
| --- | --- |
| `benchmark.<target-language>.parquet` | The localized published rows, same task set as the source; for example, `benchmark.vi.parquet`. |
| `translation_manifest.json` | Translator identity, contamination scope, and artifact hashes. |
| `stage_cache/translation_units.parquet` | Forward-translation evidence. |
| `stage_cache/backtranslation_units.parquet` | Backtranslation evidence. |
| `stage_cache/quality_metrics.parquet` | Recomputed quality-metric evidence. |

Do not treat a localized Parquet file without `translation_manifest.json` as a
localized release. To evaluate it, keep `source_run_manifest` pointed at the original
generation `run_manifest.json` and set the evaluation config's
`translation_manifest` to the localized `translation_manifest.json`. The evaluator
verifies that both records identify the same source run.

## Common Failures

| Symptom | What it means |
| --- | --- |
| `bfcl_translation_invalid` | The config named a bare Parquet file, reused an output directory, or set `remove_low_quality`. Start from `translate.yaml` and keep `config_status: resolved`. |
| Output identical to input | The no-op gate refused a translation that did not change unprotected text. Check `source_language`, `target_language`, and `localization.validation.minimum_changed_fraction`. |
| Script or guard refusal | Set `localization.validation.required_script` when the target subtag cannot infer the script, and review `forbidden_patterns`. |
| Protected value appeared in output | A tool name, argument, or other protected token leaked through a placeholder. Do not patch the table; fix the translator or guards and rerun. |

See {doc}`../reference/troubleshooting` for `bfcl_translation_invalid` and
{doc}`../explanation/pipeline-overview` for what translation may and may not change.

## Next Steps

- Score a candidate against the localized release: {doc}`run-evaluation`.
- Generation YAML that a later eval must not fork: {doc}`../reference/generate-config`.
- Source publication procedure: {doc}`publish-a-release`.
