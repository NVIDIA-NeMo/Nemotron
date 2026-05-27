---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Run nemotron steps run translate/nemo_curator end-to-end with config/default.yaml and sample chat JSONL."
topics: ["Translation", "Tutorial"]
tags: ["Tutorial", "Translation"]
content:
  type: "Tutorial"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

(getting-started-translation)=
# Getting Started With Translation

You will produce translated JSON Lines (JSONL) shards under an `output_dir` you choose, with FAITH quality scoring applied in the same run. FAITH is the translation-quality scorer name used in NVIDIA NeMo Curator documentation.

This tutorial runs `nemotron steps run translate/nemo_curator` end to end on a small sample file. You use `src/nemotron/steps/translate/nemo_curator/config/default.yaml` and CLI dotlist overrides.

:::{card}
What You Will Produce

Translation shards sit under a writable `output_dir`, with filtering governed by `faith_eval` in the same run.

^^^

Overview

1. Export `NVIDIA_API_KEY`.
2. Download the sample `train_sample.jsonl` chat file.
3. Run `nemotron steps run translate/nemo_curator -c default` with CLI overrides for paths, languages, and `server.model`.
4. Inspect `output_dir` for translated records.

{octicon}`clock;1em;sd-mr-1` Budget roughly five to fifteen minutes depending on network latency and response time from your provider and on corpus size.
The sample contains about one hundred lines.
:::

## Prerequisites

- Network access to `https://integrate.api.nvidia.com/v1`.
- `NVIDIA_API_KEY` exported in your shell.
- For local `uv run` execution with Curator/Ray, export `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`
  so Ray workers reuse the synchronized project environment.

## Sample Input File

This translation tutorial uses `train_sample.jsonl` as a compact multi-turn chat dataset.

```{literalinclude} _snippets/input/train_sample.jsonl
:language: json
:class: scrollable
```

## Procedure

1. Clone the repository, if you haven't already:

   ```console
   $ git clone https://github.com/NVIDIA-NeMo/Nemotron && cd Nemotron
   ```

1. Synchronize the dependencies:

   ```console
   $ uv sync --extra translate
   ```

1. Download `train_sample.jsonl` from the [sample file](_snippets/input/train_sample.jsonl).

   Save the file in the repository root if you want to match the `input_path` below exactly. If you save it elsewhere, change `input_path` in the commands that follow.

1. Run the translation stage.

   From the repository root, specify the `default.yaml` config and overrides by using CLI arguments.
   Set `server.model` to a model your endpoint serves.
   Replace `<api-key>` with your NVIDIA API key value, or set `NVIDIA_API_KEY` in your environment before running the command.

   ```console
   $ export NVIDIA_API_KEY="<api-key>"
   $ export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

   $ uv run --no-sync nemotron steps run translate/nemo_curator -c default \
       input_path="${PWD}/train_sample.jsonl" \
       output_dir=./output/translation-getting-started \
       source_language=en \
       target_language=hi \
       server.model=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning \
       faith_eval.enabled=false \
       faith_eval.filter_enabled=false \
       max_concurrent_requests=4
   ```

   - `input_path` and `output_dir` replace the placeholders in `default.yaml`.
   - `source_language` and `target_language` must be explicit two-letter language codes from the ISO 639-1 standard, which the International Organization for Standardization publishes.

     The starter file leaves them empty on purpose so you choose the pair at run time.
   - `server.model` is required when `backend` is `llm`.

   The default `server.url` in `default.yaml` is `https://integrate.api.nvidia.com/v1`.

1. Inspect the output.

   The `output_dir` path holds the Curator writer output when `output_format` is `jsonl`, usually several shard files instead of one consolidated JSONL file.
   Spot-check one line.

   ```console
   $ find ./output/translation-getting-started -name '*.jsonl' | head -n 1 | xargs head -n 1 | python3 -m json.tool --no-ensure-ascii
   ```

   Translated chat payloads match `text_field` set to `messages.*.content`, `output_mode` set to `both`, and `reconstruct_messages` set to `true` in `default.yaml`.

   ```{literalinclude} _snippets/output/translated.jsonl
   :language: json
   :class: scrollable
   ```

1. Optional: Print the merged configuration without running the stage.

   Pass `--dry-run` or `-d` so Curator does not execute the pipeline.

   ```console
   $ export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

   $ uv run --no-sync nemotron steps run translate/nemo_curator -d -c default \
       input_path=./train_sample.jsonl \
       output_dir=./output/translation-getting-started \
       source_language=en \
       target_language=hi \
       server.model=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning
   ```

## Next Steps

- Deeper behavior of FAITH in this pipeline: {doc}`explanation/faith-evaluation`
- Backend tuning: {doc}`how-to/run-llm-translation`
- Field wiring and `output_mode`: {doc}`how-to/configure-fields-and-output`
- FAITH thresholds and filtering: {doc}`how-to/run-faith-evaluation`
- CLI flags and overrides: {doc}`reference/cli-translation`
