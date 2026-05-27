---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "How nemotron steps run translate/nemo_curator flows through Curator readers, TranslationStage, writers, and FAITH."
topics: ["Translation", "Pipeline"]
tags: ["Explanation", "Architecture"]
content:
  type: "Explanation"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

# Pipeline Overview

This page describes how `nemotron steps run translate/nemo_curator` moves records from `input_path` into `output_dir` by using NVIDIA NeMo Curator staging primitives.

## Architecture

```{mermaid}
flowchart LR
    A[input_path] --> B[Reader]
    B --> C[TranslationStage]
    C --> D[Writer]
    D --> E[output_dir shards]
    C --> F{faith_eval.enabled?}
    F -->|yes| G[FAITH scoring LLM]
    G --> C
    F -->|no| D
```

## Reader Stage

1. Format detection follows `input_format`, which may be `auto`, `jsonl`, or `parquet`.
2. Paths may be a single file, a glob, or a homogeneous directory of shards. Never mix JSON Lines (JSONL) and Parquet in one directory when `auto` is active.

## Translation Stage

`TranslationStage` performs backend-specific translation for every location matched by `text_field`, for example `messages.*.content` wildcards inside chat arrays.

- `backend` selects `llm`, `nmt`, `google`, or `aws`.
- `segmentation_mode` chooses coarse versus fine segmentation before translation units leave the stage.
- `output_mode` controls whether replaced fields, raw metadata, or both appear on each record.

## Writer Stage

The writer emits `output_format` shards, either `jsonl` or `parquet`, under `output_dir`. Expect partitioned filenames rather than a single merged file. Downstream packing steps usually consume the directory directly.

## FAITH Coupling

When `faith_eval.enabled` is true the stage keeps an OpenAI-compatible client even if `backend` is `nmt`, `google`, or `aws`, because FAITH scoring uses the LLM configured under `server`. You can override that model string with `faith_eval.model_name`.

## Operational Reminders

- Translation failures surface as runtime errors from Curator. Rerun with smaller concurrency if providers throttle you.
- Extremely large single files may require an offline splitting stage. Mirror the guardrails codified in `step.toml`.

## Related Pages

- Hands-on first run: {doc}`../getting-started`
- Segmentation behavior: {doc}`segmentation`
- FAITH semantics: {doc}`faith-evaluation`
- YAML keyed to this flow: {doc}`../reference/translate-config`
