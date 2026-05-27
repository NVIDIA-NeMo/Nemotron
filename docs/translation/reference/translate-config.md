---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "YAML reference for translate/nemo_curator aligned with config/default.yaml."
topics: ["Translation", "Configuration"]
tags: ["Reference", "YAML"]
content:
  type: "Reference"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Translation YAML Reference

The `translate/nemo_curator` step ships `src/nemotron/steps/translate/nemo_curator/config/default.yaml` as the canonical starter profile. This page lists top-level keys you can override with `nemotron steps run translate/nemo_curator key=value` dotlists, grouped by concern, with the full baseline file inlined below.

## Default Configuration File

```{literalinclude} ../../../src/nemotron/steps/translate/nemo_curator/config/default.yaml
:language: yaml
:class: scrollable
```

## Keys Grouped by Concern

### Paths and Formats

| Key | Description |
|-----|-------------|
| `input_path` | File, glob, or homogeneous directory consumed by `JsonlReader` or `ParquetReader`. |
| `output_dir` | Directory passed to `JsonlWriter` or `ParquetWriter` in overwrite mode. |
| `input_format` | `auto`, `jsonl`, or `parquet`. |
| `output_format` | `jsonl` or `parquet`. |

### Languages and Backend

| Key | Description |
|-----|-------------|
| `source_language` / `target_language` | Required ISO 639-1 codes. Empty placeholders remind operators to set values explicitly. |
| `backend` | `llm`, `nmt`, `google`, or `aws`. |

### Translation Semantics

| Key | Description |
|-----|-------------|
| `text_field` | Dot or wildcard path describing strings to translate. The default is `messages.*.content`. |
| `output_field`, `translation_column` | Destination columns for translated text and downstream merges. |
| `output_mode` | `replaced`, `raw`, or `both`. |
| `merge_scores` | Attach FAITH outputs adjacent to translations when enabled. |
| `reconstruct_messages`, `messages_field`, `messages_content_field` | Chat reconstruction switches. |
| `segmentation_mode`, `min_segment_chars` | Segmenter behavior. Values include `coarse` and `fine`. |
| `max_concurrent_requests`, `skip_translated`, `files_per_partition`, `blocksize` | Throughput and partitioning controls surfaced to Curator readers and clients. |

### LLM Fields

Used whenever `backend=llm` or FAITH needs an OpenAI-compatible judge.

| Key | Description |
|-----|-------------|
| `server.url` | Chat-completions compatible base URL. |
| `server.model` | Model identifier. Required for `llm` translation and for FAITH unless you override the scorer model. |
| `server.api_key_env` | Environment variable housing the API secret. The default is `NVIDIA_API_KEY`. |
| `server.api_key` | Inline secret. Discouraged for shared repositories. |

### FAITH Evaluation

| Key | Description |
|-----|-------------|
| `enabled` | Turns FAITH scoring on. The starter YAML sets this to `true`. |
| `threshold` | Minimum acceptable `faith_avg` on a one-to-five scale. The starter default `2.5` is a permissive noisy-data floor. See {doc}`../explanation/faith-evaluation` for the full rubric. |
| `model_name` | Optional scorer-only model. Defaults to `server.model`. |
| `filter_enabled` | Drop failing rows when `true`. |
| `max_concurrent_requests` | Optional scorer-side concurrency limit. |
| `generation_config` | Optional OpenAI-compatible generation settings for the scorer. |

### Backend-Specific Blocks

| Block | When needed |
|-------|-------------|
| `nmt` | HTTP microservice URL, batching, timeouts. |
| `google` | Project metadata and API version. Version `v3` requires `project_id`. |
| `aws` | Region plus concurrency limits. |

## Overrides

OmegaConf dotlists merge last:

```bash
uv run nemotron steps run translate/nemo_curator -c default \
  backend=nmt \
  nmt.server_url=http://localhost:5000 \
  faith_eval.enabled=false \
  input_path=/data/chat.jsonl \
  output_dir=/data/out \
  source_language=en \
  target_language=hi
```

## Related Pages

- CLI merge rules and flags: {doc}`cli-translation`
- Record shapes and writers: {doc}`io-format`
