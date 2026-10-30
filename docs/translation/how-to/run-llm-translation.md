---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Use backend=llm with nemotron steps run translate/nemo_curator and NVIDIA_API_KEY."
topics: ["Translation", "LLM"]
tags: ["How-To", "Translation"]
content:
  type: "How-To"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

# Run LLM Translation

Use this guide when `backend` must stay `llm` and you need to point `nemotron steps run translate/nemo_curator` at an OpenAI-compatible chat-completions endpoint and model.

## Prerequisites

- Set `NVIDIA_API_KEY` in your shell when relying on the default `server.api_key_env`, for example `export NVIDIA_API_KEY="<api-key>"`.
- Confirm `server.url` matches your deployment. The `default.yaml` file targets the NVIDIA integrate API.

## Procedure

1. Start from `default.yaml` with `-c default`.
2. Override model and languages:

```bash
uv run nemotron steps run translate/nemo_curator -c default \
  backend=llm \
  input_path=/path/to/chat.jsonl \
  output_dir=/path/to/out \
  source_language=en \
  target_language=de \
  server.model=YOUR_LLM_MODEL_ID
```

3. Adjust `max_concurrent_requests` upward only after verifying the endpoint tolerates parallel completions.

## Hosted Model Hygiene

Hosted catalogs retire models frequently. Pin to identifiers your tenant currently exposes before large batch jobs.

## Related Pages

- FAITH requirements when enabled: {doc}`run-faith-evaluation`
- Full YAML reference: {doc}`../reference/translate-config`
