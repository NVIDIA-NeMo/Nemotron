---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Use backend=google or backend=aws with nemotron steps run translate/nemo_curator."
topics: ["Translation", "Cloud"]
tags: ["How-To", "Translation"]
content:
  type: "How-To"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Run Google or AWS Translation

Use this guide when `backend` is `google` or `aws` and you want managed cloud translation with credentials supplied outside YAML files.

## Shared Practices

- Inject credentials through provider-standard environment variables or instance roles. Do not paste secrets into `default.yaml`.
- Keep `google.project_id`, `google.location`, and `google.api_version` aligned with your Google Cloud Platform (GCP) setup. API version `v3` requires project metadata.
- Pin `aws.region` near your data residency requirements on Amazon Web Services (AWS).

## Google Example Skeleton

```bash
uv run nemotron steps run translate/nemo_curator -c default \
  backend=google \
  google.project_id=YOUR_PROJECT \
  google.api_version=v3 \
  google.location=global \
  input_path=/path/to/chat.jsonl \
  output_dir=/path/to/out \
  source_language=en \
  target_language=ja
```

## AWS Example Skeleton

```bash
uv run nemotron steps run translate/nemo_curator -c default \
  backend=aws \
  aws.region=us-west-2 \
  input_path=/path/to/chat.jsonl \
  output_dir=/path/to/out \
  source_language=en \
  target_language=es
```

## FAITH Reminder

Cloud backends still pair with the FAITH LLM client. Keep `server` populated whenever `faith_eval.enabled` is `true`.

## Related Pages

- YAML keys: {doc}`../reference/translate-config`
- FAITH tuning: {doc}`run-faith-evaluation`
