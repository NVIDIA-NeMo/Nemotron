---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Configuration field reference for curate/nemo_curator."
topics: ["Curation", "Reference", "Configuration"]
tags: ["Reference", "Configuration", "Curation"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Data Scientist"]
---

# curate/nemo_curator Configuration

The step reads YAML from `src/nemotron/steps/curate/nemo_curator/config/`.

| File | Purpose |
| --- | --- |
| `tiny.yaml` | Curator-container initial validation configuration. Optional filters are disabled. Override `input_glob` for local runs because the checked-in path is a container path. |
| `default.yaml` | Example Hugging Face snapshot workflow for FineWeb-Edu-style JSONL with language and word-count filters enabled. |

## Top-Level Fields

```{option} input_glob

JSONL file path or glob passed to NeMo Curator `JsonlReader`.
```

```{option} output_dir

Directory where NeMo Curator writes JSONL output shards.
```

```{option} text_field

Record field containing the text to curate.

Default: `text`.
```

```{option} dataset

Optional keyword arguments passed to `huggingface_hub.snapshot_download`.
Set `dataset: null` for local-input-only runs.

Common keys are `repo_id`, `repo_type`, `local_dir`, and `allow_patterns`.
```

```{option} language_codes

Uppercase language codes to keep.
Set `language_codes: []` to skip FastText language identification and language filtering.
When this list is non-empty, `models.fasttext_langid` must point at a FastText language identification model.
```

```{option} domains

Domains to keep through NeMo Curator `MultilingualDomainClassifier`.
Set `domains: []` to skip domain classification.
```

```{option} quality_filters

Optional quality settings.
`min_langid_score` applies when language filtering is enabled.
`min_words` and `max_words` enable word-count filtering and must be set together.
Set `quality_filters: {}` to skip word-count filtering.
```

```{option} models

Optional model and cache paths.

Common keys:

- Set `fasttext_langid` to the path of the FastText language identification model.
- Set `hf_cache_dir` to the Hugging Face model cache directory for classifier assets.
```

```{option} ray.num_cpus

Optional Ray CPU count.
If omitted, the Lepton curate profile can provide `NEMOTRON_CURATOR_RAY_NUM_CPUS`.
```

## Minimal Local Configuration

```yaml
language_codes: []
domains: []
text_field: text
input_glob: ./data/**/*.jsonl
output_dir: ./output/curated-jsonl
dataset: null
models: {}
quality_filters: {}
```

## Filtered Configuration

```yaml
language_codes:
  - EN
domains: []
text_field: text
input_glob: ./data/**/*.jsonl
output_dir: ./output/curated-jsonl
dataset: null
models:
  fasttext_langid: ./cache/models/fasttext/lid.176.bin
  hf_cache_dir: ./cache/huggingface
quality_filters:
  min_langid_score: 0.3
  min_words: 50
  max_words: 5000
```
