---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Use the dataset block in curate/nemo_curator to materialize a Hugging Face dataset snapshot before curation."
topics: ["Curation", "How-To", "Hugging Face"]
tags: ["How-To", "Curation", "Hugging Face"]
content:
  type: "How-To"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Use a Hugging Face Snapshot

Set the `dataset` block when the curation job should call `huggingface_hub.snapshot_download` before NeMo Curator reads files.

## Configuration Shape

```yaml
dataset:
  repo_id: HuggingFaceFW/fineweb-edu
  repo_type: dataset
  local_dir: ./data/fineweb-edu
  allow_patterns:
    - data/*.jsonl

input_glob: ./data/fineweb-edu/**/*.jsonl
```

The `dataset` block is passed to `snapshot_download`.
After the download finishes, `input_glob` must point at JSONL files under `dataset.local_dir`.

## Run With the Default Configuration

The default config demonstrates a FineWeb-Edu-style snapshot.
It also enables language filtering, so you must provide the FastText language identification model path if you keep `language_codes` non-empty.

```console
$ uv sync --extra curate
$ export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

$ uv run --no-sync nemotron steps run curate/nemo_curator -c default \
    dataset.local_dir="${PWD}/data/fineweb-edu" \
    input_glob="${PWD}/data/fineweb-edu/**/*.jsonl" \
    output_dir="${PWD}/output/fineweb-edu-curated" \
    models.fasttext_langid="${PWD}/cache/models/fasttext/lid.176.bin"
```

## Snapshot Without Optional Filters

For a first infrastructure run, disable filters and verify that snapshot download and JSONL IO work.

```console
$ uv run --no-sync nemotron steps run curate/nemo_curator -c default \
    dataset.local_dir="${PWD}/data/fineweb-edu" \
    input_glob="${PWD}/data/fineweb-edu/**/*.jsonl" \
    output_dir="${PWD}/output/fineweb-edu-curated" \
    language_codes=[] \
    domains=[] \
    quality_filters={}
```

## Private or Gated Datasets

If the Hugging Face repository requires authentication, export `HF_TOKEN` before running.
For remote jobs, pass `HF_TOKEN` through the environment profile.

```console
$ export HF_TOKEN="<hugging-face-token>"
```
