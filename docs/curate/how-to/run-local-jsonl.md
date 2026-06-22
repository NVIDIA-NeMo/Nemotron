---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Run curate/nemo_curator on local JSONL files."
topics: ["Curation", "How-To", "JSONL"]
tags: ["How-To", "Curation", "JSONL"]
content:
  type: "How-To"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Data Scientist"]
---

# Run Curation on Local JSONL

Use this path when your corpus already exists as local JSONL files.

## Input Requirements

Each input record must contain the configured `text_field`.
The default field name is `text`.

Example record:

```json
{"id": "doc-001", "text": "The text to keep, filter, or route downstream."}
```

## Minimal Local Run

Start with optional filters disabled.
This verifies the reader and writer path before adding model-backed filters.

```console
$ uv sync --extra curate
$ export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0

$ uv run --no-sync nemotron steps run curate/nemo_curator -c tiny \
    input_glob="${PWD}/data/my_corpus/**/*.jsonl" \
    output_dir="${PWD}/output/curated-jsonl" \
    text_field=text \
    language_codes=[] \
    domains=[] \
    quality_filters={}
```

Use an absolute path or a path relative to the repository root.
When running on a remote executor, make sure the path exists inside the container or shared mount.

## Add CPU Resources

For local runs, you can set Ray CPU count in YAML or as a CLI override:

```console
$ uv run --no-sync nemotron steps run curate/nemo_curator -c tiny \
    input_glob="${PWD}/data/my_corpus/**/*.jsonl" \
    output_dir="${PWD}/output/curated-jsonl" \
    ray.num_cpus=4
```

For generated Lepton profiles, `NEMOTRON_CURATOR_RAY_NUM_CPUS` can provide the CPU count when `ray.num_cpus` is omitted.

## Validate Output

After the run:

- Confirm that output shards exist under `output_dir`.
- Count records before and after filtering.
- Inspect a few output records to confirm the `text_field` is present and not empty.

If output is empty, run again with `language_codes=[]`, `domains=[]`, and `quality_filters={}` before enabling filters.
