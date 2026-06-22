---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "CLI reference for nemotron steps run curate/nemo_curator."
topics: ["Curation", "Reference", "CLI"]
tags: ["Reference", "CLI", "Curation"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Data Scientist"]
---

# curate/nemo_curator CLI

## Syntax

```bash
uv run --no-sync nemotron steps run curate/nemo_curator \
    [-c <config-name-or-path>] \
    [-r <run-profile> | -b <batch-profile>] \
    [-d] \
    [<dotlist-overrides>...]
```

Use `-c tiny` for a small initial validation configuration and `-c default` for the Hugging Face snapshot example.
Refer to [Nemotron Steps CLI Reference](../../train-models/reference/cli-reference.md) for the shared flag set.

## Common Commands

Show the step contract:

```console
$ uv run --no-sync nemotron steps show curate/nemo_curator
```

Run a local JSONL initial validation:

```console
$ uv run --no-sync nemotron steps run curate/nemo_curator -c tiny \
    input_glob="${PWD}/src/nemotron/steps/curate/nemo_curator/data/tiny.jsonl" \
    output_dir="${PWD}/output/curate-tiny"
```

Run on Lepton with the generated Curator profile:

```console
$ uv run --no-sync nemotron steps run curate/nemo_curator -c tiny --batch lepton_curate
```

Run against local corpus shards:

```console
$ uv run --no-sync nemotron steps run curate/nemo_curator -c tiny \
    input_glob="${PWD}/data/my_corpus/**/*.jsonl" \
    output_dir="${PWD}/output/curated-jsonl" \
    text_field=text \
    language_codes=[] \
    domains=[] \
    quality_filters={}
```

## Dotlist Overrides

All YAML fields can be overridden from the command line with `key=value` syntax.
Examples:

- `input_glob=/data/**/*.jsonl`
- `output_dir=/output/curated`
- `text_field=body`
- `language_codes=[EN]`
- `quality_filters.min_words=50`
- `quality_filters.max_words=5000`
- `ray.num_cpus=4`

Use shell quoting around globs or lists when your shell expands them unexpectedly.
