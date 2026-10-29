---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "CLI reference for the curate steps and the curate flow driver."
topics: ["Curation", "Reference", "CLI"]
tags: ["Reference", "CLI", "Curation"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Data Scientist"]
---

# Curation CLI

The curation category registers six steps, each run through `nemotron steps run`, and one flow driver, run as a Python module.

| Command | Purpose | Reference |
| --- | --- | --- |
| `nemotron steps run curate/ingest` | Normalize Parquet or JSONL and mint document identifiers | {doc}`ingest` |
| `nemotron steps run curate/profile` | Measure quality-signal distributions and write candidate policies | {doc}`profile` |
| `nemotron steps run curate/nemo_curator` | Apply language, length, and domain gates and an approved policy | {doc}`curate-config` |
| `nemotron steps run curate/audit` | Verify a curated corpus against its manifest and ledger | {doc}`audit` |
| `nemotron steps run curate/decontamination` | Remove training documents that overlap a holdout | {doc}`decontamination` |
| `nemotron steps run curate/subset` | Cut nested token-budget tiers | {doc}`subset` |
| `python -m nemotron.steps.curate.nemo_curator.scripts.run_flow` | Run the six steps from one configuration | {doc}`flow-config` |
| `python -m nemotron.steps.curate.nemo_curator.scripts.run_evaluate` | Score an approved policy against labelled documents | {doc}`../how-to/evaluate-a-policy` |

Install the dependencies once with `uv sync --extra curate`.
The decontamination similarity pass additionally needs `--extra curate-gpu` and a GPU.

## curate/nemo_curator Syntax

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
- `mode=both`
- `heuristic_filters.approved_policy=./output/vi/policy/approved_policy.yaml`

Use shell quoting around globs or lists when your shell expands them unexpectedly.

## Flow Driver Syntax

```bash
uv run --extra curate --extra xenna \
  python -m nemotron.steps.curate.nemo_curator.scripts.run_flow \
  --config <path/to/flow.yaml> [--plan]
```

The flow driver takes a configuration path and accepts no dotlist overrides; copy a configuration file and edit it.
`--plan` writes `flow_plan.json` and exits without running any step.
Refer to {doc}`flow-config`.

## Policy Evaluation Syntax

```bash
uv run --extra curate \
  python -m nemotron.steps.curate.nemo_curator.scripts.run_evaluate \
  --policy <approved_policy.yaml> \
  --labelled <labelled.jsonl> [<labelled.jsonl> ...] \
  [--language <bcp47>] [--langpack-dir <dir>] [--report <report.json>]
```

| Flag | Meaning |
| --- | --- |
| `--policy` | Required. The approved policy whose thresholds are evaluated. |
| `--labelled` | Required. One or more JSONL files of labelled documents. |
| `--language` | BCP-47 tag. Defaults to the policy's `langpack` block. |
| `--langpack-dir` | Pack root. Defaults to the policy's `langpack` block. |
| `--report` | Write the full report as JSON to this path. |

Refer to {doc}`../how-to/evaluate-a-policy`.
