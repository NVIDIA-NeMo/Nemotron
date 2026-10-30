<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# How-To Guides

Task-focused guides for `nemotron steps run byob/mcq` with the `mcq` family.

Start with {doc}`../getting-started` if you have not produced `benchmark.parquet` yet.

```{toctree}
:maxdepth: 1
:hidden:

Prepare Data <prepare-data>
Use Your Domain Data <domain-data>
Model Endpoints <custom-model-endpoints>
Tune Prompts <prompt-tuning>
Skip Stages <skip-stages>
```

## Setup and configuration

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`file-directory;1.5em;sd-mr-1` Prepare your data
:link: prepare-data
:link-type: doc
Lay out `input_dir`, text or Parquet inputs, and `target_source_mapping`.
+++
{bdg-secondary}`input_dir`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Domain corpus files
:link: domain-data
:link-type: doc
Create per-target directories of `.txt` files and match them to YAML.
+++
{bdg-secondary}`corpus`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Model endpoints
:link: custom-model-endpoints
:link-type: doc
Configure OpenAI-compatible providers for generation, judgement, expansion, validity, and filters.
+++
{bdg-secondary}`yaml`
:::

::::

## Advanced workflows

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`comment;1.5em;sd-mr-1` Prompt tuning
:link: prompt-tuning
:link-type: doc
Point `prompt_config` at a YAML file that defines stage templates.
+++
{bdg-secondary}`prompt`
:::

:::{grid-item-card} {octicon}`sync;1.5em;sd-mr-1` Skip stages
:link: skip-stages
:link-type: doc
Resume with `skip_until` and cached Parquet files.
+++
{bdg-secondary}`iteration`
:::

::::

## Workflow overview

```{mermaid}
flowchart LR
    A[Prepare data layout] --> B[Edit YAML]
    B --> C[uv run nemotron steps run byob/mcq]
    C --> D{Need translation?}
    D -->|yes| E[translate config + passthrough]
    D -->|no| F[Done]
    E --> F
    C -.->|iterate| G[skip_until]
    G --> C
```

## Related documentation

- {doc}`../explanation/index` — how each stage behaves.
- {doc}`../reference/index` — full field lists and allowed datasets.
