---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Task guides for nemotron steps run translate/nemo_curator backends and configuration."
topics: ["Translation", "How-To"]
tags: ["How-To", "Translation"]
content:
  type: "How-To"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

# How-To Guides

This section has task-focused procedures for changing backends, wiring fields, tuning segmentation, and adjusting FAITH.

For copy-paste prompts and habits when you work with a coding agent, read {doc}`../using-skills` first.
Start with {doc}`../getting-started` if you have not run the step yet.

```{toctree}
:maxdepth: 1
:hidden:

run-llm-translation
run-nmt-translation
run-google-aws-translation
configure-fields-and-output
use-fine-segmentation
run-faith-evaluation
```

Focused procedures for `nemotron steps run translate/nemo_curator`.

## Run Translation

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`cpu;1.5em;sd-mr-1` LLM backend
:link: run-llm-translation
:link-type: doc
OpenAI-compatible servers, hosted or on-prem.
+++
{bdg-secondary}`backend=llm`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` NMT HTTP service
:link: run-nmt-translation
:link-type: doc
Self-hosted `POST /translate` microservices.
+++
{bdg-secondary}`backend=nmt`
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` Google or AWS
:link: run-google-aws-translation
:link-type: doc
Managed cloud translation APIs.
+++
{bdg-secondary}`backend=google|aws`
:::

::::

## Configure and Tune

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`file-directory;1.5em;sd-mr-1` Fields and outputs
:link: configure-fields-and-output
:link-type: doc
Wildcards, `output_mode`, chat reconstruction.
+++
{bdg-secondary}`schema`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Segmentation
:link: use-fine-segmentation
:link-type: doc
Switch `segmentation_mode` deliberately.
+++
{bdg-secondary}`segmentation`
:::

::::

## FAITH Quality Gates

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` FAITH evaluation
:link: run-faith-evaluation
:link-type: doc
Thresholds, filtering, model overrides.
+++
{bdg-secondary}`faith`
:::

::::

```{mermaid}
graph LR
    A[Prepare YAML + env] --> B[nemotron steps run translate/nemo_curator]
    B --> C{Need FAITH?}
    C -->|yes| D[Tune faith_eval]
    C -->|no| E[Disable faith_eval]
    D --> B
    E --> B
```
