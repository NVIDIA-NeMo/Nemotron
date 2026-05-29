---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Concept pages for nemotron eval/model_eval: pipeline architecture, endpoint and benchmark families, and tokenizer alignment."
topics: ["Model Evaluation", "Concepts"]
tags: ["Explanation", "Model Evaluation"]
content:
  type: "Explanation"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

# Concepts

The pages in this section cover the design rules behind `eval/model_eval`.
Read them when you want to understand why the how-to pages take the actions they do, before you change a configuration default or adapt a recipe to a new deployment.

```{toctree}
:maxdepth: 1
:hidden:

pipeline-overview
endpoint-types-and-benchmarks
tokenizer-alignment
```

## Pipeline And Architecture

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Pipeline Overview
:link: pipeline-overview
:link-type: doc
Artifact flow from a checkpoint or hosted endpoint, through `eval/model_eval`, into `eval_results` on disk.
+++
{bdg-secondary}`architecture`
:::

::::

## Deployment Contract

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Endpoint Types And Benchmark Families
:link: endpoint-types-and-benchmarks
:link-type: doc
Chat versus completions endpoints, and which benchmark families match each one.
+++
{bdg-secondary}`endpoint`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Tokenizer Alignment
:link: tokenizer-alignment
:link-type: doc
Why log-probability benchmarks need a tokenizer that matches the served model.
+++
{bdg-secondary}`tokenizer`
:::

::::
