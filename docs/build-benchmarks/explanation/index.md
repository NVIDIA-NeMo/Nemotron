<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Concepts

These pages explain how the `mcq` family inside `src/nemotron/steps/byob` prepares data, runs each generation stage, and optionally translates benchmarks.

```{toctree}
:maxdepth: 2
:hidden:

pipeline-overview
Data Preparation <data-preparation>
Get the Right Questions <get-right-questions>
question-generation
quality-validation
filtering
translation
```

## Architecture

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Pipeline overview
:link: pipeline-overview
:link-type: doc
Prepare, generate, translate, and the Parquet stage cache.
+++
{bdg-secondary}`stages`
:::

::::

## Core processes

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Data preparation
:link: data-preparation
:link-type: doc
Seeds from Hugging Face plus local corpus chunks.
+++
{bdg-secondary}`few-shot`
:::

:::{grid-item-card} {octicon}`cross-reference;1.5em;sd-mr-1` Mapping targets to sources
:link: get-right-questions
:link-type: doc
`source_subjects`, weights, and optional tags.
+++
{bdg-secondary}`target_source_mapping`
:::

:::{grid-item-card} {octicon}`sparkle-fill;1.5em;sd-mr-1` Question generation
:link: question-generation
:link-type: doc
Data Designer batched calls from prepared seeds.
+++
{bdg-secondary}`generation`
:::

::::

## Quality assurance

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Validation stack
:link: quality-validation
:link-type: doc
Judgement, deduplication, distractors, coverage, outliers.
+++
{bdg-secondary}`validation`
:::

:::{grid-item-card} {octicon}`filter;1.5em;sd-mr-1` Filtering
:link: filtering
:link-type: doc
Easiness and hallucination scores with removal flags.
+++
{bdg-secondary}`filtering`
:::

::::

## Translation

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Translation
:link: translation
:link-type: doc
Curator translation, backtranslation, metrics, final schema.
+++
{bdg-secondary}`translate`
:::

::::

## Next steps

- Hands-on first run: {doc}`../getting-started`
- YAML tables: {doc}`../reference/index`
