---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Concept pages for nemotron steps run translate/nemo_curator: pipeline flow, segmentation, FAITH."
topics: ["Translation", "Concepts"]
tags: ["Explanation", "Translation"]
content:
  type: "Explanation"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Data Scientist"]
---

# Concepts

Background on how the translation pipeline fits together, how segmentation behaves, and what FAITH measures. Use these pages when how-to steps refer to behavior you want to understand before changing defaults.

```{toctree}
:maxdepth: 1
:hidden:

pipeline-overview
segmentation
faith-evaluation
```

Concept-focused explanations for `nemotron steps run translate/nemo_curator` and the `translate/nemo_curator` Curator pipeline.

## Pipeline Processing

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Pipeline overview
:link: pipeline-overview
:link-type: doc
Reader to `TranslationStage` to writer, with an optional FAITH branch.
+++
{bdg-secondary}`architecture`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Segmentation
:link: segmentation
:link-type: doc
Coarse versus fine `segmentation_mode` trade-offs.
+++
{bdg-secondary}`segmentation`
:::

::::

## Evaluation Inside Translation

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` FAITH evaluation
:link: faith-evaluation
:link-type: doc
What FAITH captures when `faith_eval.enabled` is true.
+++
{bdg-secondary}`faith`
:::

::::
