<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(train-models-explanation-index)=
# Training Concepts

This section provides conceptual material for training with Nemotron steps.
{doc}`basics` introduces fine-tuning approaches, tokenizers, the chat dataset format, and checkpoint layouts.
For the step, configuration, and environment profile model that every Nemotron command shares, see {doc}`../../steps/basics`.
The remaining pages explain how artifacts relate to one another and how the training libraries differ.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Training Basics
:link: basics
:link-type: doc
Defines supervised fine-tuning, parameter-efficient fine-tuning, reinforcement learning alignment, quantization, tokenizers, the chat dataset format, and checkpoints.
+++
{bdg-secondary}`concepts`
:::

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Artifact Graph
:link: artifact-graph
:link-type: doc
Explains how steps declare typed inputs and outputs and what common training and alignment paths look like.
+++
{bdg-secondary}`concepts`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Training Libraries
:link: training-libraries
:link-type: doc
Explains which library backs each step and how that choice determines data format, checkpoint layout, and parallelism.
+++
{bdg-secondary}`concepts`
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

basics
artifact-graph
training-libraries
```
