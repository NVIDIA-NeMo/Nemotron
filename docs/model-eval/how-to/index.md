<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(model-eval-how-to-index)=
# Model Evaluation How-To Guides

This section provides task-focused procedures for running `eval/model_eval`.
For your first run, start with {doc}`../getting-started`.
For agent-driven sessions, read {doc}`../using-skills` first.

## Choose A Guide

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`search;1.5em;sd-mr-1` Discover The Step
:link: discover-the-step
:link-type: doc
List the step, read its contract, and decide whether it applies to the task.
+++
{bdg-secondary}`discovery`
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Run A Hosted Evaluation
:link: run-hosted-evaluation
:link-type: doc
Run benchmarks against an already-running, OpenAI-compatible endpoint.
+++
{bdg-secondary}`hosted-endpoint`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Evaluate A Deployed Checkpoint
:link: evaluate-deployed-checkpoint
:link-type: doc
Choose a deployment path, deploy the endpoint, and point the step at it.
+++
{bdg-secondary}`deployment`
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

discover-the-step
run-hosted-evaluation
evaluate-deployed-checkpoint
```
