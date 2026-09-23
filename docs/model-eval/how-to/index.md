<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(model-eval-how-to-index)=
# Model Evaluation How-To Guides

This section provides task-focused procedures for running `eval/model_eval`.
For your first run, start with {doc}`../getting-started`.
For agent-driven sessions, read {doc}`../using-skills` first.
If you have not chosen between launcher mode and direct mode, read {ref}`model-eval-choose-a-mode`.

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

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Run A Direct-Mode Evaluation
:link: run-direct-mode-evaluation
:link-type: doc
Run the harness inside the step's own job, on any Nemotron backend, against an endpoint you host.
+++
{bdg-secondary}`direct-mode`
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Run A Benchmark Suite
:link: run-a-benchmark-suite
:link-type: doc
Choose a shipped suite for a base or instruct model, pin a language or subset, and select tasks with `-t`.
+++
{bdg-secondary}`direct-mode` {bdg-secondary}`suites`
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Run A Hosted Evaluation (Launcher Mode)
:link: run-hosted-evaluation
:link-type: doc
Run launcher-mode benchmarks against an already-running, OpenAI-compatible endpoint with `tiny_chat.yaml`.
+++
{bdg-secondary}`launcher-mode`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Evaluate A Deployed Checkpoint (Launcher Mode)
:link: evaluate-deployed-checkpoint
:link-type: doc
Choose a launcher deployment path, deploy the endpoint, and point the step at it.
+++
{bdg-secondary}`launcher-mode`
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

discover-the-step
run-direct-mode-evaluation
run-a-benchmark-suite
run-hosted-evaluation
evaluate-deployed-checkpoint
```
