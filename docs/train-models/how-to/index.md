<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(train-models-how-to-index)=
# Model Training How-To Guides

This section provides task-focused guides for common training workflows.
For your first run, start with {doc}`../getting-started`.

If you are new to fine-tuning concepts, read {doc}`../explanation/index` for definitions of fine-tuning approaches, data formats, and checkpoints before working through these tasks.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Run SFT with AutoModel on Custom Data
:link: run-sft-automodel
:link-type: doc
Configure `tiny.yaml` for your own JSONL data, run the step, verify the checkpoint output, and resolve common issues.
+++
{bdg-success}`30 min` {bdg-secondary}`beginner`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Choose an SFT Backend
:link: choose-sft-backend
:link-type: doc
Pick between `sft/automodel` and `sft/megatron_bridge` based on checkpoint format and scale.
+++
{bdg-secondary}`intermediate`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Choose a PEFT Backend
:link: choose-peft-backend
:link-type: doc
Pick between `peft/automodel` and `peft/megatron_bridge` based on base checkpoint format and data path.
+++
{bdg-secondary}`intermediate`
:::

:::{grid-item-card} {octicon}`milestone;1.5em;sd-mr-1` Choose an RL Alignment Step
:link: choose-rl-step
:link-type: doc
Pick between DPO, RLVR, and RLHF based on how the reward signal enters training.
+++
{bdg-secondary}`intermediate`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Run Post-Training Optimization
:link: run-optimization
:link-type: doc
Apply quantization, pruning, or distillation with Model Optimizer after a trained model passes your quality bar.
+++
{bdg-secondary}`intermediate`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Environment Profiles and Executors
:link: env-and-executors
:link-type: doc
Configure NeMo Run environment profiles for local, Slurm, and Lepton execution.
+++
{bdg-secondary}`intermediate`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Data and Checkpoint Formats
:link: data-and-checkpoint-formats
:link-type: doc
Understand the JSONL, Parquet, and checkpoint types that training steps declare in `step.toml`.
+++
{bdg-secondary}`intermediate`
:::

:::{grid-item-card} {octicon}`git-compare;1.5em;sd-mr-1` Convert Checkpoints Between Training Steps
:link: convert-checkpoints
:link-type: doc
Run a conversion step when one step produces a checkpoint layout that the next step cannot consume directly.
+++
{bdg-secondary}`intermediate`
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

Run SFT with AutoModel on Custom Data <run-sft-automodel>
Choose an SFT Backend <choose-sft-backend>
Choose a PEFT Backend <choose-peft-backend>
Choose an RL Alignment Step <choose-rl-step>
Run Post-Training Optimization <run-optimization>
Environment Profiles and Executors <env-and-executors>
Data and Checkpoint Formats <data-and-checkpoint-formats>
Convert Checkpoints <convert-checkpoints>
```
