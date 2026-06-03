<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(steps-index)=
# About Nemotron Steps

A Nemotron *step* is a named, reusable unit of work that you invoke with the `nemotron steps` CLI.
Each step declares the artifacts it consumes, the artifacts it produces, and a set of named configurations that you can run on your laptop, on a single node, or on a cluster.
Steps are the building blocks of every Nemotron pipeline.

This section is the entry point for the step model itself.
Use it to learn what a step is, to explore the available steps from the CLI, and to find the right domain section for the work you have in mind.

## The Basics

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Nemotron Steps Basics
:link: basics
:link-type: doc

Definitions of *step*, *configuration*, *environment profile*, and *artifact*.
Start here if you have not run a step before.
+++
`Concepts`
:::

:::{grid-item-card} Getting Started With Steps
:link: getting-started
:link-type: doc

List the available steps, inspect their inputs and outputs, and chain steps together.
+++
`Beginner`
:::

::::

## Building Block Steps

Pipelines are modular.
You can run a single step in isolation, and you can compose steps into longer flows.
The cards below group the available steps by the outcome they support.
Follow the link in each card for tutorials, how-to guides, concepts, and reference material in that domain.

::::::{grid} 1 1 3 3
:gutter: 2

:::::{grid-item}

::::{grid} 1
:gutter: 1

:::{grid-item-card} Synthetic Data Generation
:link: ../sdg/index
:link-type: doc

Build your own dataset
^^^

Generate supervised fine-tuning (SFT) chat data, tool-calling data, or preference pairs with NeMo Data Designer.
Backed by the `sdg/data_designer` step.
:::

:::{grid-item-card} Translation
:link: ../translation/index
:link-type: doc

Translate JSON Lines or Apache Parquet corpora with NeMo Curator, with optional faithfulness, accuracy, integrity, and translation-quality holistic (FAITH) scoring.
Backed by the `translate/nemo_curator` step.
:::

:::{grid-item-card} Data Curation and Preparation
:link: ../curate/index
:link-type: doc

Filter raw text with `curate/nemo_curator`, then tokenize and shard it with the `data_prep/pretrain_prep`, `data_prep/sft_packing`, and `data_prep/rl_prep` steps.
Use the curation docs for JSONL filtering and the training docs for data preparation.
:::

::::

:::::

:::::{grid-item}

::::{grid} 1
:gutter: 1

:::{grid-item-card} Multiple-Choice Question Benchmarks
:link: ../build-benchmarks/index
:link-type: doc

Build your own benchmarks
^^^

Generate a custom multiple-choice question (MCQ) benchmark from your own documents, with optional translation.
Backed by the `byob` step.
:::

::::

:::::

:::::{grid-item}

::::{grid} 1
:gutter: 1

:::{grid-item-card} Model Training
:link: ../train-models/index
:link-type: doc

Build your own models
^^^

Pretrain, fine-tune, align, and optimize models with the `pretrain/`, `sft/`, `peft/`, `rl/`, `optimize/`, and `convert/` step families.
:::

:::{grid-item-card} Model Evaluation
:link: ../model-eval/index
:link-type: doc

Score a trained checkpoint on standard benchmarks with NeMo Evaluator.
Backed by the `eval/model_eval` step.
:::

::::

:::::

::::::

## Shared Infrastructure

Every remote run depends on an *environment profile* that describes the cluster, the container image, the resource shape, and the mount points.
The `env/env_toml` step generates these profile files from compact YAML templates for Lepton or Slurm.
The Basics page covers profiles and the `env/env_toml` step in detail.

## I Want To

| Goal | Go To |
| --- | --- |
| Learn what a step, configuration, and profile are | [Nemotron Steps Basics](basics.md) |
| List the available steps from the CLI | [Getting Started With Steps](getting-started.md) |
| Run steps in an airgap environment | [Airgap](airgap.md) |
| Curate JSONL text | [Data Curation](../curate/index.md) |
| Generate synthetic training data | [Synthetic Data Generation](../sdg/index.md) |
| Translate a corpus | [Translation](../translation/index.md) |
| Build an MCQ benchmark | [Build MCQ Benchmarks](../build-benchmarks/index.md) |
| Fine-tune or align a model | [Model Training](../train-models/index.md) |
| Evaluate a model | [Model Evaluation](../model-eval/index.md) |
| Set up a Lepton or Slurm environment profile | [Nemotron Steps Basics](basics.md) |
