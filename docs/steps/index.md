<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# About Nemotron Steps

A Nemotron *step* is a named, reusable unit of work that you invoke with the `nemotron steps` CLI.
Each step declares the artifacts it consumes, the artifacts it produces, and a set of named configurations that you can run on your laptop, on a single node, or on a cluster.
Steps are the building blocks of every Nemotron pipeline.

This section is the entry point for the step model itself.
Use it to learn what a step is, to explore the available steps from the CLI, and to find the right domain section for the work you have in mind.

## The Basics

- [Nemotron Steps Basics](basics.md) defines *step*, *configuration*, *environment profile*, and *artifact*. Start here if you have not run a step before.
- [Getting Started With Steps](getting-started.md) shows how to list the available steps, inspect their inputs and outputs, and chain steps together.

## Building Block Steps

Pipelines are modular.
You can run a single step in isolation, and you can compose steps into longer flows.
The sections below group the available steps by the outcome they support.
Follow each link for tutorials, how-to guides, concepts, and reference material in that domain.

### [Synthetic Data Generation](../sdg/index.md)

Generate supervised fine-tuning (SFT) chat data, tool-calling data, or preference pairs with NeMo Data Designer.
Backed by the `sdg/data_designer` step.

### [Translation](../translation/index.md)

Translate JSON Lines or Apache Parquet corpora with NeMo Curator, with optional faithfulness, accuracy, integrity, and translation-quality holistic (FAITH) scoring.
Backed by the `translate/nemo_curator` step.

### [Data Curation and Preparation](../curate/index.md)

Filter raw text with `curate/nemo_curator`, then tokenize and shard it with the `data_prep/pretrain_prep`, `data_prep/sft_packing`, and `data_prep/rl_prep` steps.
Use the curation docs for JSONL filtering and the training docs for data preparation.

### [Multiple-Choice Question Benchmarks](../build-benchmarks/index.md)

Generate a custom multiple-choice question (MCQ) benchmark from your own documents, with optional translation.
Backed by the `byob/mcq` step.

### [Model Training](../train-models/index.md)

Pretrain, fine-tune, align, and optimize models with the `pretrain/`, `sft/`, `peft/`, `rl/`, `optimize/`, and `convert/` step families.

### [Model Evaluation](../model-eval/index.md)

Score a trained checkpoint on standard benchmarks with NeMo Evaluator.
Backed by the `eval/model_eval` step.

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
| Generate multiple-choice SFT data from personas | `sdg/persona_mcq` — no guide yet; see `nemotron steps show sdg/persona_mcq` and the step's README |
| Extend a tokenizer for a new language | `tokenizer_extension/*` — no guide yet; see `src/nemotron/steps/tokenizer_extension/guide.md` |
