<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

(steps-getting-started)=
# Getting Started With Steps

This guide walks through the basic CLI operations that you use to discover steps, inspect their inputs and outputs, and chain them into pipelines.
You do not need a cluster or a GPU to follow along.
Every command in this guide reads metadata only.

If you have not read [Nemotron Steps Basics](basics.md), start there for the definitions of *step*, *configuration*, *environment profile*, and *artifact*.

## Prerequisites

- Python 3.10 or later.
- [uv](https://docs.astral.sh/uv/) installed and on your path.

These prerequisites cover the basics for an introduction to using the steps.
For each activity, such as model training or data generation, refer to the getting started page of each activity for additional prerequisites.

## Getting Access to the Nemotron CLI

1. Clone the repository, if you haven't already:

   ```console
   $ git clone https://github.com/NVIDIA-NeMo/Nemotron && cd Nemotron
   ```

   Run all commands from the repository root.

1. Synchronize the common dependencies:

   ```console
   $ uv sync
   ```

1. Confirm you can run the CLI:

   ```console
   $ uv run nemotron steps --help
   ```

The output lists the step-catalog subcommands that this guide uses: `list`, `show`, and `run`.

## List the Available Steps

Use `nemotron steps list` to see every step that the CLI discovers.
The output is a table with the step identifier, the category, the consumed artifact types, the produced artifact types, and a short description.

```console
$ uv run nemotron steps list
```

Each row in the table is one building block that you can run, inspect, or chain into a pipeline.

### Filter by Category

Pass `--category` to narrow the list to one family of steps.
For example, the following command shows only the supervised fine-tuning steps.

```console
$ uv run nemotron steps list --category sft
```

The category matches the top-level folder under `src/nemotron/steps/`.
Common categories include `sft`, `peft`, `rl`, `pretrain`, `optimize`, `convert`, `data_prep`, `sdg`, `translate`, `curate`, `byob`, `eval`, and `env`.

### Emit Machine-Readable JSON

Pass `--json` to emit a JSON array instead of a table.
The JSON output is the same data the table renders, with no formatting noise, so it works well with `jq` and with agent tooling.

```console
$ uv run nemotron steps list --json
$ uv run nemotron steps list --category sft --json | jq '.[].id'
```

The `--json` flag is available on both `nemotron steps list` and `nemotron steps show`.

## Inspect a Single Step

Use `nemotron steps show <id>` to print the manifest for one step.
The output includes the description, the consumed and produced artifacts with their types and descriptions, the parameters with their defaults and choices, and the run specification, which covers the launcher, the container image, the resource shape, and the default configuration.

```console
$ uv run nemotron steps show sft/automodel
```

The same command in JSON form is the easiest way to read the full manifest programmatically.

```console
$ uv run nemotron steps show sft/automodel --json
```

The JSON document includes a `consumes` array and a `produces` array.
Each entry has a `type`, a `required` flag, and a `description`.
These two arrays are the contract that connects one step to the next.

## Chain Steps Through Artifacts

Steps fit together because their consumed and produced artifact types match.
Use the `--produces` and `--consumes` filters on `nemotron steps list` to walk the building-block graph from either direction.

To find every step that writes a `training_jsonl` artifact, run the following command.

```console
$ uv run nemotron steps list --produces training_jsonl
```

The output lists the synthetic data generation and data preparation steps that emit training JSONL.

To find every step that reads a `training_jsonl` artifact, run the following command.

```console
$ uv run nemotron steps list --consumes training_jsonl
```

The output lists the fine-tuning and reinforcement-learning (RL) steps that take training JSONL as input.

Together, the two queries show you which steps you can chain.
For example, the `sdg/data_designer` step produces `training_jsonl`, and the `sft/automodel` step consumes `training_jsonl`, so you can pipe the synthetic dataset directly into supervised fine-tuning without an intermediate conversion.

The same pattern works for other artifact types, such as `packed_parquet`, `binidx`, `checkpoint_hf`, `checkpoint_megatron`, `checkpoint_lora`, `synthetic_jsonl`, `eval_results`, and `mcq_benchmark_parquet`.

## Where To Go Next

Pick the domain section that matches the work you have in mind.

- [Synthetic Data Generation](../sdg/index.md) to generate chat data, tool-calling data, or preference pairs.
- [Translation](../translation/index.md) to translate corpora with optional faithfulness, accuracy, integrity, and translation-quality holistic (FAITH) scoring.
- [Build MCQ Benchmarks](../build-benchmarks/index.md) to generate a multiple-choice question benchmark from your own documents.
- [Model Training](../train-models/index.md) to fine-tune, align, or optimize a model.
- [Model Evaluation](../model-eval/index.md) to score a deployed checkpoint with standard benchmarks.
