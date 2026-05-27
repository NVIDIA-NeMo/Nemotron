---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Explains which training library backs each Nemotron step, and how the choice of library determines data format, checkpoint layout, and parallelism conventions."
topics: ["Training", "Explanation", "Concepts"]
tags: ["Training Libraries", "NeMo AutoModel", "Megatron-Bridge", "NeMo RL", "Model Optimizer"]
content:
  type: "Explanation"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Developer"]
---

# Training Libraries

Each Nemotron training step delegates to a specific *training library*.
The library determines how the step loads models, how it reads data, and the native checkpoint layout it writes.
Different libraries impose different data formats, checkpoint formats, and parallelism conventions, so the choice of library shapes every step downstream of it.
The Nemotron training pipeline draws on an ecosystem of four libraries, described in the sections that follow.

## NeMo AutoModel

The steps `sft/automodel` and `peft/automodel` use the NeMo AutoModel library for training centered on Hugging Face conventions.
The expected input is JSON Lines (JSONL) files in OpenAI chat format, read directly without an intermediate packing step.
The output is a Hugging Face checkpoint directory that any Hugging Face inference runtime can load as is.

## Megatron-Bridge

The steps `sft/megatron_bridge` and `peft/megatron_bridge` use the Megatron-Bridge library for large distributed training runs.
The expected input is packed Apache Parquet produced by a separate packing step.
The output is a Megatron-format checkpoint, sharded to match the parallelism settings the run used.
A separate conversion step turns the Megatron output into a Hugging Face checkpoint when the next consumer expects that layout.

## NeMo RL

The steps under `rl/nemo_rl/` delegate to the NeMo RL library for reinforcement learning alignment algorithms.
Every NeMo RL step requires a supervised fine-tuned policy checkpoint in Megatron format as the warm start.
Different alignment algorithms have different reward sources and different dataset shapes, so the consumed artifact types vary per step.

## NVIDIA Model Optimizer

The steps under `optimize/modelopt/` call the NVIDIA Model Optimizer library, which runs alongside Megatron-Bridge for export.
*Quantization* lowers the numerical precision of model weights so the model uses less memory and runs faster on the target hardware.
*Pruning* removes weights or whole structural units to shrink the model.
*Distillation* transfers quality from a larger teacher checkpoint to a smaller student checkpoint.

## Choosing a Library

Use the relevant how-to guide to map your supervised fine-tuning (SFT), parameter-efficient fine-tuning (PEFT), or reinforcement learning (RL) requirements to a library.
Treat your library choice as long-term.
Switching from one library to another requires conversion steps and a new round of performance tuning, not a single configuration flag change.

## Related Reading

- [Training Basics](basics.md) defines the training approaches and the artifact types each library consumes and produces.
- [Choose an SFT Backend](../how-to/choose-sft-backend.md) compares the two SFT libraries side by side.
- [Execution through NeMo Run](../../nemo_runspec/nemo-run.md) describes how each library is launched at runtime.
