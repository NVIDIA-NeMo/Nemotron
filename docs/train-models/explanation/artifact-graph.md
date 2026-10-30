---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Explains how training steps declare typed inputs and outputs, what the common training and alignment paths look like, and why tokenizer alignment matters across a pipeline."
topics: ["Training", "Explanation", "Concepts"]
tags: ["Artifact Graph", "Pipeline", "JSONL", "Checkpoint", "Tokenizer"]
content:
  type: "Explanation"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Developer"]
---

# Artifact Graph

For definitions of *artifact*, *step*, and how the CLI's `--produces` and `--consumes` filters let you walk the artifact graph, see [Nemotron Steps Basics](../../steps/basics.md).
This page focuses on the artifact types and common chains specific to model training.

## Common Training Paths

The supervised fine-tuning paths in the Nemotron pipeline follow one of the following two chains.

- The Hugging Face line used by `sft/automodel`: `training_jsonl` → `sft/automodel` → `checkpoint_hf`.
- The packed Megatron line used by `sft/megatron_bridge`: `training_jsonl` → packing prep → `packed_parquet` → `sft/megatron_bridge` → `checkpoint_megatron`.

A typical alignment path starts from a `checkpoint_megatron` policy, adds preference or reward-side data, runs one of the `rl/nemo_rl/...` steps, and produces a new `checkpoint_megatron`.

A typical compression path starts from `checkpoint_hf`, runs `optimize/modelopt/quantize`, and produces `checkpoint_megatron`.
Add a conversion step after quantization when the next consumer needs a Hugging Face layout again.

## Tokenizer and Chat Template Consistency

Matching artifact types is not enough for correctness.
The tokenizer, the chat template, and the maximum sequence length must stay consistent across every step that tokenizes text or loads weights for the same model line.
A mismatch often appears as a plausible training loss curve with poor downstream quality.

## Related Reading

- [Nemotron Steps Basics](../../steps/basics.md) defines the concepts of *step*, *configuration*, *environment profile*, and *artifact*.
- [Training Basics](basics.md) defines the training-specific terms, including tokenizers and chat templates.
- [Data and Checkpoint Formats](../how-to/data-and-checkpoint-formats.md) describes the on-disk layouts of each artifact type.
- [Training Libraries](training-libraries.md) describes the ecosystem of libraries that back the steps.
