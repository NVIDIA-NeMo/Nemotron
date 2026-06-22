---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Plain-language introduction to fine-tuning approaches, tokenizers, datasets, and checkpoints. For the step, configuration, and environment profile model, see Nemotron Steps Basics."
topics: ["Training", "Explanation", "Concepts"]
tags: ["Beginner", "Explanation", "SFT", "PEFT", "RL", "Quantization", "Tokenizer", "Chat Template", "JSONL", "Checkpoint"]
content:
  type: "Explanation"
  difficulty: "Beginner"
  audience: ["ML Engineer", "Developer"]
---

# Training Basics

This page defines the training-specific terms that the rest of the training documentation uses.
You do not need to read it to run a command, but every other page in this section assumes that you already understand these basics.
For the step, configuration, and environment profile model that every Nemotron command shares, see [Nemotron Steps Basics](../../steps/basics.md).

## Training Approaches

The Nemotron training pipeline packages four common training approaches.
This section defines each one in plain language.

### Supervised Fine-Tuning

*Supervised fine-tuning* (SFT) trains a base model on a dataset of instruction-and-response examples.
The training loop reads each example, asks the model to predict the desired response, and updates every weight in the model to lower the prediction error.
You start an SFT run from a base model checkpoint and a dataset of chat-formatted examples.
You finish with a fine-tuned checkpoint that prefers the response style your dataset modeled.
SFT is the usual first step after base pretraining when you adapt a model to a new task or domain.

### Parameter-Efficient Fine-Tuning

*Parameter-efficient fine-tuning* (PEFT) is fine-tuning that updates only a small number of new weights, called *adapter weights*, while keeping the base model weights frozen.
The most common form in the Nemotron pipeline is *low-rank adaptation* (LoRA), which inserts small matrices alongside the existing weight matrices and trains only those small matrices.
PEFT runs cost a fraction of full SFT in GPU memory and produce a small adapter artifact instead of a full model.
You can serve a PEFT adapter on top of the base model at inference time, or you can merge the adapter with the base model after training to produce a single deployable checkpoint.

### Reinforcement Learning Alignment

*Reinforcement learning* (RL) alignment is a follow-on training stage that takes a supervised fine-tuned model and improves it against a reward signal instead of fixed responses.
The model proposes its own responses, a scoring mechanism rates each one, and the training loop nudges the model toward higher-rated responses.
Different RL methods use different scoring mechanisms, but they all depend on a working SFT checkpoint to start from and a quality-controlled prompt or preference dataset.

### Quantization

*Quantization* lowers the numerical precision used to store model weights, for example, from sixteen-bit floating point to eight-bit floating point.
The quantized model uses less memory and runs faster on the hardware that supports the lower-precision math.
Quality drops a small amount in exchange.
You apply quantization after the model is fully trained and you have committed to a deployment target.
The choice of quantization recipe, such as `fp8` or `nvfp4`, depends on the target hardware.

## Tokenizers and Chat Templates

A *tokenizer* converts text strings into the integer identifiers the model consumes, and converts the model's output identifiers back into text.
Each model family ships with its own tokenizer.
A run that uses one tokenizer to prepare data and a different tokenizer to train will fail outright, or, worse, will train against meaningless input.
Keep the tokenizer aligned across every stage that touches the same model.

A *chat template* is a small text-substitution recipe that the tokenizer applies to a multi-turn conversation before tokenization.
It inserts role markers such as `user` and `assistant`, any special start-of-turn and end-of-turn tokens, and any fixed system instructions.
The template ships with the tokenizer.
You do not write a chat template by hand for ordinary fine-tuning.
You do need to confirm that the conversation format of your dataset matches what the chosen tokenizer's template expects.

## The Messages JSON Lines Format

Most SFT and PEFT steps in this documentation consume chat-formatted *JSON Lines* (JSONL) datasets.
JSONL is one independent JSON object per line, with no enclosing array.
Each object has a single field named `messages` that holds an ordered list of conversation turns.
Each turn has a `role` of `system`, `user`, or `assistant`, and a `content` field with the turn's text.

The following is one JSONL record covering a system instruction, a user question, and the assistant response.

```{literalinclude} ../_snippets/input/sample-messages.jsonl
:language: json
```

A real dataset has one such line for every training example.
The training loop reads the file line by line, applies the model's chat template to each `messages` list, and treats the assistant turns as the prediction targets.

## Checkpoints

A *checkpoint* is a saved snapshot of a model's weights.
A checkpoint contains every numeric parameter the model uses to make predictions, in a serialization format the training and inference code can read back.
Most training steps in the Nemotron pipeline produce one of the following three checkpoint layouts.

| Layout | When You Encounter It | What It Contains |
| --- | --- | --- |
| Hugging Face checkpoint | Output of SFT runs that use the NeMo AutoModel library. | A directory of safetensors files plus tokenizer files. The checkpoint is deployable as is to any Hugging Face inference runtime. |
| Megatron checkpoint | Output of Megatron-Bridge SFT runs, of RL runs, and of quantization runs. | A directory of distributed shards that match the parallelism settings of the training job. The checkpoint requires conversion to Hugging Face format before deployment to common inference runtimes. |
| LoRA adapter checkpoint | Output of every PEFT run. | A directory containing only the adapter weights. The adapter requires either the base model alongside it at inference time, or a merge step that combines the two into a deployable single-model checkpoint. |

Two ground rules help avoid trouble.
Always keep the original base checkpoint together with any adapter you train against it.
Always record the tokenizer version that produced the data so the inference runtime picks up the same tokenizer at serving time.

## Where To Go Next

- [Nemotron Steps Basics](../../steps/basics.md) defines the *step*, *configuration*, and *environment profile* model that every training command uses.
- [Getting Started With Training Steps](../getting-started.md) walks through a first run.
- [Reference](../reference/index.md) lists every step, configuration, and command-line option.
