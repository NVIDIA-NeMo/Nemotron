<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Run SFT with AutoModel on Custom Data

This guide walks through configuring and running the `sft/automodel` step with your own instruction data.
The step trains models in a Hugging Face checkpoint layout from OpenAI chat-formatted JSON Lines (JSONL).

Before following this guide, complete [Getting Started with Training Steps](../getting-started.md) to verify your environment profile and confirm that the sample job runs to completion.

## Prerequisites

- A completed `env.toml` at the repository root with a training profile such as `lepton_sft_automodel`.
  See [Getting Started with Training Steps](../getting-started.md) for the environment snippet and how to generate the full profile file.
- `HF_TOKEN`, `WANDB_API_KEY`, and `NVIDIA_API_KEY` exported in your shell.
- Instruction data in JSONL form where each record includes a `messages` field in OpenAI chat format.
- A Hugging Face access token if the base model is gated or must be downloaded from the Hugging Face Hub.
- Enough GPU memory for the model you select.
  Verify memory requirements before scaling beyond a tiny configuration.

## Configure the Step

1. Open `src/nemotron/steps/sft/automodel/config/tiny.yaml`.
   The checked-in defaults pull small public data slices from the Hugging Face Hub.
   Replace the `dataset` and `model` fields with your paths and model identifier.

2. Align the tokenizer and chat template with how your JSONL was built.
   The step applies the tokenizer at training time, so a mismatch between the template used during data preparation and the template applied here causes silent quality degradation or a hard failure at startup.
   See [Data and Checkpoint Formats](data-and-checkpoint-formats.md) for the canonical field names the step expects.

## Run the Step

Submit the step against your environment profile:

```console
$ uv run nemotron steps run sft/automodel -c tiny -r lepton_sft_automodel
```

Replace `lepton_sft_automodel` with the profile name from your `env.toml` when your team uses a different table key.
See [Execution through NeMo Run](../../nemo_runspec/nemo-run.md) for profile setup and scheduler behavior.

## Verify Output

The step manifest declares a `checkpoint_hf` artifact on success.
Confirm the output directory you set in the training configuration contains checkpoints, and that logs show stable loss for the duration of the run.

To inspect what the step produces, run:

```console
$ uv run nemotron steps show sft/automodel
```

The `produces` block lists `checkpoint_hf` and the path it writes to.

## Common Issues

- When you encounter CUDA out-of-memory errors, reduce batch sizes in YAML or switch to a smaller base model.
- When the trainer reports a missing chat template, pick a tokenizer that defines a template or convert your data to a format the trainer accepts.
  The `step.toml` file lists `[[errors]]` entries such as `chat_template_missing` with recovery hints.
- When you need to checkpoint to a specific directory, set `SFT_OUTPUT_DIR` in your shell before running or pass the `checkpoint.checkpoint_dir` override on the command line.

## Next Steps

- Read [Choose an SFT Backend](choose-sft-backend.md) when you need Megatron Bridge and packed Parquet instead of AutoModel and JSONL.
- Read [Data and Checkpoint Formats](data-and-checkpoint-formats.md) to understand how JSONL and checkpoints chain with other steps.
- Read [Convert Checkpoints Between Training Steps](convert-checkpoints.md) when the next step in your pipeline requires a different checkpoint layout.
