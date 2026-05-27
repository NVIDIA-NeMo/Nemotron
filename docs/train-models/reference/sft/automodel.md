---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Command-line reference for the sft/automodel training step."
topics: ["Training", "Reference", "CLI", "SFT", "AutoModel"]
tags: ["Reference", "CLI", "Steps", "SFT", "AutoModel", "Hugging Face"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Developer"]
---

# sft/automodel

This step runs supervised fine-tuning (SFT) on a Hugging Face (HF) format model from a JSON Lines (JSONL) chat dataset, using the NeMo AutoModel library.
The same step supports full fine-tuning and low-rank adaptation (LoRA) tuning, controlled by the `peft` parameter.

## Syntax

```bash
nemotron steps run sft/automodel \
    [-c <config-name-or-path>] \
    [-r <run-profile> | -b <batch-profile>] \
    [-d] \
    [--force-squash] \
    [<dotlist-overrides>...] \
    [<passthrough-args>...]
```

Refer to the [Nemotron Steps CLI Reference](../cli-reference.md) for the shared flag set.

## Configuration Files

The step ships two configuration files under `src/nemotron/steps/sft/automodel/config/`.

| File | Purpose |
| --- | --- |
| `default.yaml` | Full-shape training on `Qwen/Qwen3-30B-A3B` with the AutoModel `TrainFinetuneRecipeForNextTokenPrediction` recipe. |
| `tiny.yaml` | Short validation run with five training steps and a 64-record dataset slice from `HuggingFaceH4/ultrachat_200k`. |

Pass either name with `-c`:

```console
$ nemotron steps run sft/automodel -c tiny
$ nemotron steps run sft/automodel -c default
```

## Inputs and Outputs

| Direction | Artifact Type | Required | Description |
| --- | --- | --- | --- |
| Consumes | `training_jsonl` | Yes | Instruction data in JSONL with a `messages` field in OpenAI chat format. |
| Produces | `checkpoint_hf` | — | Hugging Face checkpoint directory. The output is a full model checkpoint when `peft=null`, and a LoRA adapter directory when `peft=lora`. |

## Supported Models

The manifest declares three reference base models.
Other Hugging Face causal language models that load through `AutoModelForCausalLM.from_pretrained` also work.

| Model | Minimum GPUs | Default | Notes |
| --- | --- | --- | --- |
| `Qwen/Qwen3-30B-A3B` | 8 | Yes | Mixture-of-experts (MoE) base model used by `config/default.yaml`. |
| `meta-llama/Llama-3.1-8B-Instruct` | 4 | No | Common dense baseline for single-node SFT and LoRA. |
| `mistralai/Mistral-7B-Instruct-v0.3` | 2 | No | Use this model when GPU count is small or iteration speed matters. |

Override the base model from the command line:

```console
$ nemotron steps run sft/automodel -c default \
    model.pretrained_model_name_or_path=meta-llama/Llama-3.1-8B-Instruct
```

## Step Parameters

The manifest declares one Nemotron-specific parameter.
Pass it as a dotlist override after the options.

```{option} peft=VALUE

Selects adapter-style training instead of full fine-tuning.
Set this value to `lora` to train a LoRA adapter, or to `null` for full SFT.

Choices: `lora`, `null`.

Default: `null`.

Example: `peft=lora`
```

You can also override any key from the compiled YAML configuration.
The frequently used keys include the following.

```{option} step_scheduler.max_steps=N

The maximum number of optimizer steps for the run.

Example: `step_scheduler.max_steps=200`
```

```{option} step_scheduler.global_batch_size=N

The global batch size across all data-parallel workers.

Example: `step_scheduler.global_batch_size=64`
```

```{option} dataset.path_or_dataset_id=<id-or-path>

The Hugging Face dataset identifier or a local path that resolves to a JSONL chat dataset.

Example: `dataset.path_or_dataset_id=/data/my-instructions.jsonl`
```

```{option} checkpoint.checkpoint_dir=PATH

The directory where the AutoModel recipe writes checkpoints.

Example: `checkpoint.checkpoint_dir=/output/qwen-sft`
```

## Strategies

The manifest records four operator strategies for `sft/automodel`.

- When the run has one or two GPUs, or memory is tight, set `peft=lora` and start from a Mistral-class model.
- When the run has three or four GPUs and the chosen model fits comfortably, consider full fine-tuning only when the resulting checkpoint size and iteration speed remain acceptable.
- When the dataset already uses OpenAI chat-format JSONL, skip `data_prep/sft_packing` and train directly from `training_jsonl`.
- When you need an immediately deployable HF checkpoint, keep the safetensors save format and the consolidated HF output layout that `config/default.yaml` sets.

## Common Errors

```{option} chat_template_missing

Cause: the tokenizer for the chosen model does not include a chat template, so the AutoModel recipe cannot render the `messages` field into prompt-and-completion form.

Recovery: choose a tokenizer with chat-template support, or convert the data to prompt-and-completion format before training.
```

```{option} oom

Cause: GPU memory is exhausted during forward or backward passes.

Recovery: set `peft=lora`, reduce `step_scheduler.global_batch_size`, or move to a smaller model.
```

## Command Examples

Run the tiny validation configuration locally:

```console
$ nemotron steps run sft/automodel -c tiny
```

Compile the default configuration on a Lepton profile without submitting the job:

```console
$ nemotron steps run sft/automodel -c default -r lepton_sft_automodel --dry-run
```

Submit an attached LoRA run on a Lepton profile with a smaller base model:

```console
$ nemotron steps run sft/automodel -c default -r lepton_sft_automodel \
    peft=lora \
    model.pretrained_model_name_or_path=mistralai/Mistral-7B-Instruct-v0.3 \
    step_scheduler.max_steps=500
```

Submit a detached full SFT on Slurm and write checkpoints to a shared scratch directory:

```console
$ nemotron steps run sft/automodel -c default -b slurm_sft_automodel \
    peft=null \
    step_scheduler.global_batch_size=64 \
    checkpoint.checkpoint_dir=/lustre/runs/qwen-sft
```

## Related Skill

Run the `nemotron-sft-automodel` skill with your agent.

## Related Documentation

- [Nemotron Steps CLI Reference](../cli-reference.md) covers the shared option set, dotlist overrides, and passthrough arguments.
- [Choose an SFT Backend](../../how-to/choose-sft-backend.md) compares `sft/automodel` to `sft/megatron_bridge`.
- [Configuration Conventions](../config-conventions.md) describes the per-step `config/` layout.

### Upstream

- [NeMo AutoModel Repository](https://github.com/NVIDIA-NeMo/Automodel)
- [NeMo AutoModel Documentation](https://docs.nvidia.com/nemo/automodel/latest/index.html)
- [NeMo AutoModel Fine-Tuning Guide](https://docs.nvidia.com/nemo/automodel/latest/guides/llm/finetune.html)
- [AutoModel Training Script](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/llm/train_ft.py)
