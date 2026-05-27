---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Command-line reference for the peft/automodel training step."
topics: ["Training", "Reference", "CLI", "PEFT", "LoRA", "AutoModel"]
tags: ["Reference", "CLI", "Steps", "PEFT", "LoRA", "AutoModel", "Hugging Face"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Developer"]
---

# peft/automodel

This step trains a low-rank adaptation (LoRA) adapter on top of a Hugging Face (HF) format base model by using the NeMo AutoModel library.
The training loop matches `sft/automodel`, with a LoRA adapter wired in by default to keep large base models practical for adapter tuning.
The step produces a `checkpoint_lora` artifact that you can merge with the base model by using the `convert/merge_lora` step.

## Syntax

```bash
nemotron steps run peft/automodel \
    [-c <config-name-or-path>] \
    [-r <run-profile> | -b <batch-profile>] \
    [-d] \
    [--force-squash] \
    [<dotlist-overrides>...] \
    [<passthrough-args>...]
```

Refer to the [Nemotron Steps CLI Reference](../cli-reference.md) for the shared flag set.

## Configuration Files

The step ships two configuration files under `src/nemotron/steps/peft/automodel/config/`.

| File | Purpose |
| --- | --- |
| `default.yaml` | Full-shape LoRA tuning on `Qwen/Qwen3-30B-A3B` with the AutoModel `TrainFinetuneRecipeForNextTokenPrediction` recipe. |
| `tiny.yaml` | Short validation run with a small dataset slice and a short training schedule. |

Pass the configuration name with `-c`:

```console
$ nemotron steps run peft/automodel -c tiny
$ nemotron steps run peft/automodel -c default
```

## Inputs and Outputs

| Direction | Artifact Type | Required | Description |
| --- | --- | --- | --- |
| Consumes | `training_jsonl` | Yes | Chat-format JSON Lines with a `messages` field. |
| Produces | `checkpoint_lora` | — | LoRA adapter weights. Merge the adapter with the base model by using `convert/merge_lora` to obtain a deployable Hugging Face checkpoint. |

## Supported Models

The manifest declares three reference base models.
Other Hugging Face causal language models that load through `AutoModelForCausalLM.from_pretrained` also work.

| Model | Minimum GPUs | Default | Notes |
| --- | --- | --- | --- |
| `Qwen/Qwen3-30B-A3B` | 8 | Yes | Mixture-of-experts (MoE) base model used by `config/default.yaml`. |
| `meta-llama/Llama-3.1-8B-Instruct` | 2 | No | Common dense baseline for single-node LoRA tuning. |
| `mistralai/Mistral-7B-Instruct-v0.3` | 1 | No | Strong default for single-GPU LoRA tuning. |

Override the base model from the command line:

```console
$ nemotron steps run peft/automodel -c default \
    model.pretrained_model_name_or_path=mistralai/Mistral-7B-Instruct-v0.3
```

## Step Parameters

The manifest declares two LoRA-specific parameters.
Pass them as dotlist overrides.

```{option} peft.dim=<n>

The LoRA rank.
Values in the eight-to-thirty-two range work well for most tasks.
Raise the rank when the downstream task is harder than the base task.

Default: `16`.

Example: `peft.dim=32`
```

```{option} peft.alpha=<n>

The LoRA alpha scaling factor.
A value equal to twice the rank works well in practice.

Default: `32`.

Example: `peft.alpha=64`
```

Frequently used dotlist overrides drawn from the AutoModel recipe include the following.

```{option} step_scheduler.max_steps=<n>

The maximum number of optimizer steps for the run.

Example: `step_scheduler.max_steps=200`
```

```{option} step_scheduler.global_batch_size=<n>

The global batch size across all data-parallel workers.

Example: `step_scheduler.global_batch_size=64`
```

```{option} dataset.path_or_dataset_id=<id-or-path>

The Hugging Face dataset identifier or a local path that resolves to a JSON Lines chat dataset.

Example: `dataset.path_or_dataset_id=/data/my-instructions.jsonl`
```

```{option} peft.dropout=<float>

The LoRA dropout rate applied during training.

Example: `peft.dropout=0.05`
```

## Strategies

The manifest records two operator strategies for `peft/automodel`.

- When the run targets a single GPU or memory is tight, keep `peft.dim` low, in the eight-to-sixteen range, and prefer a Mistral-class base model.
- When the adapter is intended for deployment, run `convert/merge_lora` after training to merge the adapter with the base model and produce a standalone Hugging Face checkpoint.

## Common Errors

```{option} oom

Cause: GPU memory is exhausted during forward or backward passes.

Recovery: reduce `peft.dim`, lower `step_scheduler.local_batch_size`, lower the maximum sequence length, or move to a smaller base model.
```

## Command Examples

Run the tiny validation configuration locally:

```console
$ nemotron steps run peft/automodel -c tiny
```

Compile the default configuration on a Lepton profile without submitting the job:

```console
$ nemotron steps run peft/automodel -c default -r lepton_peft_automodel --dry-run
```

Submit an attached LoRA run with a higher adapter rank:

```console
$ nemotron steps run peft/automodel -c default -r lepton_peft_automodel \
    peft.dim=32 \
    peft.alpha=64 \
    step_scheduler.max_steps=500
```

Submit a detached single-GPU LoRA run on a Slurm profile against a smaller base model:

```console
$ nemotron steps run peft/automodel -c default -b slurm_peft_automodel \
    model.pretrained_model_name_or_path=mistralai/Mistral-7B-Instruct-v0.3 \
    peft.dim=8 \
    peft.alpha=16
```

## Related Skill

Run the `nemotron-peft-automodel` skill with your agent.

## Related Documentation

- [Nemotron Steps CLI Reference](../cli-reference.md) covers the shared option set, dotlist overrides, and passthrough arguments.
- [Choose a PEFT Backend](../../how-to/choose-peft-backend.md) compares `peft/automodel` and `peft/megatron_bridge`.
- [peft/megatron_bridge](megatron-bridge.md) documents the Megatron-Bridge LoRA step.

### Upstream

- [NeMo AutoModel Repository](https://github.com/NVIDIA-NeMo/Automodel)
- [NeMo AutoModel Documentation](https://docs.nvidia.com/nemo/automodel/latest/index.html)
- [NeMo AutoModel PEFT Guide](https://docs.nvidia.com/nemo/automodel/latest/guides/llm/finetune.html)
- [AutoModel Training Script](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/llm/train_ft.py)
