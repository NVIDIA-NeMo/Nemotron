---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Command-line reference for the peft/megatron_bridge training step."
topics: ["Training", "Reference", "CLI", "PEFT", "LoRA", "Megatron Bridge"]
tags: ["Reference", "CLI", "Steps", "PEFT", "LoRA", "Megatron-Bridge"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Developer"]
---

# peft/megatron_bridge

This step trains a low-rank adaptation (LoRA) adapter on top of a Megatron checkpoint by using NVIDIA Megatron-Bridge.
Use this step when a full supervised fine-tune does not fit in memory at the target model size, but you still need tensor and pipeline parallelism.
The step consumes packed Apache Parquet shards produced by `data_prep/sft_packing` together with a base Megatron checkpoint, and produces a `checkpoint_lora` artifact.

## Syntax

```bash
nemotron steps run peft/megatron_bridge \
    [-c <config-name-or-path>] \
    [-r <run-profile> | -b <batch-profile>] \
    [-d] \
    [--force-squash] \
    [<dotlist-overrides>...] \
    [<passthrough-args>...]
```

Refer to the [Nemotron Steps CLI Reference](../cli-reference.md) for the shared flag set.

## Configuration Files

The step ships two configuration files under `src/nemotron/steps/peft/megatron_bridge/config/`.

| File | Purpose |
| --- | --- |
| `default.yaml` | Full-shape LoRA tuning on top of the Nano3 Megatron-Bridge finetune recipe with rank thirty-two adapters on `linear_qkv` and `linear_proj`. |
| `tiny.yaml` | Short validation run against packed Parquet shards. |

Pass the configuration name with `-c`:

```console
$ nemotron steps run peft/megatron_bridge -c tiny
$ nemotron steps run peft/megatron_bridge -c default
```

## Inputs and Outputs

| Direction | Artifact Type | Required | Description |
| --- | --- | --- | --- |
| Consumes | `packed_parquet` | Yes | Packed Parquet shards with `input_ids` and `loss_mask` columns. Produce these shards with `data_prep/sft_packing` first. |
| Consumes | `checkpoint_megatron` | Yes | A pretrained Megatron checkpoint to adapt. |
| Produces | `checkpoint_lora` | — | LoRA adapter weights. Merge the adapter with the base checkpoint by using `convert/merge_lora` to obtain a deployable Hugging Face (HF) checkpoint. |

## Step Parameters

The manifest declares two LoRA-specific parameters.
Pass them as dotlist overrides.

```{option} peft.type=<scheme>

The parameter-efficient training scheme.
Only `lora` is supported today.

Choices: `lora`.

Default: `lora`.

Example: `peft.type=lora`
```

```{option} peft.dim=<n>

The LoRA rank.

Default: `32`.

Example: `peft.dim=16`
```

Frequently used dotlist overrides drawn from the Nano3 finetune recipe include the following.

```{option} recipe.seq_length=<n>

The training sequence length applied by the recipe.
This value must match the `pack_size` you used in `data_prep/sft_packing`.

Example: `recipe.seq_length=8192`
```

```{option} recipe.tensor_model_parallel_size=<n>

The tensor-model-parallel degree applied by the Nano3 finetune recipe.

Example: `recipe.tensor_model_parallel_size=8`
```

```{option} recipe.pipeline_model_parallel_size=<n>

The pipeline-model-parallel degree applied by the Nano3 finetune recipe.

Example: `recipe.pipeline_model_parallel_size=4`
```

```{option} train.train_iters=<n>

The number of training iterations.

Example: `train.train_iters=2000`
```

```{option} train.global_batch_size=<n>

The global batch size for the training loop.

Example: `train.global_batch_size=32`
```

```{option} dataset.nano3_packed_sft_dir=<path>

The directory that contains packed Parquet shards from `data_prep/sft_packing`.
The default configuration reads this value from the `SFT_PACKED_DIR` environment variable.

Example: `dataset.nano3_packed_sft_dir=/lustre/packed/super3-sft`
```

## Strategies

The manifest records one operator strategy for `peft/megatron_bridge`.

- When a full supervised fine-tune does not fit in memory at the desired model size, switch to `peft/megatron_bridge` to keep tensor and pipeline parallelism while reducing the trainable parameter count.

## Common Errors

```{option} missing_packed_data

Cause: the training loop cannot find packed Parquet shards at the configured `dataset.nano3_packed_sft_dir`.

Recovery: run `data_prep/sft_packing` first, or override `dataset.nano3_packed_sft_dir` to point at the directory that holds the packed splits.
```

## Command Examples

Run the tiny validation configuration locally:

```console
$ nemotron steps run peft/megatron_bridge -c tiny
```

Compile the default configuration without submitting the job:

```console
$ nemotron steps run peft/megatron_bridge -c default --dry-run
```

Submit an attached LoRA run on a Lepton profile with a longer sequence length:

```console
$ nemotron steps run peft/megatron_bridge -c default -r lepton_peft_megatron_bridge \
    peft.dim=16 \
    recipe.seq_length=8192 \
    train.train_iters=2000
```

Submit a detached LoRA run on a Slurm profile with eight-way tensor parallelism:

```console
$ nemotron steps run peft/megatron_bridge -c default -b slurm_peft_megatron_bridge \
    peft.dim=32 \
    recipe.tensor_model_parallel_size=8 \
    train.global_batch_size=64
```

## Related Skill

Run the `nemotron-peft-megatron-bridge` skill with your agent.

## Related Documentation

- [Nemotron Steps CLI Reference](../cli-reference.md) covers the shared option set, dotlist overrides, and passthrough arguments.
- [Choose a PEFT Backend](../../how-to/choose-peft-backend.md) compares `peft/megatron_bridge` and `peft/automodel`.
- [peft/automodel](automodel.md) documents the NeMo AutoModel LoRA step.

### Upstream

- [Megatron-Bridge Repository](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
- [Megatron-Bridge Documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/)
- [Megatron-Bridge PEFT Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/peft.html)
- [Megatron-Bridge Packed Sequences](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/packed-sequences.html)
