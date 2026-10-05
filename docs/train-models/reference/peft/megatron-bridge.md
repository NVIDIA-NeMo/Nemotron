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
uv run nemotron steps run peft/megatron_bridge \
    [-c <config-name-or-path>] \
    [-r <run-profile> | -b <batch-profile>] \
    [-d] \
    [--force-squash] \
    [<dotlist-overrides>...] \
    [<passthrough-args>...]
```

Refer to the [Nemotron Steps CLI Reference](../cli-reference.md) for the shared flag set.

## Configuration Files

The step ships three configuration files under `src/nemotron/steps/peft/megatron_bridge/config/`.

| File | Purpose |
| --- | --- |
| `default.yaml` | Full-shape LoRA tuning on top of the Nano3 Megatron-Bridge finetune recipe with rank thirty-two adapters on `linear_qkv` and `linear_proj`. |
| `tiny.yaml` | Short validation run against packed Parquet shards. |
| `lightning35.yaml` | Overlay on `default.yaml` (`defaults: default.yaml`) for Nemotron 3.5 Lightning 30B-A3B packed LoRA at 4096 tokens in the `nemo:26.08` container: `nemotron_3_5_lightning_peft_config` recipe, adapters on `linear_qkv`, `linear_proj`, `linear_fc1`, `linear_fc2`, `in_proj`, and `out_proj`, TP 2, EP 8, 100 iterations at `train.global_batch_size: 128`, learning rate `1.0e-4`. Refer to [Nemotron 3.5 Lightning](#nemotron-35-lightning). |

Pass the configuration name with `-c`:

```console
$ uv run nemotron steps run peft/megatron_bridge -c tiny
$ uv run nemotron steps run peft/megatron_bridge -c default
```

### Nemotron 3.5 Lightning

The frozen base for `lightning35.yaml` must be a Megatron checkpoint.
Convert the Hugging Face model first with [`convert/hf_to_megatron -c lightning35`](../convert/hf-to-megatron.md), then set the following environment variables in the shell or the `env.toml` profile:

| Variable | Consumed as |
| --- | --- |
| `L35_PRETRAINED_CHECKPOINT` | `checkpoint.pretrained_checkpoint` (`${L35_PRETRAINED_CHECKPOINT}/iter_0000000`) |
| `L35_PACKED_DIR` | `dataset.packed_sequence_specs.packed_train_data_path` and `packed_val_data_path` (`${L35_PACKED_DIR}/splits/{train,valid}`) |
| `L35_OUTPUT_DIR` | `checkpoint.save` (`${L35_OUTPUT_DIR}/lora-4k-tp2-ep8`) |

The overlay sets the inherited Nano recipe arguments `recipe.packed_sequence`, `recipe.peft`, and `recipe.seq_length` to `null` and selects the adapter through `recipe.peft_scheme: lora`, because the Lightning PEFT recipe is a different callable from the Nano recipe.
The inherited `peft.dim` and `peft.alpha` values from `default.yaml` still apply.

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
- When the operator selects Nemotron 3.5 Lightning, convert the base model with `convert/hf_to_megatron -c lightning35` first and run `-c lightning35`; adapters cannot load Hugging Face weights.

## Common Errors

```{option} missing_packed_data

Cause: the training loop cannot find packed Parquet shards at the configured `dataset.nano3_packed_sft_dir`.

Recovery: run `data_prep/sft_packing` first, or override `dataset.nano3_packed_sft_dir` to point at the directory that holds the packed splits.
```

## Command Examples

Run the tiny validation configuration locally:

```console
$ uv run nemotron steps run peft/megatron_bridge -c tiny
```

Compile the default configuration without submitting the job:

```console
$ uv run nemotron steps run peft/megatron_bridge -c default --dry-run
```

Submit an attached LoRA run on a Lepton profile with a longer sequence length:

```console
$ uv run nemotron steps run peft/megatron_bridge -c default -r lepton_peft_megatron_bridge \
    peft.dim=16 \
    recipe.seq_length=8192 \
    train.train_iters=2000
```

Submit a detached LoRA run on a Slurm profile with eight-way tensor parallelism:

```console
$ uv run nemotron steps run peft/megatron_bridge -c default -b slurm_peft_megatron_bridge \
    peft.dim=32 \
    recipe.tensor_model_parallel_size=8 \
    train.global_batch_size=64
```

Submit a detached Nemotron 3.5 Lightning LoRA run from a converted Megatron checkpoint:

```console
$ L35_PRETRAINED_CHECKPOINT=/lustre/checkpoints/lightning35-megatron \
    L35_PACKED_DIR=/lustre/packed/lightning35 \
    L35_OUTPUT_DIR=/lustre/runs/lightning35 \
    uv run nemotron steps run peft/megatron_bridge -c lightning35 -b <batch-profile>
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
