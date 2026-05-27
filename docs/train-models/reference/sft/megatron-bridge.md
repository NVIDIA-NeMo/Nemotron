---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Command-line reference for the sft/megatron_bridge training step."
topics: ["Training", "Reference", "CLI", "SFT", "Megatron Bridge"]
tags: ["Reference", "CLI", "Steps", "SFT", "Megatron-Bridge", "Distributed Training"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Developer"]
---

# sft/megatron_bridge

This step runs supervised fine-tuning (SFT) on a Megatron checkpoint by using NVIDIA Megatron-Bridge.
It supports tensor, pipeline, and context parallelism for large-scale distributed training of the Nemotron model family.
The step consumes packed Apache Parquet shards produced by `data_prep/sft_packing`.

## Syntax

```bash
nemotron steps run sft/megatron_bridge \
    [-c <config-name-or-path>] \
    [-r <run-profile> | -b <batch-profile>] \
    [-d] \
    [--force-squash] \
    [<dotlist-overrides>...] \
    [<passthrough-args>...]
```

See the [Nemotron Steps CLI Reference](../cli-reference.md) for the shared flag set.

## Configuration Files

The step ships three configuration files under `src/nemotron/steps/sft/megatron_bridge/config/`.

| File | Purpose |
| --- | --- |
| `default.yaml` | Two-node Slurm functional-test configuration. Loads base weights from Hugging Face via AutoBridge with LoRA enabled (`peft: lora`). Not the programmatic default. |
| `nano3.yaml` | Production full-SFT configuration for the Nano3 model (`peft: null`, 1700 training iterations). This is the programmatic default loaded when no `-c` flag is specified. |
| `tiny.yaml` | Short validation run against packed Parquet shards on a two-node Lepton profile. |

Pass the configuration name with `-c`:

```console
$ nemotron steps run sft/megatron_bridge -c tiny
$ nemotron steps run sft/megatron_bridge -c default
```

## Inputs and Outputs

| Direction | Artifact Type | Required | Description |
| --- | --- | --- | --- |
| Consumes | `packed_parquet` | Yes | Packed SFT Parquet shards with `input_ids` and `loss_mask` columns. Produce these shards with `data_prep/sft_packing` first. |
| Consumes | `checkpoint_megatron` | No | A pretrained Megatron checkpoint or a prior Megatron SFT checkpoint. When this input is absent, the step loads weights from the Hugging Face model declared by `hf_model_path`. |
| Produces | `checkpoint_megatron` | — | A fine-tuned Megatron distributed checkpoint. |

## Supported Models

| Model | Minimum GPUs | Default | Notes |
| --- | --- | --- | --- |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | 8 | Yes | Nemotron 3 Nano with 31.6 billion total and 3.2 billion active parameters. This model is the default Nano3 path. |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | 32 | No | Nemotron 3 Super with 120.6 billion total and 12.7 billion active parameters. Typical runs start at 32 GPUs. |

## Step Parameters

The manifest declares two Nemotron-specific parameters.
Pass them as dotlist overrides.

```{option} seq_length=N

The training sequence length.
This value must match the `pack_size` you used in `data_prep/sft_packing`.

Choices: `2048`, `4096`, `8192`, `16384`, `32768`.

Default: `4096`.

Example: `seq_length=8192`
```

```{option} peft=VALUE

Selects low-rank adaptation (LoRA) tuning instead of full SFT.
Set this value to `lora` for adapter tuning, or to `null` for full fine-tuning when the model and optimizer states fit in memory.

Choices: `lora`, `null`.

Default: `null` (as set in `nano3.yaml`, the programmatic default config). The `default.yaml` functional-test config sets this to `lora`.

Example: `peft=null`
```

Frequently used dotlist overrides drawn from the underlying recipe include the following.

```{option} hf_model_path=<id-or-path>

The Hugging Face identifier or local path used to load base weights through `AutoBridge` when no Megatron checkpoint is supplied.

Example: `hf_model_path=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`
```

```{option} recipe.tensor_model_parallel_size=N

The tensor-model-parallel degree applied by the Nano3 finetune recipe.

Example: `recipe.tensor_model_parallel_size=8`
```

```{option} recipe.pipeline_model_parallel_size=N

The pipeline-model-parallel degree applied by the Nano3 finetune recipe.

Example: `recipe.pipeline_model_parallel_size=4`
```

```{option} train.global_batch_size=N

The global batch size for the training loop.

Example: `train.global_batch_size=128`
```

```{option} checkpoint.save=PATH

The directory where the Megatron-Bridge recipe writes checkpoints.

Example: `checkpoint.save=/lustre/runs/nano3-sft/checkpoints`
```

## Strategies

The manifest records the following operator strategies for `sft/megatron_bridge`.

- When the dataset has fewer than ten thousand records, lower `train.global_batch_size` and raise the number of training iterations to keep optimizer steps useful.
- When the operator wants LoRA tuning, set `peft=lora` to lower the GPU requirement and shrink the checkpoint footprint.
- When the operator selects the Super3 model, start from a 32-GPU plan with `tp=8`, `pp=4`, `cp=1`, and verify cluster topology before scaling further.
- When `seq_length > 32768`, enable hybrid context parallelism.
- When GPU memory is tight, such as on A100 40 GB hardware, enable activation checkpointing and consider central-processing-unit (CPU) offloading.
- When you want maximum throughput on H100 hardware, keep packed sequences enabled and tune overlap and sequence-packing settings before scaling up.

## Common Errors

```{option} tokenizer_mismatch

Cause: the tokenizer used during `data_prep/sft_packing` differs from the tokenizer used for training, so token identifiers do not align.

Recovery: set the `data_prep/sft_packing` tokenizer to match the training model and regenerate the packed Parquet shards.
```

```{option} oom

Cause: GPU memory is exhausted during forward, backward, or optimizer steps.

Recovery: reduce `train.global_batch_size`, increase parallelism, or reduce `seq_length`.
```

```{option} missing_packed_data

Cause: the training loop cannot find packed Parquet shards at the configured `dataset.nano3_packed_sft_dir`.

Recovery: run `data_prep/sft_packing` first, or override `dataset.nano3_packed_sft_dir` to point at the directory that holds the packed splits.
```

## Command Examples

Run the tiny validation configuration on the two-node Lepton SFT profile:

```console
$ nemotron steps run sft/megatron_bridge -c tiny -r lepton_sft_megatron_bridge
```

Compile the default configuration without submitting the job:

```console
$ nemotron steps run sft/megatron_bridge -c default --dry-run
```

Submit a detached LoRA run on Slurm with a longer sequence length:

```console
$ nemotron steps run sft/megatron_bridge -c default -b slurm_sft_megatron_bridge \
    peft=lora \
    seq_length=8192 \
    train.global_batch_size=256
```

Submit an attached run on the Super3 base model with eight-way tensor parallelism and four-way pipeline parallelism:

```console
$ nemotron steps run sft/megatron_bridge -c default -r lepton_sft_megatron_bridge \
    hf_model_path=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    recipe.tensor_model_parallel_size=8 \
    recipe.pipeline_model_parallel_size=4
```

## Related Skill

Run the `nemotron-sft-megatron-bridge` skill with your agent.

## Related Documentation

- [Nemotron Steps CLI Reference](../cli-reference.md) covers the shared option set, dotlist overrides, and passthrough arguments.
- [Choose an SFT Backend](../../how-to/choose-sft-backend.md) compares `sft/megatron_bridge` to `sft/automodel`.
- [Configuration Conventions](../config-conventions.md) describes the per-step `config/` layout.

### Upstream

- [Megatron-Bridge Repository](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
- [Megatron-Bridge Documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/)
- [Megatron-Bridge Training Entry Points](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/entry-points.html)
- [Megatron-Bridge Packed Sequences](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/packed-sequences.html)
