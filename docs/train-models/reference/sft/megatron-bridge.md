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
It also ships two experimental long-context topology references for Nemotron 3 Super that are intended for `--dry-run` planning only.

## Syntax

```bash
uv run nemotron steps run sft/megatron_bridge \
    [-c <config-name-or-path>] \
    [-r <run-profile> | -b <batch-profile>] \
    [-d] \
    [--force-squash] \
    [<dotlist-overrides>...] \
    [<passthrough-args>...]
```

See the [Nemotron Steps CLI Reference](../cli-reference.md) for the shared flag set.

## Configuration Files

The step ships five configuration files under `src/nemotron/steps/sft/megatron_bridge/config/`.

| File | Purpose |
| --- | --- |
| `default.yaml` | Generic Nano3 example for a remote cluster profile. Loads base weights from Hugging Face via AutoBridge, trains for 10 iterations at `seq_length: 4096` on packed shards, and inherits the recipe's LoRA default. This is the programmatic default loaded when no `-c` flag is specified. |
| `tiny.yaml` | Short full-SFT validation run (`peft: null`) against packed Parquet shards. |
| `lightning35.yaml` | Overlay on `default.yaml` (`defaults: default.yaml`) for Nemotron 3.5 Lightning 30B-A3B packed SFT at 4096 tokens in the `nemo:26.08` container: `nemotron_3_5_lightning_sft_config` recipe, TP 2, selective recomputation, 100 iterations at `train.global_batch_size: 128`. It does not load Hugging Face weights; refer to [Nemotron 3.5 Lightning](#nemotron-35-lightning). |
| `super3_128k.yaml` | Experimental topology reference for full-parameter Nemotron 3 Super SFT at 131,072 tokens: 64 ranks (8 nodes × 8 GPUs), TP 8, PP 1, CP 8, EP 64. Not runnable with the stock step; refer to [Long-Context Topology References](#long-context-topology-references). |
| `super3_256k.yaml` | Experimental topology reference for full-parameter Nemotron 3 Super SFT at 262,144 tokens: 128 ranks (16 nodes × 8 GPUs), TP 8, PP 2, CP 8, EP 8. Not runnable with the stock step. |

Pass the configuration name with `-c`:

```console
$ uv run nemotron steps run sft/megatron_bridge -c tiny
$ uv run nemotron steps run sft/megatron_bridge -c default
```

### Nemotron 3.5 Lightning

`lightning35.yaml` sets `hf_model_path: null` and `load_hf_weights: false`, so the base weights must already exist as a Megatron checkpoint.
Convert them first with [`convert/hf_to_megatron -c lightning35`](../convert/hf-to-megatron.md), then set the following environment variables in the shell or the `env.toml` profile:

| Variable | Consumed as |
| --- | --- |
| `L35_PRETRAINED_CHECKPOINT` | `checkpoint.pretrained_checkpoint` (`${L35_PRETRAINED_CHECKPOINT}/iter_0000000`) |
| `L35_PACKED_DIR` | `dataset.packed_sequence_specs.packed_train_data_path` and `packed_val_data_path` (`${L35_PACKED_DIR}/splits/{train,valid}`) |
| `L35_OUTPUT_DIR` | `checkpoint.save` (`${L35_OUTPUT_DIR}/sft-4k-tp2-ep8`) |

The Lightning recipe callable takes no arguments, so the overlay sets `recipe.packed_sequence` and `recipe.seq_length` to `null`; the sequence length is governed by the inherited `dataset.*` and `model.seq_length` fields, which remain 4096.

### Long-Context Topology References

`super3_128k.yaml` and `super3_256k.yaml` capture sequence lengths, distributed topologies, resource requests, packing requirements, and training settings for implementation planning.
They are placed under `config/` for discovery, but the stock step does not forward their context-parallel packing fields or preserve their `ddp.grad_reduce_in_fp32: false` setting through the named precision setup.
Use them with `--dry-run` to inspect the compiled topology; a real launch requires a runner that honors those settings.

| Config | Sequence length | Requested ranks | TP / PP / CP / EP | Packed input |
| --- | ---: | ---: | --- | --- |
| `super3_128k` | 131,072 | 64 | 8 / 1 / 8 / 64 | `$SFT_PACKED_128K_DIR/splits/{train,valid}` |
| `super3_256k` | 262,144 | 128 | 8 / 2 / 8 / 8 | `$SFT_PACKED_256K_DIR/splits/{train,valid}` |

Both references set `train.micro_batch_size: 1` and `train.global_batch_size: 1`, which assumes exactly the listed rank count so that the data-parallel size is 1.
If the world size grows, raise `train.global_batch_size` to a multiple of the resulting data-parallel size.

The packed shards must be produced with the same tokenizer as the model, and every stored subsequence must be padded so that `(subsequence_length - 1)` is divisible by `128` (`dataset.packed_sequence_specs.pad_seq_to_mult: 128`).
With TP 8, CP 8, and sequence parallelism enabled the runtime requires a multiple of 64; the references use 128 as a conservative alignment.
The stock `data_prep/sft_packing` step does not expose this alignment, so the shards must come from an alignment-aware packer.

For 256K training, GPU memory usage differs from the reference if the mixed-precision or DDP settings are not applied as written.

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
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | 8 | No | Nemotron 3.5 Lightning, a hybrid Mamba-Transformer MoE with multi-token prediction (MTP). |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | 32 | No | Nemotron 3 Super with 120.6 billion total and 12.7 billion active parameters. Typical runs at 32K context or shorter start at 32 GPUs; the 128K and 256K topology references request 64 and 128 GPUs. |

## Step Parameters

The manifest declares the following parameters.
Pass them as dotlist overrides.

The sequence length is declared in four places, and all four must stay equal to each other and to the pack size used in `data_prep/sft_packing`.

```{option} recipe.seq_length=N

The recipe sequence length.

Choices: `2048`, `4096`, `8192`, `16384`, `32768`.

Default: `4096`.

Example: `recipe.seq_length=8192`
```

```{option} dataset.seq_length=N

The dataset sequence length.
Change it together with `recipe.seq_length`, `dataset.packed_sequence_specs.packed_sequence_size`, and `model.seq_length`.

Choices: `2048`, `4096`, `8192`, `16384`, `32768`.

Default: `4096`.
```

```{option} dataset.packed_sequence_specs.packed_sequence_size=N

The offline-packed sample size.
It must match all configured sequence lengths and the prepared data.

Choices: `2048`, `4096`, `8192`, `16384`, `32768`.

Default: `4096`.
```

```{option} model.seq_length=N

The model sequence length.
Change it together with the recipe and dataset sequence lengths.

Choices: `2048`, `4096`, `8192`, `16384`, `32768`.

Default: `4096`.
```

```{option} dataset.packed_sequence_specs.packed_train_data_path=PATH

The packed training Parquet glob, usually `<sft_packing output_dir>/splits/train/*.parquet`.

Example: `dataset.packed_sequence_specs.packed_train_data_path=/lustre/packed/sft/splits/train/*.parquet`
```

```{option} peft=VALUE

Selects low-rank adaptation (LoRA) tuning instead of full SFT.
Set this value to `lora` for adapter tuning, or to `null` for full fine-tuning when the model and optimizer states fit in memory.

Choices: `lora`, `null`.

Default: `lora`, the shipped 30B recipe default. `tiny.yaml` and the Super3 references set `peft: null`.

Example: `peft=null`
```

```{option} checkpoint.pretrained_checkpoint=PATH

An optional Megatron base checkpoint or prior SFT checkpoint.
Keep it distinct from the `checkpoint.save` output path.

Example: `checkpoint.pretrained_checkpoint=/lustre/output/convert/nano3-megatron`
```

```{option} train.micro_batch_size=N

The per-rank micro batch size.
Start at `1` when validating a new distributed shape.

Default: `1`.
```

```{option} train.global_batch_size=N

The global batch size.
Keep it divisible by the resulting data-parallel size.

Example: `train.global_batch_size=128`
```

Frequently used dotlist overrides drawn from the underlying recipe include the following.

```{option} hf_model_path=<id-or-path>

The Hugging Face identifier or local path used to load base weights through `AutoBridge` when no Megatron checkpoint is supplied.

Example: `hf_model_path=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`
```

```{option} recipe.tensor_model_parallel_size=N

The tensor-model-parallel degree applied by the finetune recipe.

Example: `recipe.tensor_model_parallel_size=8`
```

```{option} recipe.pipeline_model_parallel_size=N

The pipeline-model-parallel degree applied by the finetune recipe.

Example: `recipe.pipeline_model_parallel_size=4`
```

```{option} checkpoint.save=PATH

The directory where the Megatron-Bridge recipe writes checkpoints.

Example: `checkpoint.save=/lustre/runs/nano3-sft/checkpoints`
```

## Strategies

The manifest records the following operator strategies for `sft/megatron_bridge`.

- When the dataset has fewer than ten thousand records, lower `train.global_batch_size` and raise the number of training iterations to keep optimizer steps useful.
- When consuming packed Parquet, set `recipe.packed_sequence=true` and keep `dataset.seq_length`, `dataset.packed_sequence_specs.packed_sequence_size`, `model.seq_length`, and the `data_prep/sft_packing` pack size identical.
- When the operator selects Nemotron 3.5 Lightning, convert the base model with `convert/hf_to_megatron -c lightning35` first and run `-c lightning35`; the overlay does not load Hugging Face weights.
- When the operator wants LoRA tuning, set `recipe.peft=lora` to lower the GPU requirement and shrink the checkpoint footprint.
- When the operator selects the Super3 model at 32K context or shorter, start from a 32-GPU plan with `tp=8`, `pp=4`, `cp=1`, and verify cluster topology before scaling further.
- When the sequence length exceeds 32K, use a `config/super3_*` long-context YAML for `--dry-run` planning only, then add and verify packing-alignment and mixed-precision support before launch.
- When the operator selects Super3 with 128K context, inspect `config/super3_128k.yaml` with `--dry-run` for a 64-rank TP 8, PP 1, CP 8, EP 64 topology; do not launch it with the stock step.
- When the operator selects Super3 with 256K context, inspect `config/super3_256k.yaml` with `--dry-run` for a 128-rank TP 8, PP 2, CP 8, EP 8 topology; do not launch it with the stock step.
- When the tokenizer, sequence length, TP, CP, or sequence parallelism changes, re-pack with an alignment-aware packer into a new output directory and revalidate stored lengths and loss masks.
- When GPU memory is tight, such as on A100 40 GB hardware, enable activation checkpointing and consider central-processing-unit (CPU) offloading.
- When you want maximum throughput on H100 hardware, keep packed sequences enabled and tune overlap and sequence-packing settings before scaling up.

## Common Errors

```{option} tokenizer_mismatch

Cause: the tokenizer used during `data_prep/sft_packing` differs from the tokenizer used for training, so token identifiers do not align.

Recovery: set the `data_prep/sft_packing` tokenizer to match the training model and regenerate the packed Parquet shards.
```

```{option} oom

Cause: GPU memory is exhausted during forward, backward, or optimizer steps.

Recovery: reduce `train.global_batch_size`, increase parallelism, or reduce the sequence length.
```

```{option} missing_packed_data

Cause: the training loop cannot find packed Parquet shards at the configured `dataset.packed_sequence_specs.packed_train_data_path`.

Recovery: for the standard configurations, run `data_prep/sft_packing` first or point the path at compatible packed shards. The Super3 long-context references require pre-existing shards explicitly aligned to 128.
```

```{option} bad_parallel_batch_shape

Cause: `train.global_batch_size` is not a multiple of the data-parallel size that results from the TP, PP, and CP settings.

Recovery: set `train.global_batch_size` to a multiple of the data-parallel size, and start with `train.micro_batch_size=1` while validating a new TP/PP/CP shape.
```

```{option} bad_loss_masks_or_template

Cause: the packed records carry incorrect loss masks or a mismatched chat template; the symptom is often poor SFT quality rather than a failure.

Recovery: inspect packed records from `data_prep/sft_packing` before training.
```

```{option} misaligned_packed_sequences

Cause: stored subsequences do not satisfy the alignment that the context-parallel topology requires.

Recovery: use an external or updated packer that explicitly aligns every stored subsequence; alignment is not configurable through the stock SFT step.
```

## Command Examples

Run the tiny validation configuration on the two-node Lepton SFT profile:

```console
$ uv run nemotron steps run sft/megatron_bridge -c tiny -r lepton_sft_megatron_bridge
```

Compile the default configuration without submitting the job:

```console
$ uv run nemotron steps run sft/megatron_bridge -c default --dry-run
```

Submit a detached LoRA run on Slurm with a longer sequence length, keeping all four sequence-length fields equal:

```console
$ uv run nemotron steps run sft/megatron_bridge -c default -b slurm_sft_megatron_bridge \
    peft=lora \
    recipe.seq_length=8192 \
    dataset.seq_length=8192 \
    dataset.packed_sequence_specs.packed_sequence_size=8192 \
    model.seq_length=8192 \
    train.global_batch_size=256
```

Submit an attached run on the Super3 base model with eight-way tensor parallelism and four-way pipeline parallelism:

```console
$ uv run nemotron steps run sft/megatron_bridge -c default -r lepton_sft_megatron_bridge \
    hf_model_path=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \
    recipe.tensor_model_parallel_size=8 \
    recipe.pipeline_model_parallel_size=4
```

Submit a detached Nemotron 3.5 Lightning SFT run from a converted Megatron checkpoint:

```console
$ L35_PRETRAINED_CHECKPOINT=/lustre/checkpoints/lightning35-megatron \
    L35_PACKED_DIR=/lustre/packed/lightning35 \
    L35_OUTPUT_DIR=/lustre/runs/lightning35 \
    uv run nemotron steps run sft/megatron_bridge -c lightning35 -b <batch-profile>
```

Inspect the compiled 128K Super3 topology without submitting it:

```console
$ SFT_PACKED_128K_DIR=/lustre/packed/super3-128k SFT_OUTPUT_DIR=/lustre/runs/super3-sft \
    uv run nemotron steps run sft/megatron_bridge -c super3_128k -b slurm_sft_megatron_bridge --dry-run
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
