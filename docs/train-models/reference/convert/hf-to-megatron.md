---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Command-line reference for the convert/hf_to_megatron step."
topics: ["Training", "Reference", "Checkpoint Conversion"]
tags: ["Reference", "CLI", "Steps", "Hugging Face", "Megatron"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Developer"]
---

# convert/hf_to_megatron

This step converts a Hugging Face checkpoint or model id into Megatron distributed checkpoint layout.
Use it when a downstream Megatron-Bridge step expects `checkpoint_megatron` but the upstream artifact is `checkpoint_hf`.

## Syntax

```bash
nemotron steps run convert/hf_to_megatron \
    [-c <config-name-or-path>] \
    [-r <run-profile> | -b <batch-profile>] \
    [-d] \
    [<dotlist-overrides>...]
```

Refer to the [Nemotron Steps CLI Reference](../cli-reference.md) for the shared flag set.

## Configuration Files

The step ships `config/default.yaml` under `src/nemotron/steps/convert/hf_to_megatron/config/`.
The default source model is `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`, and the default output path is derived from `CONVERT_OUTPUT_DIR`, `NEMO_RUN_DIR`, or `./output/convert`.

## Inputs and Outputs

| Direction | Artifact Type | Required | Description |
| --- | --- | --- | --- |
| Consumes | `checkpoint_hf` | Yes | A clean Hugging Face checkpoint directory or model id. |
| Produces | `checkpoint_megatron` | - | A Megatron distributed checkpoint directory. |

## Parameters

```{option} hf_model_id=<id-or-path>

Hugging Face model id or local checkpoint path to import.
Use a clean model directory, not trainer logs, optimizer state, or adapter-only output.

Example: `hf_model_id=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`
```

```{option} megatron_path=<path>

Output directory for the Megatron distributed checkpoint.
Keep this path separate from the Hugging Face source path.

Example: `megatron_path=/lustre/output/convert/nano3-megatron`
```

```{option} dtype=<dtype>

Torch dtype used during import.
Typical NVIDIA Nemotron checkpoints use `bfloat16`.

Choices: `bfloat16`, `float16`, `float32`.
Default: `bfloat16`.
```

```{option} device_map=<value>

Optional Transformers `device_map` forwarded during Hugging Face model loading, such as `auto` or `cuda:0`.
Leave unset unless the conversion host requires explicit placement.
```

```{option} trust_remote_code=<true-or-false>

Whether to trust Hugging Face custom model code when AutoBridge loads the source model configuration.

Default: `true`.
```

## Command Examples

Convert the default NVIDIA Nemotron base model into a local Megatron output directory:

```console
$ nemotron steps run convert/hf_to_megatron -c default \
    hf_model_id=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    megatron_path=./output/convert/nano3-megatron
```

Submit the conversion through a generated Lepton profile:

```console
$ nemotron steps run convert/hf_to_megatron -c default --batch lepton_convert_model \
    hf_model_id=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    megatron_path=/mnt/lustre-shared/output/convert/nano3-megatron
```

## Recovery Notes

- If the source came from LoRA training, merge the adapter into the original base first with `convert/merge_lora`.
- If tokenizer or model config files are missing, use the original Hugging Face model id as `hf_model_id`.
- If conversion fails, retry into a fresh `megatron_path` instead of reusing a partially written directory.

## Related Documentation

- [Convert Checkpoints Between Training Steps](../../how-to/convert-checkpoints.md)
- [convert/merge_lora](merge-lora.md)
- [Megatron-Bridge Documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/)
