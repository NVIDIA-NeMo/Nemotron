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

```{option} distributed=<true-or-false-or-auto>

Use the mounted multi-GPU converter instead of the single-process AutoBridge helper.
Keep this enabled for large models that cannot be materialized on one GPU.

Default: `true`.
```

```{option} tp=<int> pp=<int> ep=<int> etp=<int>

Tensor, pipeline, expert, and expert-tensor parallel sizes for the Megatron checkpoint written by the converter.
The defaults are `tp=1 pp=1 ep=8 etp=1`, matching the common Nemotron MoE conversion path.
Override these for dense models or a different target layout.

Defaults: `tp=1`, `pp=1`, `ep=8`, `etp=1`.
```

```{option} torchrun.nproc_per_node=<int>

Number of local conversion ranks when the step has to launch `torchrun` itself.
When a backend already launches the step with `torchrun`, the existing distributed world is reused.

Default: `NEMOTRON_CONVERT_NPROC_PER_NODE` or `8`.
```

```{option} script.path=<path>

Path to Megatron-Bridge's `convert_checkpoints_multi_gpu.py`.
Defaults to the path shipped in `nvcr.io/nvidia/nemo:26.04`: `/opt/Megatron-Bridge/examples/conversion/convert_checkpoints_multi_gpu.py`.
```

## Command Examples

Convert the default NVIDIA Nemotron base model into a local Megatron output directory:

```console
$ nemotron steps run convert/hf_to_megatron -c default \
    hf_model_id=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    megatron_path=./output/convert/nano3-megatron \
    tp=1 pp=1 ep=8
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
- If `distributed=true` launches multiple ranks with `tp=pp=ep=etp=1`, the step fails early because that would not shard the model. Set the real target parallelism, such as `tp=1 pp=1 ep=8 etp=1` for Nemotron MoE or `tp=8 pp=1 ep=1 etp=1` for a dense model.

## Related Documentation

- [Convert Checkpoints Between Training Steps](../../how-to/convert-checkpoints.md)
- [convert/merge_lora](merge-lora.md)
- [Megatron-Bridge Documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/)
