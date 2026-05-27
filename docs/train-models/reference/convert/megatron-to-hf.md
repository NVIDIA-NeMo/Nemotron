---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Command-line reference for the convert/megatron_to_hf step."
topics: ["Training", "Reference", "Checkpoint Conversion"]
tags: ["Reference", "CLI", "Steps", "Megatron", "Hugging Face"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Developer"]
---

# convert/megatron_to_hf

This step converts a Megatron distributed checkpoint into Hugging Face safetensors layout.
Use it when a downstream Hugging Face-native consumer expects `checkpoint_hf` but the upstream artifact is `checkpoint_megatron`.

## Syntax

```bash
nemotron steps run convert/megatron_to_hf \
    [-c <config-name-or-path>] \
    [-r <run-profile> | -b <batch-profile>] \
    [-d] \
    [<dotlist-overrides>...]
```

Refer to the [Nemotron Steps CLI Reference](../cli-reference.md) for the shared flag set.

## Configuration Files

The step ships `config/default.yaml` under `src/nemotron/steps/convert/megatron_to_hf/config/`.
The default `hf_model_id` is `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`, and `megatron_path` must be supplied for real runs.

## Inputs and Outputs

| Direction | Artifact Type | Required | Description |
| --- | --- | --- | --- |
| Consumes | `checkpoint_megatron` | Yes | A specific Megatron checkpoint directory, normally an `iter_*` directory. |
| Produces | `checkpoint_hf` | - | A Hugging Face safetensors checkpoint directory. |

## Parameters

```{option} megatron_path=<path>

Specific Megatron checkpoint directory to export.
Point this at the concrete checkpoint iteration, not the parent training run folder.

Example: `megatron_path=/lustre/runs/sft/checkpoints/iter_0000100`
```

```{option} hf_model_id=<id-or-path>

Hugging Face model id or config source used by AutoBridge to reconstruct the Hugging Face architecture and tokenizer expectations.

Example: `hf_model_id=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`
```

```{option} hf_path=<path>

Output directory for the exported Hugging Face safetensors checkpoint.

Example: `hf_path=/lustre/output/convert/sft-hf`
```

```{option} trust_remote_code=<true-or-false>

Whether to trust custom Hugging Face model code while reconstructing the export target.

Default: `true`.
```

```{option} show_progress=<true-or-false>

Whether to show Megatron-Bridge export progress output.

Default: `true`.
```

```{option} strict=<true-or-false>

Whether Megatron-Bridge should require source and target checkpoint keys to match strictly.

Default: `true`.
```

## Command Examples

Export a validated Megatron checkpoint iteration to Hugging Face layout:

```console
$ nemotron steps run convert/megatron_to_hf -c default \
    megatron_path=/path/to/megatron/iter_0000100 \
    hf_model_id=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    hf_path=./output/convert/sft-hf
```

Submit the export through a generated Lepton profile:

```console
$ nemotron steps run convert/megatron_to_hf -c default --batch lepton_convert_model \
    megatron_path=/mnt/lustre-shared/output/sft/megatron_bridge/iter_0000100 \
    hf_model_id=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    hf_path=/mnt/lustre-shared/output/convert/sft-hf
```

## Recovery Notes

- If export fails because the checkpoint is incomplete, wait for async checkpoint save to finish and retry from a complete `iter_*` directory.
- If tokenizer or config reconstruction fails, set `hf_model_id` to the original base model or config source.
- Validate the exported Hugging Face checkpoint with a small generation or evaluation job before deployment.

## Related Documentation

- [Convert Checkpoints Between Training Steps](../../how-to/convert-checkpoints.md)
- [Data and Checkpoint Formats](../../how-to/data-and-checkpoint-formats.md)
- [Megatron-Bridge Documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/)
