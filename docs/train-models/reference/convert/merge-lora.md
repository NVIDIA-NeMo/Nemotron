---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Command-line reference for the convert/merge_lora step."
topics: ["Training", "Reference", "Checkpoint Conversion", "LoRA"]
tags: ["Reference", "CLI", "Steps", "PEFT", "LoRA", "Merge"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Developer"]
---

# convert/merge_lora

This step merges a LoRA adapter with its original base checkpoint.
Use it when a downstream consumer needs a standalone checkpoint instead of an adapter plus base model pair.

## Syntax

```bash
nemotron steps run convert/merge_lora \
    [-c <config-name-or-path>] \
    [-r <run-profile> | -b <batch-profile>] \
    [-d] \
    [<dotlist-overrides>...]
```

Refer to the [Nemotron Steps CLI Reference](../cli-reference.md) for the shared flag set.

## Configuration Files

The step ships `config/default.yaml` under `src/nemotron/steps/convert/merge_lora/config/`.
The default `backend` is `auto`.
When `base_megatron_path` is set, `auto` selects the Megatron-Bridge merge path.
Otherwise, `auto` selects the Hugging Face PEFT merge path.

## Inputs and Outputs

| Direction | Artifact Type | Required | Description |
| --- | --- | --- | --- |
| Consumes | `checkpoint_lora` | Yes | LoRA adapter weights from a PEFT step. |
| Consumes | `checkpoint_hf` | Yes for `hf_peft` | Original Hugging Face base checkpoint or model/config source. |
| Consumes | `checkpoint_megatron` | Required for `megatron_bridge` | Original dense Megatron checkpoint for Megatron-Bridge adapter merge. |
| Produces | `checkpoint_hf` | - | Merged standalone Hugging Face checkpoint. |
| Produces | `checkpoint_megatron` | Optional | Merged Megatron checkpoint when `backend=megatron_bridge`. |

## Parameters

```{option} backend=<auto|hf_peft|megatron_bridge>

Adapter merge backend.
`auto` selects `megatron_bridge` when `base_megatron_path` is set, otherwise `hf_peft`.

Default: `auto`.
```

```{option} lora_checkpoint=<path>

Adapter checkpoint path produced by a PEFT step.
```

```{option} base_hf_path=<id-or-path>

Original Hugging Face base checkpoint for `backend=hf_peft`, or Hugging Face config/model source for `backend=megatron_bridge`.
Do not substitute a different base.
```

```{option} base_megatron_path=<path>

Original dense Megatron checkpoint for `backend=megatron_bridge`.
Use the base checkpoint that the adapter was trained against.
```

```{option} hf_model_id=<id-or-path>

Hugging Face model id or path used to reconstruct architecture when exporting a Megatron-Bridge merge to Hugging Face layout.

Example: `hf_model_id=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`
```

```{option} output_hf_path=<path>

Directory for the merged standalone Hugging Face checkpoint.
```

```{option} output_megatron_path=<path>

Directory for the merged Megatron checkpoint when `backend=megatron_bridge`.
```

```{option} cpu=<true-or-false>

Merge on CPU when GPU memory is tight or when running outside a training container.

Default: `true`.
```

```{option} tp=<n>

Tensor parallel size for Megatron-Bridge merge.

Default: `1`.
```

```{option} pp=<n>

Pipeline parallel size for Megatron-Bridge merge.

Default: `1`.
```

```{option} ep=<n>

Expert parallel size for Megatron-Bridge merge.

Default: `1`.
```

## Command Examples

Merge an AutoModel PEFT adapter into its original Hugging Face base:

```console
$ nemotron steps run convert/merge_lora -c default \
    backend=hf_peft \
    lora_checkpoint=/path/to/adapter_checkpoint \
    base_hf_path=/path/to/original_hf_base \
    output_hf_path=./output/convert/merged-hf
```

Merge a Megatron-Bridge PEFT adapter and export the merged result to Hugging Face layout:

```console
$ nemotron steps run convert/merge_lora -c default --batch lepton_convert_model \
    backend=megatron_bridge \
    lora_checkpoint=/mnt/lustre-shared/output/peft/megatron_bridge/adapter \
    base_megatron_path=/mnt/lustre-shared/output/base/megatron \
    hf_model_id=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    output_megatron_path=/mnt/lustre-shared/output/convert/merged-megatron \
    output_hf_path=/mnt/lustre-shared/output/convert/merged-hf
```

## Recovery Notes

- If scores differ after merge, evaluate the adapter-loaded model and the merged checkpoint separately; do not assume they are identical.
- If merge fails with missing keys or shape mismatch, verify the adapter was trained against the same base checkpoint.
- Write outputs to a fresh directory so a failed merge cannot overwrite the base or adapter.

## Related Documentation

- [Choose a PEFT Backend](../../how-to/choose-peft-backend.md)
- [convert/hf_to_megatron](hf-to-megatron.md)
- [PEFT documentation](https://github.com/huggingface/peft)
