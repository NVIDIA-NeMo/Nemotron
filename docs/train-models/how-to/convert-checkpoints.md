---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "How to choose and run Nemotron checkpoint conversion steps between Hugging Face, Megatron, and LoRA adapter layouts."
topics: ["Training", "How-To", "Checkpoint Conversion"]
tags: ["How-To", "Checkpoints", "Conversion", "Hugging Face", "Megatron", "LoRA"]
content:
  type: "How-To"
  difficulty: "Intermediate"
  audience: ["ML Engineer", "Developer"]
---

# Convert Checkpoints Between Training Steps

Use a conversion step when one step produces a checkpoint layout that the next step cannot consume directly.
The converter is an explicit pipeline step, not an implicit side effect of training.

## Choose the Converter

| Source artifact | Target artifact | Step |
| --- | --- | --- |
| `checkpoint_hf` | `checkpoint_megatron` | `convert/hf_to_megatron` |
| `checkpoint_megatron` | `checkpoint_hf` | `convert/megatron_to_hf` |
| `checkpoint_lora` plus original base checkpoint | merged `checkpoint_hf` | `convert/merge_lora` |

Common cases:

- AutoModel SFT or PEFT produces Hugging Face layout checkpoints. Use `convert/hf_to_megatron` before Megatron-Bridge consumers that require Megatron layout.
- Megatron-Bridge SFT, RL, and some optimization steps produce Megatron distributed checkpoints. Use `convert/megatron_to_hf` before Hugging Face-native evaluation, deployment, or pruning flows.
- PEFT steps produce adapter checkpoints. Use `convert/merge_lora` when deployment or evaluation needs a single merged Hugging Face checkpoint.

## Preflight Checks

Before conversion:

- Pick one validated checkpoint iteration. For Megatron exports, point `megatron_path` at the concrete `iter_*` checkpoint directory, not the parent run directory.
- Keep output paths separate from input paths. A failed conversion should never overwrite the source checkpoint.
- Keep tokenizer and chat-template provenance with the checkpoint. If the converter needs `hf_model_id`, use the original model or config source used by training.
- For LoRA merge, use the exact base checkpoint the adapter was trained against.

## Convert Hugging Face to Megatron

Use this path when a Megatron-Bridge consumer needs a Megatron distributed checkpoint.

```console
$ nemotron steps run convert/hf_to_megatron -c default \
    hf_model_id=/path/to/hf_checkpoint_or_model_id \
    megatron_path=/path/to/output_megatron_checkpoint
```

For NVIDIA Nemotron checkpoints, keep `dtype=bfloat16` unless the source checkpoint requires another dtype.

## Convert Megatron to Hugging Face

Use this path when the next consumer is Hugging Face-native evaluation, deployment, pruning, or a tool that expects safetensors.

```console
$ nemotron steps run convert/megatron_to_hf -c default \
    megatron_path=/path/to/megatron/iter_0000100 \
    hf_model_id=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    hf_path=/path/to/output_hf_checkpoint
```

The `hf_model_id` value supplies the model configuration and tokenizer expectations used to reconstruct the Hugging Face layout.

## Merge LoRA Into a Hugging Face Base

Use this path for adapters produced by Hugging Face-native PEFT flows.

```console
$ nemotron steps run convert/merge_lora -c default \
    backend=hf_peft \
    lora_checkpoint=/path/to/adapter_checkpoint \
    base_hf_path=/path/to/original_hf_base \
    output_hf_path=/path/to/merged_hf_checkpoint
```

Do not merge into a different base model, even if the architecture name matches.

## Merge Megatron-Bridge LoRA

Use this path for adapters produced by Megatron-Bridge PEFT flows.
The step can write a merged Megatron checkpoint and export a Hugging Face checkpoint when `export_hf=true`.

```console
$ nemotron steps run convert/merge_lora -c default \
    backend=megatron_bridge \
    lora_checkpoint=/path/to/lora_megatron_checkpoint \
    base_megatron_path=/path/to/original_dense_megatron_checkpoint \
    hf_model_id=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    output_megatron_path=/path/to/merged_megatron_checkpoint \
    output_hf_path=/path/to/merged_hf_checkpoint
```

Use `tp`, `pp`, and `ep` overrides when the merge must match a specific tensor, pipeline, or expert parallel layout.

## Run on a Cluster Profile

Generated environment files include one shared conversion profile per executor family.
Use the profile that matches your site:

```console
$ nemotron steps run convert/megatron_to_hf -c default --batch lepton_convert_model \
    megatron_path=/mnt/lustre-shared/output/sft/megatron_bridge/iter_0000100 \
    hf_model_id=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
    hf_path=/mnt/lustre-shared/output/convert/sft-hf
```

Equivalent profile names are `slurm_convert_model` for Slurm and `dgxcloud_convert_model` for DGX Cloud.

## Validate the Output

After conversion:

- Confirm the output directory exists and contains model weights plus tokenizer/config files for Hugging Face outputs.
- Run a small generation or evaluation smoke test before using the checkpoint for a larger training or evaluation job.
- Preserve the source checkpoint until the converted checkpoint has passed validation.

## Related Documentation

- [Data and Checkpoint Formats](data-and-checkpoint-formats.md)
- [Artifact Graph](../explanation/artifact-graph.md)
- [Conversion Step References](../reference/convert/index.md)
