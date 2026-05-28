---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Reference pages for Nemotron checkpoint conversion steps."
topics: ["Training", "Reference", "Checkpoint Conversion"]
tags: ["Reference", "Steps", "Checkpoints", "Conversion"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Developer"]
---

# Checkpoint Conversion Steps

This section documents the conversion steps registered under `src/nemotron/steps/convert/`.
Use these steps to bridge checkpoint layouts between Hugging Face, Megatron distributed checkpoints, and LoRA adapter outputs.

## Steps

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} convert/hf_to_megatron
:link: hf-to-megatron
:link-type: doc
Import a Hugging Face checkpoint or model id into Megatron distributed checkpoint layout.
+++
{bdg-success}`reference`
:::

:::{grid-item-card} convert/megatron_to_hf
:link: megatron-to-hf
:link-type: doc
Export a Megatron distributed checkpoint iteration into Hugging Face safetensors layout.
+++
{bdg-success}`reference`
:::

:::{grid-item-card} convert/merge_lora
:link: merge-lora
:link-type: doc
Merge a LoRA adapter with its original base checkpoint to produce a standalone checkpoint.
+++
{bdg-success}`reference`
:::

::::

## Related Documentation

- [Convert Checkpoints Between Training Steps](../../how-to/convert-checkpoints.md)
- [Data and Checkpoint Formats](../../how-to/data-and-checkpoint-formats.md)
- [Step Catalog](../step-catalog.md)

```{toctree}
:hidden:
:maxdepth: 1

hf-to-megatron
megatron-to-hf
merge-lora
```
