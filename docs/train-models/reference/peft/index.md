---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Reference pages for the parameter-efficient fine-tuning training steps."
topics: ["Training", "Reference", "PEFT", "LoRA"]
tags: ["Reference", "Steps", "PEFT", "LoRA"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Developer"]
---

# Parameter-Efficient Fine-Tuning Steps

This section documents the parameter-efficient fine-tuning (PEFT) steps registered under `src/nemotron/steps/peft/`.
Each step trains a low-rank adaptation (LoRA) adapter on top of a base model and produces a `checkpoint_lora` artifact.
Merge that artifact with the base model by using the `convert/merge_lora` step when you need a standalone Hugging Face (HF) checkpoint for deployment.

## Steps

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} peft/automodel
:link: automodel
:link-type: doc
LoRA tuning with the NeMo AutoModel library against Hugging Face base models and JSON Lines chat datasets.
+++
{bdg-success}`reference`
:::

:::{grid-item-card} peft/megatron_bridge
:link: megatron-bridge
:link-type: doc
LoRA tuning on top of NVIDIA Megatron-Bridge for distributed training of the Nemotron model family.
+++
{bdg-success}`reference`
:::

::::

## Related Documentation

- [Nemotron Steps CLI Reference](../cli-reference.md) covers the shared option set, dotlist overrides, and passthrough arguments.
- [Choose a PEFT Backend](../../how-to/choose-peft-backend.md) compares `peft/automodel` and `peft/megatron_bridge`.
- [Configuration Conventions](../config-conventions.md) describes the per-step `config/` layout.

```{toctree}
:hidden:
:maxdepth: 1

automodel
megatron-bridge
```
