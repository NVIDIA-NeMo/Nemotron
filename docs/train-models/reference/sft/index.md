---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Reference pages for the supervised fine-tuning training steps."
topics: ["Training", "Reference", "SFT"]
tags: ["Reference", "Steps", "SFT", "AutoModel", "Megatron-Bridge"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Developer"]
---

# Supervised Fine-Tuning Steps

This section documents the supervised fine-tuning (SFT) steps registered under `src/nemotron/steps/sft/`.
The two steps target different training libraries and consume different data formats.
Both produce checkpoints you can use as warm-start policies for reinforcement learning alignment.

## Steps

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} sft/automodel
:link: automodel
:link-type: doc
Supervised fine-tuning with the NeMo AutoModel library against Hugging Face base models and JSON Lines chat datasets.
+++
{bdg-success}`reference`
:::

:::{grid-item-card} sft/megatron_bridge
:link: megatron-bridge
:link-type: doc
Supervised fine-tuning on top of NVIDIA Megatron-Bridge for distributed training of the Nemotron model family.
+++
{bdg-success}`reference`
:::

::::

## Related Documentation

- [Nemotron Steps CLI Reference](../cli-reference.md) covers the shared option set, dotlist overrides, and passthrough arguments.
- [Choose an SFT Backend](../../how-to/choose-sft-backend.md) compares `sft/automodel` and `sft/megatron_bridge`.
- [Configuration Conventions](../config-conventions.md) describes the per-step `config/` layout.

```{toctree}
:hidden:
:maxdepth: 1

automodel
megatron-bridge
```
