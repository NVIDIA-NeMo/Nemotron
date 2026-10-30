---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Reference pages for the reinforcement learning training steps."
topics: ["Training", "Reference", "RL", "DPO", "RLVR", "RLHF"]
tags: ["Reference", "Steps", "RL", "DPO", "RLVR", "RLHF", "NeMo-RL"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Developer"]
---

# Reinforcement Learning Steps

This section documents the reinforcement learning (RL) alignment steps registered under `src/nemotron/steps/rl/nemo_rl/`.
All three steps run on NeMo-RL, consume a supervised fine-tuning (SFT) Megatron checkpoint as the warm-start policy, and produce an aligned `checkpoint_megatron` artifact.

## Steps

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} rl/nemo_rl/dpo
:link: dpo
:link-type: doc
Direct preference optimization with preference pairs.
+++
{bdg-success}`reference`
:::

:::{grid-item-card} rl/nemo_rl/rlvr
:link: rlvr
:link-type: doc
Reinforcement learning with verifiable rewards via group relative policy optimization.
+++
{bdg-success}`reference`
:::

:::{grid-item-card} rl/nemo_rl/rlhf
:link: rlhf
:link-type: doc
Reinforcement learning from human feedback with a generative reward model judge.
+++
{bdg-success}`reference`
:::

::::

## Related Documentation

- [Nemotron Steps CLI Reference](../cli-reference.md) covers the shared option set, dotlist overrides, and passthrough arguments.
- [Choose an RL Alignment Step](../../how-to/choose-rl-step.md) compares the three RL steps.
- [Configuration Conventions](../config-conventions.md) describes the per-step `config/` layout.

```{toctree}
:hidden:
:maxdepth: 1

dpo
rlvr
rlhf
```
