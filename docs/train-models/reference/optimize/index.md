---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Reference pages for the model optimization steps."
topics: ["Training", "Reference", "Optimization", "Quantization", "Pruning", "Distillation"]
tags: ["Reference", "Steps", "Optimization", "ModelOpt", "Megatron-Bridge"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Developer"]
---

# Model Optimization Steps

This section documents the model optimization steps registered under `src/nemotron/steps/optimize/modelopt/`.
The three steps wrap NVIDIA Model Optimizer through NVIDIA Megatron-Bridge to quantize, prune, and distill trained checkpoints.

## Steps

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} optimize/modelopt/quantize
:link: quantize
:link-type: doc
Post-training quantization with floating-point recipes such as fp8 and nvfp4, and integer recipes such as int8_sq and int4_awq.
+++
{bdg-success}`reference`
:::

:::{grid-item-card} optimize/modelopt/prune
:link: prune
:link-type: doc
Structured pruning by target parameter budget or explicit architecture.
+++
{bdg-success}`reference`
:::

:::{grid-item-card} optimize/modelopt/distill
:link: distill
:link-type: doc
Teacher-student distillation that recovers quality after pruning or quantization.
+++
{bdg-success}`reference`
:::

::::

## Related Documentation

- [Nemotron Steps CLI Reference](../cli-reference.md) covers the shared option set, dotlist overrides, and passthrough arguments.
- [Run Post-Training Optimization](../../how-to/run-optimization.md) explains the ordering of prune and distill, hardware targets, and quality recovery.
- [Configuration Conventions](../config-conventions.md) describes the per-step `config/` layout.

```{toctree}
:hidden:
:maxdepth: 1

quantize
prune
distill
```
