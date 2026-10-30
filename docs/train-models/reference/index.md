---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Reference index for Nemotron training steps, configuration conventions, and the CLI."
topics: ["Training", "Reference"]
tags: ["Reference", "CLI", "Configuration", "Steps"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Developer"]
---

# Training Reference

This section provides lookup material for every supervised fine-tuning (SFT), parameter-efficient fine-tuning (PEFT), reinforcement learning (RL), and optimization step packaged under `src/nemotron/steps/`.
Use these pages to find the exact CLI syntax, configuration file layout, parameters, and configuration overrides for each step.

## Reference Sections

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` Nemotron Steps CLI Reference
:link: cli-reference
:link-type: doc
Shared command-line syntax, options, dotlist overrides, and passthrough arguments for the `nemotron steps` command group.
+++
{bdg-success}`lookup`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Env Profile Generator
:link: env-profile-generator
:link-type: doc
The packaged `env/env_toml` step that generates the environment profile file every other training step consumes.
+++
{bdg-success}`setup`
:::

:::{grid-item-card} {octicon}`list-unordered;1.5em;sd-mr-1` Step Catalog
:link: step-catalog
:link-type: doc
Every step identifier, manifest path, and per-step reference link.
+++
{bdg-success}`lookup`
:::

:::{grid-item-card} {octicon}`file-code;1.5em;sd-mr-1` Configuration Conventions
:link: config-conventions
:link-type: doc
Per-step `config/` layout, CLI configuration resolution, dotlist override rules, and environment-variable expansion.
+++
{bdg-success}`lookup`
:::

::::

## Per-Category Step References

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Supervised Fine-Tuning Steps
:link: sft/index
:link-type: doc
The `sft/automodel` and `sft/megatron_bridge` references.
+++
{bdg-success}`reference`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Parameter-Efficient Fine-Tuning Steps
:link: peft/index
:link-type: doc
The `peft/automodel` and `peft/megatron_bridge` references.
+++
{bdg-success}`reference`
:::

:::{grid-item-card} {octicon}`milestone;1.5em;sd-mr-1` Reinforcement Learning Steps
:link: rl/index
:link-type: doc
The `rl/nemo_rl/dpo`, `rl/nemo_rl/rlvr`, and `rl/nemo_rl/rlhf` references.
+++
{bdg-success}`reference`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Optimization Steps
:link: optimize/index
:link-type: doc
The `optimize/modelopt/quantize`, `optimize/modelopt/prune`, and `optimize/modelopt/distill` references.
+++
{bdg-success}`reference`
:::

:::{grid-item-card} {octicon}`git-compare;1.5em;sd-mr-1` Checkpoint Conversion Steps
:link: convert/index
:link-type: doc
The `convert/hf_to_megatron`, `convert/megatron_to_hf`, and `convert/merge_lora` references.
+++
{bdg-success}`reference`
:::

::::

## Related Documentation

- [Getting Started With Training Steps](../getting-started.md) walks through a first run with the tiny configuration.
- [Execution Through NeMo Run](../../nemo_runspec/nemo-run.md) explains attached and detached execution, environment profiles, and remote job directories.
- [How-To Guides](../how-to/index.md) cover backend choice, data and checkpoint formats, and environment-and-executor setup.

```{toctree}
:hidden:
:maxdepth: 2

cli-reference
env-profile-generator
step-catalog
config-conventions
SFT Steps <sft/index>
PEFT Steps <peft/index>
RL Steps <rl/index>
Optimization Steps <optimize/index>
Checkpoint Conversion Steps <convert/index>
```
