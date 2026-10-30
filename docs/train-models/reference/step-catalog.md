---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Catalog of every Nemotron training step identifier, manifest path, and per-step reference link."
topics: ["Training", "Reference", "Steps", "Catalog"]
tags: ["Reference", "Steps", "Catalog"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Developer"]
---

# Step Catalog

This page catalogs every training step identifier registered under `src/nemotron/steps/` in the `sft`, `peft`, `rl`, and `optimize` directories.
Each row gives the step identifier, the on-disk manifest path, and the per-step reference page.

Adjacent preparation and conversion steps do not appear in this catalog.
Those step identifiers include `data_prep/sft_packing`, `data_prep/rl_prep`, and `convert/megatron_to_hf`, and they live under different directories with their own manifests.

## Supervised Fine-Tuning Steps

| Step Identifier | Manifest Path | Reference |
| --- | --- | --- |
| `sft/automodel` | `src/nemotron/steps/sft/automodel/step.toml` | [sft/automodel](sft/automodel.md) |
| `sft/megatron_bridge` | `src/nemotron/steps/sft/megatron_bridge/step.toml` | [sft/megatron_bridge](sft/megatron-bridge.md) |

## Parameter-Efficient Fine-Tuning Steps

| Step Identifier | Manifest Path | Reference |
| --- | --- | --- |
| `peft/automodel` | `src/nemotron/steps/peft/automodel/step.toml` | [peft/automodel](peft/automodel.md) |
| `peft/megatron_bridge` | `src/nemotron/steps/peft/megatron_bridge/step.toml` | [peft/megatron_bridge](peft/megatron-bridge.md) |

## Reinforcement Learning Steps

| Step Identifier | Manifest Path | Reference |
| --- | --- | --- |
| `rl/nemo_rl/dpo` | `src/nemotron/steps/rl/nemo_rl/dpo/step.toml` | [rl/nemo_rl/dpo](rl/dpo.md) |
| `rl/nemo_rl/rlvr` | `src/nemotron/steps/rl/nemo_rl/rlvr/step.toml` | [rl/nemo_rl/rlvr](rl/rlvr.md) |
| `rl/nemo_rl/rlhf` | `src/nemotron/steps/rl/nemo_rl/rlhf/step.toml` | [rl/nemo_rl/rlhf](rl/rlhf.md) |

## Optimization Steps

| Step Identifier | Manifest Path | Reference |
| --- | --- | --- |
| `optimize/modelopt/quantize` | `src/nemotron/steps/optimize/modelopt/quantize/step.toml` | [optimize/modelopt/quantize](optimize/quantize.md) |
| `optimize/modelopt/prune` | `src/nemotron/steps/optimize/modelopt/prune/step.toml` | [optimize/modelopt/prune](optimize/prune.md) |
| `optimize/modelopt/distill` | `src/nemotron/steps/optimize/modelopt/distill/step.toml` | [optimize/modelopt/distill](optimize/distill.md) |

## List Steps from the Command Line

This catalog covers the `sft`, `peft`, `rl`, and `optimize` step categories.
Running `nemotron steps list` with no filter returns all registered steps, including data preparation, evaluation, conversion, environment, and other categories not listed here.
Use the `--category`, `--consumes`, and `--produces` filters to narrow the results.

```console
$ nemotron steps list --category sft
$ nemotron steps list --consumes training_jsonl
$ nemotron steps list --produces checkpoint_megatron --json
```

## Related Documentation

- [Nemotron Steps CLI Reference](cli-reference.md) covers the `list`, `show`, and `run` subcommands.
- [Configuration Conventions](config-conventions.md) describes the per-step `config/` layout.
