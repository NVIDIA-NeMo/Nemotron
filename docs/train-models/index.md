# Model Training with Nemotron Steps

This section documents how to run supervised fine-tuning (SFT), parameter-efficient fine-tuning (PEFT), reinforcement learning (RL) alignment, and post-training optimization with Nemotron *steps*.
Each step packages a training approach, configuration files, and entry logic that you invoke through the `nemotron steps` CLI.
For the definitions of *step*, *configuration*, and *environment profile* that apply across every domain, see [Nemotron Steps Basics](../steps/basics.md).
If you are new to fine-tuning, start with [Training Basics](explanation/basics.md), which defines the training-specific terms the rest of this section uses.

## Capabilities at a Glance

| Area | Step Names | Role |
|------|-------------------------------|------|
| SFT | `sft/automodel`, `sft/megatron_bridge` | Supervised fine tuning from chat-formatted JSON Lines (JSONL) or packed Apache Parquet |
| PEFT | `peft/automodel`, `peft/megatron_bridge` | Adapter training with a smaller trainable surface |
| RL | `rl/nemo_rl/dpo`, `rl/nemo_rl/rlvr`, `rl/nemo_rl/rlhf` | Alignment after a supervised fine tuning (SFT) policy exists |
| Optimize | `optimize/modelopt/quantize`, `optimize/modelopt/prune`, `optimize/modelopt/distill` | Compression and quality recovery |

## Limitations and Restrictions

The Nemotron steps for data preparation and model training do not support local training, such as on a developer workstation.

These steps require access to at least two nodes, each equipped with 8 x NVIDIA A100 80 GB or better GPUs.
These steps support the following environments:

- Slurm
- NVIDIA DGX Cloud Lepton
- NVIDIA Run:ai

For assistance with configuring access to one of the supported computing environments, refer to [](./reference/env-profile-generator.md) or run the `nemotron-env-toml` skill with your agent.

## Learning Path

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} Training Basics
:link: explanation/basics
:link-type: doc
This page defines supervised fine-tuning, parameter-efficient fine-tuning, reinforcement learning alignment, quantization, tokenizers, the chat dataset format, and checkpoints.
+++
`Beginner Level`
:::

:::{grid-item-card} Getting Started
:link: getting-started
:link-type: doc
This guide covers installation expectations, the environment profile, and how to run your first sample job with the `tiny` configuration.
+++
`Beginner`
:::

:::{grid-item-card} How-To Guides
:link: how-to/index
:link-type: doc
These guides are task-focused. They explain how to pick a backend, wire data, and run optimization steps.
+++
`Intermediate`
:::

:::{grid-item-card} Explanation
:link: explanation/index
:link-type: doc
This material explains the basics, the artifact graph, and how the training libraries differ.
+++
`Concepts`
:::

:::{grid-item-card} Reference
:link: reference/index
:link-type: doc
This material is for lookup. It lists the step catalog, parameters, and configuration conventions.
+++
`Lookup`
:::

:::{grid-item-card} Model Training with Agents
:link: using-skill
:link-type: doc
Use the customize skill with a YAML-first plan: repo steps, then configs, then code only for gaps.
+++
`Workflow`
:::

::::

## Quick Links

- [Model Training with Agents](using-skill.md) describes how to work with the `nemotron-customize` skill for multi-stage training plans and YAML-first deliverables.
- [Getting Started](getting-started.md) describes how to verify the CLI and run a tiny configuration.
- [Execution through NeMo Run](../nemo_runspec/nemo-run.md) describes profiles, attached and detached runs, and clusters.
- [Nemotron CLI Overview](../nemotron/cli.md) describes how the wider CLI relates to configuration and overrides.
