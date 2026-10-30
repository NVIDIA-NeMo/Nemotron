---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Command-line reference for the rl/nemo_rl/rlvr training step."
topics: ["Training", "Reference", "CLI", "RL", "RLVR", "GRPO"]
tags: ["Reference", "CLI", "Steps", "RL", "RLVR", "GRPO", "NeMo-RL", "NeMo-Gym"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Developer"]
---

# rl/nemo_rl/rlvr

This step runs reinforcement learning with verifiable rewards (RLVR) by using group relative policy optimization (GRPO) on NeMo-RL.
Use this step when the downstream task has a programmatic reward signal, such as a unit-tested code generation task or a mathematics problem with a ground-truth solution.
The step produces an aligned `checkpoint_megatron` artifact.

## Syntax

```bash
nemotron steps run rl/nemo_rl/rlvr \
    [-c <config-name-or-path>] \
    [-r <run-profile> | -b <batch-profile>] \
    [-d] \
    [--force-squash] \
    [<dotlist-overrides>...] \
    [<passthrough-args>...]
```

Refer to the [Nemotron Steps CLI Reference](../cli-reference.md) for the shared flag set.

## Configuration Files

The step ships three configuration files under `src/nemotron/steps/rl/nemo_rl/rlvr/config/`.

| File | Purpose |
| --- | --- |
| `default.yaml` | Lightweight upstream group relative policy optimization (GRPO) example path. The step delegates to `/opt/nemo-rl/examples/run_grpo.py`. |
| `nemo_gym.yaml` | NeMo-Gym path that mirrors the Super3 RLVR style, using NeMo-Gym JSON Lines and resource-server reward configurations. |
| `tiny.yaml` | Short validation run with a small dataset slice. |

Pass the configuration name with `-c`:

```console
$ nemotron steps run rl/nemo_rl/rlvr -c tiny
$ nemotron steps run rl/nemo_rl/rlvr -c default
$ nemotron steps run rl/nemo_rl/rlvr -c nemo_gym
```

## Inputs and Outputs

| Direction | Artifact Type | Required | Description |
| --- | --- | --- | --- |
| Consumes | `training_jsonl` | Yes | Prompt JSON Lines with verifiable answers, such as ground-truth solutions for a mathematics task. |
| Consumes | `checkpoint_megatron` | Yes | The supervised fine-tuned policy to optimize. |
| Produces | `checkpoint_megatron` | — | The RLVR-aligned policy checkpoint. |

## Step Parameters

The manifest declares three group relative policy optimization (GRPO) parameters.
Pass them as dotlist overrides.

```{option} grpo.num_generations_per_prompt=<n>

The number of rollouts produced per prompt.
This value sets the GRPO group size.

Default: `8`.

Example: `grpo.num_generations_per_prompt=16`
```

```{option} grpo.normalize_rewards=<bool>

When set to `true`, the trainer normalizes rewards within each group before computing advantages.

Default: `true`.

Example: `grpo.normalize_rewards=false`
```

```{option} env.should_use_nemo_gym=<bool>

When set to `true`, the step switches from the upstream generic GRPO example to the NeMo-Gym GRPO runner.

Default: `false`.

Example: `env.should_use_nemo_gym=true`
```

Frequently used dotlist overrides drawn from the NeMo-RL GRPO recipe include the following.

```{option} grpo.max_num_steps=<n>

The maximum number of training steps.

Example: `grpo.max_num_steps=1000`
```

```{option} grpo.num_prompts_per_step=<n>

The number of prompts sampled per training step.

Example: `grpo.num_prompts_per_step=24`
```

```{option} grpo.use_leave_one_out_baseline=<bool>

When set to `true`, the trainer uses a leave-one-out baseline within each group when computing advantages.

Example: `grpo.use_leave_one_out_baseline=false`
```

```{option} data.train.data_path=<path>

The path to the training JSON Lines dataset.

Example: `data.train.data_path=/lustre/rlvr/train.jsonl`
```

```{option} data.validation.data_path=<path>

The path to the validation JSON Lines dataset.

Example: `data.validation.data_path=/lustre/rlvr/val.jsonl`
```

## Strategies

The manifest records two operator strategies for `rl/nemo_rl/rlvr`.

- When reward variance is low, raise `grpo.num_generations_per_prompt` and keep the leave-one-out baseline enabled.
- When the data follows the Super3 JSON Lines layout or relies on resource-server rewards, start from `config/nemo_gym.yaml` and set `data.train.data_path`, `data.validation.data_path`, and the NeMo-Gym `env.nemo_gym.config_paths` field.

## Command Examples

Run the tiny validation configuration locally:

```console
$ nemotron steps run rl/nemo_rl/rlvr -c tiny
```

Compile the default configuration without submitting the job:

```console
$ nemotron steps run rl/nemo_rl/rlvr -c default --dry-run
```

Submit an attached run on a Lepton profile with a larger group size and more training steps:

```console
$ nemotron steps run rl/nemo_rl/rlvr -c default -r lepton_rl_nemo_rl_rlvr \
    grpo.num_generations_per_prompt=16 \
    grpo.max_num_steps=1000
```

Submit a detached run on a Slurm profile with the NeMo-Gym path and explicit data paths:

```console
$ nemotron steps run rl/nemo_rl/rlvr -c nemo_gym -b slurm_rl_nemo_rl_rlvr \
    env.should_use_nemo_gym=true \
    data.train.data_path=/lustre/rlvr/train.jsonl \
    data.validation.data_path=/lustre/rlvr/val.jsonl
```

## Related Skill

Run the `nemotron-rl-nemo-rl-rlvr` skill with your agent.

## Related Documentation

- [Nemotron Steps CLI Reference](../cli-reference.md) covers the shared option set, dotlist overrides, and passthrough arguments.
- [Choose an RL Alignment Step](../../how-to/choose-rl-step.md) compares the three RL steps.

### Upstream

- [NeMo-RL GRPO Example](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/run_grpo.py)
- [NeMo-RL Repository](https://github.com/NVIDIA-NeMo/RL)
