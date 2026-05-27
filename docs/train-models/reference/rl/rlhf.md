---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Command-line reference for the rl/nemo_rl/rlhf training step."
topics: ["Training", "Reference", "CLI", "RL", "RLHF", "GRPO"]
tags: ["Reference", "CLI", "Steps", "RL", "RLHF", "GRPO", "Reward Model", "GenRM", "NeMo-RL", "NeMo-Gym"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Developer"]
---

# rl/nemo_rl/rlhf

This step runs reinforcement learning from human feedback (RLHF) on top of the NeMo-RL group relative policy optimization (GRPO) training loop, with a generative reward model judge.
The step consumes a prompt dataset for rollouts, a supervised fine-tuning (SFT) Megatron checkpoint as the warm-start policy, and a Hugging Face (HF) reward model checkpoint that NeMo-Gym serves as the judge.
The step produces an aligned `checkpoint_megatron` artifact.

## Syntax

```bash
nemotron steps run rl/nemo_rl/rlhf \
    [-c <config-name-or-path>] \
    [-r <run-profile> | -b <batch-profile>] \
    [-d] \
    [--force-squash] \
    [<dotlist-overrides>...] \
    [<passthrough-args>...]
```

Refer to the [Nemotron Steps CLI Reference](../cli-reference.md) for the shared flag set.

## Configuration Files

The step ships two configuration files under `src/nemotron/steps/rl/nemo_rl/rlhf/config/`.

| File | Purpose |
| --- | --- |
| `default.yaml` | NeMo-Gym RLHF path with GRPO sampling and a generative reward model judge that NeMo-Gym hosts. |
| `tiny.yaml` | Short validation run with a small dataset slice. |

Pass the configuration name with `-c`:

```console
$ nemotron steps run rl/nemo_rl/rlhf -c tiny
$ nemotron steps run rl/nemo_rl/rlhf -c default
```

## Inputs and Outputs

| Direction | Artifact Type | Required | Description |
| --- | --- | --- | --- |
| Consumes | `training_jsonl` | Yes | Prompt JSON Lines used for rollouts. |
| Consumes | `checkpoint_megatron` | Yes | The supervised fine-tuned policy to align. |
| Consumes | `checkpoint_hf` | Yes | The reward model checkpoint, in Hugging Face format, served as the generative reward model judge. |
| Produces | `checkpoint_megatron` | — | The RLHF-aligned policy checkpoint. |

## Step Parameters

The manifest declares two NeMo-Gym RLHF parameters.
Pass them as dotlist overrides.

```{option} grpo.num_generations_per_prompt=<n>

The number of rollouts produced per prompt.
This value sets the GRPO group size.

Default: `8`.

Example: `grpo.num_generations_per_prompt=16`
```

```{option} env.nemo_gym.genrm_model.responses_api_models.vllm_model.model=<id-or-path>

The Hugging Face identifier or local path for the generative reward model that NeMo-Gym serves through its responses API.

Default: `meta-llama/Llama-3.2-1B-Instruct` (or the value of `RL_REWARD_MODEL` when that variable is set).

Example: `env.nemo_gym.genrm_model.responses_api_models.vllm_model.model=nvidia/Nemotron-Reward-Model`
```

Frequently used dotlist overrides drawn from the NeMo-RL GRPO recipe include the following.

```{option} grpo.max_num_steps=<n>

The maximum number of training steps.

Example: `grpo.max_num_steps=1000`
```

```{option} grpo.num_prompts_per_step=<n>

The number of prompts sampled per training step.

Example: `grpo.num_prompts_per_step=32`
```

```{option} data.train.data_path=<path>

The path to the training JSON Lines dataset.

Example: `data.train.data_path=/lustre/rlhf/train.jsonl`
```

```{option} data.validation.data_path=<path>

The path to the validation JSON Lines dataset.

Example: `data.validation.data_path=/lustre/rlhf/val.jsonl`
```

## Strategies

The manifest records two operator strategies for `rl/nemo_rl/rlhf`.

- When the reward model saturates or reward hacking appears in rollouts, raise the Kullback-Leibler penalty, lower the learning rate, or add reward clipping.
- When the data follows the Super3 RLHF layout, keep `env.should_use_nemo_gym=true` and point `data.train.data_path` and `data.validation.data_path` at the prepared NeMo-Gym JSON Lines files.

## Command Examples

Run the tiny validation configuration locally:

```console
$ nemotron steps run rl/nemo_rl/rlhf -c tiny
```

Compile the default configuration without submitting the job:

```console
$ nemotron steps run rl/nemo_rl/rlhf -c default --dry-run
```

Submit an attached run on a Lepton profile with an explicit reward model identifier:

```console
$ nemotron steps run rl/nemo_rl/rlhf -c default -r lepton_rl_nemo_rl_rlhf \
    env.nemo_gym.genrm_model.responses_api_models.vllm_model.model=nvidia/Nemotron-Reward-Model \
    grpo.num_generations_per_prompt=16
```

Submit a detached run on a Slurm profile with a longer training schedule:

```console
$ nemotron steps run rl/nemo_rl/rlhf -c default -b slurm_rl_nemo_rl_rlhf \
    grpo.max_num_steps=1000 \
    data.train.data_path=/lustre/rlhf/train.jsonl \
    data.validation.data_path=/lustre/rlhf/val.jsonl
```

## Related Skill

Run the `nemotron-rl-nemo-rl-rlhf` skill with your agent.

## Related Documentation

- [Nemotron Steps CLI Reference](../cli-reference.md) covers the shared option set, dotlist overrides, and passthrough arguments.
- [Choose an RL Alignment Step](../../how-to/choose-rl-step.md) compares the three RL steps.

### Upstream

- [Super3 RLHF Recipe](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/src/nemotron/recipes/super3/stage2_rl/stage3_rlhf)
- [NeMo-RL Repository](https://github.com/NVIDIA-NeMo/RL)
