---
license: Apache-2.0
copyright: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
description: "Command-line reference for the rl/nemo_rl/dpo training step."
topics: ["Training", "Reference", "CLI", "RL", "DPO"]
tags: ["Reference", "CLI", "Steps", "RL", "DPO", "Alignment", "NeMo-RL"]
content:
  type: "Reference"
  difficulty: "All Levels"
  audience: ["ML Engineer", "Developer"]
---

# rl/nemo_rl/dpo

This step runs direct preference optimization (DPO) alignment on NeMo-RL.
The step consumes a preference dataset of chosen-and-rejected response pairs, together with a supervised fine-tuning (SFT) Megatron checkpoint that serves as the warm-start policy.
The step produces an aligned `checkpoint_megatron` artifact.

## Syntax

```bash
nemotron steps run rl/nemo_rl/dpo \
    [-c <config-name-or-path>] \
    [-r <run-profile> | -b <batch-profile>] \
    [-d] \
    [--force-squash] \
    [<dotlist-overrides>...] \
    [<passthrough-args>...]
```

Refer to the [Nemotron Steps CLI Reference](../cli-reference.md) for the shared flag set.

## Configuration Files

The step ships two configuration files under `src/nemotron/steps/rl/nemo_rl/dpo/config/`.

| File | Purpose |
| --- | --- |
| `default.yaml` | Full-shape DPO with a five-hundred-step training schedule and `meta-llama/Llama-3.2-1B-Instruct` as the policy. |
| `tiny.yaml` | Short validation run with a small dataset slice. |

Pass the configuration name with `-c`:

```console
$ nemotron steps run rl/nemo_rl/dpo -c tiny
$ nemotron steps run rl/nemo_rl/dpo -c default
```

## Inputs and Outputs

| Direction | Artifact Type | Required | Description |
| --- | --- | --- | --- |
| Consumes | `training_jsonl` | Yes | Preference JSON Lines with `prompt`, `chosen`, and `rejected` fields. |
| Consumes | `checkpoint_megatron` | Yes | The supervised fine-tuned policy to align. |
| Produces | `checkpoint_megatron` | — | The DPO-aligned policy checkpoint. |

## Step Parameters

The manifest declares one DPO-specific parameter.
Pass it as a dotlist override.

```{option} dpo.reference_policy_kl_penalty=<float>

The Kullback-Leibler (KL) penalty between the trained policy and the reference policy.
This value corresponds to β in the DPO objective.
Lower values let the policy drift further from the reference; higher values keep the policy close to the reference.

Default: `0.05`.

Example: `dpo.reference_policy_kl_penalty=0.1`
```

Frequently used dotlist overrides drawn from the NeMo-RL DPO recipe include the following.

```{option} policy.model_name=<id-or-path>

The Hugging Face identifier or local path for the policy.
The default configuration reads this value from the `RL_POLICY_MODEL` environment variable.

Example: `policy.model_name=meta-llama/Llama-3.2-3B-Instruct`
```

```{option} dpo.max_num_steps=<n>

The maximum number of training steps.

Example: `dpo.max_num_steps=1000`
```

```{option} policy.train_global_batch_size=<n>

The global batch size across all data-parallel workers.

Example: `policy.train_global_batch_size=64`
```

```{option} policy.optimizer.lr=<float>

The optimizer learning rate.

Example: `policy.optimizer.lr=1.0e-6`
```

## Strategies

The manifest records one operator strategy for `rl/nemo_rl/dpo`.

- When the loss diverges or the Kullback-Leibler divergence collapses, raise `dpo.reference_policy_kl_penalty` to a value in the 0.1-0.3 range, or lower the learning rate.

## Command Examples

Run the tiny validation configuration locally:

```console
$ nemotron steps run rl/nemo_rl/dpo -c tiny
```

Compile the default configuration without submitting the job:

```console
$ nemotron steps run rl/nemo_rl/dpo -c default --dry-run
```

Submit an attached run on a Lepton profile with a tighter Kullback-Leibler penalty:

```console
$ nemotron steps run rl/nemo_rl/dpo -c default -r lepton_rl_nemo_rl_dpo \
    dpo.reference_policy_kl_penalty=0.2 \
    dpo.max_num_steps=1000
```

Submit a detached run on a Slurm profile with a larger policy:

```console
$ nemotron steps run rl/nemo_rl/dpo -c default -b slurm_rl_nemo_rl_dpo \
    policy.model_name=meta-llama/Llama-3.2-3B-Instruct \
    policy.train_global_batch_size=64 \
    policy.optimizer.lr=1.0e-6
```

## Related Skill

Run the `nemotron-rl-nemo-rl-dpo` skill with your agent.

## Related Documentation

- [Nemotron Steps CLI Reference](../cli-reference.md) covers the shared option set, dotlist overrides, and passthrough arguments.
- [Choose an RL Alignment Step](../../how-to/choose-rl-step.md) compares the three RL steps.

### Upstream

- [NeMo-RL DPO Example](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/run_dpo.py)
- [NeMo-RL Repository](https://github.com/NVIDIA-NeMo/RL)
