---
name: nemotron-rl
description: "Choose among Nemotron NeMo-RL alignment steps: DPO, RLVR/GRPO, and RLHF with reward models. Use when planning, configuring, validating, or debugging reinforcement-learning alignment after SFT, or when reward design (verifiable, preference, or learned) decides the algorithm."
---

# Nemotron RL

Pick a NeMo-RL alignment step by **how reward is computed**. The catalog
ships three algorithms — DPO, RLVR, RLHF — all under the
[`rl/nemo_rl/`](nemo_rl/SKILL.md) subcategory.

## Steps

| Reward source | Step | Required data shape | Output |
|---|---|---|---|
| Static preference pairs (no online reward) | [`rl/nemo_rl/dpo`](nemo_rl/dpo/SKILL.md) | `{prompt, chosen, rejected}` | `checkpoint_megatron` |
| Programmatic / verifiable (math answer, tests, tool success) | [`rl/nemo_rl/rlvr`](nemo_rl/rlvr/SKILL.md) | `{prompt, answer | tests | env_metadata}` | `checkpoint_megatron` |
| Learned judge / GenRM-style comparison reward | [`rl/nemo_rl/rlhf`](nemo_rl/rlhf/SKILL.md) | `{prompt}` + reward-model `checkpoint_hf` | `checkpoint_megatron` |

All three consume an SFT-trained `checkpoint_megatron` policy as the warm
start. RLVR and RLHF additionally support **NeMo-Gym** mode for
resource-server / GenRM rewards (`env.should_use_nemo_gym=true`).

## Decision tree

- Have static preference pairs and no reward function → **DPO**.
- Reward is deterministic and codifiable (regex / unit test / answer match) → **RLVR**.
- Reward must be learned from human comparisons or a judge model → **RLHF**.
- Sovereign deployment, custom reward server, or domain-specific tool rewards
  → RLVR or RLHF with NeMo-Gym (`config/nemo_gym.yaml`).

## Pre-conditions

1. **A validated SFT policy** in `checkpoint_megatron` format. RL trains
   *deltas*, not behaviors from scratch.
2. **Validated reward design** — RL fails are usually reward-design fails,
   not launch fails. See
   [../patterns/rl-validate-rewards-before-scale.md](../patterns/rl-validate-rewards-before-scale.md).
3. **Materialized data**. If data starts as HF references, run
   [`prep/rl_prep`](../prep/rl_prep/SKILL.md) first to resolve placeholders
   into local JSONL.

## Pipeline placement

```
... → sft/megatron_bridge → prep/rl_prep → rl/nemo_rl/dpo   → checkpoint_megatron
                                          → rl/nemo_rl/rlvr  → checkpoint_megatron
                                          → rl/nemo_rl/rlhf  → checkpoint_megatron
```

Output is Megatron-format. Add [`convert/megatron_to_hf`](../convert/megatron_to_hf/step.toml)
when the next consumer (eval, deployment) expects HF.

## Workflow

1. **Env profile first** — verify the env profile for Lepton/Slurm/Ray runs
   (`env.toml` by default, or `NEMOTRON_ENV_FILE` for backend-specific files).
2. Confirm the SFT warm-start checkpoint exists and was trained on a
   compatible tokenizer and chat template.
3. Run [`prep/rl_prep`](../prep/rl_prep/SKILL.md) when data needs HF
   resolution or sharding.
4. Pick the step per the decision tree.
5. Validate the reward path on a tiny set **before scaling rollout count** —
   see the rewards pattern above.
6. Use `config/tiny.yaml` for runner validation; method-specific configs
   (`config/nemo_gym.yaml` for resource-server rewards) for production.
7. Track KL, reward variance, reward saturation, response length, and
   held-out task evals — not just reward.
8. Bookend with eval — see
   [../patterns/eval-before-and-after-training.md](../patterns/eval-before-and-after-training.md).
   For sovereign deployments judge against
   [../patterns/byob-benchmark-design.md](../patterns/byob-benchmark-design.md).

## Smoke commands

```bash
nemotron steps run rl/nemo_rl/dpo  -c tiny
nemotron steps run rl/nemo_rl/rlvr -c tiny
nemotron steps run rl/nemo_rl/rlhf -c tiny
```

## Guardrails

- Never trust reward gain alone. Score on a held-out task eval before
  scaling rollout count.
- Keep RLVR reward functions deterministic, bounded, and tested
  independently.
- Keep reward-model serving (RLHF) validated separately from policy
  optimization — failures otherwise look like the policy's fault.
- Inverted DPO preference pairs silently teach the wrong behavior. Validate
  ordering before launching.
- For NeMo-Gym, the runner switches from upstream example to in-repo
  resource-server runner via `env.should_use_nemo_gym=true`. The two paths
  have different actor topologies — don't mix configs.
