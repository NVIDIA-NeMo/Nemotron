# Nemotron RL

Pick a NeMo-RL alignment step by **how reward is computed**. The catalog
ships three algorithms — DPO, RLVR, RLHF — all under the
[`rl/nemo_rl/`](nemo_rl/README.md) subcategory.

## Developer Journey

RL starts from a trained policy and a reward/data contract. Before choosing DPO,
RLVR, or RLHF, validate the data shape and the reward path on a tiny set. Most
RL failures are schema or reward-design failures, not optimizer failures.

1. Start from a validated SFT `checkpoint_megatron`.
2. Prepare local prompt or preference JSONL through
   [`../data_prep/rl_prep/README.md`](../data_prep/rl_prep/README.md) when data
   comes from blends, HF references, or needs sharding.
3. Pick the RL step by reward source: static preference, verifiable reward, or
   learned judge.
4. Run a tiny rollout/reward validation before scaling rollout count or GPUs.
5. Track held-out task metrics, reward saturation, response length, and KL.

## Steps

| Reward source | Step | Required data shape | Output |
|---|---|---|---|
| Static preference pairs (no online reward) | [`rl/nemo_rl/dpo`](nemo_rl/dpo/README.md) | `{prompt, chosen, rejected}` | `checkpoint_megatron` |
| Programmatic / verifiable (math answer, tests, tool success) | [`rl/nemo_rl/rlvr`](nemo_rl/rlvr/README.md) | `{prompt, answer | tests | env_metadata}` | `checkpoint_megatron` |
| Learned judge / GenRM-style comparison reward | [`rl/nemo_rl/rlhf`](nemo_rl/rlhf/README.md) | `{prompt}` + reward-model `checkpoint_hf` | `checkpoint_megatron` |

All three consume an SFT-trained `checkpoint_megatron` policy as the warm
start. RLVR and RLHF additionally support **NeMo-Gym** mode for
resource-server / GenRM rewards (`env.should_use_nemo_gym=true`).

## Decision Guide

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
   [`data_prep/rl_prep`](../data_prep/rl_prep/README.md) first to resolve placeholders
   into local JSONL.

## Data And Artifact Flow

```
... → sft/megatron_bridge → data_prep/rl_prep → rl/nemo_rl/dpo   → checkpoint_megatron
                                              → rl/nemo_rl/rlvr  → checkpoint_megatron
                                              → rl/nemo_rl/rlhf  → checkpoint_megatron
```

Output is Megatron-format. Add [`convert/megatron_to_hf`](../convert/megatron_to_hf/step.toml)
when the next consumer (eval, deployment) expects HF.

```text
preference records {prompt, chosen, rejected}
  -> data_prep/rl_prep
  -> rl/nemo_rl/dpo
```

```text
prompt records + verifier fields
  -> data_prep/rl_prep
  -> rl/nemo_rl/rlvr
```

```text
prompt records + reward-model checkpoint_hf / GenRM service config
  -> data_prep/rl_prep
  -> rl/nemo_rl/rlhf
```

Keep prompt data, verifier metadata, reward-model checkpoints, and resource
server configs separate in the project layout. They are related inputs, not the
same artifact.

## Workflow

1. Confirm the SFT warm-start checkpoint exists and was trained on a
   compatible tokenizer and chat template.
2. Run [`data_prep/rl_prep`](../data_prep/rl_prep/README.md) when data needs HF
   resolution or sharding.
3. Pick the step per the decision tree.
4. Validate the reward path on a tiny set **before scaling rollout count** —
   see the rewards pattern above.
5. Use `config/tiny.yaml` for runner validation; method-specific configs
   (`config/nemo_gym.yaml` for resource-server rewards) for production.
6. For remote submission, select the profile from
   `env/env_toml/config/{lepton,slurm,dgxcloud}.yaml` or the generated env file;
   do not hardcode profile names here.
7. Track KL, reward variance, reward saturation, response length, and
   held-out task evals — not just reward.
8. Bookend with eval — see
   [../patterns/eval-before-and-after-training.md](../patterns/eval-before-and-after-training.md).
   For sovereign deployments judge against
   [../patterns/byob-benchmark-design.md](../patterns/byob-benchmark-design.md).

## Smoke commands

```bash
uv run nemotron steps run rl/nemo_rl/dpo  -c tiny --dry-run
uv run nemotron steps run rl/nemo_rl/rlvr -c tiny --dry-run
uv run nemotron steps run rl/nemo_rl/rlhf -c tiny --dry-run
```

## Project layout for generated configs

Keep every generated overlay config and any supporting code under a single
self-contained project root that also holds the local input data, so the
whole directory is rsync/scp-portable to the remote machine that will run
the alignment step.

- `<project>/config/` for generated YAML — never write into
  `src/nemotron/steps/rl/nemo_rl/<algo>/config/`; the shipped
  `default.yaml`, `tiny.yaml`, and `nemo_gym.yaml` stay as catalog
  references.
- `<project>/data/` for sharded preference / prompt / verifier JSONL
  produced by `data_prep/rl_prep`.
- Keep the SFT warm-start `checkpoint_megatron` path, RLHF reward-model
  `checkpoint_hf` path, and any NeMo-Gym resource-server configs resolvable
  under the same project root so the whole alignment job ships in one
  bundle.
- Project-root scripts only when catalog code cannot serve the request.
- Do not split generated files into home dirs, scratch dirs, or paths
  outside the project root that will not ship with the bundle.

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
