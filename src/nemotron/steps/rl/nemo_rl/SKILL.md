---
name: nemotron-rl-nemo-rl
description: Navigate the Nemotron rl/nemo_rl family for DPO, RLVR/GRPO, and RLHF. Use when working with NeMo-RL launchers, Ray execution, data schemas, reward configuration, NeMo-Gym, rollout settings, and Megatron checkpoint alignment outputs.
---

# NeMo-RL

Use the `rl/nemo_rl` family for post-SFT alignment that produces Megatron
checkpoints. All three steps share the runner at
[../../_runners/nemo_rl.py](../../_runners/nemo_rl.py).

## Algorithms

| Step | Reward shape | Recipe URL |
|---|---|---|
| [`dpo`](dpo/SKILL.md) | Static `{prompt, chosen, rejected}` preference pairs, reference-policy KL objective | `examples/run_dpo.py` (NeMo-RL) |
| [`rlvr`](rlvr/SKILL.md) | GRPO with verifiable rewards (math, tests, env success). Two runners: upstream `examples/run_grpo.py` or the in-repo NeMo-Gym runner | `examples/run_grpo.py` (or NeMo-Gym mode) |
| [`rlhf`](rlhf/SKILL.md) | GRPO with a learned judge / GenRM via NeMo-Gym | NeMo-Gym GRPO runner (in-repo) |

## Runner gating (the non-obvious part)

The shared runner has two helpers:

- `exec_nemo_rl_example(...)` — DPO uses this. Forwards `--config` and
  Hydra-style overrides to a NeMo-RL example via `os.execvp`.
- `exec_or_run_nemo_rl_grpo(...)` — RLVR/RLHF use this. Inspects the loaded
  config:
  - `env.should_use_nemo_gym = false` → exec upstream NeMo-RL example.
  - `env.should_use_nemo_gym = true` → call in-repo
    `nemo_rl_grpo_nemo_gym.run_nemo_gym_grpo(...)` directly (no exec).

The two paths have different Ray actor topologies. Don't mix configs.

The local `defaults: <yaml>` form in YAML is a small layering convenience
(single string or list); it is **not** a full Hydra composition engine.

## Workflow

1. Read the algorithm's `step.toml` for consumed artifacts and main knobs.
2. Run [`../prep/rl_prep`](../../prep/rl_prep/SKILL.md) when data starts as
   HF references or unsharded blends.
3. Use `config/tiny.yaml` for runner validation. Use `config/nemo_gym.yaml`
   (RLVR/RLHF) when resource-server or GenRM rewards are required.
4. Validate rewards on a small set before scaling rollout count — see
   [../../patterns/rl-validate-rewards-before-scale.md](../../patterns/rl-validate-rewards-before-scale.md).
5. Track KL, reward variance, reward saturation, response length, held-out
   evals.

## Common config nuances

- **Schema is strict.** GRPO runners often index required fields directly.
  Missing `grpo`, `loss_fn`, `policy`, `checkpointing`, or `logger` subkeys
  surface as runtime `KeyError`. Don't rely on hidden defaults.
- **Batch sizing must match Ray topology.** Rollout batch
  (`grpo.num_prompts_per_step * grpo.num_generations_per_prompt`) and global
  policy batch must be divisible by the policy-shard count.
- **`policy.logprob_batch_size`** is per-worker after sharding, not global.
- **Validation cadence** in DPO uses explicit `dpo.val_at_start` /
  `val_period` / `val_at_end`. Don't rely on upstream defaults when
  changing run length.
- **Output paths** should derive from env vars (`RL_OUTPUT_DIR`) so repeated
  runs don't collide.

## Local files

- `dpo/`: `step.toml`, `step.py`, `config/{default,tiny}.yaml`
- `rlvr/`: `step.toml`, `step.py`, `config/{default,tiny,nemo_gym}.yaml`
- `rlhf/`: `step.toml`, `step.py`, `config/{default,tiny}.yaml`

## Patterns to cite

- [../../patterns/rl-validate-rewards-before-scale.md](../../patterns/rl-validate-rewards-before-scale.md) — validate every reward path before scaling.
- [../../patterns/eval-before-and-after-training.md](../../patterns/eval-before-and-after-training.md) — RL must be scored on task evals, not just reward.
- [../../patterns/byob-benchmark-design.md](../../patterns/byob-benchmark-design.md) — for sovereign deployments, the eval is the BYOB.
- [../../patterns/prep-data-is-tokenizer-locked.md](../../patterns/prep-data-is-tokenizer-locked.md) — RL data sharded through `prep/rl_prep` inherits the tokenizer-lock invariant.

## Guardrails

- Keep policy checkpoints in Megatron format unless a config explicitly
  expects HF.
- Keep reward-model serving, policy rollout, and validation paths explicit.
- Treat Ray + NeMo-Gym setup as part of the experiment, not infrastructure
  to ignore.
- Don't change the runner shim unless the upstream image is upgraded and
  revalidated — Ray failures and reward-model failures look identical at
  startup.
