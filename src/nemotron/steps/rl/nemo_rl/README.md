# NeMo-RL

The shared runner for the three NeMo-RL alignment steps lives at
[../../_runners/nemo_rl.py](../../_runners/nemo_rl.py). For the alignment
journey and reward-source decision, see [`../README.md`](../README.md). This
page documents the runner gating and the config shape that all three steps
inherit.

## Algorithms

| Step | Reward shape | Upstream entry |
|---|---|---|
| [`dpo`](dpo/README.md) | Static `{prompt, chosen, rejected}` preference pairs, reference-policy KL objective | `examples/run_dpo.py` |
| [`rlvr`](rlvr/README.md) | GRPO with verifiable rewards (math, tests, env success) | `examples/run_grpo.py` or in-repo NeMo-Gym runner |
| [`rlhf`](rlhf/README.md) | GRPO with a learned judge / GenRM via NeMo-Gym | in-repo NeMo-Gym GRPO runner |

## Runner Gating

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

## Common Config Nuances

- **Schema is strict.** GRPO runners often index required fields directly.
  Missing `grpo`, `loss_fn`, `policy`, `checkpointing`, or `logger` subkeys
  surface as runtime `KeyError`. Don't rely on hidden defaults.
- **Batch sizing must match Ray topology.** Rollout batch
  (`grpo.num_prompts_per_step * grpo.num_generations_per_prompt`) and global
  policy batch must be divisible by the policy-shard count.
- **`policy.logprob_batch_size`** is per-worker after sharding, not global.
- **Validation cadence** in DPO uses explicit `dpo.val_at_start` /
  `val_period` / `val_at_end`. Don't rely on upstream defaults when changing
  run length.
- **Output paths** should derive from env vars (`RL_OUTPUT_DIR`) so repeated
  runs don't collide.

## Repository Layout

- `dpo/`: `step.toml`, `step.py`, `config/{default,tiny}.yaml`
- `rlvr/`: `step.toml`, `step.py`, `config/{default,tiny,nemo_gym}.yaml`
- `rlhf/`: `step.toml`, `step.py`, `config/{default,tiny}.yaml`

## Guardrails

- Keep policy checkpoints in Megatron format unless a config explicitly
  expects HF.
- Keep reward-model serving, policy rollout, and validation paths explicit.
- Treat Ray + NeMo-Gym setup as part of the experiment, not infrastructure to
  ignore.
- Don't change the runner shim unless the upstream image is upgraded and
  revalidated — Ray failures and reward-model failures look identical at
  startup.
