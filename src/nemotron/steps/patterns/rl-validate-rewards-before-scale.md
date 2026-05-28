---
id: rl-validate-rewards-before-scale
title: "Validate RL rewards before scaling rollouts"
tags: [rl, rewards, validation]
triggers:
  - "An RLVR reward function, NeMo-Gym resource server, or learned reward model is being added."
  - "Reward is improving but held-out examples or human review look worse."
  - "A DPO, RLVR, or RLHF run is moving from tiny validation to production rollout counts."
steps: [data_prep/rl_prep, rl/nemo_rl/dpo, rl/nemo_rl/rlvr, rl/nemo_rl/rlhf]
confidence: high
---

## When to apply

Use this before scaling any NeMo-RL job. RL failures are often not launch failures; they are reward-design failures that train the policy toward the wrong behavior.

Apply it for deterministic RLVR rewards, NeMo-Gym resource-server rewards, learned judge rewards, GenRM-style comparison rewards, and DPO preference pairs.

It is especially important when the reward source is new, when reward variance is low, or when rollouts are expensive enough that a bad job would waste meaningful cluster time.

## What to do

Create a small reward validation set with known good, known bad, boundary, and adversarial examples. Run the reward function or judge on those examples before policy optimization.

For DPO, verify prompt, chosen, and rejected ordering. Inverted pairs are silent and damaging.

For RLVR, keep reward functions deterministic, bounded, fast, and tested independently. Make answer normalization, unit-test timeouts, tool failures, and partial credit rules explicit.

For RLHF, validate reward-model serving separately from policy training. Check for verbosity bias, refusal bias, formatting shortcuts, and reward saturation.

Track reward mean, reward variance, KL, clip fraction, response length, overlong rate, success rate, and representative validation examples during early runs.

If rewards improve while task evals degrade, stop and inspect reward examples before increasing rollout count. Add constraints, tune KL, or fix the reward source instead of scaling the failure.

## Exceptions

For a pure runner smoke test, tiny configs can skip deep reward analysis. The moment the result is used to judge alignment quality, reward validation becomes mandatory.

Some exploratory reward models are intentionally noisy. In that case, document the risk and keep rollout budgets small until the signal proves useful.

## References

- Pair with `eval-before-and-after-training` for pre/post alignment comparisons against task evals (not just reward).
- Pair with `prep-data-is-tokenizer-locked` when RL data is sharded or materialized through `data_prep/rl_prep`.
- Pair with `byob-benchmark-design` — RL alignment must be scored against a held-out benchmark the reward function never saw.
- Pair with `sdg-pipeline-versioning` when synthetic preferences (Data Designer `rl_pref.yaml`) feed DPO.
- Pair with `data-quality-before-quantity` — bad reward sources scale failure faster than good rewards scale success.
- This pattern applies to DPO data quality as much as online RL reward design.
