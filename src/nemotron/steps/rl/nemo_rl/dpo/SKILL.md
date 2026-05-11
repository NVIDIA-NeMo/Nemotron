---
name: nemotron-rl-nemo-rl-dpo
description: Configure Nemotron rl/nemo_rl/dpo for Direct Preference Optimization with NeMo-RL. Use for offline preference-pair alignment from prompt, chosen, and rejected JSONL plus an SFT Megatron checkpoint, KL tuning, and preference schema validation.
---

# NeMo-RL DPO

Use `rl/nemo_rl/dpo` when the training signal is static preference pairs rather than online reward functions.

Before changing configs or code, read `step.toml` to understand the step flow, consumed and produced artifacts, important parameters, strategies, failure modes, and upstream references.

## Inputs And Outputs

- Consume `training_jsonl` with prompt, chosen, and rejected fields.
- Consume an SFT `checkpoint_megatron` policy.
- Produce a DPO-aligned `checkpoint_megatron`.
- Smoke with `nemotron steps run rl/nemo_rl/dpo -c tiny`.

## Configure

- Tune `dpo.reference_policy_kl_penalty` when KL collapses or loss diverges.
- Lower learning rate before making structural changes to the runner.
- Use `prep/rl_prep` when preference data starts as HF references or blended local files.
- Keep the reference policy aligned with the SFT policy.
- Check `src/nemotron/steps/patterns/rl-validate-rewards-before-scale.md` before trusting preference-pair training results.

## Config Nuances

- Keep validation cadence explicit with fields such as `dpo.val_at_start`, `dpo.val_period`, and `dpo.val_at_end`; do not rely on upstream defaults when changing run length.
- Keep `policy.train_global_batch_size` divisible by the active policy worker shape and micro batch size.
- Keep `cluster.num_nodes` and `cluster.gpus_per_node` aligned with the RayCluster executor shape.
- Use environment-derived output paths such as `RL_OUTPUT_DIR` for logs and checkpoints so repeated runs do not collide.

## Local Files

- Contract: `src/nemotron/steps/rl/nemo_rl/dpo/step.toml`
- Runner: `src/nemotron/steps/rl/nemo_rl/dpo/step.py`
- Configs: `src/nemotron/steps/rl/nemo_rl/dpo/config/default.yaml`, `src/nemotron/steps/rl/nemo_rl/dpo/config/tiny.yaml`

## Guardrails

- Validate chosen and rejected ordering; inverted pairs silently teach the wrong behavior.
- Keep train and validation preference distributions comparable.
- Inspect examples where the model regresses after DPO; preference data can encode style bias.
