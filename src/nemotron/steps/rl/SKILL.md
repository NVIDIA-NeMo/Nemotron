---
name: nemotron-rl
description: "Choose among Nemotron NeMo-RL alignment steps: DPO, RLVR or GRPO, and RLHF with reward models. Use when planning, configuring, validating, or debugging reinforcement-learning alignment after SFT."
---

# Nemotron RL

Use this skill to choose a NeMo-RL alignment step and prepare the correct data shape.

## Route

| Need | Step | Data |
| --- | --- | --- |
| Preference-pair alignment without online rewards | `rl/nemo_rl/dpo` | prompt, chosen, rejected |
| Verifiable programmatic rewards | `rl/nemo_rl/rlvr` | prompt plus verifier fields |
| Learned judge or GenRM reward model | `rl/nemo_rl/rlhf` | prompts plus reward-model checkpoint |

## Workflow

1. For Lepton, Slurm, Ray, or batch execution, verify the env profile file first. Default lookup uses repository-root `env.toml`; generated backend files such as `env.lepton.toml` or `env.slurm.toml` require `NEMOTRON_ENV_FILE`.
2. Start from a validated SFT policy checkpoint in `checkpoint_megatron` format.
3. Run `prep/rl_prep` when data references or splits need materialization.
4. Use `config/tiny.yaml` for startup validation and method-specific configs for production.
5. Track KL, reward variance, reward saturation, response length, and held-out examples before scaling rollout count.
6. Check `src/nemotron/steps/patterns/validate-rl-rewards-before-scale.md` before increasing rollout count or trusting reward gains.

## Guardrails

- Keep reward logic deterministic for RLVR whenever possible.
- Keep reward-model checkpoints separate from policy checkpoints for RLHF.
- Validate JSONL schema and reward behavior before launching Ray jobs.
