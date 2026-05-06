---
name: nemotron-rl-nemo-rl-rlhf
description: Configure Nemotron rl/nemo_rl/rlhf for RLHF with NeMo-RL GRPO and a learned judge or generative reward model. Use when alignment depends on reward-model checkpoints, GenRM-style comparison rewards, NeMo-Gym reward serving, KL control, or reward-model validation.
---

# NeMo-RL RLHF

Use `rl/nemo_rl/rlhf` when rewards come from a learned judge or generative reward model rather than deterministic verification.

Before changing configs or code, read `step.toml` to understand the step flow, consumed and produced artifacts, important parameters, strategies, failure modes, and upstream references.

## Inputs And Outputs

- Consume prompt `training_jsonl`.
- Consume an SFT `checkpoint_megatron` policy.
- Consume a reward-model `checkpoint_hf`.
- Produce an RLHF-aligned `checkpoint_megatron`.
- Smoke with `nemotron step run rl/nemo_rl/rlhf -c tiny`.

## Configure

- Set `env.nemo_gym.genrm_model.responses_api_models.vllm_model.model` to the reward-model path.
- Tune `grpo.num_generations_per_prompt` based on reward variance and serving cost.
- Increase KL penalty, lower learning rate, or clip rewards when reward hacking appears.
- Keep `env.should_use_nemo_gym=true` for GenRM-style comparison rewards.
- Check `src/nemotron/steps/patterns/validate-rl-rewards-before-scale.md` before changing RLHF reward or rollout behavior.

## Config Nuances

- Ensure the RL prep manifest has non-empty `train` and `val` paths; only allow train-as-validation when the run is explicitly non-evaluative.
- Keep each NeMo-Gym row compatible with the Responses API path. Data should include or be normalized into `responses_create_params` before NeMo-RL loads response data.
- Size GenRM vLLM according to available GPU memory and concurrency: tune tensor parallelism, `max_num_seqs`, `max_model_len`, and `gpu_memory_utilization` so the `genrm_model` server reaches readiness.
- Keep the runner close to NeMo-RL's upstream NeMo-Gym GRPO example; avoid local rollout or actor shims unless the target image is upgraded and revalidated.
- Keep policy, reference, GenRM, and resource-server settings explicit in YAML; hidden defaults make Ray startup failures hard to distinguish from reward-model failures.

## Local Files

- Contract: `src/nemotron/steps/rl/nemo_rl/rlhf/step.toml`
- Runner: `src/nemotron/steps/rl/nemo_rl/rlhf/step.py`
- Configs: `src/nemotron/steps/rl/nemo_rl/rlhf/config/default.yaml`, `src/nemotron/steps/rl/nemo_rl/rlhf/config/tiny.yaml`
- Recipe reference: `src/nemotron/recipes/super3/stage2_rl/`

## Guardrails

- Validate reward-model serving separately before launching policy optimization.
- Keep policy, reference, and reward-model checkpoints clearly separated in config.
- Review held-out reward examples to detect judge bias or reward saturation.
