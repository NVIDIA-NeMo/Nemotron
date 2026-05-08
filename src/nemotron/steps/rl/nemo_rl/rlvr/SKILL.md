---
name: nemotron-rl-nemo-rl-rlvr
description: Configure Nemotron rl/nemo_rl/rlvr for GRPO with verifiable rewards in NeMo-RL. Use for math, code, tool, or environment tasks where rewards come from deterministic answer checks, unit tests, NeMo-Gym resource-server rewards, or explicit verifier fields.
---

# NeMo-RL RLVR

Use `rl/nemo_rl/rlvr` when reward signals are verifiable and can be computed programmatically.

Before changing configs or code, read `step.toml` to understand the step flow, consumed and produced artifacts, important parameters, strategies, failure modes, and upstream references.

## Inputs And Outputs

- Consume prompt `training_jsonl` with verifier fields such as answers.
- Consume an SFT `checkpoint_megatron` policy.
- Produce an RLVR-aligned `checkpoint_megatron`.
- Smoke with `nemotron step run rl/nemo_rl/rlvr -c tiny`.

## Configure

- Increase `grpo.num_generations_per_prompt` when reward variance is too low.
- Keep `grpo.normalize_rewards=true` unless debugging raw reward scale.
- Use `config/nemo_gym.yaml` for resource-server rewards.
- Set `data.train.data_path`, `data.validation.data_path`, and `env.nemo_gym.config_paths` explicitly for NeMo-Gym.
- Check `src/nemotron/steps/patterns/rl-validate-rewards-before-scale.md` before changing GRPO or reward strategy.

## Config Nuances

- Keep configs aligned with the NeMo-RL image schema. GRPO runners often index required fields directly, so missing `grpo`, `loss_fn`, `policy`, `checkpointing`, or `logger` subkeys tend to surface as runtime `KeyError`s.
- Use the response data shape expected by the active NeMo-RL version: `data.train`, `data.validation`, and `data.default` with `dataset_name`, `env_name`, and `processor`. Avoid mixing that structure with legacy top-level dataset path keys.
- Keep validation sample counts numeric whenever validation is scheduled. Choose `grpo.max_val_samples` and `grpo.val_batch_size` together so validator batch math and checkpoint metrics are well defined.
- Size rollout, generation, validation, and training batches for the active Ray worker topology. The rollout batch (`grpo.num_prompts_per_step * grpo.num_generations_per_prompt`) and global policy batch sizes should be divisible by the number of policy shards.
- Treat `policy.logprob_batch_size` as a per-worker logprob microbatch, not always as the global rollout size. After global data is sharded across workers, this value must divide the per-worker shard size.
- Match `checkpointing.metric_name` to the validation metrics produced by the chosen reward path, for example accuracy-style metrics for math verifiers and reward metrics for NeMo-Gym reward servers.
- Keep distributed policy scaffolding explicit in YAML: `dtensor_cfg`, `dynamic_batching`, `sequence_packing`, optimizer `kwargs`, scheduler, generation `vllm_cfg`, and checkpoint save format. Hidden defaults make Ray actor failures hard to diagnose.

## Local Files

- Contract: `src/nemotron/steps/rl/nemo_rl/rlvr/step.toml`
- Runner: `src/nemotron/steps/rl/nemo_rl/rlvr/step.py`
- Configs: `src/nemotron/steps/rl/nemo_rl/rlvr/config/default.yaml`, `src/nemotron/steps/rl/nemo_rl/rlvr/config/tiny.yaml`, `src/nemotron/steps/rl/nemo_rl/rlvr/config/nemo_gym.yaml`

## Guardrails

- Validate reward functions on sample rollouts before training.
- Keep reward outputs bounded and deterministic when possible.
- Avoid ambiguous reward fields; schema drift tends to surface as poor learning rather than clear failures.
