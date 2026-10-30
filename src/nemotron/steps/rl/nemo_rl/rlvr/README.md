# NeMo-RL RLVR

Use `rl/nemo_rl/rlvr` when reward signals are verifiable and can be computed programmatically.

Use this README for workflow and pitfalls; use `step.toml` for the exact artifact, parameter, strategy, and error manifest before editing configs or code.

## Inputs And Outputs

- Consume prompt `training_jsonl` with verifier fields such as answers.
- Consume an SFT `checkpoint_megatron` policy.
- Produce an RLVR-aligned `checkpoint_megatron`.
- Smoke with `nemotron steps run rl/nemo_rl/rlvr -c tiny`.

## CLI And Overlay Knobs

Start from `config/tiny.yaml` for runner validation. Use `config/nemo_gym.yaml`
when resource-server rewards are required. In a project overlay, developers
usually change:

- `data.train.data_path` and `data.validation.data_path`: prompt JSONL with
  verifier fields.
- `grpo.num_prompts_per_step` and `grpo.num_generations_per_prompt`: rollout
  batch size and reward variance.
- `policy.logprob_batch_size`: per-worker logprob microbatch after sharding.
- `env.should_use_nemo_gym`: switch to the NeMo-Gym runner only with matching
  resource-server config.
- `env.nemo_gym.config_paths`: resource-server configs for NeMo-Gym mode.

Example shape:

```bash
uv run nemotron steps run rl/nemo_rl/rlvr \
  -c <project>/config/rlvr.yaml \
  data.train.data_path=<rl-prep>/train.jsonl \
  data.validation.data_path=<rl-prep>/validation.jsonl
```

Related patterns:

- Check `src/nemotron/steps/patterns/rl-validate-rewards-before-scale.md` before changing GRPO or reward strategy.

## Config Nuances

- Keep configs aligned with the NeMo-RL image schema. GRPO runners often index required fields directly, so missing `grpo`, `loss_fn`, `policy`, `checkpointing`, or `logger` subkeys tend to surface as runtime `KeyError`s.
- Use the response data shape expected by the active NeMo-RL version: `data.train`, `data.validation`, and `data.default` with `dataset_name`, `env_name`, and `processor`. Avoid mixing that structure with legacy top-level dataset path keys.
- Keep validation sample counts numeric whenever validation is scheduled. Choose `grpo.max_val_samples` and `grpo.val_batch_size` together so validator batch math and checkpoint metrics are well defined.
- Size rollout, generation, validation, and training batches for the active Ray worker topology. The rollout batch (`grpo.num_prompts_per_step * grpo.num_generations_per_prompt`) and global policy batch sizes should be divisible by the number of policy shards.
- Treat `policy.logprob_batch_size` as a per-worker logprob microbatch, not always as the global rollout size. After global data is sharded across workers, this value must divide the per-worker shard size.
- Match `checkpointing.metric_name` to the validation metrics produced by the chosen reward path, for example accuracy-style metrics for math verifiers and reward metrics for NeMo-Gym reward servers.
- Keep distributed policy scaffolding explicit in YAML: `dtensor_cfg`, `dynamic_batching`, `sequence_packing`, optimizer `kwargs`, scheduler, generation `vllm_cfg`, and checkpoint save format. Hidden defaults make Ray actor failures hard to diagnose.

## Run It

Smoke first to validate wiring, imports, data access, and output paths:

```bash
uv run nemotron steps run rl/nemo_rl/rlvr -c tiny --dry-run
```

Then run the real job from a project overlay:

```bash
uv run nemotron steps run rl/nemo_rl/rlvr \
  -c <project>/config/rl_nemo_rl_rlvr.yaml
```

## Repository Layout

- Manifest: `src/nemotron/steps/rl/nemo_rl/rlvr/step.toml`
- Runner: `src/nemotron/steps/rl/nemo_rl/rlvr/step.py`
- Configs: `src/nemotron/steps/rl/nemo_rl/rlvr/config/default.yaml`, `src/nemotron/steps/rl/nemo_rl/rlvr/config/tiny.yaml`, `src/nemotron/steps/rl/nemo_rl/rlvr/config/nemo_gym.yaml`

## Guardrails

- Validate reward functions on sample rollouts before training.
- Do not mix NeMo-Gym resource-server config with the upstream generic GRPO
  data schema.
- Keep reward outputs bounded and deterministic when possible.
- Avoid ambiguous reward fields; schema drift tends to surface as poor learning rather than clear failures.
