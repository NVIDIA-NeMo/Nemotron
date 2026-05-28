# NeMo-RL RLHF

Use `rl/nemo_rl/rlhf` when rewards come from a learned judge or generative reward model rather than deterministic verification.

Use this README for workflow and pitfalls; use `step.toml` for the exact artifact, parameter, strategy, and error manifest before editing configs or code.

## Inputs And Outputs

- Consume prompt `training_jsonl`.
- Consume an SFT `checkpoint_megatron` policy.
- Consume a reward-model `checkpoint_hf`.
- Produce an RLHF-aligned `checkpoint_megatron`.
- Smoke with `nemotron steps run rl/nemo_rl/rlhf -c tiny`.

## CLI And Overlay Knobs

Start from `config/tiny.yaml` for runner validation, then move to a project
overlay for real GenRM / reward-model serving. Developers usually change:

- `data.train.data_path` and `data.validation.data_path`: prompt JSONL prepared
  for the Responses API path.
- `env.nemo_gym.genrm_model.responses_api_models.vllm_model.model`: reward-model
  checkpoint or served model path.
- `env.nemo_gym.genrm_model.vllm`: tensor parallelism, memory, max sequence, and
  concurrency settings for GenRM serving.
- `grpo.num_generations_per_prompt`: reward variance versus serving cost.
- KL, reward clipping, learning rate, and output/logging directories.

Example shape:

```bash
uv run nemotron steps run rl/nemo_rl/rlhf \
  -c <project>/config/rlhf.yaml \
  data.train.data_path=<rl-prep>/train.jsonl \
  env.nemo_gym.genrm_model.responses_api_models.vllm_model.model=<reward-model>
```

Related patterns:

- Check `src/nemotron/steps/patterns/rl-validate-rewards-before-scale.md` before changing RLHF reward or rollout behavior.

## Config Nuances

- Ensure the RL prep manifest has non-empty `train` and `val` paths; only allow train-as-validation when the run is explicitly non-evaluative.
- Keep each NeMo-Gym row compatible with the Responses API path. Data should include or be normalized into `responses_create_params` before NeMo-RL loads response data.
- Size GenRM vLLM according to available GPU memory and concurrency: tune tensor parallelism, `max_num_seqs`, `max_model_len`, and `gpu_memory_utilization` so the `genrm_model` server reaches readiness.
- Keep the runner close to NeMo-RL's upstream NeMo-Gym GRPO example; avoid local rollout or actor shims unless the target image is upgraded and revalidated.
- Keep policy, reference, GenRM, and resource-server settings explicit in YAML; hidden defaults make Ray startup failures hard to distinguish from reward-model failures.

## Run It

Smoke first to validate wiring, imports, data access, and output paths:

```bash
uv run nemotron steps run rl/nemo_rl/rlhf -c tiny --dry-run
```

Then run the real job from a project overlay:

```bash
uv run nemotron steps run rl/nemo_rl/rlhf \
  -c <project>/config/rl_nemo_rl_rlhf.yaml
```

## Repository Layout

- Manifest: `src/nemotron/steps/rl/nemo_rl/rlhf/step.toml`
- Runner: `src/nemotron/steps/rl/nemo_rl/rlhf/step.py`
- Configs: `src/nemotron/steps/rl/nemo_rl/rlhf/config/default.yaml`, `src/nemotron/steps/rl/nemo_rl/rlhf/config/tiny.yaml`
- Recipe reference: `src/nemotron/recipes/super3/stage2_rl/`

## Guardrails

- Validate reward-model serving separately before launching policy optimization.
- Keep policy, reference, and reward-model checkpoints clearly separated in config.
- Do not use train data as validation unless the run is explicitly marked
  non-evaluative.
- Review held-out reward examples to detect judge bias or reward saturation.
