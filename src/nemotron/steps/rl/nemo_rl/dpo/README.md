# NeMo-RL DPO

Use `rl/nemo_rl/dpo` when the training signal is static preference pairs rather than online reward functions.

Use this README for workflow and pitfalls; use `step.toml` for the exact artifact, parameter, strategy, and error manifest before editing configs or code.

## Inputs And Outputs

- Consume `training_jsonl` with prompt, chosen, and rejected fields.
- Consume an SFT `checkpoint_megatron` policy.
- Produce a DPO-aligned `checkpoint_megatron`.
- Smoke with `nemotron steps run rl/nemo_rl/dpo -c tiny`.

## CLI And Overlay Knobs

Start from `config/tiny.yaml` for runner validation and `config/default.yaml`
for the production-shaped example. In a project overlay, developers usually
change:

- `data.train.prompt_file` and `data.validation.prompt_file`: sharded DPO JSONL
  from `data_prep/rl_prep`.
- `dpo.reference_policy_kl_penalty`: raise when KL collapses or loss diverges.
- `policy.train_global_batch_size`: keep divisible by policy worker and micro
  batch shape.
- `cluster.num_nodes` and `cluster.gpus_per_node`: match the selected Ray env
  profile.
- Output/logging directories, ideally through environment-derived paths.

Example shape:

```bash
uv run nemotron steps run rl/nemo_rl/dpo \
  -c <project>/config/dpo.yaml \
  data.train.prompt_file=<rl-prep>/train.jsonl \
  data.validation.prompt_file=<rl-prep>/validation.jsonl
```

Related patterns:

- Check `src/nemotron/steps/patterns/rl-validate-rewards-before-scale.md` before trusting preference-pair training results.

## Config Nuances

- Keep validation cadence explicit with fields such as `dpo.val_at_start`, `dpo.val_period`, and `dpo.val_at_end`; do not rely on upstream defaults when changing run length.
- Keep `policy.train_global_batch_size` divisible by the active policy worker shape and micro batch size.
- Keep `cluster.num_nodes` and `cluster.gpus_per_node` aligned with the RayCluster executor shape.
- Use environment-derived output paths such as `RL_OUTPUT_DIR` for logs and checkpoints so repeated runs do not collide.

## Run It

Smoke first to validate wiring, imports, data access, and output paths:

```bash
uv run nemotron steps run rl/nemo_rl/dpo -c tiny --dry-run
```

Then run the real job from a project overlay:

```bash
uv run nemotron steps run rl/nemo_rl/dpo \
  -c <project>/config/rl_nemo_rl_dpo.yaml
```

## Repository Layout

- Manifest: `src/nemotron/steps/rl/nemo_rl/dpo/step.toml`
- Runner: `src/nemotron/steps/rl/nemo_rl/dpo/step.py`
- Configs: `src/nemotron/steps/rl/nemo_rl/dpo/config/default.yaml`, `src/nemotron/steps/rl/nemo_rl/dpo/config/tiny.yaml`

## Guardrails

- Validate chosen and rejected ordering; inverted pairs silently teach the wrong behavior.
- Keep `policy.train_global_batch_size` divisible by the active policy worker
  shape and micro batch size.
- Keep train and validation preference distributions comparable.
- Inspect examples where the model regresses after DPO; preference data can encode style bias.
