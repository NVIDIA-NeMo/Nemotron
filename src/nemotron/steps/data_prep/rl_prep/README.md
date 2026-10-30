# RL Prep

Use `data_prep/rl_prep` before NeMo-RL when prompt or preference data needs HF resolution, local materialization, or split sharding.

Use this README for workflow and pitfalls; use `step.toml` for the exact artifact, parameter, strategy, and error manifest before editing configs or code.

## Inputs And Outputs

- Consume `training_jsonl` through an RL data blend.
- Produce sharded `training_jsonl` ready for `rl/nemo_rl/dpo`, `rl/nemo_rl/rlvr`, or `rl/nemo_rl/rlhf`.
- Smoke with `nemotron steps run data_prep/rl_prep -c tiny`.

## CLI And Overlay Knobs

Start from `config/tiny.yaml` for wiring and `config/default.yaml` for the
production-shaped example. In a project overlay, developers usually change:

- `blend_path`: source prompt/preference blend.
- `num_shards_per_split`: size shards for dataset size and filesystem behavior.
- `resolve_hf_placeholders`: keep `true` when the training cluster cannot reach
  the Hub.
- `compression`: choose only if downstream readers support it.
- `max_rows`: useful for representative smoke runs.

Example shape:

```bash
uv run nemotron steps run data_prep/rl_prep \
  -c <project>/config/rl_prep.yaml \
  blend_path=<project>/data/rl_blend.json \
  resolve_hf_placeholders=true
```

Related patterns:

- Check `src/nemotron/steps/patterns/prep-data-is-tokenizer-locked.md` before changing RL data layout.
- Check `src/nemotron/steps/patterns/rl-validate-rewards-before-scale.md` before scaling RL jobs from prepared data.

## Run It

Smoke first to validate wiring, imports, data access, and output paths:

```bash
uv run nemotron steps run data_prep/rl_prep -c tiny --dry-run
```

Then run the real job from a project overlay:

```bash
uv run nemotron steps run data_prep/rl_prep \
  -c <project>/config/data_prep_rl_prep.yaml
```

## Repository Layout

- Manifest: `src/nemotron/steps/data_prep/rl_prep/step.toml`
- Runner: `src/nemotron/steps/data_prep/rl_prep/step.py`
- Configs: `src/nemotron/steps/data_prep/rl_prep/config/default.yaml`, `src/nemotron/steps/data_prep/rl_prep/config/tiny.yaml`
- Sample blend: `src/nemotron/steps/data_prep/rl_prep/data/blend_tiny.json`

## Guardrails

- Validate output JSONL records from every split before launching RL.
- Preserve split names expected by the RL config.
- Keep DPO preference ordering and RLVR verifier fields explicit.
