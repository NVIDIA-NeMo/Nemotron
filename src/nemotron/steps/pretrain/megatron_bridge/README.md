# Megatron-Bridge Pretrain

Use `pretrain/megatron_bridge` when model size, sequence length, or throughput requires Megatron distributed parallelism.

Use this README for workflow and pitfalls; use `step.toml` for the exact artifact, parameter, strategy, and error manifest before editing configs or code.

## Inputs And Outputs

- Consume `binidx` data and `blend.json` from `data_prep/pretrain_prep`.
- Optionally initialize from a base checkpoint or HF weights for continued pretraining.
- Produce `checkpoint_megatron`.
- Validate data loading, parallelism, and checkpoint output with a short run before scaling token budget.

## CLI And Overlay Knobs

Start from `config/tiny.yaml` for launch validation and `config/default.yaml`
for the production-shaped topology. In a project overlay, developers usually
change:

- `dataset.data_paths`: prep-emitted `blend.json`, not packed Parquet.
- `seq_length`, `recipe.seq_length`, `model.seq_length`, and dataset sequence
  length: keep all values aligned.
- `load_hf_weights` or checkpoint fields: set explicitly for CPT or resume.
- `train.micro_batch_size`, `train.global_batch_size`, and TP/PP/CP/EP sizes:
  size them against the env profile.
- Checkpoint save directory, validation cadence, learning-rate schedule, and
  train iterations.

Example shape:

```bash
uv run nemotron steps run pretrain/megatron_bridge \
  -c <project>/config/pretrain_megatron_bridge.yaml \
  dataset.data_paths=<prep-output>/blend.json \
  seq_length=<planned-seq-length>
```

Related patterns:

- Check `src/nemotron/steps/patterns/pretrain-token-budget-before-scale.md` before changing distributed strategy.
- Check `src/nemotron/steps/patterns/prep-data-is-tokenizer-locked.md` before reusing bin/idx data.

## Config Nuances

- Keep `recipe.seq_length`, `model.seq_length`, and `dataset.seq_length` identical; Bridge validates the model and dataset values before setup.
- Set `dataset.data_paths` to the bin/idx `blend.json` from `data_prep/pretrain_prep`, not SFT packed Parquet.
- For Qwen/Nemotron MoE runs, keep `model.sequence_parallel: true` with tensor parallelism.
- If Transformer Engine userbuffers are enabled on a system without CUDA multicast support, set `run.env.env_vars.UB_SKIPMC: "1"` or default it in `step.py` before Bridge initialization.
- Use `train.global_batch_size` as a multiple of data-parallel size; start with `train.micro_batch_size: 1` when validating a new parallelism shape.

## Run It

Smoke first to validate wiring, imports, data access, and output paths:

```bash
uv run nemotron steps run pretrain/megatron_bridge -c tiny --dry-run
```

Then run the real job from a project overlay:

```bash
uv run nemotron steps run pretrain/megatron_bridge \
  -c <project>/config/pretrain_megatron_bridge.yaml
```

## Repository Layout

- Manifest: `src/nemotron/steps/pretrain/megatron_bridge/step.toml`
- Runner: `src/nemotron/steps/pretrain/megatron_bridge/step.py`
- Configs: `src/nemotron/steps/pretrain/megatron_bridge/config/default.yaml`, `src/nemotron/steps/pretrain/megatron_bridge/config/tiny.yaml`
- Shared runner: `src/nemotron/steps/_runners/megatron_bridge.py`

## Guardrails

- Run `data_prep/pretrain_prep` first unless compatible bin/idx data already exists.
- Verify data paths and checkpoint writes on the target executor before long jobs.
- Convert Megatron checkpoints only when the downstream consumer requires HF layout.
