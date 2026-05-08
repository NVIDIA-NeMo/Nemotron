---
name: nemotron-pretrain-megatron-bridge
description: Configure Nemotron pretrain/megatron_bridge for large-scale pretraining or continued pretraining with NVIDIA Megatron-Bridge. Use for bin/idx data, Megatron checkpoint output, TP/PP/CP/EP distributed training, Nemotron recipe overrides, or HF weight initialization.
---

# Megatron-Bridge Pretrain

Use `pretrain/megatron_bridge` when model size, sequence length, or throughput requires Megatron distributed parallelism.

Before changing configs or code, read `step.toml` to understand the step flow, consumed and produced artifacts, important parameters, strategies, failure modes, and upstream references.

## Inputs And Outputs

- Consume `binidx` data and `blend.json` from `prep/pretrain_prep`.
- Optionally initialize from a base checkpoint or HF weights for continued pretraining.
- Produce `checkpoint_megatron`.
- Validate data loading, parallelism, and checkpoint output with a short run before scaling token budget.

## Configure

- Keep `seq_length` aligned with the data and token budget.
- Set `load_hf_weights` or checkpoint paths explicitly for continued pretraining.
- Start from the closest Megatron-Bridge recipe and override only required knobs.
- Tune tensor, pipeline, context, and expert parallelism before scaling global batch.
- Check `src/nemotron/steps/patterns/pretrain-token-budget-before-scale.md` before changing distributed strategy.
- Check `src/nemotron/steps/patterns/prep-data-is-tokenizer-locked.md` before reusing bin/idx data.

## Config Nuances

- Keep `recipe.seq_length`, `model.seq_length`, and `dataset.seq_length` identical; Bridge validates the model and dataset values before setup.
- Set `dataset.data_paths` to the bin/idx `blend.json` from `prep/pretrain_prep`, not SFT packed Parquet.
- For Qwen/Nemotron MoE runs, keep `model.sequence_parallel: true` with tensor parallelism.
- If Transformer Engine userbuffers are enabled on a system without CUDA multicast support, set `run.env.env_vars.UB_SKIPMC: "1"` or default it in `step.py` before Bridge initialization.
- Use `train.global_batch_size` as a multiple of data-parallel size; start with `train.micro_batch_size: 1` when validating a new parallelism shape.

## Local Files

- Contract: `src/nemotron/steps/pretrain/megatron_bridge/step.toml`
- Runner: `src/nemotron/steps/pretrain/megatron_bridge/step.py`
- Configs: `src/nemotron/steps/pretrain/megatron_bridge/config/default.yaml`, `src/nemotron/steps/pretrain/megatron_bridge/config/tiny.yaml`
- Shared runner: `src/nemotron/steps/_runners/megatron_bridge.py`

## Guardrails

- Run `prep/pretrain_prep` first unless compatible bin/idx data already exists.
- Verify data paths and checkpoint writes on the target executor before long jobs.
- Convert Megatron checkpoints only when the downstream consumer requires HF layout.
