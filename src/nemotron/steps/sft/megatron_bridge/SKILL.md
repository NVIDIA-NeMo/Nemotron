---
name: nemotron-sft-megatron-bridge
description: Configure Nemotron sft/megatron_bridge for distributed supervised fine-tuning with Megatron-Bridge. Use for packed Parquet SFT, large models, TP/PP/CP parallelism, Nemotron Nano3 or Super3 recipe patterns, and Megatron checkpoint output.
---

# Megatron-Bridge SFT

Use `sft/megatron_bridge` when distributed training strategy and packed-sequence throughput matter more than HF-native simplicity.

Before changing configs or code, read `step.toml` to understand the step flow, consumed and produced artifacts, important parameters, strategies, failure modes, and upstream references.

## Inputs And Outputs

- Consume `packed_parquet` from `prep/sft_packing`.
- Optionally consume a `checkpoint_megatron` base or prior checkpoint.
- Produce `checkpoint_megatron`.
- Validate packed data, parallelism, and checkpoint output with a short run before scaling.

## Configure

- Keep `seq_length` equal to the prep step's `pack_size`.
- Start Nano3 plans around the existing recipe defaults; scale Super3-like plans only after short validation runs pass.
- Tune tensor, pipeline, and context parallelism before scaling global batch.
- The shipped 30B default uses `peft=lora` to fit the starter topology; set `recipe.peft=null` and remove the top-level `peft:` block only when full SFT fits.
- Check `src/nemotron/steps/patterns/prepared-data-is-tokenizer-locked.md` before reusing packed data.
- Check `src/nemotron/steps/patterns/pack-variable-length.md` when packing efficiency is part of the decision.

## Config Nuances

- Set `recipe.packed_sequence: true` when consuming packed Parquet.
- Keep `dataset.seq_length`, `dataset.packed_sequence_specs.packed_sequence_size`, and `model.seq_length` equal.
- Use `model.sequence_parallel: true` for MoE plus tensor parallelism.
- Start with `train.micro_batch_size: 1` when validating a new distributed shape and choose `train.global_batch_size` as a multiple of the resulting data-parallel size.

## Local Files

- Contract: `src/nemotron/steps/sft/megatron_bridge/step.toml`
- Runner: `src/nemotron/steps/sft/megatron_bridge/step.py`
- Configs: `src/nemotron/steps/sft/megatron_bridge/config/default.yaml`, `src/nemotron/steps/sft/megatron_bridge/config/tiny.yaml`
- Recipe reference: `src/nemotron/recipes/nano3/stage1_sft/`

## Guardrails

- Run `prep/sft_packing` first unless a compatible packed dataset already exists.
- Repack data after tokenizer, template, or sequence length changes.
- Convert Megatron checkpoints to HF format before HF-native evaluation or deployment.
