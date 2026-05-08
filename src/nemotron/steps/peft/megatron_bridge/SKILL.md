---
name: nemotron-peft-megatron-bridge
description: Configure Nemotron peft/megatron_bridge for distributed LoRA adapter training on Megatron-Bridge. Use when full Megatron SFT is too memory-heavy but packed Parquet data, Megatron checkpoints, TP/PP scaling, or Megatron-compatible adapter output are required.
---

# Megatron-Bridge PEFT

Use `peft/megatron_bridge` when adapter training must stay in the Megatron-Bridge ecosystem.

Before changing configs or code, read `step.toml` to understand the step flow, consumed and produced artifacts, important parameters, strategies, failure modes, and upstream references.

## Inputs And Outputs

- Consume `packed_parquet` from `prep/sft_packing`.
- Consume a base `checkpoint_megatron`.
- Produce `checkpoint_lora`.
- Validate with a short adapter run before scaling data, rank, or sequence length.

## Configure

- Keep `peft.type=lora`.
- Start with the default `peft.dim`, then reduce it if memory is tight.
- Keep packed-data tokenizer and sequence length aligned with the base model.
- Merge or convert adapters when downstream consumers need HF model layout.
- Check `src/nemotron/steps/patterns/prep-data-is-tokenizer-locked.md` before reusing packed data.
- Check `src/nemotron/steps/patterns/peft-adapter-merge-discipline.md` before merging or converting adapters.

## Config Nuances

- Set `checkpoint.pretrained_checkpoint` to a real Megatron checkpoint directory for PEFT; placeholders and parent output directories fail validation.
- Use `load_hf_weights: false` when PEFT starts from `checkpoint.pretrained_checkpoint`; use HF loading only when deliberately bootstrapping from HF weights.
- Keep `model.sequence_parallel: true` when `model.tensor_model_parallel_size > 1` and MoE is enabled.
- When checkpoint save reliability matters more than async throughput, prefer `checkpoint.async_save: false`, `checkpoint.fully_parallel_save: false`, `checkpoint.save_optim: false`, and `checkpoint.save_rng: false`.
- `dataset.packed_sequence_specs.packed_train_data_path` should point at `splits/train/*.parquet` produced by `prep/sft_packing`.

## Local Files

- Contract: `src/nemotron/steps/peft/megatron_bridge/step.toml`
- Runner: `src/nemotron/steps/peft/megatron_bridge/step.py`
- Configs: `src/nemotron/steps/peft/megatron_bridge/config/default.yaml`, `src/nemotron/steps/peft/megatron_bridge/config/tiny.yaml`

## Guardrails

- Run `prep/sft_packing` first unless a compatible packed dataset already exists.
- Use `sft/megatron_bridge` instead when the user explicitly needs full fine-tuning.
- Keep the base Megatron checkpoint path separate from adapter output paths.
