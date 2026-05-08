---
name: nemotron-prep-pretrain-binidx
description: Configure the Nemotron prep/pretrain_prep step that tokenizes HF or local text blends into Megatron bin/idx shards and a blend.json for pretrain/automodel or pretrain/megatron_bridge. Use when preparing pretraining or continued-pretraining data, rebuilding tokenizer-locked corpora, or validating data splits.
---

# Pretrain Bin/Idx Prep

Use `prep/pretrain_prep` when downstream pretraining expects Megatron `binidx` data.

Before changing configs or code, read `step.toml` to understand the step flow, consumed and produced artifacts, important parameters, strategies, failure modes, and upstream references.

## Inputs And Outputs

- Consume curated text data through a blend file, usually local JSONL, parquet, or HF dataset references.
- Produce bin/idx shards plus `blend.json` split metadata.
- Validate tokenization and emitted `blend.json` on a small subset before full prep.

## Configure

- Set `tokenizer.model` to the downstream pretraining model tokenizer.
- Tune `num_shards` for target filesystem and trainer throughput.
- Leave `max_doc_tokens` unset unless the data policy requires truncation.
- Point pretrain configs at the emitted `blend.json`.
- Check `src/nemotron/steps/patterns/prep-data-is-tokenizer-locked.md` before changing tokenization, split, or sharding behavior.
- Check `src/nemotron/steps/patterns/pretrain-token-budget-before-scale.md` before creating production pretraining data.

## Config Nuances

- Emit and preserve `blend.json`; both AutoModel and Megatron Bridge pretrain configs should point at this file.
- Reduce `tokenization_workers` when Ray actor memory is tight.
- Keep `valid_shards`, `test_shards`, and `split_seed` explicit for deterministic prep.
- Rebuild bin/idx whenever `tokenizer.model` or `sequence_length` assumptions change.

## Local Files

- Contract: `src/nemotron/steps/prep/pretrain_prep/step.toml`
- Runner: `src/nemotron/steps/prep/pretrain_prep/step.py`
- Configs: `src/nemotron/steps/prep/pretrain_prep/config/default.yaml`, `src/nemotron/steps/prep/pretrain_prep/config/tiny.yaml`
- Sample blend: `src/nemotron/steps/prep/pretrain_prep/data/blend_tiny.json`

## Guardrails

- Treat bin/idx as tokenizer-locked; rebuild it when the tokenizer changes.
- Keep train, validation, and test split names consistent with downstream pretrain configs.
- Validate token counts, empty-document rates, and split leakage before a full pretraining run.
