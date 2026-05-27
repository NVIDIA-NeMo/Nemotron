---
name: nemotron-data-prep-sft-packing
description: Configure the Nemotron data_prep/sft_packing step that applies chat templates, tokenizes training JSONL, and emits Megatron-Bridge packed Parquet shards for SFT or PEFT. Use when preparing data for sft/megatron_bridge, peft/megatron_bridge, packed sequence training, loss-mask validation, or sequence-length alignment.
---

# SFT Packing

Use `data_prep/sft_packing` when the downstream step consumes `packed_parquet`, especially `sft/megatron_bridge` or `peft/megatron_bridge`.

Before changing configs or code, read `step.toml` to understand the step flow, consumed and produced artifacts, important parameters, strategies, failure modes, and upstream references.

## Inputs And Outputs

- Consume `training_jsonl` with OpenAI-style `messages`.
- Produce `packed_parquet` shards with token ids and loss masks.
- Validate formatted prompts, loss masks, and packed output on a small subset before full prep.

## Configure

- Set `tokenizer` to the same tokenizer used by downstream training and evaluation.
- Set `pack_size` equal to downstream `seq_length`.
- Set `chat_template` to the target model family or template path.
- Lower `num_shards` for small samples so shards remain useful.
- Keep `train_ratio`, `valid_ratio`, and `test_ratio` explicit when downstream
  training expects stable split directories.
- Check `src/nemotron/steps/patterns/prep-data-is-tokenizer-locked.md` before reusing packed data after tokenizer, template, or sequence-length changes.
- Check `src/nemotron/steps/patterns/sft-sequence-packing.md` when deciding whether packing is useful for a corpus.

## Config Nuances

- `pack_size` must match Megatron Bridge `dataset.seq_length`, `packed_sequence_size`, and `model.seq_length`.
- Megatron Bridge configs should consume `splits/train/*.parquet`; AutoModel SFT/PEFT should consume JSONL instead.
- Repack after tokenizer or chat-template changes, even if the source JSONL is unchanged.

## Local Files

- Contract: `src/nemotron/steps/data_prep/sft_packing/step.toml`
- Runner: `src/nemotron/steps/data_prep/sft_packing/step.py`
- Configs: `src/nemotron/steps/data_prep/sft_packing/config/default.yaml`, `src/nemotron/steps/data_prep/sft_packing/config/tiny.yaml`
- Sample blend: `src/nemotron/steps/data_prep/sft_packing/data/blend_tiny.json`

## Avoid

- Do not use this step for AutoModel SFT or PEFT; those steps read `training_jsonl` directly.
- Do not reuse packed data after changing tokenizer, template, or sequence length.
- Do not treat packing efficiency as correctness; inspect loss masks and formatted prompts too.
