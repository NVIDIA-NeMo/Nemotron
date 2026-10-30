# SFT Packing

Use `data_prep/sft_packing` when the downstream step consumes `packed_parquet`, especially `sft/megatron_bridge` or `peft/megatron_bridge`.

Use this README for workflow and pitfalls; use `step.toml` for the exact artifact, parameter, strategy, and error manifest before editing configs or code.

## Inputs And Outputs

- Consume `training_jsonl` with OpenAI-style `messages`.
- Produce `packed_parquet` shards with token ids and loss masks.
- Validate formatted prompts, loss masks, and packed output on a small subset before full prep.

## CLI And Overlay Knobs

Start from `config/tiny.yaml` for wiring and `config/default.yaml` for the
production-shaped example. In a project overlay, developers usually change:

- `tokenizer`: the exact tokenizer used by downstream training.
- `pack_size`: must match downstream Megatron-Bridge `seq_length`.
- `chat_template`: model-family template or explicit template path.
- Source blend / input data paths and output directory.
- `num_shards`, `train_ratio`, `valid_ratio`, and `test_ratio`.

Example shape:

```bash
uv run nemotron steps run data_prep/sft_packing \
  -c <project>/config/sft_packing.yaml \
  tokenizer=<model-tokenizer> \
  pack_size=<train-seq-length>
```

Related patterns:

- Check `src/nemotron/steps/patterns/prep-data-is-tokenizer-locked.md` before reusing packed data after tokenizer, template, or sequence-length changes.
- Check `src/nemotron/steps/patterns/sft-sequence-packing.md` when deciding whether packing is useful for a corpus.

## Config Nuances

- `pack_size` must match Megatron Bridge `dataset.seq_length`, `packed_sequence_size`, and `model.seq_length`.
- Megatron Bridge configs should consume `splits/train/*.parquet`; AutoModel SFT/PEFT should consume JSONL instead.
- Repack after tokenizer or chat-template changes, even if the source JSONL is unchanged.

## Run It

Smoke first to validate wiring, imports, data access, and output paths:

```bash
uv run nemotron steps run data_prep/sft_packing -c tiny --dry-run
```

Then run the real job from a project overlay:

```bash
uv run nemotron steps run data_prep/sft_packing \
  -c <project>/config/data_prep_sft_packing.yaml
```

## Repository Layout

- Manifest: `src/nemotron/steps/data_prep/sft_packing/step.toml`
- Runner: `src/nemotron/steps/data_prep/sft_packing/step.py`
- Configs: `src/nemotron/steps/data_prep/sft_packing/config/default.yaml`, `src/nemotron/steps/data_prep/sft_packing/config/tiny.yaml`
- Sample blend: `src/nemotron/steps/data_prep/sft_packing/data/blend_tiny.json`

## Avoid

- Do not use this step for AutoModel SFT or PEFT; those steps read `training_jsonl` directly.
- Do not reuse packed data after changing tokenizer, template, or sequence length.
- Do not treat packing efficiency as correctness; inspect loss masks and formatted prompts too.
