# Pretrain Bin/Idx Prep

Use `data_prep/pretrain_prep` when downstream pretraining expects Megatron `binidx` data.

Use this README for workflow and pitfalls; use `step.toml` for the exact artifact, parameter, strategy, and error manifest before editing configs or code.

## Inputs And Outputs

- Consume curated text data through a blend file, usually local JSONL, parquet, or HF dataset references.
- Produce bin/idx shards plus `blend.json` split metadata.
- Validate tokenization and emitted `blend.json` on a small subset before full prep.

## CLI And Overlay Knobs

Start from `config/tiny.yaml` for wiring and `config/default.yaml` for the
production-shaped example. In a project overlay, developers usually change:

- `blend_path`: source text blend with local or HF-backed entries.
- `tokenizer.model`: tokenizer used by the downstream pretraining model.
- `num_shards`: match filesystem and trainer throughput.
- `valid_shards`, `test_shards`, and `split_seed`: keep validation reproducible.
- `max_doc_tokens` and `text_field`: set only when the data policy requires it.

Example shape:

```bash
uv run nemotron steps run data_prep/pretrain_prep \
  -c <project>/config/pretrain_prep.yaml \
  blend_path=<project>/data/blend.json \
  tokenizer.model=<model-tokenizer>
```

Related patterns:

- Check `src/nemotron/steps/patterns/prep-data-is-tokenizer-locked.md` before changing tokenization, split, or sharding behavior.
- Check `src/nemotron/steps/patterns/pretrain-token-budget-before-scale.md` before creating production pretraining data.

## Config Nuances

- Emit and preserve `blend.json`; both AutoModel and Megatron Bridge pretrain configs should point at this file.
- Reduce `tokenization_workers` when Ray actor memory is tight.
- Keep `valid_shards`, `test_shards`, and `split_seed` explicit for deterministic prep.
- Rebuild bin/idx whenever `tokenizer.model` or `sequence_length` assumptions change.

## Run It

Smoke first to validate wiring, imports, data access, and output paths:

```bash
uv run nemotron steps run data_prep/pretrain_prep -c tiny --dry-run
```

Then run the real job from a project overlay:

```bash
uv run nemotron steps run data_prep/pretrain_prep \
  -c <project>/config/data_prep_pretrain_prep.yaml
```

## Repository Layout

- Manifest: `src/nemotron/steps/data_prep/pretrain_prep/step.toml`
- Runner: `src/nemotron/steps/data_prep/pretrain_prep/step.py`
- Configs: `src/nemotron/steps/data_prep/pretrain_prep/config/default.yaml`, `src/nemotron/steps/data_prep/pretrain_prep/config/tiny.yaml`
- Sample blend: `src/nemotron/steps/data_prep/pretrain_prep/data/blend_tiny.json`

## Guardrails

- Treat bin/idx as tokenizer-locked; rebuild it when the tokenizer changes.
- Keep train, validation, and test split names consistent with downstream pretrain configs.
- Validate token counts, empty-document rates, and split leakage before a full pretraining run.
