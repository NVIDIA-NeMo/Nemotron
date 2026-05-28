# Megatron-Bridge SFT

Use `sft/megatron_bridge` when distributed training strategy and packed-sequence throughput matter more than HF-native simplicity.

Use this README for workflow and pitfalls; use `step.toml` for the exact artifact, parameter, strategy, and error manifest before editing configs or code.

## Inputs And Outputs

- Consume `packed_parquet` from `data_prep/sft_packing`.
- Optionally consume a `checkpoint_megatron` base or prior checkpoint.
- Produce `checkpoint_megatron`.
- Validate packed data, parallelism, and checkpoint output with a short run before scaling.

## CLI And Overlay Knobs

Start from `config/tiny.yaml` for launch validation and `config/default.yaml`
for the production-shaped topology. In a project overlay, developers usually
change:

- `dataset.packed_sequence_specs.packed_train_data_path`: packed Parquet glob,
  usually `<packed>/splits/train/*.parquet`.
- `seq_length`, `dataset.seq_length`, and packed sequence size: keep these equal
  to the `data_prep/sft_packing` `pack_size`.
- `checkpoint.pretrained_checkpoint`: optional Megatron base or resume checkpoint.
- `peft`: keep LoRA only when intentionally running adapter-style SFT; set full
  SFT explicitly when memory allows.
- `train.micro_batch_size`, `train.global_batch_size`, and model parallel sizes:
  keep them compatible with the selected env profile.

Example shape:

```bash
uv run nemotron steps run sft/megatron_bridge \
  -c <project>/config/sft_megatron_bridge.yaml \
  dataset.packed_sequence_specs.packed_train_data_path='<packed>/splits/train/*.parquet' \
  seq_length=<pack-size>
```

Related patterns:

- Check `src/nemotron/steps/patterns/prep-data-is-tokenizer-locked.md` before reusing packed data.
- Check `src/nemotron/steps/patterns/sft-sequence-packing.md` when packing efficiency is part of the decision.

## Config Nuances

- Set `recipe.packed_sequence: true` when consuming packed Parquet.
- Keep `dataset.seq_length`, `dataset.packed_sequence_specs.packed_sequence_size`, and `model.seq_length` equal.
- Use `model.sequence_parallel: true` for MoE plus tensor parallelism.
- Start with `train.micro_batch_size: 1` when validating a new distributed shape and choose `train.global_batch_size` as a multiple of the resulting data-parallel size.
- Inspect data_prep loss masks before trusting loss curves from a new template
  or tool-call format.

## Run It

Smoke first to validate wiring, imports, data access, and output paths:

```bash
uv run nemotron steps run sft/megatron_bridge -c tiny --dry-run
```

Then run the real job from a project overlay:

```bash
uv run nemotron steps run sft/megatron_bridge \
  -c <project>/config/sft_megatron_bridge.yaml
```

## Repository Layout

- Manifest: `src/nemotron/steps/sft/megatron_bridge/step.toml`
- Runner: `src/nemotron/steps/sft/megatron_bridge/step.py`
- Configs: `src/nemotron/steps/sft/megatron_bridge/config/default.yaml`, `src/nemotron/steps/sft/megatron_bridge/config/tiny.yaml`
- Recipe reference: `src/nemotron/recipes/nano3/stage1_sft/`

## Guardrails

- Run `data_prep/sft_packing` first unless a compatible packed dataset already exists.
- Repack data after tokenizer, template, or sequence length changes.
- Convert Megatron checkpoints to HF format before HF-native evaluation or deployment.
