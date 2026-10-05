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
for the generic Nano example. `config/lightning35.yaml` is a `defaults: default.yaml`
overlay for Nemotron 3.5 Lightning 30B-A3B packed SFT at 4k: it swaps the recipe
target, TP/recompute, and `L35_*` data/checkpoint paths, and nulls Nano HF load.
Use a named profile or project overlay for a different model or distributed
shape. Developers usually change:

- `dataset.packed_sequence_specs.packed_train_data_path`: packed Parquet glob,
  usually `<packed>/splits/train/*.parquet`.
- `seq_length`, `dataset.seq_length`, and packed sequence size: keep these equal
  to the `data_prep/sft_packing` `pack_size`.
- `checkpoint.pretrained_checkpoint`: optional Megatron base or resume checkpoint.
- `peft`: keep LoRA only when intentionally running adapter-style SFT; set full SFT explicitly when memory allows.
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

- Set `recipe.packed_sequence: true` when consuming packed Parquet. Lightning
  overlays null inherited Nano recipe kwargs (`packed_sequence`, `seq_length`)
  because `nemotron_3_5_lightning_sft_config` is zero-arg.
- Keep `dataset.seq_length`, `dataset.packed_sequence_specs.packed_sequence_size`, and `model.seq_length` equal.
- Use `model.sequence_parallel: true` for MoE plus tensor parallelism.
- Start with `train.micro_batch_size: 1` when validating a new distributed shape and choose `train.global_batch_size` as a multiple of the resulting data-parallel size.
- Inspect data_prep loss masks before trusting loss curves from a new template
  or tool-call format.

## Experimental Nemotron 3 Super Long-Context Topology References

Two non-runnable YAMLs capture full-parameter Super3 sequence lengths,
distributed topologies, resources, packing requirements, and training settings
for implementation planning:

| Config | Sequence length | Requested ranks | TP / PP / CP / EP |
|---|---:|---:|---:|
| `super3_128k` | 131,072 | 64 | 8 / 1 / 8 / 64 |
| `super3_256k` | 262,144 | 128 | 8 / 2 / 8 / 8 |

These files are intentionally placed under `config/` for discovery. With the
stock GA_v2 runner, use them only for dry-run planning: a real launch requires
a runner that honors their context-parallel packing and DDP precision settings.

Both reference configurations use a micro batch size of `1` and a global batch size of `1`, assuming the exact rank counts specified in the configurations: **8 × 8-GPU nodes for 128K** and **16 × 8-GPU nodes for 256K**. Before launching training, verify through a dry run that the expected resource topology is preserved. If the world size is increased, adjust the global batch size so that it is a multiple of the resulting data-parallel size.

Before training, pad every stored subsequence so that `(subsequence_length - 1)` is divisible by `128`. For `TP=8`, `CP=8`, and sequence parallelism enabled, the runtime requires alignment to a multiple of `64`; these reference configurations use `128` as a conservative alignment.

For 256K training, the GPU memory usage may differ if the mixed-precision or DDP settings are not configured as expected.

## Run It

Smoke first to validate wiring, imports, data access, and output paths:

```bash
uv run nemotron steps run sft/megatron_bridge -c tiny --dry-run
```

Then run the real job from a project overlay. Lightning SFT (`-c lightning35`)
does not AutoBridge Hugging Face weights: convert the base model with
`convert/hf_to_megatron` first, then point `checkpoint.pretrained_checkpoint`
at that Megatron checkpoint.

```bash
uv run nemotron steps run sft/megatron_bridge \
  -c <project>/config/sft_megatron_bridge.yaml

uv run nemotron steps run sft/megatron_bridge -c lightning35 --batch <profile>
```

## Repository Layout

- Manifest: `src/nemotron/steps/sft/megatron_bridge/step.toml`
- Runner: `src/nemotron/steps/sft/megatron_bridge/step.py`
- Configs:
  - `src/nemotron/steps/sft/megatron_bridge/config/default.yaml`
  - `src/nemotron/steps/sft/megatron_bridge/config/tiny.yaml`
  - `src/nemotron/steps/sft/megatron_bridge/config/lightning35.yaml` (`defaults: default.yaml` overlay)
  - `src/nemotron/steps/sft/megatron_bridge/config/super3_128k.yaml`
  - `src/nemotron/steps/sft/megatron_bridge/config/super3_256k.yaml`
- Recipe reference: `src/nemotron/recipes/nano3/stage1_sft/`, `src/nemotron/recipes/lightning35/stage1_sft/`

## Guardrails

- Run `data_prep/sft_packing` first unless a compatible packed dataset already exists.
- Repack data after tokenizer, template, or sequence length changes.
- Convert Hugging Face weights to Megatron with `convert/hf_to_megatron` before
  `-c lightning35`; that overlay does not load from Hugging Face.
- Convert Megatron checkpoints to HF format before HF-native evaluation or deployment.
