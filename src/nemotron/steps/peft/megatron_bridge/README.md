# Megatron-Bridge PEFT

Use `peft/megatron_bridge` when adapter training must stay in the
Megatron-Bridge ecosystem: packed Parquet input, Megatron checkpoint warm start,
and distributed TP/PP/CP-style scaling.

Use this README for workflow and pitfalls; use `step.toml` for the exact artifact, parameter, strategy, and error manifest before editing configs or code.

## Inputs And Outputs

- Consume `packed_parquet` from `data_prep/sft_packing`.
- Consume a base `checkpoint_megatron`.
- Produce `checkpoint_lora`.
- Validate with a short adapter run before scaling data, rank, or sequence length.

```text
training_jsonl
  -> data_prep/sft_packing
  -> packed_parquet
  + checkpoint_megatron base
  -> peft/megatron_bridge
  -> Megatron-format LoRA adapter
  -> merge/export path when HF deployment is required
```

The packed data, base checkpoint, adapter output, and eventual merged export
should be separate paths. Mixing them makes resume, merge, and evaluation hard
to reason about.

## CLI And Overlay Knobs

Start from `config/tiny.yaml` for launch validation and `config/default.yaml`
for the production-shaped Nano topology. `config/lightning35.yaml` is a
`defaults: default.yaml` overlay for Nemotron 3.5 Lightning 30B-A3B packed LoRA
at 4k: it keeps LoRA dim/alpha and `hf_load`, and overrides the Lightning
recipe, extra target modules, TP/EP, and `L35_*` paths. In a project overlay,
the first fields developers usually change are:

- `checkpoint.pretrained_checkpoint`: concrete Megatron base checkpoint.
- `dataset.packed_sequence_specs.packed_train_data_path`: normally
  `data_prep/sft_packing` output under `splits/train/*.parquet`.
- `peft.dim`: adapter rank; lower it before changing runner code for OOMs.
- `load_hf_weights`: keep `false` for normal Megatron checkpoint PEFT starts.
- `model.tensor_model_parallel_size`, `model.pipeline_model_parallel_size`,
  `model.context_parallel_size`, and `train.global_batch_size`: keep these
  consistent with the selected env profile.
- `checkpoint.save`: adapter output path, not the base checkpoint path.

Example shape:

```bash
uv run nemotron steps run peft/megatron_bridge \
  -c <project>/config/peft_megatron_bridge.yaml \
  checkpoint.pretrained_checkpoint=<megatron-base-iter> \
  dataset.packed_sequence_specs.packed_train_data_path='<packed>/splits/train/*.parquet'
```

Related patterns:

- Check `src/nemotron/steps/patterns/prep-data-is-tokenizer-locked.md` before reusing packed data.
- Check `src/nemotron/steps/patterns/peft-adapter-merge-discipline.md` before merging or converting adapters.

## Config Nuances

- Set `checkpoint.pretrained_checkpoint` to a real Megatron checkpoint directory for PEFT; placeholders and parent output directories fail validation.
- Keep `load_hf_weights: false` for PEFT: the frozen base must be a Megatron checkpoint (`checkpoint.pretrained_checkpoint`). PEFT cannot bootstrap from HF weights — convert HF to Megatron first (run SFT or `convert/hf_to_megatron`), then point `pretrained_checkpoint` at the result.
- Keep `model.sequence_parallel: true` when `model.tensor_model_parallel_size > 1` and MoE is enabled.
- When checkpoint save reliability matters more than async throughput, prefer `checkpoint.async_save: false`, `checkpoint.fully_parallel_save: false`, `checkpoint.save_optim: false`, and `checkpoint.save_rng: false`.
- `dataset.packed_sequence_specs.packed_train_data_path` should point at `splits/train/*.parquet` produced by `data_prep/sft_packing`.
- Packed `pack_size`, model `seq_length`, and packed sequence size must match
  the assumptions used by the SFT packing step.
- Lightning overlays null inherited Nano recipe kwargs (`packed_sequence`,
  `peft`, `seq_length`) because `nemotron_3_5_lightning_peft_config` is not the
  Nano PEFT callable.

## Run It

Smoke first to validate wiring, imports, data access, and output paths:

```bash
uv run nemotron steps run peft/megatron_bridge -c tiny --dry-run
```

Then run the real job from a project overlay. Convert the Hugging Face base to
Megatron with `convert/hf_to_megatron` before PEFT (including `-c lightning35`);
adapters cannot load HF weights.

```bash
uv run nemotron steps run peft/megatron_bridge \
  -c <project>/config/peft_megatron_bridge.yaml

uv run nemotron steps run peft/megatron_bridge -c lightning35 --batch <profile>
```

## Repository Layout

- Manifest: `src/nemotron/steps/peft/megatron_bridge/step.toml`
- Runner: `src/nemotron/steps/peft/megatron_bridge/step.py`
- Configs:
  - `src/nemotron/steps/peft/megatron_bridge/config/default.yaml`
  - `src/nemotron/steps/peft/megatron_bridge/config/tiny.yaml`
  - `src/nemotron/steps/peft/megatron_bridge/config/lightning35.yaml` (`defaults: default.yaml` overlay)

## Guardrails

- Run `data_prep/sft_packing` first unless a compatible packed dataset already exists.
- Convert Hugging Face weights to Megatron with `convert/hf_to_megatron` before
  PEFT; the frozen base must be a Megatron checkpoint.
- Use `sft/megatron_bridge` instead when the user explicitly needs full fine-tuning.
- Keep the base Megatron checkpoint path separate from adapter output paths.
- Plan the HF export/merge path before training if the adapter must become a
  deployable `checkpoint_hf`.
