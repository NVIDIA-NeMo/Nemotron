# AutoModel PEFT

Use `peft/automodel` for LoRA tuning when the data is chat-format JSONL and the
base model is Hugging Face-native. This is the shortest adapter path: no packing
step, no Megatron conversion before training, and an HF PEFT adapter as output.

Use this README for workflow and pitfalls; use `step.toml` for the exact artifact, parameter, strategy, and error manifest before editing configs or code.

## Inputs And Outputs

- Consume `training_jsonl`.
- Produce `checkpoint_lora`.
- Validate with a short adapter run before scaling data, rank, or sequence length.

```text
chat training_jsonl + HF base model
  -> peft/automodel
  -> HF LoRA adapter
  -> convert/merge_lora if deployment needs checkpoint_hf
```

Skip `data_prep/sft_packing`; packed Parquet is for the Megatron-Bridge path.

## CLI And Overlay Knobs

Start from `config/tiny.yaml` for wiring and `config/default.yaml` for the
production-shaped example. In a project overlay, the first fields developers
usually change are:

- `model.pretrained_model_name_or_path`: HF base or local checkpoint.
- `dataset.path_or_dataset_id`: chat JSONL path or dataset ID.
- `peft.dim`: adapter rank.
- `peft.alpha`: adapter scaling; usually `2 * peft.dim`.
- Training length, batch size, and output/checkpoint directories.

Example shape:

```bash
uv run nemotron steps run peft/automodel \
  -c <project>/config/peft_automodel.yaml \
  model.pretrained_model_name_or_path=<hf-or-local-base> \
  dataset.path_or_dataset_id=<chat-jsonl>
```

Related patterns:

- Check `src/nemotron/steps/patterns/sft-small-dataset-prefer-lora.md` before choosing LoRA for small datasets.
- Check `src/nemotron/steps/patterns/peft-adapter-merge-discipline.md` before merging adapters.

## Config Nuances

- Keep PEFT and SFT AutoModel on the same `model.pretrained_model_name_or_path`; PEFT should differ mainly in `peft`, data path, and iteration count.
- For Qwen MoE runs, use `model.backend.experts: torch_mm` and `model.backend.dispatcher: torch` unless DeepEP support has been validated in the target container.
- Prefer `distributed.activation_checkpointing: false` while dispatcher and checkpoint-recompute behavior are still being validated.

## Run It

Smoke first to validate wiring, imports, data access, and output paths:

```bash
uv run nemotron steps run peft/automodel -c tiny --dry-run
```

Then run the real job from a project overlay:

```bash
uv run nemotron steps run peft/automodel \
  -c <project>/config/peft_automodel.yaml
```

## Repository Layout

- Manifest: `src/nemotron/steps/peft/automodel/step.toml`
- Runner: `src/nemotron/steps/peft/automodel/step.py`
- Configs: `src/nemotron/steps/peft/automodel/config/default.yaml`, `src/nemotron/steps/peft/automodel/config/tiny.yaml`

## Guardrails

- Do not run `data_prep/sft_packing`; this step consumes JSONL directly.
- Reduce rank and sequence length before changing the training wrapper for OOMs.
- Treat the adapter as a separate artifact until merge and eval have passed,
  and preserve base/tokenizer/rank/alpha provenance with it.
- Do not merge the adapter into a newer or different base just because the model
  name matches.
