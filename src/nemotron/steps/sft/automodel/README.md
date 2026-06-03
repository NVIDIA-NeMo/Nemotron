# AutoModel SFT

Use `sft/automodel` for Hugging Face-format models that can train directly from `training_jsonl`.

Use this README for workflow and pitfalls; use `step.toml` for the exact artifact, parameter, strategy, and error manifest before editing configs or code.

## Inputs And Outputs

- Consume chat-format `training_jsonl`.
- Produce `checkpoint_hf`.
- Validate chat formatting and checkpoint output with a short run before scaling data or sequence length.

## CLI And Overlay Knobs

Start from `config/tiny.yaml` for wiring and `config/default.yaml` for the
production-shaped example. In a project overlay, developers usually change:

- `model.pretrained_model_name_or_path`: HF base or local checkpoint.
- `dataset.path_or_dataset_id`: chat JSONL path or dataset ID.
- `peft`: `null` for full SFT, `lora` only when using this step for adapter-style tuning.
- `dataloader.collate_fn`: keep the chat collater unless the data is deliberately pre-tokenized.
- Training length, batch size, output directories, and launcher settings.

Example shape:

```bash
uv run nemotron steps run sft/automodel \
  -c <project>/config/sft_automodel.yaml \
  model.pretrained_model_name_or_path=<hf-or-local-base> \
  dataset.path_or_dataset_id=<chat-jsonl>
```

Related patterns:

- Check `src/nemotron/steps/patterns/eval-before-and-after-training.md` before comparing SFT results.
- Check `src/nemotron/steps/patterns/sft-small-dataset-prefer-lora.md` when deciding between LoRA and full SFT.

## Config Nuances

- AutoModel SFT consumes JSONL directly through `dataset.path_or_dataset_id`; do not point it at packed Parquet.
- Use `dataloader.collate_fn: nemo_automodel.components.datasets.utils.default_collater` for chat datasets.
- Keep base model and backend settings aligned with PEFT AutoModel when comparing full SFT and LoRA behavior.

## Run It

Smoke first to validate wiring, imports, data access, and output paths:

```bash
uv run nemotron steps run sft/automodel -c tiny --dry-run
```

Then run the real job from a project overlay:

```bash
uv run nemotron steps run sft/automodel \
  -c <project>/config/sft_automodel.yaml
```

## Repository Layout

- Manifest: `src/nemotron/steps/sft/automodel/step.toml`
- Runner: `src/nemotron/steps/sft/automodel/step.py`
- Configs: `src/nemotron/steps/sft/automodel/config/default.yaml`, `src/nemotron/steps/sft/automodel/config/tiny.yaml`

## Guardrails

- Do not add `data_prep/sft_packing`; AutoModel reads JSONL directly.
- Keep `dataloader.collate_fn` on the chat collater unless you intentionally
  provide pre-tokenized data.
- Reduce batch size or switch to LoRA before changing unrelated training logic for OOMs.
- Inspect formatted conversations before trusting loss curves.
