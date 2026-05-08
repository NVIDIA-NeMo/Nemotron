---
name: nemotron-sft-automodel
description: Configure Nemotron sft/automodel for supervised fine-tuning with NeMo AutoModel on chat-format JSONL. Use for HF-native checkpoints, smaller GPU counts, rapid iteration, full SFT, LoRA-style tuning from the AutoModel stack, or direct training_jsonl inputs.
---

# AutoModel SFT

Use `sft/automodel` for Hugging Face-format models that can train directly from `training_jsonl`.

Before changing configs or code, read `step.toml` to understand the step flow, consumed and produced artifacts, important parameters, strategies, failure modes, and upstream references.

## Inputs And Outputs

- Consume chat-format `training_jsonl`.
- Produce `checkpoint_hf`.
- Validate chat formatting and checkpoint output with a short run before scaling data or sequence length.

## Configure

- Default `sft/automodel` is full fine-tuning (`peft=null`); use `peft/automodel` or add a LoRA `peft:` block when adapter training is intended.
- Keep `peft=lora` for memory-constrained runs or fast adapter experiments.
- Choose a tokenizer with chat-template support or preprocess prompts explicitly.
- Use the AutoModel launcher and executor settings when moving from local to cluster execution.
- Check `src/nemotron/steps/patterns/eval-before-and-after-training.md` before comparing SFT results.
- Check `src/nemotron/steps/patterns/sft-small-dataset-prefer-lora.md` when deciding between LoRA and full SFT.

## Config Nuances

- AutoModel SFT consumes JSONL directly through `dataset.path_or_dataset_id`; do not point it at packed Parquet.
- Use `dataloader.collate_fn: nemo_automodel.components.datasets.utils.default_collater` for chat datasets.
- Keep base model and backend settings aligned with PEFT AutoModel when comparing full SFT and LoRA behavior.

## Local Files

- Contract: `src/nemotron/steps/sft/automodel/step.toml`
- Runner: `src/nemotron/steps/sft/automodel/step.py`
- Configs: `src/nemotron/steps/sft/automodel/config/default.yaml`, `src/nemotron/steps/sft/automodel/config/tiny.yaml`

## Guardrails

- Do not add `prep/sft_packing`; AutoModel reads JSONL directly.
- Reduce batch size or switch to LoRA before changing unrelated training logic for OOMs.
- Inspect formatted conversations before trusting loss curves.
