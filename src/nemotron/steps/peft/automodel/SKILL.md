---
name: nemotron-peft-automodel
description: Configure Nemotron peft/automodel for AutoModel LoRA adapter training on chat-format JSONL. Use for memory-efficient HF model adaptation, single-node experiments, adapter checkpoints, LoRA rank or alpha tuning, and later HuggingFace checkpoint merging.
---

# AutoModel PEFT

Use `peft/automodel` for LoRA tuning when the data is chat-format JSONL and the base model is Hugging Face-native.

Before changing configs or code, read `step.toml` to understand the step flow, consumed and produced artifacts, important parameters, strategies, failure modes, and upstream references.

## Inputs And Outputs

- Consume `training_jsonl`.
- Produce `checkpoint_lora`.
- Validate with a short adapter run before scaling data, rank, or sequence length.

## Configure

- Set `model.pretrained_model_name_or_path` and keep that exact base recorded
  with the adapter for later merge.
- Set `dataset.path_or_dataset_id` to chat-format JSONL.
- Start with `peft.dim=8` or `16` on tight memory, then increase for harder tasks.
- Keep `peft.alpha` near `2 * peft.dim` unless there is a reason to tune it.
- Use smaller base models for single-GPU experiments.
- Merge with `convert/merge_lora` when deployment requires a standalone HF checkpoint.
- Check `src/nemotron/steps/patterns/sft-small-dataset-prefer-lora.md` before choosing LoRA for small datasets.
- Check `src/nemotron/steps/patterns/peft-adapter-merge-discipline.md` before merging adapters.

## Config Nuances

- Keep PEFT and SFT AutoModel on the same `model.pretrained_model_name_or_path`; PEFT should differ mainly in `peft`, data path, and iteration count.
- For Qwen MoE runs, use `model.backend.experts: torch_mm` and `model.backend.dispatcher: torch` unless DeepEP support has been validated in the target container.
- Prefer `distributed.activation_checkpointing: false` while dispatcher and checkpoint-recompute behavior are still being validated.

## Local Files

- Contract: `src/nemotron/steps/peft/automodel/step.toml`
- Runner: `src/nemotron/steps/peft/automodel/step.py`
- Configs: `src/nemotron/steps/peft/automodel/config/default.yaml`, `src/nemotron/steps/peft/automodel/config/tiny.yaml`

## Guardrails

- Do not run `data_prep/sft_packing`; this step consumes JSONL directly.
- Reduce rank and sequence length before changing the training wrapper for OOMs.
- Treat the adapter as a separate artifact until merge and eval have passed,
  and preserve base/tokenizer/rank/alpha provenance with it.
