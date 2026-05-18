---
name: nemotron-convert-merge-lora
description: Configure convert/merge_lora to merge a LoRA adapter into its original Hugging Face base checkpoint and produce a standalone HF checkpoint.
---

# Merge LoRA

Use `convert/merge_lora` when a downstream consumer needs a standalone
`checkpoint_hf` instead of a separate adapter artifact.

Before changing configs or code, read `step.toml` for the artifact contract,
parameters, strategies, and failure modes.

## Inputs And Outputs

- Consume `checkpoint_lora` plus the original HF base checkpoint.
- Produce a merged HF checkpoint suitable for HF-native eval, deployment, or
  conversion to Megatron.

## Configure

- Set `lora_checkpoint` to the adapter output from the PEFT run.
- Set `base_hf_path` to the exact base model used during adapter training.
- Set `output_hf_path` to a fresh directory.
- Use CPU merge for memory-constrained or non-training environments.

## Guardrails

- Never merge into a different base, even if the model name looks compatible.
- Evaluate after merge; adapter-loaded and merged-model scores can differ.
- Keep tokenizer, chat template, LoRA rank, alpha, and target module provenance
  with the merged artifact.
