---
name: nemotron-convert-merge-lora
description: Configure convert/merge_lora to merge a LoRA adapter into its original base checkpoint and produce a standalone HF checkpoint.
---

# Merge LoRA

Use `convert/merge_lora` when a downstream consumer needs a standalone
`checkpoint_hf` instead of a separate adapter artifact.

Before changing configs or code, read `step.toml` for the artifact contract,
parameters, strategies, and failure modes.

## Inputs And Outputs

- Consume `checkpoint_lora` plus the original base checkpoint.
- With `backend=hf_peft`, consume the original HF base and write HF output
  directly.
- With `backend=megatron_bridge`, consume the original dense Megatron base,
  write a merged Megatron checkpoint, then export it to HF when `export_hf=true`.

## Configure

- Keep `backend=auto` unless you want to force a merge path.
- Set `backend=hf_peft` for AutoModel/HuggingFace PEFT adapters.
- Set `backend=megatron_bridge` for Megatron-Bridge adapters.
- Set `lora_checkpoint` to the adapter output from the PEFT run.
- For HF PEFT, set `base_hf_path` to the exact base model used during adapter
  training and `output_hf_path` to a fresh directory.
- For Megatron-Bridge, set `base_megatron_path`, `hf_model_id` or
  `hf_model_path`, `output_megatron_path`, and `output_hf_path`.
- Use CPU merge for memory-constrained or non-training environments when
  parallelism is 1.

## Guardrails

- Never merge into a different base, even if the model name looks compatible.
- Evaluate after merge; adapter-loaded and merged-model scores can differ.
- Keep tokenizer, chat template, LoRA rank, alpha, and target module provenance
  with the merged artifact.
