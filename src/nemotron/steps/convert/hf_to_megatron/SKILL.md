---
name: nemotron-convert-hf-to-megatron
description: Configure convert/hf_to_megatron to import a Hugging Face safetensors checkpoint into Megatron distributed checkpoint layout for Megatron-Bridge consumers.
---

# HF To Megatron Conversion

Use `convert/hf_to_megatron` when a downstream Megatron-Bridge step needs
`checkpoint_megatron` but the upstream artifact is `checkpoint_hf`.

Before changing configs or code, read `step.toml` for the artifact contract,
parameters, strategies, and failure modes.

## Inputs And Outputs

- Consume a clean HF checkpoint directory or model id.
- Produce a Megatron distributed checkpoint in a fresh output directory.
- Keep tokenizer and model config files resolvable during import.

## Configure

- Set `hf_model_id` to the HF model id or local checkpoint path.
- Set `megatron_path` to a new output directory.
- Keep `torch_dtype=bfloat16` for typical Nemotron/NVIDIA checkpoints unless a source
  model requires another dtype.
- Set `device_map` only when the installed Megatron-Bridge/Transformers stack
  expects one for local loading.
- Merge LoRA adapters before importing them into Megatron layout.

## Guardrails

- Do not import trainer-state directories, optimizer folders, or adapter-only
  outputs.
- Do not write the Megatron output under the HF source directory.
- Keep `trust_remote_code=true` only for model repos you trust and whose
  architecture is supported by the installed Megatron-Bridge AutoBridge.
