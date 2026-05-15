---
name: nemotron-convert-megatron-to-hf
description: Configure convert/megatron_to_hf to export a Megatron distributed checkpoint iteration into Hugging Face safetensors layout for evaluation, deployment, optimization, or adapter merge.
---

# Megatron To HF Conversion

Use `convert/megatron_to_hf` when a downstream HF-native step needs
`checkpoint_hf` but the upstream artifact is `checkpoint_megatron`.

Before changing configs or code, read `step.toml` for the artifact contract,
parameters, strategies, and failure modes.

## Inputs And Outputs

- Consume a specific Megatron checkpoint iteration, normally an `iter_*`
  directory.
- Produce a standalone HF safetensors checkpoint.
- Preserve tokenizer and config expectations from the original HF model id.

## Configure

- Set `megatron_path` to the concrete checkpoint iteration, not the parent run
  directory.
- Set `hf_model_id` to the original model/config source when the checkpoint
  lacks enough HF metadata.
- Set `hf_path` to a fresh export directory.

## Guardrails

- Do not export while async checkpoint save is still in progress.
- Do not guess among multiple checkpoint iterations; pick the validated one.
- Validate that the exported HF checkpoint loads before using it for eval or
  deployment.
