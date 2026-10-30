---
id: checkpoint-before-convert
title: "Save a clean checkpoint before conversion"
tags: [convert, safety]
triggers:
  - "You are about to convert a checkpoint between HuggingFace and Megatron formats."
  - "A conversion step is destructive, time-consuming, or dependent on external tooling."
  - "The converted artifact will be used for downstream serving or training."
steps: [convert/megatron_to_hf, convert/hf_to_megatron]
confidence: high
---

## When to apply

Apply this before any format conversion stage. Conversion is often treated as bookkeeping, but in practice it is a risky boundary: sharding assumptions, tokenizer assets, dtype handling, and naming conventions can all fail.

Use it whenever a checkpoint is valuable enough that recreating it would cost time, GPU money, or irreproducible experimental context.

This is especially important when converting large distributed checkpoints, resuming from partial training outputs, or preparing a model for external delivery.

## What to do

Write and verify a clean source checkpoint before running the converter. That means all expected files are present, metadata is complete, and the checkpoint can at least be enumerated or reloaded in its native format.

Keep the original checkpoint immutable. Treat conversion outputs as derived artifacts, not replacements for the source of truth.

Version the source and converted outputs separately. Include step names, source format, target format, and tokenizer/version metadata in the directory naming or manifest.

Validate the converted checkpoint immediately with a lightweight load or generation smoke test. A conversion that "completed successfully" can still produce an unusable artifact.

If the checkpoint will be used for serving, also verify tokenizer files, special-token configs, and any generation config objects expected by downstream runtimes.

Log dtype, tensor parallel assumptions, and model variant details alongside the conversion. These details are often forgotten until a later failure forces a forensic reconstruction.

## Exceptions

For disposable local experiments, a redundant backup may feel heavy. Even then, prefer at least one retained source checkpoint before deleting or overwriting anything.

If storage is the main concern, keep the most recent known-good source checkpoint and prune older ones intentionally. Do not rely on converted outputs as the only recoverable copy.

Conversion can still fail after a backup because of incompatible configs or missing tokenizer assets. The point of this pattern is recoverability, not guaranteed success.

## References

- Applies directly to `convert/megatron_to_hf` and `convert/hf_to_megatron`.
- Natural companion to post-conversion smoke evaluation and deployment validation.
- Treat conversion as a safety boundary, not a trivial file rewrite.
