---
id: convert-checkpoint-safety
title: "Save a clean checkpoint before conversion"
tags: [convert, safety, format-bridge]
triggers:
  - "You are about to convert a checkpoint between HuggingFace and Megatron formats."
  - "A LoRA adapter is about to be merged into a base model."
  - "A converted artifact will feed evaluation, RL warm-start, or deployment."
  - "The conversion runs against a long or distributed-checkpoint output."
steps: [convert/megatron_to_hf, convert/hf_to_megatron, convert/merge_lora]
confidence: high
---

## When to apply

Apply this before any `convert/*` stage. Conversion looks like bookkeeping — bytes in, bytes out — but it crosses real boundaries: shard layout, tokenizer files, dtype, generation config, and tensor naming. Each is a place where a successful-looking conversion produces an unusable artifact.

This matters most when the source checkpoint is expensive to recreate: a distributed Megatron run with thousands of shards, an SFT-aligned policy, or a LoRA trained against a specific base. Treat conversion as a one-way step you will eventually want to redo.

It also matters whenever the converted artifact will be used to **make decisions**: serving traffic, RL warm-start, eval scores reported externally. A silently broken converted checkpoint corrupts every downstream signal.

## What to do

**Verify the source before converting.** The source must be enumerable in its native format: every expected shard present, metadata complete, tokenizer assets bundled. Run a tiny load — Megatron's checkpoint resume, or `from_pretrained` for HF — before piping into `convert/*`.

**Treat the source as immutable.** Never overwrite. The converted output is a derived artifact. If the conversion fails or the consumer rejects the output, you must still have the source.

**Name the source and target by what they are, not where they came from.** Prefer `nano3-sft-iter-50k-bf16-megatron/` and `nano3-sft-iter-50k-bf16-hf/` over `output/` and `output_converted/`. Include format, dtype, and provenance in the directory name; failure forensics depend on it.

**Smoke-test the converted output immediately.** A successful conversion is not a working conversion. Run, in order:
1. Load the converted checkpoint in its target runtime.
2. Generate one short response from a known prompt.
3. Compare against a generation from the source. Outputs should be near-identical (modulo dtype rounding).

If step 3 fails, the conversion broke quietly — usually tokenizer mismatch, missing `generation_config.json`, or a tensor-name remap.

**Validate the conversion-specific failure surface.** Each `convert/*` step has its own:
- `convert/megatron_to_hf` → bundle `tokenizer*` files explicitly; verify dtype matches the source; check that the iter_* directory was the actual conversion source, not the parent run dir.
- `convert/hf_to_megatron` → verify TP/PP shape matches the consumer's expected parallelism; check chat template registration.
- `convert/merge_lora` → confirm the base checkpoint is the **same one** the adapter was trained against. Different bases silently merge into garbage. See `peft-adapter-merge-discipline`.

**Keep dtype and parallelism metadata visible.** Write a short `CONVERSION.md` or `manifest.json` next to the output recording: source path, source format, source dtype, target format, target dtype, TP/PP if applicable, conversion command, conversion timestamp. Conversion bugs are usually metadata bugs.

**Plan the rollback.** If conversion fails or downstream rejects the output, your recovery path must already exist: keep the source intact, keep at least one prior known-good converted checkpoint, document which downstream consumer last accepted which output.

## Exceptions

For disposable local experiments, full provenance is overkill — but **never overwrite the source**. Even a 10-minute SFT smoke run is cheaper to convert than to retrain.

If storage is tight, prune older converted outputs deliberately rather than the source. The source can always be reconverted; conversions cannot recover a deleted source.

A successful backup does not guarantee a successful conversion. Tokenizer drift, model-config mismatches, or sharding bugs can still bite. The point of this pattern is recoverability, not infallibility.

## References

- Pair with `peft-adapter-merge-discipline` whenever a `convert/merge_lora` is in the chain.
- Pair with `eval-before-and-after-training` to confirm the converted checkpoint scores match the source.
- Pair with `prep-data-is-tokenizer-locked` when conversion happens between prep and a downstream trainer (the tokenizer must survive the bridge).
- Pair with `pretrain-token-budget-before-scale` if conversion is part of moving a pretrain checkpoint between AutoModel and Megatron-Bridge.
