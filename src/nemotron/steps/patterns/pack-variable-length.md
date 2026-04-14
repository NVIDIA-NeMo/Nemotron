---
id: pack-variable-length
title: "Pack variable-length SFT data"
tags: [prep, sft, efficiency]
triggers:
  - "Training examples range from very short to very long sequences."
  - "GPU utilization is poor because padding dominates batches."
  - "You are preparing data for Megatron-Bridge SFT with packed inputs available."
steps: [prep/sft_packing]
confidence: high
---

## When to apply

Use packing when the SFT corpus has a wide length distribution: short chats mixed with long instructions, QA pairs mixed with multi-turn conversations, or templated samples mixed with long-form reasoning traces.

It is most valuable when naive batching would waste large amounts of compute on padding tokens. This shows up as low GPU utilization, unexpectedly low tokens/sec, or many batches whose effective content is much smaller than the configured sequence length.

Apply it early in Megatron-oriented pipelines because packing changes both upstream prep assumptions and downstream training efficiency.

## What to do

Measure the token-length histogram first. Do not enable packing blindly; confirm that the data really has enough variance for packing to matter.

Choose a pack size that matches the intended training sequence length. Mismatches between packing config and training config are a common source of silent waste and hard failures.

Preserve label masks and sample boundaries carefully. Good packing improves utilization without teaching the model to predict across unrelated examples.

Inspect a few packed outputs directly. Verify that EOS handling, loss masking, and conversation boundaries match the training recipe's expectations.

Recompute throughput estimates after packing. Teams often discover they can increase effective tokens processed per GPU-hour without changing the cluster footprint.

Keep the tokenizer fixed across packing and training. A tokenizer mismatch invalidates the packed artifacts and creates confusing downstream errors.

Retain a tiny un-packed debug slice if possible. It makes troubleshooting easier when you need to isolate whether a training problem comes from data content or packing logic.

## Exceptions

Skip packing when sequence lengths are already tight and uniform. If nearly every example is close to the same length, the added complexity may buy little.

Do not use packing as a band-aid for poor data formatting. Broken chat templates, malformed JSONL, or inconsistent turns should be fixed before packing.

If the downstream trainer or evaluation path cannot respect boundaries or masks correctly, packing can hurt quality. In that case, prioritize correctness over utilization.

## References

- Most directly relevant to `prep/sft_packing` and downstream Megatron-Bridge SFT.
- Sequence packing is often the highest-leverage efficiency improvement for heterogeneous chat corpora.
- Revisit this pattern whenever the data mix changes substantially across customers or domains.
