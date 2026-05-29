---
id: peft-adapter-merge-discipline
title: "Keep adapter artifacts separate until merge is validated"
tags: [peft, lora, convert, deployment]
triggers:
  - "A LoRA or PEFT run needs to produce a standalone deployable checkpoint."
  - "An adapter checkpoint is about to be merged into a base model."
  - "Downstream evaluation or serving expects a full HuggingFace checkpoint rather than an adapter."
steps: [peft/automodel, peft/megatron_bridge, convert/merge_lora]
confidence: high
---

## When to apply

Apply this whenever LoRA or another adapter is part of the training path. Adapter checkpoints are lightweight and convenient, but they are not the same artifact as a merged full model.

Use it when moving from training to evaluation, export, or deployment. Many downstream tools expect a standalone HuggingFace checkpoint, while others can load a base model plus adapter.

It also matters when several domain adapters are trained from the same base model. Mixing up base and adapter versions can invalidate comparisons.

## What to do

Record the base checkpoint, adapter checkpoint, tokenizer, LoRA rank, alpha, target modules, and training data version together.

Keep the adapter output immutable until a merge succeeds. Treat merged checkpoints as derived artifacts, not replacements for the adapter or base.

After merging, run the same focused eval used for the adapter path. Confirm that merged quality matches adapter-loaded quality closely enough for the intended use.

Validate tokenizer files, generation config, dtype, and model config after merge. Merge failures often appear later as serving or evaluation problems.

If the deployment path needs a Megatron checkpoint, plan the conversion explicitly rather than assuming the adapter merge will satisfy every consumer.

## Exceptions

If the downstream system natively supports base-plus-adapter loading, a merged checkpoint may not be necessary. Still keep the adapter metadata clear.

If the adapter is a disposable local experiment, minimal tracking may be enough, but do not reuse it in a larger comparison without reconstructing the base and data lineage.

## References

- Pair with `sft-small-dataset-prefer-lora` when deciding whether adapter tuning is the right training mode.
- Pair with `convert-checkpoint-safety` when merge feeds a `convert/*` step (the merged HF checkpoint is itself a source that conversion must respect).
- Pair with `eval-before-and-after-training` to compare base, adapter-loaded, and merged checkpoints fairly — never compare adapter-loaded scores against merged scores assuming they're identical.
- Pair with `sft-data-blending` — domain adapters trained off different blends of the same base must be tracked by blend, not just by base.
- Pair with `byob-benchmark-design` for sovereign deployments where the merged checkpoint is what serves traffic.
