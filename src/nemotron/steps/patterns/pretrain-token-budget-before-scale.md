---
id: pretrain-token-budget-before-scale
title: "Define the pretraining token budget before scaling"
tags: [pretrain, planning, scaling, budget]
triggers:
  - "You are planning pretraining or continued pretraining beyond a smoke test."
  - "A pretrain config is being scaled from tiny/local execution to multi-GPU or multi-node execution."
  - "You need to choose between pretrain/automodel and pretrain/megatron_bridge."
  - "Cluster cost or wall-clock budget is being requested for a pretrain run."
steps: [data_prep/pretrain_prep, pretrain/automodel, pretrain/megatron_bridge]
confidence: high
---

## When to apply

Apply this before any pretraining run that costs more than a smoke test. Pretraining is the highest-cost stage in the catalog — tokens × seq_length × global_batch × wall-clock × GPU-count compounds quickly. Decisions made here cannot be tuned away later.

This is a from-scratch and CPT-baseline planning pattern. For CPT-specific blend ratios, base-model preservation, and forgetting checks see `cpt-data-blend-scoping`. Use this pattern to fix the budget *contract*; use the CPT pattern to fix the *blend* inside that contract.

Apply it before requesting cluster capacity. Hardware shape, cluster duration, and checkpoint storage all flow from the budget — the budget should not flow from what's available.

## What to do

**Write the budget down before code changes.** A reusable template:

```
target_tokens     = <e.g. 10B>
seq_length        = <e.g. 4096>
global_batch_size = <e.g. 2048>     # tokens per step = seq_length × gbs = 8.4M
train_iters       = target_tokens / (seq_length × gbs)   # ~1200
warmup_iters      = ~1-2% of train_iters
lr_schedule       = cosine to ~10% of peak by end of training
ckpt_every_iters  = ~10% of train_iters
val_every_iters   = ~5% of train_iters
expected_throughput = <tokens/sec/GPU from a representative profile>
expected_walltime  = train_iters / (throughput × num_gpus / (seq_length × gbs))
```

If any field is missing, the budget is a hope, not a plan.

**Choose the backend by scale and format.** Both `pretrain/automodel` and `pretrain/megatron_bridge` produce a usable model:
- **AutoModel** when checkpoints must land in HuggingFace format, when the cluster is ≤1 node, or when iteration speed > peak throughput.
- **Megatron-Bridge** when model size or sequence length forces tensor / pipeline / context / expert / sequence parallelism, when multi-node training is unavoidable, or when Nano3/Super3 recipe parity matters.

The decision is sticky — switching backends mid-run is itself a conversion problem. See `convert-checkpoint-safety`.

**Run a representative short job before the long run.** Not the tiny.yaml smoke (that proves runner imports). A 10–60 minute job at production sequence length and parallelism that lets you measure:
- Tokens/second/GPU (validate your throughput assumption).
- Validation loss movement (validate the data is teaching anything).
- Checkpoint save and restore (validate that 10% of train_iters won't lose the run).
- Memory headroom (validate parallelism choice has spare).

If any of these surprises you, fix it before the long run, not during.

**Hold a clean validation corpus.** Validation data must never appear in training. For CPT, see also `cpt-data-blend-scoping` for general-capability validation slices. For from-scratch, hold out at least one out-of-distribution slice so you can detect overfitting to the training mix.

**Plan the restart policy.** Cluster jobs fail. Decide before the run: how often to checkpoint, how many to retain, where they live, and what the resume command looks like. A run that loses 6 hours to a NCCL hang and has no resumable checkpoint is a 6-hour loss; a planned resume is a 5-minute one.

**For continued pretraining, use a much lower learning rate.** CPT lr is typically 5–10× lower than from-scratch (1e-5 to 5e-5 — see the `pretrain/automodel` step.toml CPT strategy). Higher rates accelerate forgetting. The full CPT discipline lives in `cpt-data-blend-scoping`.

## Exceptions

Tiny configs are still enough for runner validation. Don't require a full token-budget exercise when the goal is import / launch / config smoke testing.

For exploratory research, the budget can be approximate at first — but it must still be documented well enough that another run can reproduce the scale. "We trained for a while" is not a budget.

If the user is on a fixed cluster duration (e.g. "we have 4 H100 nodes for 48 hours"), the budget flips: wall-clock becomes the constraint, target_tokens the output. Measure throughput first, then derive how many tokens fit.

## References

- Pair with `cpt-data-blend-scoping` when continuing pretraining of an existing base model.
- Pair with `prep-data-is-tokenizer-locked` before reusing existing bin/idx artifacts.
- Pair with `eval-before-and-after-training` so pretraining gains and regressions are measured against the base checkpoint.
- Pair with `multilingual-tokenizer-check` if the corpus is non-English (token-budget assumptions break when tokens-per-character is much lower than English).
- Pair with `convert-checkpoint-safety` if the pretraining output crosses HF and Megatron formats.
- Pair with `byob-benchmark-design` if the pretraining target is a sovereign or domain-specific deployment.
