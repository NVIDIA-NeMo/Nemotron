---
id: eval-bookends
title: "Evaluate before and after training"
tags: [eval, pipeline-structure]
triggers:
  - "You are about to train or adapt a model and need to prove improvement."
  - "A pipeline includes SFT, RL, conversion, or any quality-changing stage."
  - "You need to compare multiple training runs fairly."
steps: []
confidence: high
---

## When to apply

Apply this to nearly every training pipeline. If a stage can change model behavior, you want a baseline measurement before the change and a follow-up measurement after it.

This is especially important when changes are subtle: LoRA vs full SFT, prompt-format cleanup, data curation, packing choices, checkpoint conversion, or multilingual tuning. Small gains and regressions are easy to miss without paired measurements.

Use it whenever stakeholders ask, "Did training actually help?" The answer should come from comparable metrics, not from a few hand-picked generations.

## What to do

Run a lightweight baseline evaluation on the starting checkpoint before training begins. Use the same tokenizer, inference settings, prompt format, and benchmark slice you plan to use afterward.

Snapshot the exact eval configuration. Record dataset version, decoding settings, temperature, max tokens, stop sequences, and any system prompt or chat template details.

After training, rerun the same evaluation first. Keep the first comparison maximally controlled before exploring broader benchmark suites.

Report both aggregate metrics and targeted examples. Numerical changes show trend; example outputs show whether the model became more useful, more brittle, or more verbose.

Add task-specific checks alongside standard benchmarks. A support bot may need format compliance, refusal behavior, and terminology fidelity more than generic leaderboard movement.

If you are iterating quickly, keep a cheap smoke eval set and a slower canonical eval set. The smoke set catches obvious regressions during development; the canonical set supports decisions.

Store pre/post eval artifacts next to the run outputs so later checkpoint merges, exports, or deployment changes can still be compared against the original training impact.

## Exceptions

Skip the pre-training baseline only when the upstream checkpoint has already been evaluated with an identical harness and those artifacts are still trustworthy and accessible.

Do not compare before/after numbers if the eval harness changed. A benchmark version bump, different prompt formatting, or different decoding settings can make the comparison misleading.

For very expensive evaluations, reduce scope rather than skipping the baseline entirely. A smaller but controlled pre/post comparison is better than no comparison.

## References

- This is a cross-cutting pipeline rule, not a single-step recommendation.
- Pair naturally with `eval/model_eval`, but the principle applies even when using custom or internal eval harnesses.
- Preserve eval comparability before adding exploratory benchmarks or qualitative demos.
