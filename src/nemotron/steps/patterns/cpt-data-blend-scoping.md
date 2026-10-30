---
id: cpt-data-blend-scoping
title: "Scope the CPT data blend before scaling pretraining tokens"
tags: [pretrain, cpt, data-blend, sovereign]
triggers:
  - "You are continuing pretraining of a base model on a sovereign or domain corpus."
  - "A CPT blend mixes domain text (legal, medical, government, finance) with general web/Wikipedia data."
  - "The training token budget is being chosen for a continued-pretraining run."
  - "Catastrophic forgetting on general benchmarks is a concern after CPT."
steps: [data_prep/pretrain_prep, pretrain/automodel, pretrain/megatron_bridge]
confidence: high
---

## When to apply

Apply this whenever continued pretraining (CPT) is part of a sovereign-AI customization plan — adapting a strong base model to a target language, jurisdiction, or domain corpus. CPT is unforgiving: too few domain tokens fail to shift behavior, too many damage the base model's general capabilities.

This is most relevant when the corpus mixes very different distributions: native-language scripts with English code samples, legal/medical jargon with general web text, or sovereign news with multilingual Wikipedia. The blend ratio drives every downstream signal.

Apply it before shard count, sequence length, or learning-rate decisions are locked in. Those choices flow from the blend, not the other way around.

## What to do

**Define the target shift first.** Write down the specific behaviors you want CPT to produce: target-language fluency, domain terminology, citation conventions, or culturally-aware responses. Vague goals ("make it better at Hindi") produce vague blends.

**Bound the token budget by base-model size, not corpus size.** Useful starting heuristics for CPT on a 7B–30B base:
- Conservative shift: 1–5B tokens of domain data, blended 30–50% with general data.
- Moderate shift: 5–20B tokens, blended 40–60%.
- Aggressive shift: 20–50B tokens, blended 50–70%, with explicit forgetting checks.

These are starting points. Validate with a short run before committing the full budget — see `pretrain-token-budget-before-scale`.

**Always blend with general data.** A pure domain run accelerates forgetting. Mix in general web, Wikipedia, or the original pretraining-style corpus at the ratios above. Document the per-source weight in `blend.json` and version it with the prepared `binidx`.

**Hold out a sovereign-aware validation slice.** Validation must include both target-domain prompts (to measure shift) and general benchmarks the base model already passes (to detect forgetting). The base model's score on the general slice is your forgetting baseline.

**Lock the tokenizer before blending.** A CPT run that changes the tokenizer is a from-scratch run, not CPT. If the target language has poor base-model tokenizer coverage, see `multilingual-tokenizer-check` and consider whether CPT is the right tool at all.

**Lower the learning rate.** CPT learning rates are typically 5–10× lower than from-scratch (1e-5 to 5e-5 range — see the `pretrain/automodel` step.toml CPT strategy). Higher rates accelerate forgetting.

**Checkpoint frequently and evaluate at each.** Forgetting is monotonic and often irreversible past a certain point. Save a checkpoint every few hundred million tokens, run the validation slice, and stop early if general capability degrades past tolerance.

## Exceptions

If the goal is from-scratch pretraining (not continuation), the budget logic flips — sheer token volume matters more, and base-model-derived heuristics don't apply. Use `pretrain-token-budget-before-scale` directly.

If the corpus is small (sub-1B tokens) and the goal is style/terminology adaptation rather than knowledge injection, SFT or PEFT may match CPT behavior at a fraction of the cost. See `sft-small-dataset-prefer-lora`.

For sovereign deployments where licensing is the primary driver (training only on cleared corpora), the blend ratios may be constrained by what's legally usable rather than what's optimal — document the constraint and adjust expectations.

## References

- Pair with `pretrain-token-budget-before-scale` for the broader budget/restart/validation discipline.
- Pair with `prep-data-is-tokenizer-locked` so the prepared `binidx` survives blend changes.
- Pair with `multilingual-tokenizer-check` before CPT on a non-English corpus.
- Pair with `data-quality-before-quantity` when the domain corpus is messy or scraped.
- Pair with `eval-before-and-after-training` to measure base capability before and after CPT.
