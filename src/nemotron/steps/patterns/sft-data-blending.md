---
id: sft-data-blending
title: "Blend SFT data deliberately across capabilities"
tags: [sft, data-blend, sovereign, capabilities]
triggers:
  - "An SFT corpus combines instruction-following, chat, tool use, reasoning, and domain-specific data."
  - "You are mixing translated/synthetic data with curated human-written data."
  - "Sovereign / regional SFT data is being blended with broader open-source instruction sets."
  - "After SFT the model loses one capability while gaining another."
steps: [prep/sft_packing, sft/automodel, sft/megatron_bridge, peft/automodel, peft/megatron_bridge]
confidence: high
---

## When to apply

Apply this whenever SFT data comes from more than one source. Sovereign customization almost always blends: target-language conversations, English instruction sets, domain tool-use traces, reasoning datasets, refusal/safety data, and synthetic preference responses can all land in the same training run.

This pattern matters most when capabilities trade against each other: a mostly-English instruction set will pull a target-language model back toward English; a tool-call-heavy blend can degrade chat fluency; a reasoning-heavy blend can hurt brevity on conversational tasks.

Apply it before `prep/sft_packing`. The blend ratios decide what goes into the packed Parquet; reshuffling after packing means repacking.

## What to do

**Inventory capabilities, not just sources.** Before blending, label each source by what capability it teaches: target-language fluency, instruction-following, multi-turn chat, tool calls, reasoning chains, refusal, citation, etc. Two datasets from different sources can teach the same capability — count by capability, not by file.

**Set ratios from the deployment mix, not from data availability.** If 70% of inference traffic is target-language chat and 10% is tool calls, the SFT blend should approximate that distribution. Loading 50% English instruction data because "we have it" trains for the wrong distribution.

**Cap any single source.** No single source should exceed ~40–50% of the blend, even if it's high-quality. Concentration creates style monocultures the model overfits to.

**Mix synthetic with human-written data.** Pure synthetic SFT (Data Designer output, distilled responses) tends to produce stylistically narrow models. Blend with at least 20–30% real human-written or expert-curated data when available — see `sdg-pipeline-versioning`.

**Keep capability slices balanced.** If reasoning, tool use, and chat each need to work, each capability should have enough rows to register against the blend. A 5K-row reasoning slice in a 500K-row corpus is a rounding error.

**Translate, don't paraphrase, when localizing.** When mixing translated open-source data with target-language native data, run `translate/nemo_skills` with FAITH scoring (see the step's strategies) and keep faith ≥ 0.7. Low-faith translations dilute the language signal.

**Validate the blend before packing.** Sample 100 rows proportional to the planned blend and inspect. If the sample doesn't look like what you want the model to do, the full blend won't either.

**Tag every row with source + capability.** This pays off when an SFT run regresses one capability — you can identify which slice changed.

## Exceptions

For tightly-scoped domain SFT (single capability, single source, e.g. "answer support questions in product X's voice"), aggressive blending can hurt. A single high-quality source is fine when the goal is narrow.

When data is genuinely scarce in the target language, blending in English is unavoidable. Document the ratio and accept the partial English drift; that's the price of insufficient native data.

For PEFT / LoRA, blend rules are softer — adapters don't rewrite the base model, so capability trade-offs are smaller. See `sft-small-dataset-prefer-lora` and `peft-adapter-merge-discipline`.

## References

- Pair with `prep-data-is-tokenizer-locked` so the packed Parquet captures the blend ratios.
- Pair with `sft-sequence-packing` once blend ratios are decided — packing efficiency depends on the blended length distribution.
- Pair with `multilingual-tokenizer-check` when a target-language slice is in the blend.
- Pair with `data-quality-before-quantity` per source before blending.
- Pair with `sdg-pipeline-versioning` when synthetic data is part of the blend.
- Pair with `eval-before-and-after-training` to measure each capability before and after SFT.
- Pair with `sft-small-dataset-prefer-lora` when the blend is small enough to favor PEFT.
