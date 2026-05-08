---
id: sdg-pipeline-versioning
title: "Version synthetic data generation as a pipeline"
tags: [sdg, data-quality, reproducibility]
triggers:
  - "Synthetic SFT, tool-use, prompt, or preference data is being generated."
  - "A Data Designer config is moving from preview mode to a production-scale generation job."
  - "Generated data will feed SFT, DPO, RLVR, RLHF, or downstream data prep."
  - "A second SDG run needs to reproduce or extend an earlier corpus."
steps: [sdg/data_designer, prep/sft_packing, prep/rl_prep, sft/automodel, sft/megatron_bridge, rl/nemo_rl/dpo]
confidence: high
---

## When to apply

Apply this whenever synthetic data lands in the training corpus. SDG is a pipeline (seeds → prompts → models → projections → validators), not a one-off prompt — and like every pipeline it accumulates behaviorally meaningful state that's invisible from the output JSONL alone.

This matters most for sovereign or domain-specific generation, where the same SDG job will be re-run as the deployment expands (new languages, new tool calls, new policy classes). A synthetic corpus that can't be regenerated identically is a liability.

It also matters for preference and tool-use data, where schema correctness — not text fluency — is what determines whether the corpus is usable downstream.

## What to do

**Version six things together** with the generated output:
1. Seed file(s) + their content hash.
2. Prompt templates (system + user prompts as they appear in the config).
3. Model alias, model id, and inference parameters (temperature, top_p, max_tokens).
4. Validators / output projection rules.
5. The Data Designer config file itself.
6. The generated JSONL plus its row count.

A single git commit or single tagged directory is the cheapest way; a manifest file (`SDG_RUN.toml`) recording all six is the most auditable.

**Always preview before scaling.** Run with `--preview` (or `tiny.yaml`) until the projection schema is right. The quality bugs that show up in 10 records become expensive at 10,000 records.

**Pin the output schema explicitly.** SFT data should project to OpenAI `messages` if downstream is `prep/sft_packing` or `sft/automodel`. DPO data must project to `{prompt, chosen, rejected}` for `prep/rl_prep` → `rl/nemo_rl/dpo`. Tool-use data uses `structured_messages` with `messages` + `tools`. Don't generate ambiguous schemas hoping a downstream consumer will untangle them.

**Curate seed data deliberately.** Seeds are the single biggest lever on output diversity:
- Representative of the target deployment audience.
- Licensed for the intended use.
- Free of private or PII data unless explicitly approved.
- Balanced across target capabilities (don't seed 80% Q&A and complain that the model only learned Q&A).

**Drive diversity with structure, not temperature.** Sampler columns, persona columns, task-category columns, difficulty fields, and explicit constraints produce more diverse outputs than raising temperature. High-temperature generation produces incoherent records; structured diversity produces useful ones.

**Validate what you'll actually train on.** Before the corpus feeds training, check:
- Schema validity (every row has the projection's required fields).
- Safety / refusal compliance for the deployment audience.
- Duplication rate (near-duplicates; not just exact).
- Language distribution (target-language rows are actually in the target language).
- Length distribution (not dominated by very short or very long outliers).
- Train/test contamination against the eval set — see `byob-benchmark-design`.
- For tool-use: every assistant tool call has a matching `tool_call_id`, tool arguments are JSON strings (not nested objects), final assistant answer reflects the tool response.

**Hold a non-synthetic eval slice.** Synthetic corpora teach the patterns in the corpus — including the patterns of the generator model. A held-out human-written or task-eval set is the only honest measure of whether SDG is helping. See `byob-benchmark-design`.

## Exceptions

For credential or API testing ("does the generator even respond?"), a tiny unversioned preview is fine. Don't promote those records into training data without reconstructing the config and lineage.

If generated records are illustrative documentation samples (not training inputs), full versioning is overkill.

For exploratory generation where the goal is understanding what the generator will produce, version the config but skip the heavy validation pass — the output isn't going into training yet.

## References

- Pair with `data-quality-before-quantity` before scaling synthetic volume.
- Pair with `sft-data-blending` when synthetic data joins curated human data in the SFT blend.
- Pair with `prep-data-is-tokenizer-locked` before packing generated chat data for Megatron-Bridge SFT.
- Pair with `rl-validate-rewards-before-scale` when generated preferences feed `rl/nemo_rl/dpo` or `rl/nemo_rl/rlhf`.
- Pair with `byob-benchmark-design` to keep synthetic training data and the held-out eval set genuinely separate.
