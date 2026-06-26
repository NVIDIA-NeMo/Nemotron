---
id: byob-benchmark-design
title: "Build your own benchmark for sovereign deployment claims"
tags: [eval, byob, sovereign, validation]
triggers:
  - "A sovereign or domain-specific deployment needs evaluation that public benchmarks don't cover."
  - "Standard NLU benchmarks (IFEval, GPQA, MMLU) miss the target language, jurisdiction, or domain."
  - "Stakeholders are asking 'how do we know it works for our use case?' and standard scores don't answer."
  - "A `benchmark_parquet` artifact is being designed for use with eval/model_eval."
steps: [eval/model_eval]
confidence: high
---

## When to apply

Apply this whenever the deployment audience cannot be evaluated by off-the-shelf benchmarks. Sovereign-AI customizations almost always land here: a Hindi government chatbot, a Brazilian legal assistant, a Saudi medical advisor, a Korean support bot — none have a native public benchmark that proves they work.

This is the rule: if the model is being deployed for a specific population, jurisdiction, language, or domain, a Build-Your-Own-Benchmark (BYOB) is part of the customization journey, not optional.

Apply it before training, not after. A benchmark designed retrospectively almost always rationalizes the model's existing strengths and undercounts its real failure modes.

## What to do

**Start from the deployment promise.** Write down what stakeholders, regulators, or product owners expect the model to do. Each claim becomes a benchmark slice. "Answers Hindi tax questions accurately" → a tax-question slice. "Refuses unauthorized advice in Tagalog" → a refusal slice.

**Define the artifact concretely.** A BYOB lives as `benchmark_parquet` (declared in `types.toml`) — a Parquet file with prompts, expected answers or rubrics, metadata (capability tag, difficulty, jurisdiction), and a versioning column. Treat it like training data: tokenizer-locked, version-controlled, and reproducible.

**Cover four slice types.** A useful BYOB has all of these:
- **Capability slices** — one per skill the model is meant to demonstrate (Q&A, summarization, tool calls, reasoning).
- **Failure-mode slices** — known traps: prompt injection, refusal evasion, factual hallucination on known-difficult topics, low-resource sub-language variants.
- **Regression slices** — small fixed sets the model passed before customization. If post-customization scores drop here, capability has degraded.
- **Distribution slices** — sampled from real or representative inference traffic, not synthetic. These catch the gap between what training data looks like and what users actually send.

**Size by intent, not by data availability.** 200–500 high-quality items per slice is a reasonable starting size. 50 is too few to detect movement; 5,000 is overkill for human review and dilutes per-item attention.

**Write the rubric before generating answers.** For each item, the expected behavior should be specified before any model is run on it — even if the rubric is "any of these three answer keys is acceptable" or "must cite a source from this list." Authoring rubrics post-hoc invites confirmation bias.

**Hold the BYOB private.** Never train, calibrate, validate, or do hyperparameter search on the BYOB. It is the contract with the deployment audience and must remain blind to the training pipeline. If it leaks into training data, build a fresh one — see `eval-before-and-after-training`.

**Score with multiple judges.** For non-deterministic tasks: a programmatic check (regex, exact match, schema validation) plus an LLM judge plus human review on a sample. Disagreements among judges are the signal — they identify items where the rubric or the model is ambiguous.

**Version the BYOB with the model.** Every model release pairs with a BYOB version. When the deployment scope expands, the BYOB grows; when it narrows, slices may retire. Track which model version is certified against which BYOB version.

## Exceptions

If the deployment is genuinely covered by an existing public benchmark in the target language and domain (rare for sovereign use cases — common for English research), use it and skip this pattern.

For very early prototyping ("does the runner even start?"), public benchmarks via `eval/model_eval` defaults are fine. The BYOB matters when claiming deployment-readiness, not during plumbing validation.

If the deployment is internal-only, low-stakes, or experimental, a small held-out validation set may be enough — but document that as a known gap, not as a substitute for a real BYOB.

## References

- Pair with `eval-before-and-after-training` so BYOB scores exist before and after every quality-changing stage.
- Pair with `data-quality-before-quantity` when authoring BYOB items — the same quality bar applies.
- Pair with `multilingual-tokenizer-check` for non-English BYOBs (tokenization issues affect generation quality on the BYOB itself).
- Pair with `rl-validate-rewards-before-scale` when the BYOB will judge an RL-trained model.
- Pair with `sdg-pipeline-versioning` if synthetic items contribute to the BYOB (with caution; non-synthetic items must dominate).
