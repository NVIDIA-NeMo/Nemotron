---
id: custom-mcq-benchmark-byob
title: "Build a domain MCQ benchmark with BYOB few-shot seeds"
tags: [benchmark, byob, mcq, evaluation]
triggers:
  - "You have domain text (subjects, manuals, lectures) and need a multiple-choice benchmark comparable to public MCQ suites."
  - "You want few-shot style grounded in existing Hugging Face benchmarks such as MMLU, MMLU-Pro, Global-MMLU, or GPQA."
  - "You plan to evaluate or ship a model on custom categories and need reproducible Parquet in a standard MCQ layout."
steps: [benchmark/byob]
confidence: high
---

## When to apply

Apply when generic public benchmarks do not cover your domain, but you still want question format, difficulty, and distractor structure aligned with established MCQ tasks.

Use it when stakeholders require traceability from passages to questions, optional human review before treating outputs as a benchmark, and filters for hallucination or trivially easy items.

Skip or defer BYOB when you only need zero-shot prompts on raw text without structured MCQ export, or when NeMo Data Designer and model endpoints for generation and judging are not available.

## What to do

Pick a supported few-shot source dataset (`ALLOWED_HF_DATASETS` in `nemotron/steps/byob/constants.py`) and configure `target_source_mapping` so target subjects map to your corpus layout under `input_dir`.

Run the pipeline from seed generation through judging; enable semantic deduplication, distractor validity, coverage checks, and easiness or hallucination filters when quality matters more than raw volume.

Export final `benchmark_parquet` in the MMLU-Pro-style column layout and schedule human-in-the-loop review before using the file as a production benchmark or merging into training mixes.

Keep experiment names, config YAML, and intermediate Parquet under `output_dir`/`expt_name` so runs stay reproducible and comparable.

## Exceptions

If your goal is only to evaluate on existing Hugging Face tasks, prefer `eval/model_eval` against published suites rather than generating new questions.

When the corpus is tiny or highly redundant, expect weaker diversity; widen targets or relax filters only after validating that metrics still reflect intended difficulty.

## References

- Catalog step `benchmark/byob` and artifact type `benchmark_parquet` in `types.toml`.
- Runnable entry: `use-case-examples/build-your-own-benchmark/build_mcq_benchmark.ipynb` (multi-stage pipeline in **section 3 — MCQ generation pipeline**; also `use-case-examples/build-your-own-benchmark/README.md`).
