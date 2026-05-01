---
name: byob
description: Generate and translate bring-your-own MCQ benchmarks from domain documents with a modular benchmark-family runtime. Use when a user asks to create an MCQ benchmark, translate a BYOB benchmark, or extend the flow to a new benchmark family such as GSM8K.
when_to_use: Use for requests like "create a custom benchmark from these documents", "run BYOB MCQ generation", "translate the generated benchmark", "add a GSM8K-style BYOB family", or "keep the benchmark schema intact". Do not use for ordinary training-corpus translation.
compatibility: Runtime requires BYOB dependencies for Data Designer, pandas, parquet IO, embedding models, and optional translation metrics.
metadata:
  owner: nemotron
  workflow-step: byob
---

# BYOB

Use this skill to create or translate benchmark artifacts while keeping benchmark-family logic easy for coding agents to change.

## Default

1. Read [references/STEP.md](references/STEP.md) for the artifact contract.
2. Start from [config/default.yaml](config/default.yaml) for MCQ generation or [config/translate.yaml](config/translate.yaml) for translation.
3. Run `python -m nemotron.steps.byob.scripts.run --family mcq --stage prepare --config CONFIG`.
4. Run `python -m nemotron.steps.byob.scripts.run --family mcq --stage generate --config CONFIG`.
5. Translate an existing benchmark with `--stage translate` and a translation config.

## Change Points

- Add new benchmark families under `runtime/benchmark_families/<family>/`.
- Before adding a new family, answer the questions in [references/new-family-checklist.md](references/new-family-checklist.md).
- Register the family in `runtime/benchmark_families/registry.py`.
- Keep `scripts/runtime.py` as a dispatcher only; family-specific schema, prompts, postprocessing, and export code belong in family modules.
- Use `adapter.py` only for schema bridging when composing BYOB with other skills.

## Gotchas

- Do not merge the whole runtime into `scripts/runtime.py`; that blocks future GSM8K-style extensions.
- Keep `question_id`, `question`, `options`, `answer_index`, `answer`, `cot_content`, `src`, and `category` stable in final MCQ parquet outputs.
- Do not drop staged rows inline during translation reassembly. Filtering belongs after rows are restored.
- Resume with `--skip-until` only when the expected cached parquet for the previous stage already exists.
- Use deterministic seeds for sampling and distractor shuffling when comparing benchmark runs.

## Validate

- Run `python -m nemotron.steps.byob.scripts.validate`.
- Run `python -m nemotron.steps.byob.scripts.run --list-families`.
- Confirm final generation outputs `benchmark_raw.parquet` and `benchmark.parquet`.
- Confirm translated outputs preserve row count unless `remove_low_quality` is intentionally enabled.

## Load More Only If Needed

- [references/guide.md](references/guide.md) for orchestration details
- [references/benchmark-schema.md](references/benchmark-schema.md) for MCQ schema rules
- [references/new-family-checklist.md](references/new-family-checklist.md) for GSM8K-style or non-MCQ extensions
- [references/quality-and-filtering.md](references/quality-and-filtering.md) for quality gates
- [patterns/index.yaml](patterns/index.yaml) for skill-local routing hints
