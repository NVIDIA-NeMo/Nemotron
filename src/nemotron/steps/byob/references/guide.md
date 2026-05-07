# BYOB Guide

## Generation

Install runtime dependencies before running BYOB:

```bash
uv sync --extra byob
```

Prepare source documents as one directory or parquet file per target subject. In the MCQ family, each target
subject maps to few-shot source subjects from the configured Hugging Face benchmark. MCQ stage orchestration
lives in `runtime/benchmark_families/mcq/pipeline.py`. The generation run:

1. Samples few-shot examples and target document chunks into `seed.parquet`.
2. Generates candidate MCQs with Data Designer.
3. Judges candidate quality.
4. Applies semantic deduplication with Curator's current semantic dedup workflow.
5. Optionally expands distractors from four choices to ten.
6. Optionally measures document coverage.
7. Checks distractor validity and semantic outliers.
8. Applies hallucination and easiness filters.
9. Exports the final MCQ benchmark parquet.

Run generation through the installed CLI:

```bash
nemotron byob --family mcq --stage prepare --config src/nemotron/steps/byob/config/default.yaml
nemotron byob --family mcq --stage generate --config src/nemotron/steps/byob/config/default.yaml
```

When editing semantic deduplication, keep the Curator imports aligned with the runtime:
`nemo_curator.backends.ray_data.RayDataExecutor`,
`nemo_curator.backends.ray_actor_pool.RayActorPoolExecutor`, and
`nemo_curator.stages.deduplication.semantic.SemanticDeduplicationWorkflow`.

## Translation

The translation run expects an MCQ benchmark parquet with stable question IDs and option lists. BYOB flattens
questions and choices into text rows, Curator experimental translation translates those rows, BYOB reassembles
the MCQ schema, Curator computes configured backtranslation quality metrics, and BYOB exports the final
translated benchmark.

Run translation through the same CLI with `--stage translate` and a translation config. Keep
Curator settings under `translation_model_config`; BYOB does not maintain a separate translation engine or mode selector.

## Extending To Another Family

Read [new-family-checklist.md](new-family-checklist.md) first. Answer the schema, source-data,
quality-gate, and validation questions before editing code. Then add a new package under
`runtime/benchmark_families/<family>/`, implement the family-specific schema, prompts, postprocessing,
export code, and stage orchestration, then register a `BenchmarkFamilySpec` in
`runtime/benchmark_families/registry.py`. Keep CLI parsing and stage dispatch unchanged.
