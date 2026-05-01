---
id: nemotron.steps.byob
version: 0.1
owner: nemotron
summary: Generate and translate bring-your-own MCQ benchmarks from domain documents.
entrypoint:
  kind: cli
  command: nemotron byob
  module: nemotron.steps.byob.scripts.run
consumes:
  - type: benchmark_source_corpus
  - type: benchmark_parquet
    required: false
produces:
  - type: mcq_benchmark_parquet
  - type: translated_mcq_benchmark_parquet
    required: false
parameters:
  required:
    - family
    - stage
    - config
  important:
    - skip_until
    - target_source_mapping
    - filtering_model_configs
compute:
  shape: single-node plus model endpoints
  optional_services:
    - OpenAI-compatible LLM endpoint
    - embedding model downloads
    - Curator experimental translation backend
source:
  - repo: Nemotron
    path: src/nemotron/steps/byob/scripts/run.py
  - repo: Nemotron
    path: src/nemotron/steps/byob/runtime/benchmark_families/mcq/
---

# BYOB Step Contract

BYOB produces benchmark parquet artifacts from user-provided domain documents. The current registered
family is `mcq`; new families should add isolated modules under `runtime/benchmark_families/`.

The MCQ generation path writes stage cache parquet files under `output_dir/expt_name/stage_cache/`, then
writes `benchmark_raw.parquet` and filtered `benchmark.parquet`.

The translation path reads an existing benchmark parquet, flattens MCQ text for Curator experimental
translation, writes translation and quality cache files, then writes translated `benchmark_raw.parquet`
and `benchmark.parquet`.
