---
id: nemotron.steps.byob
version: 0.3
owner: nemotron
summary: Generate BYOB MCQ or function-calling benchmarks from family-specific inputs.
entrypoint:
  kind: cli
  command: nemotron steps run byob/<family>
  module: nemotron.steps.byob.scripts.run
consumes:
  - type: benchmark_source_corpus
  - type: benchmark_parquet
    required: false
  - type: oracle_pack
    required: false
produces:
  - type: mcq_benchmark_parquet
  - type: translated_mcq_benchmark_parquet
    required: false
  - type: bfcl_benchmark_parquet
    required: false
  - type: bfcl_run_manifest
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
  dependency_extra: byob
  python: ">=3.11 for BYOB Curator runtime dependencies"
  optional_services:
    - OpenAI-compatible LLM endpoint
    - embedding model downloads
    - Curator semantic deduplication backend
    - Curator experimental translation backend
source:
  - repo: Nemotron
    path: src/nemotron/steps/byob/scripts/run.py
  - repo: Nemotron
    path: src/nemotron/steps/byob/scripts/runtime.py
  - repo: Nemotron
    path: src/nemotron/steps/byob/runtime/benchmark_families/mcq/pipeline.py
  - repo: Nemotron
    path: src/nemotron/steps/byob/runtime/benchmark_families/bfcl/pipeline.py
---

# BYOB Step Contract

BYOB produces benchmark parquet artifacts through the registered `mcq` and `bfcl`
families. MCQ consumes domain documents; BFCL consumes executable oracle packs.
New families should add isolated modules under `runtime/benchmark_families/`.
The generic dispatcher is `scripts/runtime.py`; each family owns stage
orchestration in its own `pipeline.py`.

Install BYOB runtime dependencies explicitly with `uv sync --extra byob` or
`pip install ".[byob]"`. The base Nemotron install does not pull Data Designer,
Curator, RAPIDS semantic deduplication packages, embedding models, or translation
metric packages unless the `byob` extra is selected.

The MCQ generation path writes stage cache parquet files under `output_dir/expt_name/stage_cache/`, then
writes `benchmark_raw.parquet` and filtered `benchmark.parquet`.

Semantic deduplication computes embeddings with Curator, then runs
`nemo_curator.backends.ray_actor_pool.RayActorPoolExecutor` for KMeans,
`nemo_curator.backends.ray_data.RayDataExecutor` for embedding and pairwise stages,
and `nemo_curator.stages.deduplication.semantic.SemanticDeduplicationWorkflow`
for orchestration.

The translation path reads an existing benchmark parquet, flattens MCQ text for Curator experimental
translation, writes translation and quality cache files, then writes translated `benchmark_raw.parquet`
and `benchmark.parquet`.
