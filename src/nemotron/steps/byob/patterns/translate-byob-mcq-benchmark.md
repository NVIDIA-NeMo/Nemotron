---
id: translate-byob-mcq-benchmark
title: Translate an existing BYOB MCQ benchmark
tags: [byob, benchmark, translation]
triggers:
  - Translate a generated MCQ benchmark while keeping the benchmark schema.
  - Add backtranslation metrics to a BYOB benchmark.
steps: [byob]
confidence: high
---

Use the MCQ family `translate` stage with `config/translate.yaml` as the template. Curator experimental
translation owns text translation and round-trip metrics; BYOB owns MCQ flattening, reassembly, and final
parquet schema. Keep this flow in `runtime/benchmark_families/mcq/pipeline.py`.
