---
id: create-byob-mcq-from-domain-corpus
title: Create BYOB MCQ benchmark from domain documents
tags: [byob, benchmark, mcq]
triggers:
  - Generate an MCQ benchmark from custom domain text.
  - Build a benchmark from documents grouped by subject.
steps: [byob]
confidence: high
---

Use the MCQ family with `prepare` followed by `generate`. Start from `config/default.yaml`.
MCQ orchestration lives in `runtime/benchmark_families/mcq/pipeline.py`.
If the request touches semantic deduplication, use Curator's `RayDataExecutor`,
`RayActorPoolExecutor`, and package-level `SemanticDeduplicationWorkflow`.
