# Benchmark Report

Evaluation plan and current validation status for the `nemotron-retrieval-recipes` skill, which guides agents through Nemotron `embed` and `rerank` retrieval recipe work.

## Status

This is a new skill. The committed benchmark state is intentionally limited to the current 14-case dataset and local validation checks. Final with-skill versus without-skill live results have not been published yet for this skill.

Because the underlying recipe stages can run for hours, recipe wall-clock completion is not the primary benchmark signal. The benchmark should measure whether the skill improves agent behavior: routing to the right recipe family, grounding recommendations in the current checkout, using safe dry-runs before expensive stages, preserving secrets, interpreting metrics correctly, and handing off long-running execution clearly.

## Evaluation Targets

| Agent harness | Status |
| --- | --- |
| Codex | 14-case with/without run pending |
| Claude Code | 14-case with/without run pending |

## Metrics

Use the configured live-eval metrics for paired runs:

- `security`
- `skill_execution`
- `skill_efficiency`
- `accuracy`
- `goal_accuracy`
- `behavior_check`

Publication reporting should also include task completion rate plus agent-harness wall-clock time and token consumption when available. Those values measure evaluation cost, not expected recipe training runtime.

## Test Tasks

The dataset contains 14 realistic task cases in `evals/evals.json`:

| Task | Type |
| --- | --- |
| `nemotron-retrieval-recipes-embed-plan-001` | Positive: embedding recipe planning |
| `nemotron-retrieval-recipes-rerank-choice-001` | Positive: embedder vs reranker choice |
| `nemotron-retrieval-recipes-deploy-debug-001` | Positive: reranker NIM deployment debugging |
| `nemotron-retrieval-recipes-negative-001` | Negative: unrelated factual question |
| `nemotron-retrieval-recipes-negative-vector-db-001` | Negative: generic vector database advice |
| `nemotron-retrieval-recipes-secret-handling-001` | Positive: secret-safe Stage 0 planning |
| `nemotron-retrieval-recipes-stale-artifacts-001` | Positive: stale artifact diagnosis |
| `nemotron-retrieval-recipes-prereq-gap-001` | Positive: prerequisite gate before GPU work |
| `nemotron-retrieval-recipes-remote-batch-001` | Positive: remote batch planning |
| `nemotron-retrieval-recipes-metrics-nuance-001` | Positive: nuanced metrics interpretation |
| `nemotron-retrieval-recipes-stage-readiness-001` | Positive: stage readiness from raw docs |
| `nemotron-retrieval-recipes-export-boundary-001` | Positive: TensorRT export boundary debugging |
| `nemotron-retrieval-recipes-long-running-boundary-001` | Positive: long-running execution handoff |
| `nemotron-retrieval-recipes-docs-integration-001` | Positive: docs-to-checkout reconciliation |

## Current Checks

| Check | Result |
| --- | --- |
| Eval dataset structure | Passed for 14 cases |
| Static skill-quality check | Passed, 100/100 |
| Command freshness checklist | Passed in the current checkout |

Command freshness covered read-only help and dry-run checks for the documented `embed` and `rerank` recipe flows. Generated run artifacts should stay out of source control.

## Publication Gate

Before publication, run paired with-skill and without-skill evaluation on the 14-case dataset for both target agent harnesses, then update this report with aggregate scores, uplift, task completion rate, agent-harness wall-clock time, token consumption, and any skipped-agent limitations.
