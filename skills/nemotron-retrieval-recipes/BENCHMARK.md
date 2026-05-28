# Benchmark Report

Evaluation of the `nemotron-retrieval-recipes` skill, which guides agents through public Nemotron embedding and reranking retrieval recipe work.

## Conformance Status

This report follows the functional skill evaluation approach: define realistic task cases, run the agent harness with and without the skill, compare aggregate metrics, and publish only reusable fixtures plus the summary report. Generated run artifacts stay out of source control.

Current publication status: Codex evaluation is recorded for the original task set; Claude Code with-skill live evaluation is recorded for the 12-case expanded set. Full Claude Code lift analysis is still pending because the baseline leg hit evaluation harness failures. The dataset now contains 14 cases; the two newest cases cover long-running job boundaries and docs-to-checkout reconciliation and still need live with/without runs.

Because the underlying recipe stages can run for hours, recipe wall-clock completion is not the primary benchmark signal. Report wall-clock time and token consumption for the agent harness as publication metadata, but judge lift mainly by whether the skill improves routing, repo grounding, safety around expensive execution, metric interpretation, and concrete run handoff.

## Agents Used

| Agent harness | Model | Status |
| --- | --- | --- |
| Codex | Configured evaluation model | Evaluated on original 4-case set |
| Claude Code | Configured evaluation model | With-skill live run recorded for 12 cases; baseline/lift run invalid due harness failures |

## Metrics Used

Default live-eval metrics:

- `security`
- `skill_execution`
- `skill_efficiency`
- `accuracy`
- `goal_accuracy`
- `behavior_check`

Publication reporting should also include task completion rate plus agent-harness wall-clock time and token consumption when available. Static validation also uses deterministic skill-quality checks.

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

## Results

| Metric | Num | Codex | Claude Code |
| --- | ---: | --- | --- |
| Eval dataset validation | 14 tasks | Passed local structure validation | Not agent-specific |
| Static quality score | 1 skill | 100/100 after rename/layout change | Not agent-specific |
| Command freshness checklist | 6 representative commands | Passed manually after rename/layout change in current checkout | Not agent-specific |
| Live overall score (original set) | 4 tasks | 0.90 with skill vs 0.66 without skill (+0.24) | Baseline/lift pending |
| Live with-skill score (expanded set) | 12 tasks | Not rerun | 0.86 |
| Task completion | 12-task Claude run | Not rerun | 12/12 scored attempts with skill |
| Agent wall-clock and token cost | 14-task target | Pending next full run | Pending next full run |
| Security | 12 tasks | 1.00 on original set | 1.00 |
| Skill execution | 12 tasks | Not rerun | 0.90 |
| Efficiency | 12 tasks | Not rerun | 0.77 |
| Accuracy | 12 tasks | Not rerun | 0.95 |
| Goal accuracy | 12 tasks | Not rerun | 0.85 |
| Behavior check | 12 tasks | Not rerun | 0.69 |

## Experiential Design Iteration

A follow-up design iteration used three Codex-native trace-bundle trials: one baseline rerank-selection task without the skill, one with-skill embedding first-pass planning task, and one with-skill rerank NIM deployment-debug task. Automated trajectory review was not available, so the review used saved trace bundles instead. Two trials produced partial trajectory evidence because their subagent command runner could not spawn shell processes; the deployment-debug trial produced complete trace evidence with source/config inspection, CLI help, and dry-runs. The reusable deltas from those traces were folded into the skill references and pitfalls.

## Notes

The eval setup compares with-skill and without-skill performance, keeps generated `evals/results/` output out of source control, and uses task prompts that do not explicitly name the skill. The committed `evals/evals.json` file is the reusable test dataset; aggregate results are summarized here rather than committing full run directories or raw provider traces.

The Claude Code with-skill live run completed 12/12 scored attempts after explicit approval for live evaluation. Full Claude Code with/without lift was attempted, but the baseline leg was invalid because two baseline attempts failed in the evaluation harness (`AgentTimeoutError` and a tool-schema API error). Do not use the partial lift scores for pass/fail or model comparison.

Before publication, rerun live with/without evaluation on the 14-case dataset for both Codex and Claude Code, then update the result table with task completion, agent wall-clock time, and token consumption. Local structure validation, static evaluation, and command freshness checks passed after the rename/layout change.
