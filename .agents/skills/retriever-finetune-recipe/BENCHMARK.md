# Evaluation Report

Evaluation of the `retriever-finetune-recipe` skill, which guides agents through Nemotron embedding and reranking fine-tuning recipe work.

## Conformance Status

This report follows the functional skill evaluation approach: define realistic task cases, run the agent harness with and without the skill, compare aggregate metrics, and publish only reusable fixtures plus the summary report. Generated run artifacts stay out of source control.

Current publication status: Codex evaluation is recorded; Claude Code with-skill live evaluation is recorded. Full Claude Code lift analysis is still pending because the baseline leg hit evaluation harness failures.

## Agents Used

| Agent harness | Model | Status |
| --- | --- | --- |
| Codex | Configured evaluation model | Evaluated |
| Claude Code | Configured evaluation model | With-skill live run recorded; baseline/lift run invalid due harness failures |

## Metrics Used

Default live-eval metrics:

- `security`
- `skill_execution`
- `skill_efficiency`
- `accuracy`
- `goal_accuracy`
- `behavior_check`

Static validation also uses deterministic skill-quality checks.

## Test Tasks

The dataset contains 12 realistic task cases in `evals/evals.json`:

| Task | Type |
| --- | --- |
| `retriever-finetune-recipe-embed-plan-001` | Positive: embedding recipe planning |
| `retriever-finetune-recipe-rerank-choice-001` | Positive: embedder vs reranker choice |
| `retriever-finetune-recipe-deploy-debug-001` | Positive: reranker NIM deployment debugging |
| `retriever-finetune-recipe-negative-001` | Negative: unrelated factual question |
| `retriever-finetune-recipe-negative-vector-db-001` | Negative: generic vector database advice |
| `retriever-finetune-recipe-secret-handling-001` | Positive: secret-safe Stage 0 planning |
| `retriever-finetune-recipe-stale-artifacts-001` | Positive: stale artifact diagnosis |
| `retriever-finetune-recipe-prereq-gap-001` | Positive: prerequisite gate before GPU work |
| `retriever-finetune-recipe-remote-batch-001` | Positive: remote batch planning |
| `retriever-finetune-recipe-metrics-nuance-001` | Positive: nuanced metrics interpretation |
| `retriever-finetune-recipe-stage-readiness-001` | Positive: stage readiness from raw docs |
| `retriever-finetune-recipe-export-boundary-001` | Positive: TensorRT export boundary debugging |

## Results

| Metric | Num | Codex | Claude Code |
| --- | ---: | --- | --- |
| Static quality score | 1 skill | 100/100 | Not agent-specific |
| Command freshness checklist | 6 representative commands | Passed manually in current checkout | Not agent-specific |
| Live overall score (original set) | 4 tasks | 0.90 with skill vs 0.66 without skill (+0.24) | Baseline/lift pending |
| Live with-skill score (expanded set) | 12 tasks | Not rerun | 0.86 |
| Security | 12 tasks | 1.00 on original set | 1.00 |
| Skill execution | 12 tasks | Not rerun | 0.90 |
| Efficiency | 12 tasks | Not rerun | 0.77 |
| Accuracy | 12 tasks | Not rerun | 0.95 |
| Goal accuracy | 12 tasks | Not rerun | 0.85 |
| Behavior check | 12 tasks | Not rerun | 0.69 |

## Experiential Design Iteration

A follow-up design iteration used three Codex-native trace-bundle trials: one baseline rerank-selection task without the skill, one with-skill embedding first-pass planning task, and one with-skill rerank NIM deployment-debug task. MCP-backed trajectory review was not available, so the review used saved trace bundles instead. Two trials produced partial trajectory evidence because their subagent command runner could not spawn shell processes; the deployment-debug trial produced complete trace evidence with source/config inspection, CLI help, and dry-runs. The reusable deltas from those traces were folded into the skill references and pitfalls.

## Notes

The eval setup compares with-skill and without-skill performance, keeps generated `evals/results/` output out of source control, and uses task prompts that do not explicitly name the skill. The eight newer cases expand adversarial coverage; rerun live evaluation before replacing the original-set aggregate scores. The committed `evals/evals.json` file is the reusable test dataset; aggregate results are summarized here rather than committing full run directories or raw provider traces.

The Claude Code with-skill live run completed 12/12 scored attempts after explicit approval for provider-backed evaluation. Full Claude Code with/without lift was attempted, but the baseline leg was invalid because two baseline attempts failed in the evaluation harness (`AgentTimeoutError` and a tool-schema API error). Do not use the partial lift scores for pass/fail or model comparison.
