# Evaluation Report

Evaluation of the `retriever-finetune-recipe` skill, which guides agents through Nemotron embedding and reranking fine-tuning recipe work.

## Conformance Status

This report follows the functional skill evaluation approach: define realistic task cases, run the agent harness with and without the skill, compare aggregate metrics, and publish only reusable fixtures plus the summary report. Generated run artifacts stay out of source control.

Current publication status: Codex evaluation is recorded; Claude Code evaluation is still pending and should be completed, or an explicit exception documented, before final publication.

## Agents Used

| Agent harness | Model | Status |
| --- | --- | --- |
| Codex | Configured evaluation model | Evaluated |
| Claude Code | Configured evaluation model | Required before publication; pending explicit live-run approval |

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

The dataset contains 8 realistic task cases in `evals/evals.json`:

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

## Results

| Metric | Num | Codex | Claude Code |
| --- | ---: | --- | --- |
| Static quality score | 1 skill | 100/100 | Not agent-specific |
| Command freshness | 8 commands | Passed in current checkout | Not agent-specific |
| Live overall score (original set) | 4 tasks | 0.90 with skill vs 0.66 without skill (+0.24) | Pending |
| Security (original set) | 4 tasks | 1.00 | Pending |

## Experiential Design Iteration

A follow-up design iteration used three Codex-native trace-bundle trials: one baseline rerank-selection task without the skill, one with-skill embedding first-pass planning task, and one with-skill rerank NIM deployment-debug task. MCP-backed trajectory review was not available, so the review used saved trace bundles instead. Two trials produced partial trajectory evidence because their subagent command runner could not spawn shell processes; the deployment-debug trial produced complete trace evidence with source/config inspection, CLI help, and dry-runs. The reusable deltas from those traces were folded into the skill references and pitfalls.

## Notes

The eval setup compares with-skill and without-skill performance, keeps generated `evals/results/` output out of source control, and uses task prompts that do not explicitly name the skill. The four newer cases expand adversarial coverage; rerun live evaluation before replacing the original-set aggregate scores. The committed `evals/evals.json` file is the reusable test dataset; aggregate results are summarized here rather than committing full run directories or raw provider traces.

The Claude Code run is not recorded here yet because live evaluation sends workspace skill/eval content to configured model providers and requires explicit approval in this environment.
