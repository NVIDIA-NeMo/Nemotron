# Evaluation Report

Evaluation of the `retriever-finetune-recipe` skill, which guides agents through Nemotron embedding and reranking fine-tuning recipe work.

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

The dataset contains 4 task-shaped cases in `evals/evals.json`:

| Task | Type |
| --- | --- |
| `retriever-finetune-recipe-embed-plan-001` | Positive: embedding recipe planning |
| `retriever-finetune-recipe-rerank-choice-001` | Positive: embedder vs reranker choice |
| `retriever-finetune-recipe-deploy-debug-001` | Positive: reranker NIM deployment debugging |
| `retriever-finetune-recipe-negative-001` | Negative: unrelated factual question |

## Results

| Metric | Num | Codex | Claude Code |
| --- | ---: | --- | --- |
| Static quality score | 1 skill | 100/100 | Not agent-specific |
| Live overall score | 4 tasks | 0.90 with skill vs 0.66 without skill (+0.24) | Pending |
| Security | 4 tasks | 1.00 | Pending |

## Notes

The eval setup compares with-skill and without-skill performance, keeps generated `evals/results/` output out of source control, and uses task prompts that do not explicitly name the skill.

The Claude Code run is not recorded here yet because live evaluation sends workspace skill/eval content to configured model providers and requires explicit approval in this environment.
