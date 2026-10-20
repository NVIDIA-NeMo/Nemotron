# Skill Benchmark: nemotron-retrieval-recipes

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemotron-retrieval-recipes`
- Evaluation date: 2026-10-20
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 12 evaluation tasks (10 positive, 2 negative)
- Dataset digest: `sha256:4fb0dda3af2ac86b9c5427dd6ccbee42c8c5b9e8988259e68571be7f1223112d` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 97.5% — baseline ran, but no comparable score was available; uplift unavailable | 94.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 95.8% → 100.0% (+4.2 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 58.3% → 100.0% (+41.7 points) | 80.0% → 96.7% (+16.7 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 93.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 58.5% → 90.7% (+32.2 points) | 65.6% → 85.0% (+19.4 points) |
| Efficiency | 96.7% — baseline ran, but no comparable score was available; uplift unavailable | 96.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,309,378 | 1,291,821 | N/A | N/A | skill 7/12; base 7/12 |
| claude-code | nemotron-retrieval-recipes-export-boundary-001 | 107,570 | 91,583 | +15,987 | +17.46% | skill 1/1; base 1/1 |
| claude-code | nemotron-retrieval-recipes-llama-profile-export-001 | 107,016 | 32,314 | +74,702 | +231.18% | skill 1/1; base 1/1 |
| claude-code | nemotron-retrieval-recipes-multimodal-input-plan-001 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | nemotron-retrieval-recipes-multimodal-rerank-handoff-001 | 441,694 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| claude-code | nemotron-retrieval-recipes-multimodal-resume-001 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | nemotron-retrieval-recipes-multimodal-serving-drift-001 | N/A | 434,426 | N/A | N/A | skill 0/1; base 1/1 |
| claude-code | nemotron-retrieval-recipes-negative-nonrecipe-nim-001 | 182,664 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| claude-code | nemotron-retrieval-recipes-negative-vector-db-001 | N/A | 151,473 | N/A | N/A | skill 0/1; base 1/1 |
| claude-code | nemotron-retrieval-recipes-nemotron3-deploy-001 | 106,748 | 91,211 | +15,537 | +17.03% | skill 1/1; base 1/1 |
| claude-code | nemotron-retrieval-recipes-remote-batch-001 | 107,752 | 243,878 | -136,126 | -55.82% | skill 1/1; base 1/1 |
| claude-code | nemotron-retrieval-recipes-rerank-choice-001 | 255,934 | 246,936 | +8,998 | +3.64% | skill 1/1; base 1/1 |
| claude-code | nemotron-retrieval-recipes-rerank-sdg-governance-001 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | All cases | 918,765 | 694,056 | N/A | N/A | skill 7/12; base 7/12 |
| codex | nemotron-retrieval-recipes-export-boundary-001 | 96,164 | 83,316 | +12,848 | +15.42% | skill 1/1; base 1/1 |
| codex | nemotron-retrieval-recipes-llama-profile-export-001 | 95,004 | 30,812 | +64,192 | +208.33% | skill 1/1; base 1/1 |
| codex | nemotron-retrieval-recipes-multimodal-input-plan-001 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | nemotron-retrieval-recipes-multimodal-rerank-handoff-001 | N/A | 178,062 | N/A | N/A | skill 0/1; base 1/1 |
| codex | nemotron-retrieval-recipes-multimodal-resume-001 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | nemotron-retrieval-recipes-multimodal-serving-drift-001 | 316,558 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| codex | nemotron-retrieval-recipes-negative-nonrecipe-nim-001 | 117,721 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| codex | nemotron-retrieval-recipes-negative-vector-db-001 | N/A | 120,228 | N/A | N/A | skill 0/1; base 1/1 |
| codex | nemotron-retrieval-recipes-nemotron3-deploy-001 | 85,272 | 65,254 | +20,018 | +30.68% | skill 1/1; base 1/1 |
| codex | nemotron-retrieval-recipes-remote-batch-001 | 90,842 | 101,165 | -10,323 | -10.20% | skill 1/1; base 1/1 |
| codex | nemotron-retrieval-recipes-rerank-choice-001 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | nemotron-retrieval-recipes-rerank-sdg-governance-001 | 117,204 | 115,219 | +1,985 | +1.72% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,228,143 | 1,985,877 | N/A | N/A | skill 14/24; base 14/24 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 2 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 12 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SECURITY/External Transmission (E1): Data Exfiltration: curl -X POST http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": ["hello"], "model": "nvidia/nemotron-3-embed-1b", "input_type": "query"}'

# With -c llama, u (`references/embed.md:306`)
- **MEDIUM** SECURITY/External Transmission (E1): Data Exfiltration: curl -X POST http://localhost:8000/v1/ranking \
  -H 'Content-Type: application/json' \
  -d '{"model": "nvidia/llama-nemotron-rerank-1b-v2", "query": {"text": "what is AI?"}, "passages": [{"text": "A (`references/rerank.md:224`)

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
