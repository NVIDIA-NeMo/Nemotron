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
- Dataset digest: `sha256:03b2236004cabe1fdd06ed4343ecd2a7bd420a21dee7765bace9daae9c486b1c` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 91.9% — baseline ran, but no comparable score was available; uplift unavailable | 94.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 91.7% → 100.0% (+8.3 points) |
| Correctness | 63.3% → 100.0% (+36.7 points) | 71.7% → 96.7% (+25.0 points) |
| Discoverability | 99.8% — baseline ran, but no comparable score was available; uplift unavailable | 93.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 61.8% → 91.0% (+29.2 points) | 63.5% → 88.7% (+25.2 points) |
| Efficiency | 68.9% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,225,550 | 1,826,088 | N/A | N/A | skill 7/12; base 7/12 |
| claude-code | nemotron-retrieval-recipes-export-boundary-001 | 231,192 | 91,888 | +139,304 | +151.60% | skill 1/1; base 1/1 |
| claude-code | nemotron-retrieval-recipes-llama-profile-export-001 | 104,774 | 33,836 | +70,938 | +209.65% | skill 1/1; base 1/1 |
| claude-code | nemotron-retrieval-recipes-multimodal-input-plan-001 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | nemotron-retrieval-recipes-multimodal-rerank-handoff-001 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | nemotron-retrieval-recipes-multimodal-resume-001 | N/A | 828,996 | N/A | N/A | skill 0/1; base 1/1 |
| claude-code | nemotron-retrieval-recipes-multimodal-serving-drift-001 | 1,129,205 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| claude-code | nemotron-retrieval-recipes-negative-nonrecipe-nim-001 | 89,827 | 183,427 | -93,600 | -51.03% | skill 1/1; base 1/1 |
| claude-code | nemotron-retrieval-recipes-negative-vector-db-001 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | nemotron-retrieval-recipes-nemotron3-deploy-001 | 104,964 | 150,638 | -45,674 | -30.32% | skill 1/1; base 1/1 |
| claude-code | nemotron-retrieval-recipes-remote-batch-001 | 187,567 | 185,345 | +2,222 | +1.20% | skill 1/1; base 1/1 |
| claude-code | nemotron-retrieval-recipes-rerank-choice-001 | 378,021 | 351,958 | +26,063 | +7.41% | skill 1/1; base 1/1 |
| claude-code | nemotron-retrieval-recipes-rerank-sdg-governance-001 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | All cases | 970,529 | 953,005 | N/A | N/A | skill 7/12; base 7/12 |
| codex | nemotron-retrieval-recipes-export-boundary-001 | 92,610 | 98,559 | -5,949 | -6.04% | skill 1/1; base 1/1 |
| codex | nemotron-retrieval-recipes-llama-profile-export-001 | 60,070 | 73,702 | -13,632 | -18.50% | skill 1/1; base 1/1 |
| codex | nemotron-retrieval-recipes-multimodal-input-plan-001 | N/A | 284,730 | N/A | N/A | skill 0/1; base 1/1 |
| codex | nemotron-retrieval-recipes-multimodal-rerank-handoff-001 | 369,374 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| codex | nemotron-retrieval-recipes-multimodal-resume-001 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | nemotron-retrieval-recipes-multimodal-serving-drift-001 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | nemotron-retrieval-recipes-negative-nonrecipe-nim-001 | 73,734 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| codex | nemotron-retrieval-recipes-negative-vector-db-001 | N/A | 137,319 | N/A | N/A | skill 0/1; base 1/1 |
| codex | nemotron-retrieval-recipes-nemotron3-deploy-001 | 103,188 | 66,104 | +37,084 | +56.10% | skill 1/1; base 1/1 |
| codex | nemotron-retrieval-recipes-remote-batch-001 | 89,886 | 99,072 | -9,186 | -9.27% | skill 1/1; base 1/1 |
| codex | nemotron-retrieval-recipes-rerank-choice-001 | 181,667 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| codex | nemotron-retrieval-recipes-rerank-sdg-governance-001 | N/A | 193,519 | N/A | N/A | skill 0/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 3,196,079 | 2,779,093 | N/A | N/A | skill 14/24; base 14/24 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 6 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 12 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'author' is ignored; use 'metadata.author' (`skills/nemotron-retrieval-recipes/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'tags' is ignored; use 'metadata.tags' (`skills/nemotron-retrieval-recipes/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/nemotron-retrieval-recipes/SKILL.md`)
- **MEDIUM** SECURITY/External Transmission (E1): Data Exfiltration: curl -X POST http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input": ["hello"], "model": "nvidia/nemotron-3-embed-1b", "input_type": "query"}'

# With -c llama, u (`references/embed.md:286`)
- **MEDIUM** SECURITY/Autonomous Decision Making (EA2): Excessive Agency: never ask user (`references/remote.md:9`)
- 1 additional finding(s) are available in the full evaluation artifacts.

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
