# EVAL.md — nemotron-policy-generator

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

How to run the eval set for this skill, what it measures, and how to interpret the result.

## Quick run

```bash
# From the repo root
nv-aces run \
  --skill skills/nemotron-policy-generator \
  --dataset skills/nemotron-policy-generator/evals/evals.json \
  --harness claude-code \
  --model sonnet \
  --baseline without-skill \
  --output BENCHMARK.md
```

Run it twice, once for each agent harness the catalog supports (Claude Code and Codex), per the publishing guide:

```bash
nv-aces run --skill skills/nemotron-policy-generator \
  --dataset skills/nemotron-policy-generator/evals/evals.json \
  --harness codex --model gpt-5 --baseline without-skill --output BENCHMARK.md
```

`nv-aces` ships five evaluators out of the box: `skill_execution`, `skill_efficiency`, `accuracy`, `goal_accuracy`, `behavior_check`. NV-BASE rolls these up into the five human-readable NVIDIA evaluation dimensions: Security, Correctness, Discoverability, Effectiveness, Efficiency.

## What this eval set measures

The `cases` array in `evals.json` mixes positive cases (where the agent should trigger this skill and produce a policy artifact) with negative cases (where it should stay silent). The split exists because trigger accuracy under distractor load is the hard problem — selection accuracy degrades sharply when many skills are installed, per Liu et al. (arXiv 2604.04323) cited in the publishing guide.

Positive cases exercise:

- **pos-001** — minimal "keywords only" input → clean V2 map → text policy for Reasoning-4B. The most common shape.
- **pos-002** — multimodal + multilingual BYO with custom categories → exercises Nemotron-3 emit block and modality_notes population.
- **pos-003** — extend an existing policy → exercises version bump + diff summary behavior.
- **pos-004** — labeling rubric → exercises the "primary use case" branch where binary severity is appropriate.

Negative cases exercise the explicit "Do not activate" boundary stated in SKILL.md:

- **neg-001** — policy evaluation (review, not generation)
- **neg-002** — legal advice (out of scope by design)
- **neg-003** — benchmark / test (separate skill)
- **neg-004** — wholly unrelated LLM task (distractor)

## Acceptance bar

- **skill_execution** ≥ 0.95 on positive cases (agent reads the right SKILL.md and follows the workflow order).
- **behavior_check** ≥ 0.85 average across the expected_behavior steps. Below 0.85 indicates the workflow steps in SKILL.md need to be tightened.
- **accuracy** ≥ 0.85 against ground_truth on positive cases (LLM-judge rubric).
- **trigger precision** = 1.0 on negative cases (zero false activations). False positives on the negative set are a release blocker — they pollute the catalog's trigger-accuracy baseline.

## When to update this dataset

- Whenever a Nemotron content-safety model ships a new capability that changes how the skill should emit (e.g., when Nemotron-3's `/think` mode lands at Computex 2026, add a positive case exercising the combined `/think + /categories` emit block).
- Whenever a new sibling skill in the catalog creates a confusion boundary — add a distractor case that uses keywords from the sibling.
- Whenever a real customer interaction surfaces a misfire — capture the prompt as a new case so the same misfire doesn't ship again.

## Related

- `BENCHMARK.md` — the report produced by running this eval set.
- `evals.json` — the dataset.
- `SKILL.md` — the skill being evaluated.
