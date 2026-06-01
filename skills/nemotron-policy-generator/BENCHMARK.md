# BENCHMARK.md — nemotron-policy-generator

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

Evaluation report for `nemotron-policy-generator` v1.0.0. Per the NVIDIA Skills Publishing Onboarding Guide, every published skill ships a BENCHMARK.md showing the with-skill vs without-skill comparison so the uplift is visible.

> **Status: placeholder.** Tables below are scaffolding. Populate via `nv-aces run … --output BENCHMARK.md` (see `evals/EVAL.md`) or fill in manually after team-owned evaluation. Either way, this file lands in the skill directory and surfaces in the catalog.

## Harness configuration

| Field | Value |
|---|---|
| Skill version | 1.0.0 |
| Eval dataset | `evals/evals.json` (8 cases: 4 positive, 4 negative) |
| Agent harness A | Claude Code — `claude-sonnet-4-5` (TBD) |
| Agent harness B | Codex — `gpt-5` (TBD) |
| Evaluator framework | NV-ACES (Skill Execution, Skill Efficiency, Accuracy, Goal Accuracy, Behavior Check) |
| Date evaluated | _TBD_ |
| Evaluator(s) | Shyamala Prayaga \<sprayaga@nvidia.com\> |

## Results — Claude Code (sonnet)

### With skill installed

| Case ID | Skill Execution | Skill Efficiency | Accuracy | Goal Accuracy | Behavior Check | Tokens | Wall-clock (s) |
|---|---|---|---|---|---|---|---|
| pos-001-rough-keywords | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| pos-002-multimodal-byo | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| pos-003-extend-existing | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| pos-004-eval-rubric | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| neg-001-eval-existing | n/a (silent expected) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| neg-002-legal-advice | n/a (silent expected) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| neg-003-benchmark-test | n/a (silent expected) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| neg-004-unrelated-llm | n/a (silent expected) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **Aggregate** | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### Without skill (baseline)

| Case ID | Accuracy | Goal Accuracy | Behavior Check | Tokens | Wall-clock (s) |
|---|---|---|---|---|---|
| pos-001-rough-keywords | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| pos-002-multimodal-byo | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| pos-003-extend-existing | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| pos-004-eval-rubric | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **Aggregate** | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### Uplift (with-skill minus without-skill)

| Metric | Δ |
|---|---|
| Accuracy | _TBD_ |
| Goal Accuracy | _TBD_ |
| Behavior Check | _TBD_ |
| Tokens (lower is better) | _TBD_ |
| Wall-clock (lower is better) | _TBD_ |

## Results — Codex (gpt-5)

Same table shape as above, populated when Codex eval run completes.

## Five-dimension rollup (NV-BASE)

| Dimension | Score | Notes |
|---|---|---|
| Security | _TBD_ | NV-BASE scan result; secrets, PII, prompt injection, spec compliance |
| Correctness | _TBD_ | accuracy + goal_accuracy weighted |
| Discoverability | _TBD_ | trigger precision on negative cases; trigger recall on positive cases |
| Effectiveness | _TBD_ | uplift vs without-skill baseline |
| Efficiency | _TBD_ | token + wall-clock vs baseline |

## Known limitations and follow-ups

- Today's dataset focuses on the four most common policy-generation shapes. Multi-skill composition (this skill + a downstream eval skill) is out of scope until JFP's stack-level evaluation framework lands.
- BENCHMARK.md does not detect SKILL.md drift against the Nemotron Content Safety V2 taxonomy. Per the publishing guide, drift is team-owned: refresh this skill (and re-run evals) on every Nemotron-Content-Safety / Nemotron-3-Content-Safety model release.
- Trigger evals on `neg-004-unrelated-llm` should be re-run whenever a new Nemotron-adjacent skill ships to the catalog, in case the distractor distribution shifts.
