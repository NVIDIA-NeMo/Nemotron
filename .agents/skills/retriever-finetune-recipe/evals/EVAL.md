# Evaluation Guidance

This skill follows a functional skill evaluation approach: define realistic tasks, run the chosen agent harness with and without the skill, compare outcomes, and report uplift.

## Evaluation Goals

The evaluation is functional: it checks whether agents use the skill when retrieval recipe expertise is needed, avoid it for unrelated tasks, and produce better task outcomes with the skill than without it.

## Dataset Rules

- Keep prompts realistic. Do not name the skill in user prompts.
- Include positive cases for embedding planning, reranker selection, deployment debugging, stale artifact diagnosis, secret-safe setup, and prerequisite gating.
- Include negative cases where the skill should not activate, including unrelated factual questions and generic vector database advice.
- Keep `expected_skill`, `ground_truth`, and ordered `expected_behavior` entries explicit enough for deterministic and judge-based grading.
- Do not commit generated `evals/results/` output; commit only reusable fixtures and summary reports.

## Required Checks

Before publication, run the configured skill evaluation harness in the configured evaluation environment:

- validate the skill and eval dataset structure,
- run static skill-quality checks,
- run `.agents/skills/retriever-finetune-recipe/scripts/check-command-freshness.sh` when the checkout has the recipe CLI installed,
- run live with-skill and without-skill evaluation,
- cover both Codex and Claude Code, or document why an agent was skipped.

The live evaluation sends skill and eval prompts to the configured model providers. Get explicit approval before running it in an environment where workspace content is sensitive.

## Reporting

After live evaluation, update `BENCHMARK.md` with:

- agent harness and model versions, or public-safe aliases when model route names should not be published,
- metric names,
- test dataset size,
- with-skill score, without-skill score, and uplift,
- limitations or skipped agents.
