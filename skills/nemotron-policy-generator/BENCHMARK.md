# Evaluation Report

Evaluation of the `nemotron-policy-generator` skill before publication through NVSkills-Eval.

This benchmark summarizes 3-Tier Evaluation from NVSkills-Eval results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `nemotron-policy-generator`
- Evaluation date: 2026-06-03
- NVSkills-Eval profile: `external`
- Environment: `local`
- Dataset: 11 evaluation tasks
- Attempts per task: 2
- Pass threshold: 50%
- Overall verdict: FAIL

## Agents Used

- `claude-code`
- `codex`

## Metrics Used

Reported benchmark dimensions:

- Security: checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access.
- Correctness: checks whether the agent follows the expected workflow and produces the correct final output.
- Discoverability: checks whether the agent loads the skill when relevant and avoids using it when irrelevant.
- Effectiveness: checks whether the agent performs measurably better with the skill than without it.
- Efficiency: checks whether the agent uses fewer tokens and avoids redundant work.

Underlying evaluation signals used in this run:

- `security` (Security): checks for unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): verifies that the agent loaded the expected skill and workflow.
- `skill_efficiency` (Efficiency): checks routing quality, decoy avoidance, and redundant tool usage.
- `accuracy` (Accuracy): grades final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): checks whether the overall user task completed successfully.
- `behavior_check` (Behavior Check): verifies expected behavior steps, including safety expectations.
- `token_efficiency` (Token Efficiency): compares token usage with and without the skill.

## Test Tasks

The benchmark dataset contained 11 evaluation tasks:

- Positive tasks: 6 tasks where the skill was expected to activate.
- Negative tasks: 5 tasks where no skill was expected.
- Unlabeled tasks: 0 tasks where positive/negative intent could not be inferred.

Task composition is derived from the evaluation dataset when possible. Entries with `expected_skill` set are treated as positive skill-activation cases, while entries with `expected_skill: null` are treated as negative activation cases.

## Results

| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 8 | 95% (+20%) | 98% (+16%) |
| Correctness | 8 | 89% (-0%) | 79% (+10%) |
| Discoverability | 8 | 90% (+11%) | 77% (+2%) |
| Effectiveness | 8 | 82% (+1%) | 68% (+14%) |
| Efficiency | 8 | 75% (+14%) | 68% (+4%) |

Score values show skill-assisted performance. Values in parentheses show uplift versus the no-skill baseline when baseline data is available.

## Tier 1: Static Validation Summary

Tier 1 validation passed with observations. NVSkills-Eval ran 9 checks and found 8 total findings.

Top findings:

- MEDIUM QUALITY/quality_discoverability: Description uses first/second person (`skills/nemotron-policy-generator/SKILL.md`)
- LOW QUALITY/quality_discoverability: Description very long (352 chars, recommend 50-150) (`skills/nemotron-policy-generator/SKILL.md`)
- LOW QUALITY/quality_discoverability: No '## Purpose' section (`skills/nemotron-policy-generator/SKILL.md`)
- LOW QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/nemotron-policy-generator/SKILL.md`)
- LOW QUALITY/quality_reliability: No limitations documented (`skills/nemotron-policy-generator/SKILL.md`)

## Tier 2: Deduplication Summary

Tier 2 validation reported findings. NVSkills-Eval ran 2 checks and found 1 total findings.

Top findings:

- HIGH DUPLICATE/duplicate: Duplicate content found across references/content_safety_taxonomy.md and references/target_models.md:
  "## How severity maps to model output" in references/content_safety_taxonomy.md (lines 162-169)
  vs "## Severity (runtime concept, not model output)" in references/target_models.md (lines 51-53) (`references/content_safety_taxonomy.md:162`)

## Publication Recommendation

The skill should be reviewed before NVSkills-Eval publication. Skill owners should address the findings above and rerun NVSkills-Eval to refresh this benchmark.
