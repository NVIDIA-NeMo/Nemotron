## Description: <br>
Generates BYO custom safety policies for NVIDIA Nemotron content-safety guardrails — Nemotron-Content-Safety-Reasoning-4B (text) and multimodal Nemotron-3-Content-Safety. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers and safety engineers generating custom content-safety policies, JSON taxonomies, and inference prompts for NVIDIA Nemotron guardrail models. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Content Safety Taxonomy (V2)](references/content_safety_taxonomy.md) <br>
- [Policy Patterns](references/policy_patterns.md) <br>
- [Target Models](references/target_models.md) <br>
- [Nemotron Developer Repository](https://nvidia-nemo.github.io/Nemotron/dev/) <br>


## Skill Output: <br>
**Output Type(s):** [Files, Configuration instructions] <br>
**Output Format:** [Markdown policy, JSON taxonomy, and plain-text system prompts] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Outputs validated against assets/policy_json_schema.json] <br>

## Evaluation Agents Used: <br>
- Claude Code (`claude-code`) <br>
- Codex (`codex`) <br>



## Evaluation Tasks: <br>
Evaluated against 11 NVSkills-Eval tasks (6 positive, 5 negative) with 2 attempts per task and a 50% pass threshold. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies that the agent loaded the expected skill and workflow. <br>
- `skill_efficiency`: Checks routing quality, decoy avoidance, and redundant tool usage. <br>
- `accuracy`: Grades final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the overall user task completed successfully. <br>
- `behavior_check`: Verifies expected behavior steps, including safety expectations. <br>
- `token_efficiency`: Compares token usage with and without the skill. <br>



## Evaluation Results: <br>
| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 8 | 95% (+20%) | 98% (+16%) |
| Correctness | 8 | 89% (-0%) | 79% (+10%) |
| Discoverability | 8 | 90% (+11%) | 77% (+2%) |
| Effectiveness | 8 | 82% (+1%) | 68% (+14%) |
| Efficiency | 8 | 75% (+14%) | 68% (+4%) |

## Testing Completed: <br>
**[x] Agent Red-Teaming** <br>
**[ ] Network Security** <br>
**[ ] Product Security** <br>

## Skill Version(s): <br>
0.1.0 (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
