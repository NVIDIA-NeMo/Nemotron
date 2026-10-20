## Description: <br>
Use for Nemotron embed/rerank recipes, including Nemotron 3.5 multimodal SDG, tuning, evaluation, and deployment. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and ML engineers use this skill to plan, run, debug, tune, evaluate, export, and deploy public Nemotron embedding and reranking recipes, including Nemotron 3.5 multimodal profiles. Planning uses the bundled references without a checkout or network access. Recipe execution requires a prepared Nemotron source checkout. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** Only for stages that use protected services; planning requires none. <br>
**Credential Type(s):** API key for hosted generation or authenticated model/container access, as required by the selected workflow. <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Embedding Recipe Reference](references/embed.md) <br>
- [Rerank Recipe Reference](references/rerank.md) <br>
- [Evaluation Reference](references/evaluation.md) <br>
- [Remote Execution Reference](references/remote.md) <br>
- [Multimodal Data Reference](references/multimodal-data.md) <br>
- [NVIDIA Nemotron Documentation Snapshot](https://github.com/NVIDIA-NeMo/Nemotron/blob/e68db1595a9f078a3fb48f8216fa0481d1c862a2/docs/index.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Separates preview (dry-run) commands from execution commands and includes compact run reports for dry-runs and real runs.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
12 evaluation tasks (10 positive, 2 negative), one attempt per task, each run in an isolated sandbox pod, with and without the skill. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? <br>
- Correctness: Is the answer correct? <br>
- Discoverability: Was the right skill loaded when needed? <br>
- Effectiveness: Did the skill help complete the task? (equal-weight goal completion and expected workflow adherence) <br>
- Efficiency: Did it avoid wasted tool calls and token usage? <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
These results describe version 0.3.0. Live agent evaluation has not been rerun for the portability changes in version 0.4.0. <br>

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 91.9% (uplift unavailable) | 94.8% (uplift unavailable) |
| Security | 100.0% → 100.0% (±0.0 points) | 91.7% → 100.0% (+8.3 points) |
| Correctness | 63.3% → 100.0% (+36.7 points) | 71.7% → 96.7% (+25.0 points) |
| Discoverability | 99.8% (uplift unavailable) | 93.5% (uplift unavailable) |
| Effectiveness | 61.8% → 91.0% (+29.2 points) | 63.5% → 88.7% (+25.2 points) |
| Efficiency | 68.9% (uplift unavailable) | 95.0% (uplift unavailable) |

## Skill Version(s): <br>
0.4.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
