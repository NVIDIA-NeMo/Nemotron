# Evaluation Guidance

This skill follows a functional skill evaluation approach: define realistic tasks, run the chosen agent harness with and without the skill, compare outcomes, and report uplift.

## Evaluation Goals

The evaluation is functional: it checks whether agents use the skill when public Nemotron retrieval recipe expertise is needed, avoid it for unrelated tasks, and produce better task outcomes with the skill than without it. Because these recipes are long-running, wall-clock recipe completion is not the main success signal. The useful lift is whether the skill helps the agent route to the right recipe family, use bundled guidance for planning, verify the available runtime before execution, choose safe dry-runs before expensive stages, preserve secrets, interpret metrics correctly, and hand off long-running execution with clear run reports.

## Dataset Rules

- Keep prompts realistic. Do not name the skill in user prompts.
- Keep the suite at 12 cases unless a new failure mode justifies the extra live CI cost. Replace or extend an existing case when it covers the same decision.
- Use realistic recipe tasks and nearby negative cases. Each case must protect a distinct behavior in the coverage table below.
- Keep `expected_skill`, `expected_output`, and `assertions` explicit. Grade task outcomes in both the with-skill and no-skill arms; do not require skill invocation for a positive case to earn correctness credit.
- Do not commit generated `evals/results/` output; commit only reusable fixtures and summary reports.

## Case Coverage and Cost

The suite contains 10 positive and two negative cases. With one attempt, two
agents, and both with-skill and no-skill arms, it requires 48 agent trials.
The previous 20-case suite required 80 under the same settings.

Each ID below has the prefix `nemotron-retrieval-recipes-`:

| Case ID | Distinct behavior protected |
| --- | --- |
| `multimodal-input-plan-001` | Plan without a checkout or installed CLI; distinguish directory discovery, paired records, and PDF preprocessing, then provide a future bounded preview. |
| `multimodal-rerank-handoff-001` | Reuse embedding preparation with explicit input paths, a separate rerank artifact root, and the supported stage sequence. |
| `multimodal-serving-drift-001` | Compare fixed candidates and raw scores across local and vLLM evaluation, with correct Docker GPU selection and preserved results. |
| `multimodal-resume-001` | Detect producer identity changes before resume and preserve existing artifacts. |
| `rerank-choice-001` | Choose embedding or reranking from candidate recall and ordering quality. |
| `llama-profile-export-001` | Preserve the Llama profile's ONNX/NIM contract across stages. |
| `nemotron3-deploy-001` | Preserve default Nemotron 3 direct deployment and its endpoint-specific evaluation contract. |
| `export-boundary-001` | Localize a legacy TensorRT regression after checkpoint/ONNX parity. |
| `remote-batch-001` | Use scheduler GPU allocation and a durable execution handoff for a supported text profile. |
| `rerank-sdg-governance-001` | Check confidential-corpus egress permission and protect ambient credentials. |
| `negative-vector-db-001` | Avoid activation for adjacent generic vector database advice. |
| `negative-nonrecipe-nim-001` | Avoid activation for unrelated embedding-server troubleshooting. |

Input readiness, command freshness, and metric hygiene are checked within the
relevant tasks. Separate trivial factual negatives and overlapping planning or
prerequisite cases add little coverage for their execution cost.

## Usefulness Rubric

Score agent usefulness above raw runtime. Strong with-skill trajectories should:

- activate on public Nemotron `embed`/`rerank` recipe tasks and stay inactive for generic retrieval or vector database advice,
- use bundled guidance for planning without a checkout, then verify the selected runtime before execution,
- choose `embed` vs `rerank` from the retrieval failure mode,
- recommend help/dry-run checks before API, GPU, Docker, Slurm, NIM, or other long-running work,
- handle secrets through environment configuration without asking users to paste values,
- separate preview commands, execution commands, polling cadence, and compact run reports.

## Required Checks

Before publication, run the configured skill evaluation harness in the configured evaluation environment:

- validate the skill and eval dataset structure,
- copy the skill outside the repository and check that bundled references resolve and external documentation links are commit-pinned,
- run static skill-quality checks,
- verify documented command examples with read-only `uv run --no-sync ... --help` and `-d` dry-runs when the checkout has the recipe CLI installed,
- run live with-skill and without-skill evaluation,
- cover both Codex and Claude Code, or document why an agent was skipped.

The live evaluation sends skill and eval prompts to the configured model providers. Get explicit approval before running it in an environment where workspace content is sensitive.

## Command Freshness Checklist

Use the current checkout rather than memory. Run the smallest relevant subset of these commands when recipe CLI drift is a concern:

```bash
uv run --no-sync nemotron embed --help
uv run --no-sync nemotron embed run -c nemotron-3 -d --from sdg --to prep
uv run --no-sync nemotron embed run -c nemotron-3 -d --from prep --to eval
uv run --no-sync nemotron rerank --help
uv run --no-sync nemotron rerank run -c default -d --from prep --to eval
uv run --no-sync nemotron rerank eval -c default -d eval_nim=true eval_base=false
```

## Reporting

Review the CI-generated `BENCHMARK.md` before merge. Do not hand-maintain a committed benchmark report for this skill unless the repository process changes.

The generated benchmark should include:

- agent harness and model versions, or public-safe aliases when model route names should not be published,
- metric names,
- test dataset size,
- with-skill score, without-skill score, and uplift,
- task completion and wall-clock/token data for the agent harness, making clear that this is agent-evaluation cost rather than expected recipe training runtime,
- limitations or skipped agents.

Treat any remaining generated findings as public-facing. Tier 1 high/critical findings must be fixed before merge unless they are confirmed false positives; lower-tier findings should still be reviewed and either fixed, accepted as non-blocking risk, or identified as false positives or run-to-run variance.
