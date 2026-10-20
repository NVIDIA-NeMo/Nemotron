---
name: nemotron-retrieval-recipes
license: Apache-2.0
metadata:
  version: "0.4.0"
  author: "NVIDIA Nemotron Team <noreply@nvidia.com>"
  tags:
    - nemotron
    - retrieval
    - fine-tuning
    - embeddings
    - reranking
description: Use for Nemotron embed/rerank recipes, including Nemotron 3.5 multimodal SDG, tuning, evaluation, and deployment. Not for generic retrieval.
---

# Nemotron Retrieval Recipes

Invocation: `$nemotron-retrieval-recipes`.

## Purpose

Use this skill to plan, run, and debug public Nemotron embedding and reranking recipes. The bundled references support planning without a Nemotron checkout or network access. Before execution, verify the installed CLI's capabilities and the selected configuration because recipe interfaces can change.

Use it only for tasks tied to the public Nemotron `embed` or `rerank` recipe flow. If the request is unrelated retrieval theory, generic vector database selection, generic benchmark advice, or non-recipe Docker/Slurm/NIM troubleshooting, stop with a short scope note and do not inspect recipe files in that turn.

## Security Notes

Use `Bash` for repo-scoped inspection, help, dry-run, and user-approved execution commands. Do not run API, GPU, Docker, Slurm, NIM, or other long-running work unless the user explicitly asks for it. Before Stage 0 SDG for either family, confirm the user's data-governance policy permits sending corpus content to the configured inference endpoints; otherwise use an approved private or air-gapped path. Never run broad environment dumps or commands that expose secret values. Prefer dotlist overrides and config review over editing recipe defaults.

## Runtime and Source Priority

Resolve `references/` links relative to this skill folder, not the working
directory. External source links are optional, commit-pinned background.
Do not require repository documentation to answer a planning question.

Execution examples assume a prepared Nemotron checkout. At the bundled
revision, the CLI resolves recipe scripts from the working directory: run
from the checkout root even if `nemotron` is installed globally. Use
`uv run nemotron` there, or `nemotron` directly in an activated environment.
Do not run `uv sync` in an unrelated project. Without a checkout or runtime,
provide a plan from the bundled references and identify setup as an execution
prerequisite. Never require a checkout just to explain the workflow.

For a new execution workspace, this setup selects the bundled revision. Use
the user's existing checkout and revision when they already chose one:

```bash
git clone https://github.com/NVIDIA-NeMo/Nemotron.git
cd Nemotron
git checkout e68db1595a9f078a3fb48f8216fa0481d1c862a2
uv sync --python 3.12
```

Resolve conflicts in this order:

1. User-selected runtime: installed CLI help and resolved configs, plus source files when a matching checkout is available.
2. Bundled references in this skill.
3. User-provided docs or saved snippets.
4. Memory.

Before execution, confirm the command, profile, input paths, and required
environment exist. Missing runtime dependencies block execution, not planning.

## Inputs

Gather only what the requested task needs:

- Required: the goal or failure, recipe family, model profile, and target stage.
- Optional, depending on the stage: corpus or prepared-data paths, artifact root, checkpoint, execution mode, GPU IDs, and whether credentials are configured.
- Prefer explicit user choices over profile defaults. Infer available context from the conversation and checkout; ask only for missing information that changes the next action. Never request secret values.

## Instructions

1. Identify the recipe family.
   - Use `references/embed.md` for embedding, embed, bi-encoder, vector search, first-stage retrieval, low Recall@k, missing relevant documents, NIM embeddings, or `nemotron embed`.
   - Use `references/rerank.md` for rerank, reranker, cross-encoder, second-stage retrieval, acceptable recall but poor top-rank ordering, low nDCG with good Recall, or `nemotron rerank`.
   - Use both references when comparing families or planning work that spans both families.
2. Choose the model profile within the selected family.
   - For embed, use `-c nemotron-3` for Nemotron 3 Embed, `-c llama` for Llama Nemotron Embed, or `-c nemotron-3.5` for Nemotron 3.5 Embed with text, images, or paired documents.
   - For rerank, use `-c default` for Llama Nemotron Rerank or `-c nemotron-3.5` for Nemotron 3.5 Rerank.
   - For Nemotron 3.5 reranking from raw documents, also load the embedding reference's multimodal workflow: embedding SDG and preparation produce the shared training inputs. With prepared inputs, start from the rerank reference's handoff.
   - When the CLI is available, run `nemotron <family> info` if model identity is unclear. Read the selected profile's section in the family reference before composing commands.
   - Carry the selected profile and `artifact_root` through every stage; never combine artifacts from different profiles.
3. Choose the model family to tune from the retrieval failure mode.
   - Prefer embedding fine-tuning when relevant documents are absent from the candidate set.
   - Prefer reranker fine-tuning when relevant documents are retrieved but ordered poorly near the top.
   - For production retrieval stacks, remember that these are complementary: embed first, rerank candidates second.
4. Identify the intent: plan a run, execute a stage, debug a failure, tune hyperparameters, interpret metrics, export/deploy a model, inspect configs, or propose dotlist overrides. Select any additional references below for that task before composing commands.
5. Before execution, inspect CLI help and dry runs. When a matching checkout is available, use these optional implementation locations for debugging:
   - Recipe files: `src/nemotron/recipes/<embed|rerank>/`
   - CLI files: `src/nemotron/cli/commands/<embed|rerank>/`
   - Configs: `src/nemotron/recipes/<family>/stage*/config/<profile>.yaml`
   - Help and dry runs: `uv run nemotron <family> --help`, `uv run nemotron <family> <stage> -c <profile> -d`

## Prerequisites

Check only the prerequisites for the requested stage:

- Source checkout setup, when needed: `uv sync --python 3.12` at the checkout root; preserve isolated stage environments. Skip synchronization if that environment is already prepared.
- Stage 0 synthetic data generation (SDG): `NVIDIA_API_KEY` and the selected profile's model-role configuration.
- Stages 1–3 GPU work: CUDA/NVIDIA driver availability and enough VRAM.
- Stage 4 export: the NeMo Export-Deploy container when using TensorRT. Default Nemotron 3 Embed skips this stage; Nemotron 3.5 has no export profile.
- Stage 5 deploy: Docker plus the selected backend's image and artifact contract. Load the family reference before requiring NGC credentials.
- Remote execution: root `env.toml` profile for `--run` or `--batch`; load `references/remote.md` when remote scheduling, logs, or GPU placement matter.

## Safe Workflow

1. Resolve the inputs above and verify the family reference's stage contracts.
2. When the runtime is available, start with cheap checks before expensive work. Otherwise, present these as checks to run after installation:
   - `uv run nemotron <family> --help`
   - `uv run nemotron <family> <stage> --help`
   - `uv run nemotron <family> <stage> -c <profile> -d`
   - `uv run nemotron <family> run -c <profile> -d --from <stage> --to <stage>`
   - The inherited `-c` and `-d` options work with `run`, even when they are absent from `run --help` output. Confirm with `run -c <model-profile> -d ...` when unsure.
   - In an already prepared checkout, `uv run --no-sync ... --help` or `uv run --no-sync ... -d` can avoid unexpected dependency sync during read-only checks.
3. Use dotlist overrides instead of editing defaults unless the user asks for reusable config changes. Keep the selected profile, artifact root, sequence length, prefixes, pooling/normalization, prompt templates, and hard-negative counts consistent across stages.
4. Avoid launching API, GPU, Docker, Slurm, NIM, or long-running jobs unless the user explicitly asked to run them. Offer or run dry-runs, config review, and small pilots first.
5. For local Python execution, scope requested GPU IDs with `CUDA_VISIBLE_DEVICES=<ids>`. For Docker deployment, select the host GPU explicitly with `gpus='device=0'`, replacing `0` with the requested device; `CUDA_VISIBLE_DEVICES` in the shell does not select the container's GPU. For `--run` or `--batch`, configure scheduler resources such as `gpus_per_node` in the selected `env.toml` profile and let the scheduler assign devices; do not assume submit-shell `CUDA_VISIBLE_DEVICES` propagates remotely.
6. For supported multi-stage runs, use `uv run nemotron <family> run -c <profile> --from <stage> --to <stage>`. Check the family reference for profile-specific stage and executor restrictions. Stop at evaluation before choosing export or deployment.
7. When evaluating quality, compare against the base model on a fixed held-out evaluation set before recommending deployment. Do not substitute a standalone public-benchmark eval for the recipe's own Stage 3 evaluation.
8. For long-running SDG, prep, finetune, or eval work, start the process in a session-safe way and poll at human-scale intervals: roughly 60 seconds for small pilots and 120-300 seconds for larger runs.

## Troubleshooting

Localize the failing stage, then inspect its config, required inputs, output
directory, and corresponding CLI wrapper or `run_uv.py`. Use the family reference's
recovery guidance before rerunning or replacing artifacts.

## References

Select only the guidance needed for the chosen profile and stage. All bundled
references are linked here; the individual guides contain procedures and contracts.

- [Embedding reference](references/embed.md): read for embedding tasks; choose its Nemotron 3.5 or text-profile section for stage contracts, commands, and recovery.
- [Rerank reference](references/rerank.md): read for reranking tasks; choose its Nemotron 3.5 or default text-profile section for data handoffs and deployment.
- [Evaluation reference](references/evaluation.md): read when interpreting metrics, comparing checkpoints or endpoints, or assessing deployment readiness.
- [Remote reference](references/remote.md): read when scheduling supported profiles, selecting GPUs, finding logs, or managing long-running jobs.
- [Multimodal data reference](references/multimodal-data.md): read for Nemotron 3.5 source records, generator/judge settings, bundle handoffs, or missing artifacts. For full hash, content, and split validation, use its [export audit procedure](references/multimodal-data.md#audit-an-export); normal stage handoffs do not repeat that audit.

## Examples

User asks: "Recall is decent, but nDCG is poor and the right passage is around rank 40. Should I tune embed or rerank?"

Load `references/rerank.md` and `references/evaluation.md`, explain that acceptable recall with poor top-rank ordering points to reranker tuning, then offer a cheap preview before training.

```bash
uv run nemotron rerank run -c default -d --from prep --to eval
```

## Limitations

- Bundled references describe Nemotron commit `e68db1595a9f078a3fb48f8216fa0481d1c862a2`. Verify commands, flags, defaults, and output paths against the selected runtime before execution.
- This skill does not provide datasets, checkpoints, credentials, GPU capacity, Docker images, or NIM services.

## Output Style

For planning or debugging recommendations, use this shape when it helps: `Decision`, `Why`, `Required inputs`, `Preview command`, `Execution command`, `Avoid`, and `Next step`. Omit fields that are irrelevant to a short answer.

Give concrete commands and file paths. State assumptions, expected inputs, expected outputs, and the cheapest validation step that proves the next action is ready. For long-running stages, separate preview commands from execution commands so the user can choose deliberately.

When reporting a dry-run or real run, include a compact run report: command, mode, config, dotlist overrides, input paths, output paths, validation signal or metric file, and next cheapest check. Include the checkout commit when it is available.
