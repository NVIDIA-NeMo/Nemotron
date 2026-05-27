---
name: retriever-finetune-recipe
version: "0.1.0"
author: "NVIDIA"
tags:
  - nemotron
  - retrieval
  - fine-tuning
  - embeddings
  - reranking
tools:
  - shell
  - read
  - write
description: Use when planning, debugging, tuning, evaluating, exporting, or deploying Nemotron `embed`/`rerank` retrieval recipes.
---

# Retriever Fine-Tune Recipe

## Purpose

Use this skill to work with Nemotron embedding and reranking fine-tuning recipes in a source checkout or installed package. Prefer the current checkout over memory, because the recipe CLI, configs, containers, and output paths are actively changing. Treat each recipe family as available only after its recipe directory and matching CLI files are present.

## Prerequisites

- Repo environment: `uv sync --all-extras` or the smallest relevant extra documented by the checkout.
- Stage 0 SDG: `NVIDIA_API_KEY`; never ask users to paste secret values.
- Stage 1-4 GPU work: CUDA/NVIDIA driver availability and enough VRAM.
- Stage 4 export: NeMo Export-Deploy container when using TensorRT.
- Stage 5 deploy: Docker, NGC access, and `NGC_API_KEY`.
- Remote execution: root `env.toml` profile for `--run` or `--batch`.

## First Decisions

1. Identify the recipe family.
   - Use `references/embed.md` for embedding, embed, bi-encoder, vector search, first-stage retrieval, low Recall@k, missing relevant documents, NIM embeddings, or `nemotron embed`.
   - Use `references/rerank.md` for rerank, reranker, cross-encoder, second-stage retrieval, acceptable recall but poor top-rank ordering, low nDCG with good Recall, or `nemotron rerank`.
   - Use both references only when the user asks about both families or asks which family to choose.
2. Choose the model to tune from the retrieval failure mode.
   - Prefer embedding fine-tuning when relevant documents are absent from the candidate set.
   - Prefer reranker fine-tuning when relevant documents are retrieved but ordered poorly near the top.
   - For production retrieval stacks, remember that these are complementary: embed first, rerank candidates second.
3. Identify the intent: plan a run, execute a stage, debug a failure, tune hyperparameters, interpret metrics, export/deploy a model, or modify recipe code/configs.
4. Inspect the current public surface before acting:
   - Recipe files: `src/nemotron/recipes/<embed|rerank>/`
   - CLI files: `src/nemotron/cli/commands/<embed|rerank>/`
   - Default configs: `src/nemotron/recipes/<family>/stage*/config/default.yaml`
   - Help and dry runs: `uv run nemotron <family> --help`, `uv run nemotron <family> <stage> -c default -d`

## Safe Workflow

1. Gather only context relevant to the task: corpus path, existing SDG/training/eval data, target stage range, output directory, checkpoint path, execution mode, GPU IDs, and whether required secrets are configured. Never ask users to paste secret values.
2. Start with cheap checks before expensive work:
   - `uv run nemotron <family> --help`
   - `uv run nemotron <family> <stage> --help`
   - `uv run nemotron <family> <stage> -c default -d`
   - `uv run nemotron <family> run -c default -d --from <stage> --to <stage>`
3. Check prerequisites for the requested stage:
   - Repo environment: `uv sync --all-extras` or the smallest relevant extra if documented by the repo.
   - Stage 0 SDG: `NVIDIA_API_KEY`.
   - Stage 1-4 GPU work: CUDA/NVIDIA driver availability and enough VRAM.
   - Stage 4 export: the NeMo Export-Deploy container when using TensorRT.
   - Stage 5 deploy: Docker, NGC access, and `NGC_API_KEY`.
   - Remote execution: root `env.toml` profile for `--run` or `--batch`.
4. Use dotlist overrides instead of editing defaults unless the user asks for reusable config changes. Keep sequence length, prefixes, pooling/normalization, prompt templates, and hard-negative counts consistent across stages.
5. Avoid launching API, GPU, Docker, Slurm, NIM, or long-running jobs unless the user explicitly asked to run them. Offer or run dry-runs, config review, and small pilots first.
6. If the user specifies GPU IDs, scope every stage command with `CUDA_VISIBLE_DEVICES=<ids>`.
7. For multi-stage local runs, prefer `uv run nemotron <family> run -c default --from <stage> --to <stage>`. The default `run` target stops at `eval`; `export` and `deploy` are opt-in.
8. When evaluating quality, compare against the base model on a fixed held-out evaluation set before recommending deployment. Do not substitute a standalone public-benchmark eval for the recipe's own Stage 3 evaluation.
9. For long-running SDG, prep, finetune, or eval work, start the process in a session-safe way and poll at human-scale intervals: roughly 60 seconds for small pilots and 120-300 seconds for larger runs.
10. For failures, load `PITFALLS.md`, localize the failing stage, then inspect the stage config, expected inputs, output directory, and corresponding CLI wrapper or `run_uv.py`.

## References

- `references/embed.md`: embedding recipe stages, commands, defaults, output paths, and operating patterns.
- `references/rerank.md`: rerank recipe stages, commands, defaults, output paths, and operating patterns.
- `references/evaluation.md`: metric interpretation, comparison hygiene, and deployment readiness checks.
- `PITFALLS.md`: common failures and recovery moves for SDG, prep, training, eval, export, deploy, and CLI setup.

## Example Usage

User asks: "Recall is decent, but nDCG is poor and the right passage is around rank 40. Should I tune embed or rerank?"

Load `references/rerank.md` and `references/evaluation.md`, explain that acceptable recall with poor top-rank ordering points to reranker tuning, then offer a cheap preview before training.

```bash
uv run nemotron rerank run -c default -d --from prep --to eval
```

## Troubleshooting

For failures, load `PITFALLS.md` first. Localize the failing stage, then inspect the stage config, expected inputs, output directory, and corresponding CLI wrapper or `run_uv.py`.

## Limitations

- This skill guides agents; it does not bundle recipe code or run GPU, Docker, NIM, Slurm, or provider API work on its own.
- It assumes the current checkout contains the `embed` or `rerank` recipe before treating that family as available.
- It should not trigger for unrelated retrieval theory, generic vector database setup, or benchmark evaluation that is not tied to the Nemotron recipe flow.

## Output Style

Give concrete commands and file paths. State assumptions, expected inputs, expected outputs, and the cheapest validation step that proves the next action is ready. For long-running stages, separate preview commands from execution commands so the user can choose deliberately.
