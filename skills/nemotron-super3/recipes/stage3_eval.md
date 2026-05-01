# Stage 3 Recipe Summary — Evaluation

This file maps the released evaluation stage to:

- `src/nemotron/recipes/super3/stage3_eval/`
- `src/nemotron/cli/commands/super3/eval.py`

---

## What makes evaluation different

Unlike the training stages, evaluation is **not** driven by a recipe script with a `[tool.runspec]` header.

Instead, the CLI command:

1. parses and resolves the config,
2. injects artifact and env.toml context,
3. writes `job.yaml` / `eval.yaml`,
4. calls `nemo-evaluator-launcher` directly.

That is the most important implementation fact to remember.

---

## Main source files

| Path | Role |
|---|---|
| `src/nemotron/recipes/super3/stage3_eval/README.md` | stage README |
| `.../config/default.yaml` | evaluator config |
| `src/nemotron/cli/commands/super3/eval.py` | visible execution logic |
| `docs/nemotron/super3/evaluate.md` | broader release docs and benchmark context |

---

## Default config highlights

From `stage3_eval/config/default.yaml`:

| Setting | Value |
|---|---|
| Default model artifact | `super3-rl-model:latest` |
| Container | `nvcr.io/nvidia/nemo:26.02.nemotron_3_super` |
| Deployment backend | generic NeMo Ray deployment |
| GPUs | 8 |
| Tensor parallelism | 1 |
| Expert parallelism | 8 |
| Port | 1235 |

### Default task list

| Task name |
|---|
| `adlr_mmlu` |
| `adlr_arc_challenge_llama_25_shot` |
| `hellaswag` |
| `openbookqa` |
| `adlr_winogrande_5_shot` |

---

## CLI examples exposed by the repo

```bash
uv run nemotron super3 eval --run <profile>
uv run nemotron super3 eval --run <profile> run.model=sft:v2
uv run nemotron super3 eval --run <profile> -t adlr_mmlu -t hellaswag
uv run nemotron super3 eval --dry-run
```

This is the quickest answer when a user asks how to run evaluation.

---

## What `eval.py` actually does

`src/nemotron/cli/commands/super3/eval.py` is worth citing because it makes the execution logic visible.

At a high level it:

- parses recipe config metadata,
- builds a job config with provenance,
- injects W&B env mappings if needed,
- resolves artifacts such as `${art:model,path}`,
- strips the Nemotron-only `run` section,
- passes the remaining config to `run_eval()` from `nemo-evaluator-launcher`.

So evaluation is a config-compilation + launcher-delegation flow, not a standard training launch.

---

## Artifact and deployment story

The stage expects a model artifact, most commonly the latest RL output.

| Artifact role | Default |
|---|---|
| Model to evaluate | `super3-rl-model:latest` |
| Checkpoint path | `${art:model,path}` |
| Deployment command | NeMo Ray in-framework deployment |

The deployment command in the YAML is a good citation for users asking how Super3 is actually served during evaluation.

---

## Important caveat

The release docs explicitly say this stage covers a **subset** of the full paper benchmark suite. It is meant for development validation and reproducibility support, not as the sole source of every paper number.

So when a user asks “does this stage reproduce the paper tables?”, the accurate answer is:

- it reproduces part of the evaluation surface,
- the broader suite lives in NeMo Evaluator configs and reproducibility docs.

---

## Best next file

- `../paper/evaluation.md` for the benchmark interpretation and larger result tables.
- `../paper/quantization.md` if the user is really asking about BF16 vs FP8/NVFP4 comparisons.
