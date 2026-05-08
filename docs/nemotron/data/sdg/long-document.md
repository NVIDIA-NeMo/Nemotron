# Long-Document SDG

The long-document SDG recipe generates synthetic VLM training data from PDF documents — improving long-document understanding capabilities measured against [MMLongBench-Doc](https://arxiv.org/abs/2407.01523). Output is a corpus of question/answer pairs grounded in real document images, optionally judged for quality by a frontier model.

Built as nine PEP-723 `uv run`-able scripts that double as Nemotron CLI commands (`nemotron data sdg long-document <stage>`), and dispatchable to Slurm via [NeMo-Run](../../../nemo_runspec/nemo-run.md). Each producer stage can also auto-deploy its required vLLM endpoint via `--serve` — see below.

## Pipeline

```
01 seed ──┬─ 02 ocr ──── 03 text-qa ─────┐
          ├─ 04 classify ── 05 visual-qa ┤
          ├─ 06 single-page-qa ──────────┤── 09 judge
          ├─ 07 windowed-qa ─────────────┤
          └─ 08 whole-doc-qa ────────────┘
```

| Stage | Description | Model |
|---|---|---|
| `seed` | Download PDFs from FinePDFs, render pages to PNG, produce per-page / windowed / whole-document seed parquets | CPU-only |
| `ocr` | OCR with text + bbox metadata via Nemotron-Parse | `nvidia/NVIDIA-Nemotron-Parse-v1.1` |
| `text-qa` | QA pairs from OCR-transcribed text | `openai/gpt-oss-120b` |
| `page-classification` | Page-type and reasoning-complexity classification | `Qwen/Qwen3-VL-30B-A3B-Instruct` |
| `visual-qa` | Visual QA grounded in page images | `Qwen/Qwen3-VL-235B-A22B-Thinking-FP8` |
| `single-page-qa` | Anchored single-page QA across Text/Table/Chart/Image/Layout | same |
| `windowed-qa` | Multi-page sliding-window QA | same |
| `whole-document-qa` | Whole-document cross-page QA | same |
| `judge` | LLM-as-a-judge scoring of any QA output | any OpenAI-compatible frontier endpoint |

Stages 02–08 are CPU clients to a vLLM endpoint; stage 01 is CPU-only; stage 09 hits a third-party frontier API. Output of every stage is a parquet file consumable by downstream training recipes.

## Two ways to run a stage

Each stage works in two modes:

1. **Standalone via `uv`** — drop into any environment with `uv` and a vLLM endpoint:

   ```bash
   uv run --no-project 02-nemotron-parse-ocr-sdg.py \
       --config config/02-ocr.yaml \
       vllm_endpoint=http://localhost:8000/v1 \
       seed_path=./seed_data/seed_per_page.parquet \
       num_records=100
   ```

2. **Through the Nemotron CLI** — same recipes, dispatched on Slurm:

   ```bash
   nemotron data sdg long-document ocr --batch <profile> -c 02-ocr \
       vllm_endpoint=http://compute-node:8000/v1 \
       seed_path=/lustre/.../seed.parquet \
       num_records=100
   ```

Configuration is YAML + Hydra-style `key=value` overrides validated by a Pydantic `<Stage>Config` class — `nemotron data sdg long-document <stage> --help` renders the full field table.

## Auto-deploy with `--serve`

Producer stages (`ocr`, `text-qa`, `page-classification`, `visual-qa`, `single-page-qa`, `windowed-qa`, `whole-document-qa`) accept `--serve`. When passed, the CLI composes a multi-task NeMo-Run experiment:

- A **serve task** on a GPU partition brings vLLM up, picks a free TCP port at runtime, polls both `/health` and `/v1/models` to confirm the served model is registered, then publishes its endpoint to a sentinel file on shared storage.
- A **client task** (the recipe) waits on the sentinel, injects `vllm_endpoint=<url>` into its config, runs the recipe, and on exit signals the serve task to clean up.

```bash
# OCR — auto-deploys nvidia/NVIDIA-Nemotron-Parse-v1.1 on a GPU node,
# runs the recipe against it, tears the deployment down on exit.
nemotron data sdg long-document ocr --batch prep --serve \
    -c 02-ocr \
    seed_path=/lustre/.../seed_per_page.parquet \
    num_records=100
```

Override the default deployment with `--serve-config <name>` (configs live in `recipes/data/sdg/long-document/deployment/`).

`--serve` is not offered for `seed` (CPU-only, no model) or `judge` (frontier endpoint, third-party hosted).

## Cluster operational guidance

The `seed` stage and the `--serve` *client* tasks are CPU-only. On clusters whose default partitions require GPUs (e.g. NVIDIA's dlw cluster, where `interactive` and `batch` reject CPU-only jobs), use a profile that extends the cluster profile with CPU partitions. dlw ships `[prep]`:

```toml
[prep]
extends = "dlw"
run_partition = "cpu"
batch_partition = "cpu"
```

Use `--batch prep` / `--run prep` for these recipes:

```bash
nemotron data sdg long-document seed --batch prep -c 01-seed ...
nemotron data sdg long-document ocr  --batch prep --serve -c 02-ocr ...
```

The serve task always lands on a GPU partition (the cluster's `sdg_serve_partition` from `env.toml`, defaulting to `interactive`); the client task uses the env profile's regular `run_partition` / `batch_partition`.

## Getting Started

The recipe scripts live in:

```
src/nemotron/recipes/data/sdg/long-document/
```

See the [recipe README](https://github.com/NVIDIA/nemotron/tree/main/src/nemotron/recipes/data/sdg/long-document) for full per-stage documentation, deployment-config schema, troubleshooting, and full-pipeline examples (both manual-vLLM and `--serve` styles).

## Prerequisites

- `uv` installed (recipes resolve PEP 723 inline deps at run time).
- For producers (02–08): an OpenAI-compatible vLLM endpoint serving the recipe's required model — either operator-launched, or auto-deployed by `--serve`.
- For the judge (09): an OpenAI-compatible frontier endpoint with a valid API key in an env var.
- For Slurm dispatch: the [`nemotron-evaluator-launcher`-style env.toml profile](../../../nemo_runspec/nemo-run.md) for your cluster.

## After SDG

Once the pipeline runs, the resulting parquet files can be:

- Published to Hugging Face Hub as a public dataset.
- Stored in internal Lustre and registered as a Nemotron / W&B artifact.
- Consumed directly by training recipes via `dataset.path` or HF-dataset-id config.

The recipe README has copy-pasteable templates for both publish paths.

## Further Reading

- [Recipe README](https://github.com/NVIDIA/nemotron/tree/main/src/nemotron/recipes/data/sdg/long-document) — comprehensive per-stage docs, config schemas, troubleshooting.
- [Deployment config schema](https://github.com/NVIDIA/nemotron/tree/main/src/nemotron/recipes/data/sdg/long-document/deployment) — `--serve` deployment YAML reference.
- [MMLongBench-Doc paper](https://arxiv.org/abs/2407.01523) — the benchmark this dataset targets.
- [Nemotron-CC](../curation/nemotron-cc.md) — the sibling pretraining-data curation recipe.
- [Execution through NeMo-Run](../../../nemo_runspec/nemo-run.md) — how the `--run` / `--batch` dispatch works.
