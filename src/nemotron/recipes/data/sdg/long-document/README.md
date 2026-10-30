# Long-Document SDG Pipeline

Synthetic data generation (SDG) recipes for building high-quality VLM training data from PDF documents — improves long-document understanding capabilities. Progress is tracked using the [MMLongBench-Doc](https://arxiv.org/abs/2407.01523) benchmark.

Each stage is:

- A **standalone `uv run`-able** PEP 723 script — drop into any environment with `uv` and a vLLM endpoint and it works.
- **Wired into the Nemotron CLI** as `nemotron data sdg long-document <stage>` so it can dispatch via nemo-run onto Slurm with `--run <profile>` / `--batch <profile>`.

Configuration is YAML + OmegaConf dotlist overrides, validated by a Pydantic `<Stage>Config` class. The default YAML for each stage lives under `config/`.

Operators have two options for the model server: **launch vLLM yourself** (cookbook-style — see [Manual model launch](#manual-model-launch)) or **let the CLI auto-deploy** via [`--serve`](#auto-deploy-with---serve), which composes a multi-task `nemo_run.Experiment` with a vLLM serve job + the recipe client job.

## Pipeline overview

```text
01 seed ──┬─ 02 ocr ──── 03 text-qa ─────┐
          ├─ 04 classify ── 05 visual-qa ┤
          ├─ 06 single-page-qa ──────────┤── 09 judge
          ├─ 07 windowed-qa ─────────────┤
          └─ 08 whole-doc-qa ────────────┘
```

### Seed outputs

`01-seed-dataset-preparation.py` produces three parquet files:

| File | Granularity | Consumed by |
|---|---|---|
| `seed_per_page.parquet` | one row per page | 02, 03, 04, 05, 06 |
| `seed_windowed.parquet` | one row per sliding window of pages | 07 |
| `seed_whole_document.parquet` | one row per document | 08 |

All seed files share a `png_images_base64` column containing a JSON array of base64-encoded PNG strings.

### Stages

| Stage | Script | Purpose | Model |
|---|---|---|---|
| `seed` | `01-seed-dataset-preparation.py` | Build per-page / windowed / whole-document seed parquets from FinePDFs | CPU-only |
| `ocr` | `02-nemotron-parse-ocr-sdg.py` | OCR extraction with text + bounding-box metadata | `nvidia/NVIDIA-Nemotron-Parse-v1.1` |
| `text-qa` | `03-text-qa-sdg.py` | Text QA from OCR-transcribed text | `openai/gpt-oss-120b` |
| `page-classification` | `04-page-classification-sdg.py` | Page-type and reasoning-complexity classification | `Qwen/Qwen3-VL-30B-A3B-Instruct` |
| `visual-qa` | `05-visual-qa-sdg.py` | Visual QA generation | `Qwen/Qwen3-VL-235B-A22B-Thinking-FP8` |
| `single-page-qa` | `06-single-page-qa-sdg.py` | Anchored single-page QA | `Qwen/Qwen3-VL-235B-A22B-Thinking-FP8` |
| `windowed-qa` | `07-multi-page-windowed-qa-sdg.py` | Multi-page sliding-window QA | `Qwen/Qwen3-VL-235B-A22B-Thinking-FP8` |
| `whole-document-qa` | `08-whole-document-qa-sdg.py` | Whole-document cross-page QA | `Qwen/Qwen3-VL-235B-A22B-Thinking-FP8` |
| `judge` | `09-frontier-judge-sdg.py` | LLM-as-a-judge scoring for QA outputs | any OpenAI-compatible frontier endpoint |

## Running a stage

Every stage accepts a YAML config and Hydra-style `key=value` dotlist overrides on the CLI. Discover the full config surface with `--help`:

```bash
nemotron data sdg long-document <stage> --help        # rich Pydantic field table + examples
uv run --no-project 0X-…sdg.py --help                  # standalone argparse help
```

### Standalone via `uv`

Self-contained — PEP 723 inline deps are resolved by `uv`. No Nemotron repo needed in the run environment.

```bash
# 01 — seed (CPU-only)
uv run --no-project 01-seed-dataset-preparation.py \
    --config config/01-seed.yaml \
    output_dir=./seed_data \
    num_docs=10
```

```bash
# 02 — OCR against an operator-launched vLLM
uv run --no-project 02-nemotron-parse-ocr-sdg.py \
    --config config/02-ocr.yaml \
    vllm_endpoint=http://localhost:8000/v1 \
    seed_path=./seed_data/seed_per_page.parquet \
    num_records=50 \
    artifact_path=./ocr_output
```

Same shape for `03`–`08`: `--config config/0X-<stage>.yaml` + dotlist `vllm_endpoint=… seed_path=… num_records=… artifact_path=…`.

`09` is shaped slightly differently (frontier endpoint, not vLLM):

```bash
uv run --no-project 09-frontier-judge-sdg.py \
    --config config/09-judge.yaml \
    seed_path=./single_page_qa_output.parquet \
    endpoint=https://generativelanguage.googleapis.com/v1beta \
    model_id=gemini-2.5-flash-preview-04-17 \
    api_key_env=GEMINI_API_KEY \
    artifact_path=./judged_single_page_qa.parquet
```

### Through the Nemotron CLI

The same stages, dispatched via `nemo-run`. Add `--run <profile>` (attached) or `--batch <profile>` (detached) to land the job on a Slurm cluster — see your `env.toml` for available profiles.

```bash
nemotron data sdg long-document seed --batch prep \
    -c 01-seed num_docs=50 \
    output_dir=/lustre/.../sdg/seeds
```

For producer stages and the judge, you have two options for the model endpoint:

#### Option A — point at a manually-launched vLLM

Same flow as standalone, plus `--run <profile>` / `--batch <profile>`. The recipe job is a CPU client; the GPUs are on the operator-supplied vLLM endpoint.

```bash
nemotron data sdg long-document ocr --batch prep \
    -c 02-ocr \
    vllm_endpoint=http://compute-node-0001:8000/v1 \
    seed_path=/lustre/.../seeds/seed_per_page.parquet \
    num_records=100 \
    artifact_path=/lustre/.../sdg/ocr
```

#### Option B — let the CLI auto-deploy with `--serve`

Producer stages (`ocr`, `text-qa`, `page-classification`, `visual-qa`, `single-page-qa`, `windowed-qa`, `whole-document-qa`) accept `--serve`. When passed, the CLI:

1. Reads the deployment config from `deployment/<name>.yaml` (the default `<name>` for each stage is registered in `_deployment.py:STAGE_DEFAULT_DEPLOYMENT`; override with `--serve-config <name>`).
2. Composes a multi-task `nemo_run.Experiment`:
   - **Serve task** on a GPU partition: brings vLLM up, picks a free TCP port, polls `/health` *and* `/v1/models` to confirm the served model is registered, publishes `host:port` to a sentinel file on shared storage.
   - **Client task** (the recipe): waits on the sentinel, injects `vllm_endpoint=<url>` into its config, runs the recipe.
3. Cleans up: client trap-on-exit writes `<sentinel>.done` → serve task SIGTERMs vLLM and exits cleanly. Slurm walltime backstops abnormal exits.

Drop the `vllm_endpoint=…` argument when using `--serve` — it's set automatically.

```bash
# OCR: brings up nvidia/NVIDIA-Nemotron-Parse-v1.1 on a GPU partition,
# runs the recipe against it, tears the deployment down on exit.
nemotron data sdg long-document ocr --batch prep --serve \
    -c 02-ocr \
    seed_path=/lustre/.../seeds/seed_per_page.parquet \
    num_records=100 \
    artifact_path=/lustre/.../sdg/ocr

# Override the default deployment (e.g. test a different vLLM tuning):
nemotron data sdg long-document single-page-qa --batch prep --serve \
    --serve-config qwen3-vl-30b \
    -c 06-single-page-qa \
    seed_path=/lustre/.../seeds/seed_per_page.parquet \
    num_records=10000
```

`--serve` is **not** offered for `seed` (CPU-only, no model) or `judge` (frontier endpoint, third-party hosted).

## Cluster operational guidance

### Partition selection

The `seed` stage and the `--serve` *client* tasks are CPU-only. On clusters where the default partitions require GPUs (e.g. NVIDIA's dlw cluster, where `interactive` and `batch` both reject CPU-only jobs), use a profile that extends the cluster profile with CPU partitions. dlw ships one out of the box:

```toml
[prep]
extends = "dlw"
run_partition = "cpu"
batch_partition = "cpu"
```

So on dlw use `--batch prep` / `--run prep` for these recipes:

```bash
nemotron data sdg long-document seed --batch prep -c 01-seed ...
nemotron data sdg long-document ocr  --batch prep --serve -c 02-ocr ...
```

The `--serve` *serve* task always lands on a GPU partition (the cluster's `sdg_serve_partition` in `env.toml`, defaulting to `interactive`); the *client* task uses the regular `run_partition` / `batch_partition` of the env profile.

### Lustre paths

- `seed_path`, `output_dir`, `artifact_path` should all be absolute paths on shared storage (Lustre on dlw). Relative paths resolve to the slurm job's working directory, which is generally not what you want.
- The serve sentinel lives at `${remote_job_dir}/sdg-deploy/<recipe-name>/<timestamp>/endpoint`.

### HF / API tokens

- `HF_TOKEN` for any model that pulls from gated HF repos.
- For `judge`: set the env var named in `api_key_env` to your frontier API key. The recipe reads it inside the slurm job.

These env vars must be propagated through your env.toml profile's `env_vars` block.

## Auto-deploy: deployment configs

Deployment YAMLs live under `deployment/<name>.yaml` and validate against the `DeploymentConfig` Pydantic schema in `cli/commands/data/sdg/long_document/_deployment.py`. The shape mirrors Evaluator-launcher's `vllm.yaml` so the two orgs can converge later.

| Stage | Default deployment | Model |
|---|---|---|
| `ocr` | `nemotron-parse-v1.1` | `nvidia/NVIDIA-Nemotron-Parse-v1.1` |
| `text-qa` | `gpt-oss-120b` | `openai/gpt-oss-120b` |
| `page-classification` | `qwen3-vl-30b` | `Qwen/Qwen3-VL-30B-A3B-Instruct` |
| `visual-qa` | `qwen3-vl-235b` | `Qwen/Qwen3-VL-235B-A22B-Thinking-FP8` |
| `single-page-qa` | `qwen3-vl-235b` | same |
| `windowed-qa` | `qwen3-vl-235b` | same |
| `whole-document-qa` | `qwen3-vl-235b` | same |

See [`deployment/README.md`](./deployment/README.md) for the full schema. To override a deployment knob without forking the YAML, copy it and pass `--serve-config <new-name>`.

### Port selection (important)

Default `port: null` means **the serve task picks a free TCP port at runtime** (via `python3 -c "import socket; ..."`). This is the safe default — pinning a specific port (e.g. 8000) is a footgun on shared GPU nodes where DCGM exporter / Prometheus / similar agents commonly bind well-known ports. Pin a port only if you're on a dedicated node where collisions are impossible.

The dynamically-selected port is woven into the published endpoint URL the client receives — operators don't have to reason about it.

## Manual model launch

If you'd rather launch vLLM yourself and pass `vllm_endpoint=…` explicitly (Option A above), copy-pasteable starting points follow. Adjust partition names, mount points, GPU counts, and token lengths to match your environment.

### `nvidia/NVIDIA-Nemotron-Parse-v1.1` — used by `ocr`

The vLLM server must be launched with a chat template that injects Nemotron-Parse special tokens. Save this single line as `chat_template.jinja`:

```jinja
{% for message in messages %}{% if message["role"] == "user" %}{{ "</s><s><predict_bbox><predict_classes><output_markdown>" }}{% endif %}{% endfor %}
```

Local docker:

```bash
docker run --gpus all -p 8000:8000 \
    -v $(pwd)/chat_template.jinja:/chat_template.jinja \
    -e HF_TOKEN=$HF_TOKEN \
    --entrypoint bash \
    vllm/vllm-openai:v0.14.1 \
    -c "pip install open-clip-torch albumentations timm && \
        vllm serve nvidia/NVIDIA-Nemotron-Parse-v1.1 \
        --tensor-parallel-size 1 \
        --max-model-len 9000 \
        --gpu-memory-utilization 0.85 \
        --max-num-seqs 128 \
        --chat-template /chat_template.jinja \
        --trust-remote-code"
```

Slurm + Pyxis (1× H100):

```bash
srun --partition=interactive --nodes=1 --ntasks=1 --gres=gpu:1 \
     --time=04:00:00 \
     --container-image=vllm/vllm-openai:v0.14.1 \
     --container-mounts=/lustre:/lustre \
     bash -c "pip install open-clip-torch albumentations timm && \
              vllm serve nvidia/NVIDIA-Nemotron-Parse-v1.1 \
                --tensor-parallel-size 1 \
                --max-model-len 9000 \
                --gpu-memory-utilization 0.85 \
                --chat-template /lustre/.../chat_template.jinja \
                --trust-remote-code"
```

### `openai/gpt-oss-120b` — used by `text-qa`

```bash
docker run --gpus all -p 8000:8000 \
    -e HF_TOKEN=$HF_TOKEN \
    vllm/vllm-openai:latest \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.80 \
    --reasoning-parser openai_gptoss
```

### `Qwen/Qwen3-VL-30B-A3B-Instruct` — used by `page-classification`

```bash
docker run --gpus all -p 8000:8000 \
    -e HF_TOKEN=$HF_TOKEN \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 128000 \
    --gpu-memory-utilization 0.95 \
    --limit-mm-per-prompt '{"video": 0}' \
    --trust-remote-code
```

### `Qwen/Qwen3-VL-235B-A22B-Thinking-FP8` — used by `visual-qa`, `single-page-qa`, `windowed-qa`, `whole-document-qa`

```bash
docker run --gpus all -p 8000:8000 \
    -e HF_TOKEN=$HF_TOKEN \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-VL-235B-A22B-Thinking-FP8 \
    --tensor-parallel-size 4 \
    --max-model-len 50000 \
    --gpu-memory-utilization 0.90 \
    --reasoning-parser deepseek_r1 \
    --limit-mm-per-prompt '{"video": 0}' \
    --trust-remote-code
```

### Frontier judge endpoint — used by `judge`

`09-frontier-judge-sdg.py` accepts any OpenAI-compatible endpoint via `endpoint=…`, plus `model_id=…` and `api_key_env=…` (the name of the env var holding the API key). This repository does not prescribe a default model; use one approved for your environment.

## Full-pipeline example (manual vLLM)

```bash
# 1. Seed the pipeline.
uv run --no-project 01-seed-dataset-preparation.py \
    --config config/01-seed.yaml \
    output_dir=./seed_data \
    num_docs=1000 \
    subset=eng_Latn

# 2. Launch the vLLMs you'll need (see Manual model launch above).
#    Note the host:port for each.

# 3. Run producers in parallel.
uv run --no-project 02-nemotron-parse-ocr-sdg.py \
    --config config/02-ocr.yaml \
    vllm_endpoint=http://node-a:8000/v1 \
    seed_path=./seed_data/seed_per_page.parquet \
    num_records=10000 \
    artifact_path=./ocr_output

uv run --no-project 06-single-page-qa-sdg.py \
    --config config/06-single-page-qa.yaml \
    vllm_endpoint=http://node-b:8000/v1 \
    seed_path=./seed_data/seed_per_page.parquet \
    num_records=100000 \
    artifact_path=./single_page_qa_output

# … and so on for 03, 04, 05, 07, 08.

# 4. Judge a QA output.
uv run --no-project 09-frontier-judge-sdg.py \
    --config config/09-judge.yaml \
    seed_path=./single_page_qa_output/.../generated.parquet \
    endpoint=https://generativelanguage.googleapis.com/v1beta \
    model_id=gemini-2.5-flash-preview-04-17 \
    api_key_env=GEMINI_API_KEY \
    artifact_path=./judged_single_page_qa
```

## Full-pipeline example (`--serve` auto-deploy on Slurm)

```bash
# 1. Seed (CPU partition).
nemotron data sdg long-document seed --batch prep \
    -c 01-seed \
    output_dir=/lustre/.../sdg/seeds \
    num_docs=1000

# 2. OCR — auto-deploys nemotron-parse on a GPU node.
nemotron data sdg long-document ocr --batch prep --serve \
    -c 02-ocr \
    seed_path=/lustre/.../sdg/seeds/seed_per_page.parquet \
    num_records=10000 \
    artifact_path=/lustre/.../sdg/ocr

# 3. Single-page QA — auto-deploys Qwen3-VL-235B on 4× GPUs.
nemotron data sdg long-document single-page-qa --batch prep --serve \
    -c 06-single-page-qa \
    seed_path=/lustre/.../sdg/seeds/seed_per_page.parquet \
    num_records=100000 \
    artifact_path=/lustre/.../sdg/single_page_qa

# 4. Judge.
nemotron data sdg long-document judge --batch prep \
    -c 09-judge \
    seed_path=/lustre/.../sdg/single_page_qa/.../generated.parquet \
    endpoint=https://generativelanguage.googleapis.com/v1beta \
    model_id=gemini-2.5-flash-preview-04-17 \
    api_key_env=GEMINI_API_KEY \
    artifact_path=/lustre/.../sdg/judged_single_page_qa
```

## Troubleshooting

### `Cannot find GPU specification, you may not submit a job not requesting GPUs in a non-CPU partition`

Slurm rejected a CPU-only recipe because the env profile points at a GPU-only partition. Use the `prep` profile (or any CPU-friendly profile your env.toml provides) instead of the base cluster profile. See [Partition selection](#partition-selection).

### `Address already in use` on the serve task

Should be impossible with the default `port: null` (dynamic). If you've pinned a port in the deployment YAML, another process on the GPU node owns that port — drop the pinning so the runtime picks a free ephemeral port.

### Recipe fails with `model 'X' could not be found`

The recipe got an endpoint URL that points at a server which doesn't actually serve the expected model. With `--serve` this should never happen — the serve bash polls `/v1/models` and won't publish the sentinel until the served model is registered. With manual vLLM, double-check your `--served-model-name` matches what the recipe sends as `model_id` / `model_alias`.

### Container squash takes forever / pyxis flakes

vLLM images are large (15-20 GB squashed). First-run squash can take 5-15 min on the build partition. Subsequent runs reuse the cached squashfs. Pyxis container-startup failures (`pyxis: couldn't start container`) are usually transient — retry once.

### Standalone scripts fail with `pyarrow` / `data_designer` import errors

The recipe scripts use PEP 723 inline deps. Make sure you invoked `uv run --no-project <script>` (not bare `python <script>`) so `uv` resolves the inline deps before running.

## Publishing the output

After running the pipeline, publish the resulting parquet outputs through one of the following paths.

### Public path: Hugging Face Hub

```bash
export HF_TOKEN=...
hf auth login --token "$HF_TOKEN"

hf repo create nvidia/long-document-understanding-sdg-v1 \
    --repo-type dataset --private=false

hf upload nvidia/long-document-understanding-sdg-v1 \
    ./published_dataset --repo-type dataset
```

Recommended contents for `./published_dataset`:

- seed parquet files if you want to expose the starting point
- generated parquet outputs from the QA stages
- judged parquet outputs
- a dataset card describing source PDFs, filtering, and model variants used

### Private path: internal storage plus artifact registration

```bash
mkdir -p /lustre/team/datasets/long-document-understanding-sdg/v1
cp -R ./published_dataset/. /lustre/team/datasets/long-document-understanding-sdg/v1/
```

If your environment uses Nemotron artifact logging:

```bash
nemotron kit log-artifact data \
    --name omni3-long-document-sdg \
    --path /lustre/team/datasets/long-document-understanding-sdg/v1
```

Otherwise register manually through your standard W&B flow with `wandb.log_artifact`.

## Consumption

Downstream training recipes consume the published dataset by artifact or dataset ID. For Omni-style configs:

```yaml
run:
  data: omni3-long-document-sdg:latest
dataset:
  path: ${art:data,path}
  # or, if consuming directly from HF Hub:
  # hf_dataset_id: nvidia/long-document-understanding-sdg-v1
```

The SDG pipeline itself stays consumer-agnostic.

## Layout

```
long-document/
├── README.md                     ← you are here
├── config/                       ← per-stage YAML defaults
│   ├── 01-seed.yaml
│   ├── 02-ocr.yaml
│   └── …
├── deployment/                   ← auto-deploy configs (--serve)
│   ├── README.md                 ← deployment-config schema
│   ├── nemotron-parse-v1.1.yaml
│   ├── gpt-oss-120b.yaml
│   ├── qwen3-vl-30b.yaml
│   └── qwen3-vl-235b.yaml
├── _recipe_config.py             ← shared YAML+OmegaConf+Pydantic loader
├── 01-seed-dataset-preparation.py
├── 02-nemotron-parse-ocr-sdg.py
├── …
└── 09-frontier-judge-sdg.py
```

The CLI plumbing lives at `src/nemotron/cli/commands/data/sdg/long_document/`.
