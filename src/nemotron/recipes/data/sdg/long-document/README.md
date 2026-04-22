# Long-Document SDG Pipeline

This directory scaffolds the long-document synthetic data generation (SDG) pipeline described in `designs/long-document-sdg-pipeline.md`.

The intended steady-state shape matches `src/nemotron/recipes/data/curation/nemotron-cc/`:

- plain `uv run`-able scripts
- no package wiring
- no Nemotron CLI wiring
- model serving documented in this README rather than shipped as framework code

## Status

The upstream recipe bodies currently live in a private GitLab project:

- `https://gitlab-master.nvidia.com/sdg-research/sdg-share/-/tree/main/sdgs/long-document/public_recipes/`

That source was not accessible from this environment, so the nine scripts in this directory are scaffold stubs with the expected filenames, CLI flags, and release-time TODOs. Port the upstream bodies into these files before release.

## Pipeline overview

```text
01 seed ──┬─ 02 ocr ─── 03 text-qa ─────┐
          ├─ 04 classify ── 05 visual-qa ┤
          ├─ 06 single-page-qa ──────────┤── 09 judge
          ├─ 07 windowed-qa ─────────────┤
          └─ 08 whole-doc-qa ────────────┘
```

### Seed outputs

`01-seed-dataset-preparation.py` is expected to produce three parquet files:

| File | Granularity | Consumed by |
|---|---|---|
| `seed_per_page.parquet` | one row per page | 02, 03, 04, 05, 06 |
| `seed_windowed.parquet` | one row per sliding window of pages | 07 |
| `seed_whole_document.parquet` | one row per document | 08 |

All seed files share a `png_images_base64` column containing a JSON array of base64-encoded PNG strings.

## Scripts

| Script | Purpose | Model / endpoint |
|---|---|---|
| `01-seed-dataset-preparation.py` | Build per-page, windowed, and whole-document seed parquet files | CPU-only |
| `02-nemotron-parse-ocr-sdg.py` | OCR extraction with text and bounding-box metadata | `nvidia/NVIDIA-Nemotron-Parse-v1.1` |
| `03-text-qa-sdg.py` | Text QA from OCR-transcribed text | `openai/gpt-oss-120b` |
| `04-page-classification-sdg.py` | Page-type and reasoning-complexity classification | `Qwen/Qwen3-VL-30B-A3B-Instruct` |
| `05-visual-qa-sdg.py` | Visual QA generation | `Qwen/Qwen3-VL-235B-A22B-Thinking-FP8` |
| `06-single-page-qa-sdg.py` | Anchored single-page QA generation | `Qwen/Qwen3-VL-235B-A22B-Thinking-FP8` |
| `07-multi-page-windowed-qa-sdg.py` | Sliding-window multi-page QA generation | `Qwen/Qwen3-VL-235B-A22B-Thinking-FP8` |
| `08-whole-document-qa-sdg.py` | Whole-document cross-page QA generation | `Qwen/Qwen3-VL-235B-A22B-Thinking-FP8` |
| `09-frontier-judge-sdg.py` | LLM-as-a-judge scoring for QA outputs | any OpenAI-compatible frontier endpoint |

## Serving philosophy

This recipe does **not** ship a serve script, endpoint registry, or `ServeArtifact`.

Operators launch the model endpoint themselves, note the host and port, and pass that value to the recipe via:

- `--vllm-endpoint` for producer steps
- `--frontier-endpoint` for the judge step

That is the same cookbook style used by `data/curation/nemotron-cc/step_4-sdg.py`.

## Model launch recipes

The commands below are copy-pasteable starting points. Adjust partition names, mount points, GPU counts, and token lengths to match your environment.

### Qwen3-VL-235B-A22B-Thinking-FP8

Used by: `05`, `06`, `07`, `08`

#### Local docker

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

#### Slurm + Pyxis

```bash
srun --partition=interactive \
     --nodes=1 \
     --ntasks=1 \
     --gres=gpu:4 \
     --time=24:00:00 \
     --container-image=vllm/vllm-openai:latest \
     --container-mounts=/lustre:/lustre \
     vllm serve Qwen/Qwen3-VL-235B-A22B-Thinking-FP8 \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size 4 \
        --max-model-len 50000 \
        --gpu-memory-utilization 0.90 \
        --reasoning-parser deepseek_r1 \
        --limit-mm-per-prompt '{"video": 0}' \
        --trust-remote-code
```

### Qwen3-VL-30B-A3B-Instruct

Used by: `04`

#### Local docker

```bash
docker run --gpus all -p 8000:8000 \
    -e HF_TOKEN=$HF_TOKEN \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"video": 0}' \
    --trust-remote-code
```

#### Slurm + Pyxis

```bash
srun --partition=interactive \
     --nodes=1 \
     --ntasks=1 \
     --gres=gpu:1 \
     --time=24:00:00 \
     --container-image=vllm/vllm-openai:latest \
     --container-mounts=/lustre:/lustre \
     vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size 1 \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.90 \
        --limit-mm-per-prompt '{"video": 0}' \
        --trust-remote-code
```

### gpt-oss-120b

Used by: `03`

#### Local docker

```bash
docker run --gpus all -p 8000:8000 \
    -e HF_TOKEN=$HF_TOKEN \
    vllm/vllm-openai:latest \
    --model openai/gpt-oss-120b \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90
```

#### Slurm + Pyxis

```bash
srun --partition=interactive \
     --nodes=1 \
     --ntasks=1 \
     --gres=gpu:4 \
     --time=24:00:00 \
     --container-image=vllm/vllm-openai:latest \
     --container-mounts=/lustre:/lustre \
     vllm serve openai/gpt-oss-120b \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size 4 \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.90
```

### NVIDIA-Nemotron-Parse-v1.1

Used by: `02`

#### Local docker

```bash
docker run --gpus all -p 8000:8000 \
    -e HF_TOKEN=$HF_TOKEN \
    vllm/vllm-openai:latest \
    --model nvidia/NVIDIA-Nemotron-Parse-v1.1 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"video": 0}' \
    --trust-remote-code
```

#### Slurm + Pyxis

```bash
srun --partition=interactive \
     --nodes=1 \
     --ntasks=1 \
     --gres=gpu:1 \
     --time=24:00:00 \
     --container-image=vllm/vllm-openai:latest \
     --container-mounts=/lustre:/lustre \
     vllm serve nvidia/NVIDIA-Nemotron-Parse-v1.1 \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size 1 \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.90 \
        --limit-mm-per-prompt '{"video": 0}' \
        --trust-remote-code
```

### Frontier judge endpoint

`09-frontier-judge-sdg.py` accepts any OpenAI-compatible endpoint via `--frontier-endpoint`. This repository does not prescribe a default model; use the frontier model approved for your environment.

## Invocation examples

### 1. Prepare seeds

```bash
uv run 01-seed-dataset-preparation.py \
    --output-dir ./seed_data \
    --num-docs 10000
```

### 2. Launch the required model endpoint(s)

Pick the model that matches the recipe you plan to run:

- OCR: `nvidia/NVIDIA-Nemotron-Parse-v1.1`
- text QA: `openai/gpt-oss-120b`
- page classification: `Qwen/Qwen3-VL-30B-A3B-Instruct`
- visual, single-page, windowed, whole-document QA: `Qwen/Qwen3-VL-235B-A22B-Thinking-FP8`
- judge: your own frontier endpoint

### 3. Run producer steps

```bash
uv run 02-nemotron-parse-ocr-sdg.py \
    --vllm-endpoint http://localhost:8000/v1 \
    --seed-path seed_data/seed_per_page.parquet \
    --output-dir ./ocr_output
```

```bash
uv run 03-text-qa-sdg.py \
    --vllm-endpoint http://localhost:8001/v1 \
    --seed-path seed_data/seed_per_page.parquet \
    --num-records 50000 \
    --output-dir ./text_qa_output
```

```bash
uv run 04-page-classification-sdg.py \
    --vllm-endpoint http://localhost:8002/v1 \
    --seed-path seed_data/seed_per_page.parquet \
    --num-records 50000 \
    --output-dir ./page_classification_output
```

```bash
uv run 06-single-page-qa-sdg.py \
    --vllm-endpoint http://compute-node-0001:8000/v1 \
    --seed-path seed_data/seed_per_page.parquet \
    --num-records 100000 \
    --output-dir ./single_page_qa_output
```

```bash
uv run 07-multi-page-windowed-qa-sdg.py \
    --vllm-endpoint http://compute-node-0002:8000/v1 \
    --seed-path seed_data/seed_windowed.parquet \
    --num-records 25000 \
    --output-dir ./windowed_qa_output
```

```bash
uv run 08-whole-document-qa-sdg.py \
    --vllm-endpoint http://compute-node-0003:8000/v1 \
    --seed-path seed_data/seed_whole_document.parquet \
    --num-records 10000 \
    --output-dir ./whole_doc_qa_output
```

### 4. Judge a QA output

```bash
uv run 09-frontier-judge-sdg.py \
    --frontier-endpoint http://frontier-host:8000/v1 \
    --input ./single_page_qa_output/generated.parquet \
    --output ./judged_single_page_qa.parquet
```

## Publishing the output

After running the pipeline, publish the resulting parquet outputs through one of the following paths.

### Public path: Hugging Face Hub

```bash
export HF_TOKEN=...
hf auth login --token "$HF_TOKEN"

hf repo create nvidia/long-document-understanding-sdg-v1 \
    --repo-type dataset \
    --private=false

hf upload nvidia/long-document-understanding-sdg-v1 \
    ./published_dataset \
    --repo-type dataset
```

Recommended contents for `./published_dataset`:

- seed parquet files if you want to expose the starting point
- generated parquet outputs from the QA stages
- judged parquet outputs
- a dataset card describing source PDFs, filtering, and model variants used

### Private path: internal storage plus artifact registration

Copy the final dataset bundle to internal storage, for example:

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

If not, register it manually through your standard W&B flow, for example with `wandb.log_artifact`.

## Consumption

Downstream training recipes should consume the published dataset by artifact or dataset ID. For Omni-style configs that typically looks like:

```yaml
run:
  data: omni3-long-document-sdg:latest
dataset:
  path: ${art:data,path}
  # or, if consuming directly from HF Hub:
  # hf_dataset_id: nvidia/long-document-understanding-sdg-v1
```

The SDG pipeline itself stays consumer-agnostic.
