## Nemotron-CC Data Curation

This directory contains the recipe for curating datasets similar to the [Nemotron-CC datasets](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2). The pipeline processes raw Common Crawl snapshots through extraction, deduplication, quality classification, and synthetic data generation.

### Requirements

- [NeMo Curator](https://github.com/NVIDIA/NeMo-Curator) 1.2.0 (26.04 release) or newer ([install instructions](https://docs.nvidia.com/nemo/curator/latest/admin/installation.html))
- GPU(s) for steps 2a, 2b, and 3 (deduplication and classification)
- [Cargo/Rust](https://doc.rust-lang.org/cargo/getting-started/installation.html) for step 2c (building `deduplicate-text-datasets`)
- For step 4, one of: GPU(s) to host a local inference server (default), an OpenAI-compatible endpoint (self-hosted vLLM/NIM or cloud), or an [NVIDIA Build](https://build.nvidia.com/) API key

### Pipeline Overview

| # | Script | Compute | Output |
|---|--------|---------|--------|
| 1 | [`step_1-download_extract.py`](#step-1-download-extract-and-clean) | CPU | `data/cleaned_extracted/` |
| 2a | [`step_2a-exact_dedup.py`](#step-2a-exact-deduplication) | GPU + CPU | `data/exact_deduplicated/` |
| 2b | [`step_2b-fuzzy_dedup.py`](#step-2b-fuzzy-deduplication) | GPU + CPU | `data/fuzzy_deduplicated/` |
| 2c | [`step_2c-substring_dedup/`](#step-2c-substring-deduplication) | CPU | `data/substring_deduped/` |
| 3 | [`step_3-quality_classification.py`](#step-3-quality-classification) | GPU + CPU | `data/quality_labeling/bucketed_results/` |
| 4 | [`step_4-sdg.py`](#step-4-synthetic-data-generation) | GPU or external API | `data/sdg_output/` |

---

#### Step 1: Download, Extract, and Clean

A CPU-only pipeline that produces clean text from raw web data:

- Downloads Common Crawl snapshots (WARC files) and extracts text using JusText.
- Annotates each document with a language using a FastText language identification model.
- Fixes mojibake (encoding issues) via Unicode reformatting.

**Resources:** CPU-only. Recommend at least 2GB RAM per worker to prevent OOM.

---

#### Step 2a: Exact Deduplication

Exact deduplication via document hashing. Run `--identify` then `--remove`.

| Phase | Compute | Scale tested / notes |
|-------|---------|----------------------|
| `--identify` | GPU | 8× H100 for a single snapshot (~4-10TB). ~128× 80GB GPUs recommended for full Common Crawl. |
| `--remove` | CPU, ≥6GB RAM/worker | Reads cached duplicate IDs and filters the original dataset. |

---

#### Step 2b: Fuzzy Deduplication

Fuzzy deduplication using MinHash + LSH. Run `--identify` then `--remove`.

| Phase | Compute | Scale tested / notes |
|-------|---------|----------------------|
| `--identify` | GPU | 8× H100 for a single snapshot (~1-8TB exact-deduped). |
| `--remove` | CPU, ≥6GB RAM/worker | Filters using connected-components results. |

---

#### Step 2c: Substring Deduplication

CPU-only exact substring deduplication using [Google Research's deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets) ([paper](https://arxiv.org/abs/2107.06499)). Removes duplicate substrings within and across documents using suffix arrays.

**Resources:** CPU-only. Requires 2-3× the input dataset size in RAM and 10-15× on disk. Recommend splitting data into 100GB chunks.

See the [step_2c README](./step_2c-substring_dedup/README.md) for detailed instructions and debugging tips.

---

#### Step 3: Quality Classification

Ensemble quality scoring and bucketing into 20 quality tiers. Run `--classify` then `--ensemble`.

| Phase | Compute | Scale tested / notes |
|-------|---------|----------------------|
| `--classify` | GPU, ≥80GB VRAM | Filters to English, runs three classifiers in parallel. Tested at 64× H100 per snapshot; embarrassingly parallel — scale up/down freely. |
| `--ensemble` | CPU | Computes token-weighted percentile thresholds and per-document max across classifiers. Tested at `fraction=0.1` on 200GB RAM; reduce sampling fraction if OOM. |

Classifiers used:
- [FineWebNemotronEduClassifier](https://huggingface.co/nvidia/nemocurator-fineweb-nemotron-4-edu-classifier)
- [FineWebMixtralEduClassifier](https://huggingface.co/nvidia/nemocurator-fineweb-mixtral-edu-classifier)
- [FastText quality filter (`fasttext-oh-eli5`)](https://huggingface.co/mlfoundations/fasttext-oh-eli5)

**Output layout:** `data/quality_labeling/bucketed_results/ensemble-max-int={0-19}/` partitioned by bucket (0 = lowest, 19 = highest).

---

#### Step 4: Synthetic Data Generation

LLM-based synthetic data generation on the highest-quality documents (buckets 18 and 19). Four generation tasks:

| Task | Description | Max Input / Output Tokens |
|------|-------------|---------------------------|
| `diverse_qa` | Generates diverse QA pairs (yes/no, open-ended, multiple-choice, comparison, comprehension, problem-solving) | 1000 / 600 |
| `distill` | Condenses text while preserving key information, technical terms, and examples | 2000 / 1600 |
| `extract_knowledge` | Rewrites text as textbook/Wikipedia-style passages focused on factual content | 1400 / 1400 |
| `knowledge_list` | Extracts organized bulleted lists of key facts, concepts, and statistics | 1000 / 600 |

Each task is an independent pipeline (preprocess → LLM generate → postprocess → write). `--task all` runs the four sequentially; they can also be launched as parallel processes.

**LLM backends** — pick one:

| Backend | How to select | Notes |
|---------|---------------|-------|
| Local inference server (default) | (default) | Spins up a Ray Serve + vLLM deployment of `--model-name` on the local GPU cluster. No API key. Tune with `--tensor-parallel-size`, `--min-replicas`, `--max-replicas`; bump `--max-concurrent-requests` (try 256–512) if GPU utilization is low. |
| Existing OpenAI-compatible endpoint | `--no-serve-model --base-url <url>` | Self-hosted vLLM/TRT-LLM/NIM or any OpenAI-compatible cloud provider. `--api-key` forwarded if set. |
| [NVIDIA Build](https://build.nvidia.com/) | `--no-serve-model` | Uses the default `--base-url`. Requires `--api-key` (or `NVIDIA_API_KEY`). Default `--model-name` is not on NVIDIA Build — set `--model-name` (and `--tokenizer`) to a model that is. |

> **Note on `--tokenizer`:** The tokenizer is loaded via Hugging Face `AutoTokenizer`, so `--tokenizer` must be a Hugging Face repo id (or local path to HF tokenizer files), regardless of which backend you pick. If `--tokenizer` is not set, it defaults to `--model-name`, which in some cases is not a valid HF tokenizer path — for example `--model-name meta/llama-3.3-70b-instruct` needs `--tokenizer meta-llama/Llama-3.3-70B-Instruct` set explicitly.

**Defaults:**

- **Default model:** [`Qwen/Qwen3-30B-A3B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507).
- **Resources:** With `--serve-model`, GPU(s) for vLLM. Otherwise CPU-only; just needs network access to the chosen endpoint.
