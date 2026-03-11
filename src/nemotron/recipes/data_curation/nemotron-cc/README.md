## Nemotron-CC Data Curation

This directory contains the recipe for curating datasets similar to the [Nemotron-CC datasets](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2). The pipeline processes raw Common Crawl snapshots through extraction, deduplication, quality classification, and synthetic data generation.

### Requirements

- [NeMo Curator](https://github.com/NVIDIA/NeMo-Curator) > 1.1.0 ([install from main](https://docs.nvidia.com/nemo/curator/latest/admin/installation.html))
- GPU(s) for steps 2a, 2b, and 3 (deduplication and classification)
- [Cargo/Rust](https://doc.rust-lang.org/cargo/getting-started/installation.html) for step 2c (building `deduplicate-text-datasets`)
- Access to an OpenAI-compatible LLM endpoint for step 4 (NVIDIA NIM, vLLM, or cloud API)

### Pipeline Overview

#### Step 1: Download, Extract, and Clean (`step_1-download_extract.py`)

A CPU-only pipeline that produces clean text from raw web data:

- Downloads Common Crawl snapshots (WARC files) and extracts text using JusText.
- Annotates each document with a language using a FastText language identification model.
- Fixes mojibake (encoding issues) via Unicode reformatting.
- **Output:** `data/cleaned_extracted/`
- **Resources:** CPU-only. We recommend each worker has at least 2GB of RAM to prevent OOM errors.

#### Step 2a: Exact Deduplication (`step_2a-exact_dedup.py`)

Exact deduplication using document hashing:

- **Phase 1 (`--identify`):** Hashes every document and identifies exact duplicates.
  - **Resources:** Requires GPU(s) for accelerated hashing. For a single snapshot (~4-10TB) extracted we tested with 8 H100 GPUs. For all of Common Crawl we recommend ~128 GPUs with 80GB VRAM per GPU.
- **Phase 2 (`--remove`):** Removes duplicate documents, keeping one copy.
  - **Resources:** CPU-only. Reads duplicate IDs from the cache directory and filters the original dataset. We recommend each worker has at-least 6GB of RAM to prevent OOM errors.
- **Output:** `data/exact_deduplicated/`.

#### Step 2b: Fuzzy Deduplication (`step_2b-fuzzy_dedup.py`)

Fuzzy deduplication using MinHash + LSH:

- **Phase 1 (`--identify`):** Identify near duplicate docs using MinHash-LSH based duplicate identification.
  - **Resources:** Requires GPU(s). For a single snapshot (~1-8TB) exact deduplicated we tested with 8 H100 GPUs.
- **Phase 2 (`--remove`):** Removes fuzzy duplicates based on connected components.
  - **Resources:** CPU-only. Reads duplicate IDs and filters the original dataset. We recommend each worker has at-least 6GB of RAM to prevent OOM errors.
- **Output:** `data/fuzzy_deduplicated/`.

#### Step 2c: Substring Deduplication (`step_2c-substring_dedup/`)

CPU-only exact substring deduplication using [Google Research's deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets) ([paper](https://arxiv.org/abs/2107.06499)). Removes duplicate substrings within and across documents using suffix arrays.

- **Resources:** CPU-only. Requires 2-3x the input dataset size in RAM and 10-15x in disk space. We recommend splitting data into 100GB chunks.
- **Output:** `data/substring_deduped/`.

See the [step_2c README](./step_2c-substring_dedup/README.md) for detailed instructions and debugging tips.

#### Step 3: Quality Classification (`step_3-quality_classification.py`)

Ensemble quality scoring and bucketing into 20 quality tiers:

- **Phase 1 (`--classify`):** Filters to English, then runs three quality classifiers in parallel:
  - [FineWebNemotronEduClassifier](https://huggingface.co/nvidia/nemocurator-fineweb-nemotron-4-edu-classifier)
  - [FineWebMixtralEduClassifier](https://huggingface.co/nvidia/nemocurator-fineweb-mixtral-edu-classifier)
  - [FastText quality filter (`fasttext-oh-eli5`)](https://huggingface.co/mlfoundations/fasttext-oh-eli5)
  - **Resources:** Requires GPU(s) for the neural classifiers. For a single snapshot we tested with 64 H100 GPUs. This scale is embarrassingly parallel so use fewer/more GPUs as needed with at least 80GB VRAM per GPU.
- **Phase 2 (`--ensemble`):** Computes token-weighted percentile thresholds from sampled classification scores, maps float scores to integer bins (0-19), takes the per-document max across classifiers as the ensemble score.
  - **Resources:** CPU-only. Reads classification results and computes thresholds and bucketing. Tested with max `fraction=0.1` on a machine with 200GB ram. For OOM errors would recommend reducing the sampling fraction.
- **Output:** `data/quality_labeling/bucketed_results/ensemble-max-int={0-19}/` partitioned by quality bucket (0 = lowest, 19 = highest).

#### Step 4: Synthetic Data Generation (`step_4-sdg.py`)

LLM-based synthetic data generation on the highest-quality documents (buckets 18 and 19). This is a CPU-only pipeline — LLM inference happens via API calls to an external endpoint (NVIDIA Integrate, or a self-hosted OAI endpoint compatible server).

Four generation tasks:

| Task | Description | Max Input / Output Tokens |
|------|-------------|---------------------------|
| `diverse_qa` | Generates diverse QA pairs (yes/no, open-ended, multiple-choice, comparison, comprehension, problem-solving) | 1000 / 600 |
| `distill` | Condenses text while preserving key information, technical terms, and examples | 2000 / 1600 |
| `extract_knowledge` | Rewrites text as textbook/Wikipedia-style passages focused on factual content | 1400 / 1400 |
| `knowledge_list` | Extracts organized bulleted lists of key facts, concepts, and statistics | 1000 / 600 |

Each task runs as an independent pipeline (preprocessing, LLM generation, postprocessing, write). When `--task all` is used, the four tasks run sequentially. They can also be run as separate processes in parallel.

- **Default model:** [`Qwen/Qwen3-30B-A3B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
- **Output:** `data/sdg_output/<task_name>/`.
- **Resources:** CPU-only for the script itself. Requires access to an LLM endpoint.
