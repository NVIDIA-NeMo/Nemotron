# Embedding Model Fine-Tuning Recipe

A complete 6-stage pipeline for fine-tuning and deploying embedding models on domain-specific data using synthetic data generation.

> **Full documentation**: See the [Embedding Recipe Guide](../../../docs/nemotron/embed/README.md) for detailed configuration, troubleshooting, and best practices.

## Overview

This recipe fine-tunes NVIDIA's [Llama-Nemotron-Embed-1B-v2](https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2) embedding model on your own domain data. By the end of this pipeline, you'll have a domain-adapted embedding model that excels at retrieving relevant documents from your specific corpus.

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           YOUR DOCUMENT CORPUS                              │
│                    (Text files: .txt, .md, etc.)                            │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              STAGE 0: SYNTHETIC DATA GENERATION (retriever-sdg)             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────────────────┐ │
│  │ Document Chunks │ →  │  LLM Generation │ →  │ Q&A Pairs + Evaluations  │ │
│  │                 │    │  (NVIDIA API)   │    │                          │ │
│  └─────────────────┘    └─────────────────┘    └──────────────────────────┘ │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: TRAINING DATA PREPARATION                       │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────────────────┐ │
│  │ Train/Val/Test  │ →  │  Hard Negative  │ →  │   Multi-hop Unrolling    │ │
│  │     Split       │    │     Mining      │    │                          │ │
│  └─────────────────┘    └─────────────────┘    └──────────────────────────┘ │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STAGE 2: MODEL FINE-TUNING (Automodel)                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  Contrastive Learning: Query → Positive Documents vs Hard Negatives     ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STAGE 3: EVALUATION (BEIR)                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │     Compare Base vs Fine-tuned Model on IR Metrics (nDCG, Recall)       ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 4: EXPORT (ONNX/TensorRT)                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │     Export Model to ONNX and TensorRT for Optimized Inference           ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 5: DEPLOY (NIM)                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │     Launch NIM Container with Custom Model for Production Inference     ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

| Stage | Command | Description | Output |
|-------|---------|-------------|--------|
| [Stage 0: SDG](./stage0_sdg/) | `nemotron embed sdg` | Validate corpus, generate synthetic Q&A pairs from documents | Q&A pairs with quality scores |
| [Stage 1: Data Prep](./stage1_data_prep/) | `nemotron embed prep` | Convert, mine hard negatives, unroll | Training-ready data |
| [Stage 2: Finetune](./stage2_finetune/) | `nemotron embed finetune` | Fine-tune embedding model | Model checkpoint |
| [Stage 3: Eval](./stage3_eval/) | `nemotron embed eval` | Evaluate on retrieval metrics | Metrics comparison |
| [Stage 4: Export](./stage4_export/) | `nemotron embed export` | Export to ONNX/TensorRT | Optimized inference models |
| [Stage 5: Deploy](./stage5_deploy/) | `nemotron embed deploy` | Deploy NIM with custom model | Running inference service |

## Quick Start

### 1. Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/NVIDIA/Nemotron.git
cd Nemotron
uv sync --all-extras
```

### 2. Set Environment

```bash
export LD_LIBRARY_PATH=""
export NVIDIA_API_KEY=nvapi-your_key_here
```

### 3. Run the Pipeline

```bash
# Stage 0: Generate synthetic Q&A pairs (sample corpus auto-downloaded from HuggingFace)
nemotron embed sdg -c default

# Or use your own documents:
# nemotron embed sdg -c default corpus_dir=/path/to/your/docs

# Stage 1: Prepare training data (convert, mine hard negatives, unroll)
nemotron embed prep -c default

# Stage 2: Fine-tune the embedding model
nemotron embed finetune -c default

# Stage 3: Evaluate base vs fine-tuned model
nemotron embed eval -c default

# Stage 4: Export to ONNX/TensorRT for deployment
nemotron embed export -c default

# Stage 5: Deploy NIM with custom model
nemotron embed deploy -c default
```

### Using NVIDIA's Pre-Generated Dataset

NVIDIA provides a ready-to-use synthetic retrieval dataset on HuggingFace: [Retrieval-Synthetic-NVDocs-v1](https://huggingface.co/datasets/nvidia/Retrieval-Synthetic-NVDocs-v1). This dataset was generated from NVIDIA's publicly available content using the same SDG pipeline (Stage 0) in this recipe, and contains ~15K documents with 105K+ question-answer pairs across multiple reasoning types.

If you want to fine-tune on NVIDIA-related content, you can **skip Stage 0 entirely** and start directly from Stage 1:

```bash
# Download the pre-generated dataset
python -c "
from datasets import load_dataset
ds = load_dataset('nvidia/Retrieval-Synthetic-NVDocs-v1', split='train')
ds.to_json('./output/embed/stage0_sdg/nv_docs_sdg.json')
"

# Start from Stage 1 using the downloaded data
nemotron embed prep -c default sdg_input_path=./output/embed/stage0_sdg

# Continue with the rest of the pipeline
nemotron embed finetune -c default
nemotron embed eval -c default
```

This is useful for quickly getting started or benchmarking the pipeline without needing an NVIDIA API key.

## Prerequisites

- **GPU**: NVIDIA GPU with at least 80GB VRAM (e.g., A100, H100) — Stage 0 uses NVIDIA API (no GPU required)
- **Python**: 3.12 or later
- **UV**: Package manager ([installation](https://docs.astral.sh/uv/))
- **NVIDIA API Key**: Required for synthetic data generation — sign up at [build.nvidia.com](https://build.nvidia.com)

## Stage Documentation

- [Stage 0: SDG](./stage0_sdg/) — Synthetic data generation
- [Stage 1: Data Prep](./stage1_data_prep/) — Training data preparation
- [Stage 2: Finetune](./stage2_finetune/) — Model fine-tuning
- [Stage 3: Eval](./stage3_eval/) — Evaluation
- [Stage 4: Export](./stage4_export/) — ONNX/TensorRT export
- [Stage 5: Deploy](./stage5_deploy/) — NIM deployment

## Further Reading

- [Full Documentation](../../../docs/nemotron/embed/README.md) — Detailed configuration, troubleshooting, and best practices
- [Retrieval-Synthetic-NVDocs-v1](https://huggingface.co/datasets/nvidia/Retrieval-Synthetic-NVDocs-v1) — Pre-generated synthetic retrieval dataset on NVIDIA content
- [Recipes Overview](../README.md) — General information about Nemotron recipes
