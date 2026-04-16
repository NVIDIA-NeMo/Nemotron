# Reranking Model Fine-Tuning Recipe

A complete 6-stage pipeline for fine-tuning and deploying cross-encoder reranking models on domain-specific data using synthetic data generation.

## Overview

This recipe fine-tunes NVIDIA's [Llama-Nemotron-Rerank-1B-v2](https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2) cross-encoder reranking model on your own domain data. By the end of this pipeline, you'll have a domain-adapted reranker that improves retrieval precision by re-scoring candidate documents returned by a first-stage retriever.

### Why Fine-Tune Reranking Models?

In a typical retrieval pipeline, a fast embedding model retrieves a broad set of candidate documents, then a cross-encoder reranker re-scores each query–document pair to improve ranking quality. Fine-tuning the reranker adapts it to:

- Understand domain-specific relevance signals and terminology
- Better discriminate between subtly relevant and irrelevant documents
- Improve precision at the top of the ranked list (nDCG@k) on your specific corpus

### Embedding vs. Reranking

| Aspect | Embedding Model | Reranking Model |
|--------|----------------|-----------------|
| **Architecture** | Bi-encoder (encodes query and document separately) | Cross-encoder (encodes query and document together) |
| **Speed** | Fast (single encoding per document, offline indexable) | Slower (joint encoding per query–document pair) |
| **Accuracy** | Good for broad recall | Higher precision at top ranks |
| **Role** | First-stage retrieval | Second-stage re-ranking |

The two models are complementary — use the embedding model to cast a wide net, then the reranker to sort the catch.

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
│              STAGE 2: MODEL FINE-TUNING (Cross-Encoder, Automodel)          │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  Classification Loss: Query+Passage → Relevance Score                  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STAGE 3: EVALUATION (BEIR + Reranking)                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │   Dense Retrieval → Re-rank → Measure nDCG@k Improvement               ││
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
│  │     Launch NIM Container with Custom Model for Ranking API              ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

| Stage | Command | Description | Output |
|-------|---------|-------------|--------|
| [Stage 0: SDG](./stage0_sdg/) | `nemotron rerank sdg` | Validate corpus, generate synthetic Q&A pairs from documents | Q&A pairs with quality scores |
| [Stage 1: Data Prep](./stage1_prep/) | `nemotron rerank prep` | Convert, mine hard negatives, unroll | Training-ready data |
| [Stage 2: Finetune](./stage2_finetune/) | `nemotron rerank finetune` | Fine-tune cross-encoder reranking model | Model checkpoint |
| [Stage 3: Eval](./stage3_eval/) | `nemotron rerank eval` | Evaluate reranking improvement over first-stage retrieval | Metrics comparison |
| [Stage 4: Export](./stage4_export/) | `nemotron rerank export` | Export to ONNX/TensorRT | Optimized inference models |
| [Stage 5: Deploy](./stage5_deploy/) | `nemotron rerank deploy` | Deploy NIM ranking service | Running inference service |

## Installation

### 1. Install UV Package Manager

This project **requires [UV](https://docs.astral.sh/uv/)** as its package manager. UV automatically creates and manages a virtual environment under the repository root, and each pipeline stage uses its own isolated environment as well. **Do not use `pip install`** — the project relies on UV workspaces and per-stage dependency isolation.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Install Nemotron

```bash
# Clone the repository
git clone https://github.com/NVIDIA/Nemotron.git
cd Nemotron

# UV creates a virtual environment at .venv/ and installs all dependencies
uv sync --all-extras
```

### 3. Get Your NVIDIA API Key

The SDG stage (Stage 0) uses NVIDIA's hosted LLM APIs for synthetic data generation.

1. Sign up at [build.nvidia.com](https://build.nvidia.com)
2. Create an API key
3. Set the environment variable:

```bash
export NVIDIA_API_KEY=nvapi-your_key_here
```

### 4. Configure Execution Profiles (Optional)

For Docker or Slurm execution, create `env.toml` in the **repository root directory**.

**Minimal configuration (local execution only):**
```toml
[wandb]
project = "my-reranking-project"
entity = "my-username"
```

**Full configuration with Docker and Slurm support:**
See the [Execution Profiles](#execution-profiles) section below.

## Preparing Your Corpus

### Supported Formats

- Any text files, default: .txt, .md, .text, and files with no extension
- Documents should be UTF-8 encoded
- Files are processed recursively from the corpus directory

### Corpus Size Recommendations

| Corpus Size | Documents | Expected Results |
|-------------|-----------|------------------|
| **Minimum** | 50-100 docs (~50K tokens) | Basic domain adaptation |
| **Recommended** | 500+ docs | Good domain coverage |
| **Optimal** | 1000+ docs | Best performance |

### Document Organization

Organize your documents in a directory structure:

```bash
data/corpus/
├── doc1.txt
├── doc2.md
└── subdirectory/
    └── doc3.txt
```

All files matching the `file_extensions` config (default: `.txt,.md`) will be processed recursively.

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with at least 80GB VRAM (e.g., A100, H100)
  - Stage 0 uses NVIDIA API (no GPU required)
  - Stages 1-5: Require GPU for model inference and training
- **CPU**: Modern multi-core processor (16+ cores recommended)
- **Memory**: 128GB+ RAM recommended
- **Storage**: ~50GB free disk space for outputs, models, and containers

### Software Requirements

- **Python**: 3.12 or later
- **UV**: Package manager (installation instructions above)
- **NVIDIA API Key**: Required for synthetic data generation
- **NVIDIA GPU Drivers**: Latest drivers for your GPU
- **Docker** (optional): For containerized execution or NIM deployment
- **Slurm** (optional): For cluster execution

### Expected Runtime & Resources

| Stage | GPU VRAM | CPU | Notes |
|-------|----------|-----|-------|
| Stage 0 (SDG) | N/A | 8+ cores | Uses API (no local GPU); runtime varies by dataset size |
| Stage 1 (Data Prep) | 40GB | 16+ cores | Hard negative mining on GPU; runtime varies by dataset size |
| Stage 2 (Finetune) | 80GB | 16+ cores | Runtime varies by dataset size and epochs |
| Stage 3 (Eval) | 40GB | 8+ cores | Dense retrieval + reranking; runtime varies by dataset size |
| Stage 4 (Export) | 40GB | 8+ cores | TensorRT export requires NGC container |
| Stage 5 (Deploy) | 40GB | 4+ cores | NIM container initialization |

### LLM API Usage (Stage 0)

Stage 0 uses LLM APIs for synthetic data generation. By default, it uses NVIDIA's hosted LLMs:

- **Default provider**: NVIDIA API (free tier available at [build.nvidia.com](https://build.nvidia.com))
- **Default model**: `nvidia/nemotron-3-nano-30b-a3b` (fast, reliable for structured generation)
- **Usage**: ~4 API calls per document (artifact extraction, QA generation, dedup, quality eval)
- **Cost**: Free tier has rate limits; contact NVIDIA for production usage

## Quick Start

### Local Execution

```bash
# Set environment (important for CUDA compatibility)
export LD_LIBRARY_PATH=""
export NVIDIA_API_KEY=nvapi-your_key_here

# Stage 0: Generate synthetic Q&A pairs from your documents
nemotron rerank sdg -c default corpus_dir=/path/to/your/docs

# Stage 1: Prepare training data (convert, mine hard negatives, unroll)
nemotron rerank prep -c default

# Stage 2: Fine-tune the cross-encoder reranking model
nemotron rerank finetune -c default

# Stage 3: Evaluate base vs fine-tuned reranker
nemotron rerank eval -c default

# Stage 4: Export to ONNX/TensorRT for deployment
nemotron rerank export -c default

# Stage 5: Deploy NIM with custom model
nemotron rerank deploy -c default

# Optional: Verify NIM accuracy matches checkpoint
nemotron rerank eval -c default eval_nim=true eval_base=false
```

### Preview Commands (Dry Run)

```bash
# See what would be executed without running
nemotron rerank finetune -c default --dry-run
```

## Pipeline Flexibility

Stages are designed to run sequentially, but you can start from any stage if you have the required inputs:

| Start From | Requirement | Use Case |
|------------|-------------|----------|
| **Stage 0** | Document corpus | Full pipeline from scratch |
| **Stage 1** | Q&A pairs (JSON) | Skip SDG if you have labeled data or use [NVIDIA's pre-generated dataset](#using-nvidias-pre-generated-dataset) |
| **Stage 2** | Training data (Automodel format) | Skip data prep if data is ready |
| **Stage 3** | Model checkpoint | Evaluate existing checkpoint |
| **Stage 4** | Model checkpoint | Export existing model |
| **Stage 5** | Exported model (ONNX/TensorRT) | Deploy existing model |

See individual stage configs for input format requirements.

### Using NVIDIA's Pre-Generated Dataset

NVIDIA provides a ready-to-use synthetic retrieval dataset on Hugging Face: [Retrieval-Synthetic-NVDocs-v1](https://huggingface.co/datasets/nvidia/Retrieval-Synthetic-NVDocs-v1). This dataset was generated from NVIDIA's publicly available content using the same SDG pipeline (Stage 0) in this recipe, and contains ~15K documents with 105K+ question-answer pairs across multiple reasoning types.

If you want to fine-tune a reranking model on NVIDIA-related content, you can **skip Stage 0 entirely** and start directly from Stage 1:

```bash
# Download the pre-generated dataset
python -c "
from datasets import load_dataset
ds = load_dataset('nvidia/Retrieval-Synthetic-NVDocs-v1', split='train')
ds.to_json('./output/rerank/stage0_sdg/nv_docs_sdg.json')
"

# Start from Stage 1 (data preparation) using the downloaded data
nemotron rerank prep -c default sdg_input_path=./output/rerank/stage0_sdg

# Continue with the rest of the pipeline
nemotron rerank finetune -c default
nemotron rerank eval -c default
```

## Execution Modes

The rerank recipe supports multiple execution modes for flexibility between local development and production cluster runs.

### Local Execution (Default)

Run directly on your local machine with GPU:

```bash
nemotron rerank finetune -c default
nemotron rerank eval -c default
```

### Docker Execution

Run inside a Docker container with GPU passthrough using `--run local-docker`:

```bash
# Runs the command inside a Docker container with GPU access
nemotron rerank finetune -c default --run local-docker

# All stages support Docker execution
nemotron rerank sdg -c default --run local-docker
nemotron rerank prep -c default --run local-docker
nemotron rerank eval -c default --run local-docker
```

> **Note**: Requires `local-docker` profile in `env.toml` (see [Execution Profiles](#execution-profiles) below)

### Slurm Batch Execution

Submit jobs to a Slurm cluster for production workloads:

```bash
# Attached execution (waits for completion, streams logs via SSH)
nemotron rerank finetune -c default --run my-cluster

# Detached execution (submits job and exits immediately)
nemotron rerank finetune -c default --batch my-cluster

# Run full pipeline on cluster
nemotron rerank sdg -c default --batch my-cluster
nemotron rerank prep -c default --batch my-cluster
nemotron rerank finetune -c default --batch my-cluster
nemotron rerank eval -c default --batch my-cluster
```

### Execution Profiles

Execution profiles are defined in `env.toml` in the **repository root directory**.

**Example `env.toml` for local and cluster execution:**

```toml
# Weights & Biases configuration (optional but recommended)
[wandb]
project = "my-reranking-project"
entity = "my-team"

# Local Docker execution profile
[local-docker]
executor = "docker"
container_image = "nvcr.io/nvidia/pytorch:25.12-py3"
runtime = "nvidia"  # Enable GPU passthrough
ipc_mode = "host"
shm_size = "16g"
mounts = [
    "./data:/workspace/data",
    "./output:/workspace/output"
]

# Slurm cluster execution profile
[my-cluster]
executor = "slurm"
account = "my-account"
partition = "interactive"
batch_partition = "batch"
container_image = "nvcr.io/nvidia/pytorch:25.12-py3"
tunnel = "ssh"
host = "cluster.example.com"
user = "username"
remote_job_dir = "/shared/path/to/jobs"
mounts = ["/shared:/shared"]
```

### Runtime Overrides

Override execution settings on the command line:

```bash
# Use more GPUs
nemotron rerank finetune -c default --run my-cluster run.env.gpus_per_node=4

# Use different partition
nemotron rerank finetune -c default --batch my-cluster run.env.partition=batch

# Override time limit
nemotron rerank finetune -c default --batch my-cluster run.env.time=08:00:00
```

### Interactive Debugging

Stage files to the cluster for interactive debugging:

```bash
# Stage files without executing
nemotron rerank finetune -c default --run my-cluster --stage

# Then SSH to cluster and run manually
ssh cluster.example.com
cd /path/to/staged/files
./run.sh
```

## Configuration

Each stage has a `config/` directory with YAML configuration files.

| File | Purpose |
|------|---------|
| `default.yaml` | Production-ready configuration |

### Key Configuration Options

**Stage 0: SDG**
```yaml
corpus_id: my_corpus           # Identifier for your corpus
corpus_dir: ./data/corpus      # Path to your documents
file_extensions: ".txt,.md"    # File types to process
output_dir: ./output/rerank/stage0_sdg  # Path to save the generated data
artifact_extraction_model: nvidia/nemotron-3-nano-30b-a3b  # LLM for document artifacts extraction
qa_generation_model: nvidia/nemotron-3-nano-30b-a3b        # LLM for QA generation
quality_judge_model: nvidia/nemotron-3-nano-30b-a3b        # LLM for QA quality evaluation
```

**Stage 1: Data Prep**
```yaml
base_model: nvidia/llama-nemotron-embed-1b-v2  # Embedding model for hard negative mining
quality_threshold: 7.0         # Minimum Q&A quality score (0-10)
hard_negatives_to_mine: 5      # Number of hard negatives per query
query_max_length: 512          # Max query tokens
passage_max_length: 512        # Max passage tokens
train_ratio: 0.8               # Training data split (80%)
val_ratio: 0                   # Validation split (0% — maximizes train/test for small data)
test_ratio: 0.2                # Test split (20%)
```

**Stage 2: Finetune**
```yaml
base_model: nvidia/llama-nemotron-rerank-1b-v2
num_epochs: 3
global_batch_size: 128
learning_rate: 3.0e-6
lr_warmup_steps: 100
lr_decay_style: cosine
weight_decay: 0.01
rerank_max_length: 512         # Max tokens for concatenated query+passage
prompt_template: "question:{query} \n \n passage:{passage}"
train_n_passages: 5            # 1 positive + 4 hard negatives
num_labels: 1
temperature: 1.0
```

> **Warning — Overfitting risk**: The default `num_epochs: 3` is set for the small example dataset shipped with this recipe, where fewer epochs may not produce a visible training signal. For most real-world datasets, **1–2 epochs is sufficient** and 3 epochs carries a high risk of overfitting. Lower this value when working with your own data (e.g., `nemotron rerank finetune -c default num_epochs=1`).

**Stage 3: Eval**
```yaml
base_model: nvidia/llama-nemotron-rerank-1b-v2   # Base reranker for comparison
retrieval_model: nvidia/llama-nemotron-embed-1b-v2  # First-stage retriever
k_values: [1, 5, 10, 100]     # K values for nDCG@k, Recall@k
top_k: 100                    # Number of first-stage candidates to re-rank
max_length: 512                # Max sequence length
eval_base: true                # Evaluate base reranker
eval_finetuned: true           # Evaluate fine-tuned reranker
eval_nim: false                # Evaluate NIM endpoint
```

**Stage 4: Export**
```yaml
model_path: ./output/rerank/stage2_finetune/checkpoints/LATEST/model/consolidated
export_to_trt: false           # Export to TensorRT (requires nemo:25.07+ container)
quant_cfg: null                # Quantization: null, "fp8", "int8_sq"
trt_opt_batch: 16              # Optimal batch size for TRT
trt_opt_seq_len: 256           # Optimal sequence length for TRT
```

**Stage 5: Deploy**
```yaml
nim_image: nvcr.io/nim/nvidia/llama-nemotron-rerank-1b-v2:1.10.0
model_dir: ./output/rerank/stage4_export/onnx  # Path to exported model
host_port: 8000                # Port for NIM ranking API
detach: false                  # Run in background
```

### Overriding Configuration

Override config values on the command line:

```bash
# Override training epochs
nemotron rerank finetune -c default num_epochs=5

# Override learning rate
nemotron rerank finetune -c default learning_rate=1e-5

# Override multiple values
nemotron rerank finetune -c default num_epochs=2 learning_rate=1e-5

# Force specific attention implementation
nemotron rerank finetune -c default attn_implementation=flash_attention_2
```

## CLI Commands

### Workspace Info

```bash
# Display workflow overview
nemotron rerank info
```

### Data

```bash
# Generate synthetic Q&A pairs from documents
nemotron rerank sdg -c default corpus_dir=/path/to/docs

# Prepare training data (convert, mine, unroll)
nemotron rerank prep -c default sdg_input_path=/path/to/sdg
```

### Training

```bash
# Fine-tune the cross-encoder reranking model
nemotron rerank finetune -c default train_data_path=/path/to/data
```

### Evaluation

```bash
# Evaluate base and fine-tuned rerankers
nemotron rerank eval -c default finetuned_model_path=/path/to/checkpoint
```

### Export

```bash
# Export model to ONNX
nemotron rerank export -c default model_path=/path/to/checkpoint

# Export to ONNX only (skip TensorRT)
nemotron rerank export -c default export_to_trt=false

# Export with FP8 quantization
nemotron rerank export -c default quant_cfg=fp8
```

### Deploy

```bash
# Deploy NIM with custom ONNX model (foreground)
nemotron rerank deploy -c default

# Deploy in background (detached mode)
nemotron rerank deploy -c default detach=true

# Deploy with custom model directory
nemotron rerank deploy -c default model_dir=./output/rerank/stage4_export/onnx

# Stop the NIM container
docker stop nemotron-rerank-nim
```

### Verify NIM Accuracy

```bash
# Evaluate NIM endpoint against fine-tuned checkpoint
nemotron rerank eval -c default eval_nim=true eval_base=false

# The output will show if NIM metrics match the checkpoint
# ok       indicates metrics match within tolerance (0.03 for @1, 0.01 for @5+)
# MISMATCH indicates potential accuracy loss beyond ONNX/TensorRT conversion noise
```

### Test the Deployed Service

```bash
curl -X POST http://localhost:8000/v1/ranking \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/llama-nemotron-rerank-1b-v2",
    "query": {"text": "what is AI?"},
    "passages": [
      {"text": "AI is artificial intelligence"},
      {"text": "The weather is sunny today"}
    ]
  }'
```

## Output Structure

After running the full pipeline:

```
output/rerank/
├── stage0_sdg/                    # Synthetic Q&A pairs
│   └── generated_batch*.json
├── stage1_prep/                   # Training-ready data
│   ├── train.json                 # Original training data
│   ├── train_mined.automodel.json # With hard negatives
│   ├── train_mined.automodel_unrolled.json  # Final training file
│   ├── corpus/                    # Document corpus
│   └── eval_beir/                 # BEIR-format evaluation data
├── stage2_finetune/               # Model checkpoints
│   └── checkpoints/
│       └── LATEST/model/consolidated/  # Final model
├── stage3_eval/                   # Evaluation results
│   └── eval_results.json
└── stage4_export/                 # Exported models
    ├── onnx/                      # ONNX model files
    │   └── model.onnx
    └── tensorrt/                  # TensorRT engine (if enabled)
        └── model.plan
```

## Evaluation Metrics

The evaluation stage measures reranking quality using a two-stage approach: first-stage dense retrieval followed by cross-encoder re-ranking. This mirrors how rerankers are used in production.

| Metric | Description | Range |
|--------|-------------|-------|
| **nDCG@k** | Normalized Discounted Cumulative Gain (ranking quality) | 0.0-1.0 |
| **Recall@k** | Fraction of relevant documents in top-k results | 0.0-1.0 |
| **Precision@k** | Fraction of retrieved documents that are relevant | 0.0-1.0 |
| **MAP@k** | Mean Average Precision | 0.0-1.0 |

Higher scores indicate better re-ranking performance. The key metric to watch is **nDCG@10**, which captures how well the reranker promotes relevant documents to the top of the list.


## Key Components

| Component | Purpose | Repository |
|-----------|---------|------------|
| retriever-sdg | Synthetic data generation using NeMo Data Designer | [GitHub](https://github.com/NVIDIA/NeMo-Data-Designer) |
| Automodel | Cross-encoder model training framework | [GitHub](https://github.com/NVIDIA/NeMo-Automodel) |
| BEIR | Evaluation framework for information retrieval | [GitHub](https://github.com/beir-cellar/beir) |
| NeMo Export-Deploy | ONNX/TensorRT export for optimized inference | [GitHub](https://github.com/NVIDIA/NeMo-Export-Deploy) |
| NVIDIA NIM | Production inference microservice with ranking API | [Developer Site](https://developer.nvidia.com/nim) |

## Base Model

| Property | Value |
|----------|-------|
| Model | nvidia/llama-nemotron-rerank-1b-v2 |
| Parameters | ~1B |
| Architecture | Cross-encoder (sequence classification) |
| Max Sequence Length | 512 (concatenated query + passage) |
| Pooling | Average |
| HuggingFace | [Model Card](https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2) |

## Further Reading

- [NeMo Data Designer Documentation](https://github.com/NVIDIA/NeMo-Data-Designer) - Synthetic data generation framework
- [Automodel Documentation](https://github.com/NVIDIA/NeMo-Automodel) - Model training framework
- [BEIR Benchmark](https://github.com/beir-cellar/beir) - Information retrieval evaluation
- [NVIDIA NIM Documentation](https://developer.nvidia.com/nim) - Production inference microservices
- [Llama-Nemotron-Rerank-1B-v2 Model Card](https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2) - Base model details
- [Retrieval-Synthetic-NVDocs-v1 Dataset](https://huggingface.co/datasets/nvidia/Retrieval-Synthetic-NVDocs-v1) - Pre-generated synthetic retrieval dataset

## Support

For issues, questions, or contributions:
- **Issues**: [GitHub Issues](https://github.com/NVIDIA/Nemotron/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NVIDIA/Nemotron/discussions)
- **Documentation**: [Nemotron Documentation](https://github.com/NVIDIA/Nemotron)
