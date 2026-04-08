# Embedding Model Fine-Tuning Recipe

A complete 6-stage pipeline for fine-tuning and deploying embedding models on domain-specific data using synthetic data generation.

## Overview

This recipe fine-tunes NVIDIA's [Llama-Nemotron-Embed-1B-v2](https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2) embedding model on your own domain data. By the end of this pipeline, you'll have a domain-adapted embedding model that excels at retrieving relevant documents from your specific corpus.

### Why Fine-Tune Embedding Models?

Pre-trained embedding models work well for general-purpose retrieval, but may underperform on specialized domains with unique terminology, document structures, or query patterns. Fine-tuning adapts the model to:

- Understand domain-specific vocabulary and concepts
- Better match the types of queries your users will ask
- Improve retrieval accuracy on your specific document corpus

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
project = "my-embedding-project"
entity = "my-username"
```

**Full configuration with Docker and Slurm support:**
See the [Execution Profiles](#execution-profiles) section below.

## Preparing Your Corpus

### Supported Formats

- Text files: `.txt`, `.md`, `.rst`
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

### Document Quality Tips

- **Length**: Aim for 200-2000 tokens per document
- **Content**: Ensure documents are representative of your domain
- **Diversity**: Include various document types/topics from your domain
- **Quality**: Clean, well-formatted text yields better synthetic Q&A pairs

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with at least 80GB VRAM (e.g., A100, H100)
  - Stages 0 uses NVIDIA API (no GPU required)
  - Stage 1-5: Require GPU for model inference and training
- **CPU**: Modern multi-core processor (16+ cores recommended)
- **Memory**: 128GB+ RAM recommended
- **Storage**: ~50GB free disk space for outputs, models, and containers

### Software Requirements

- **Python**: 3.12 or later
- **UV**: Package manager (installation instructions above)
- **NVIDIA API Key**: Required for synthetic data generation
- **NVIDIA GPU Drivers**: Latest drivers for your GPU
- **Docker** (optional): For containerized execution
- **Slurm** (optional): For cluster execution

### Expected Runtime & Resources

| Stage | GPU VRAM | CPU | Notes |
|-------|----------|-----|-------|
| Stage 0 (SDG) | N/A | 8+ cores | Uses API (no local GPU); runtime varies by dataset size |
| Stage 1 (Data Prep) | 40GB | 16+ cores | Hard negative mining on GPU; runtime varies by dataset size |
| Stage 2 (Finetune) | 80GB | 16+ cores | Runtime varies by dataset size and epochs |
| Stage 3 (Eval) | 40GB | 8+ cores | Evaluation metrics computation; runtime varies by dataset size |
| Stage 4 (Export) | 40GB | 8+ cores | TensorRT export requires NGC container |
| Stage 5 (Deploy) | 40GB | 4+ cores | NIM container initialization |

**Total disk space**: ~50GB for outputs, model checkpoints, and containers

**Runtime**: Highly dependent on dataset size. Expect longer runtimes for larger corpora and more training epochs.
 - For small dataset (e.g. nv_pp_random with ~70 input files), it can take ~30 minutes for Stage 0 (SDG) with the default setup. Changing to other LLM endpoints or tune `max_parallel_requests_for_gen` can affect the runtime, rate limit failures, and generation quality. It can take 10-20 minutes for Stage 2 (Finetune) with the default setup.
 - For large dataset (e.g. 10K+ input files), it can take tens of hours or 1-2 days for Stage 0 (SDG) and 5-10 hours for Stage 2 (Finetune). Changing model endpoints, type and number of GPUs (and other fine-tune parameters) can affect the runtime.


### LLM API Usage (Stage 0)

Stage 0 uses LLM APIs for synthetic data generation. By default, it uses NVIDIA's hosted LLMs:

- **Default provider**: NVIDIA API (free tier available at [build.nvidia.com](https://build.nvidia.com))
- **Default model**: `nvidia/nemotron-3-nano-30b-a3b` (fast, reliable for structured generation)
- **Usage**: ~4 API calls per document (artifact extraction, QA generation, dedup, quality eval)
- **Cost**: Free tier has rate limits; contact NVIDIA for production usage
- **Progress**: Built-in progress logging shows completion %, records/second, and ETA per stage
- **Other providers**: NeMo Data Designer supports multiple providers (OpenAI, OpenRouter, etc.)
  - Customize provider settings in the config file
  - See [default provider settings](https://nvidia-nemo.github.io/DataDesigner/latest/concepts/models/default-model-settings/) for configuration options

## Quick Start

### Local Execution

```bash
# Set environment (important for CUDA compatibility)
export LD_LIBRARY_PATH=""
export NVIDIA_API_KEY=nvapi-your_key_here

# Stage 0: Generate synthetic Q&A pairs from your documents
nemotron embed sdg -c default corpus_dir=/path/to/your/docs

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

# Optional: Verify NIM accuracy matches checkpoint
nemotron embed eval -c default eval_nim=true eval_base=false
```

### Preview Commands (Dry Run)

```bash
# See what would be executed without running
nemotron embed finetune -c default --dry-run
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

See individual stage READMEs for input format requirements.

### Using NVIDIA's Pre-Generated Dataset

NVIDIA provides a ready-to-use synthetic retrieval dataset on Hugging Face: [Retrieval-Synthetic-NVDocs-v1](https://huggingface.co/datasets/nvidia/Retrieval-Synthetic-NVDocs-v1). This dataset was generated from NVIDIA's publicly available content using the same SDG pipeline (Stage 0) in this recipe, and contains ~15K documents with 105K+ question-answer pairs across multiple reasoning types.

If you want to fine-tune an embedding model on NVIDIA-related content, you can **skip Stage 0 entirely** and start directly from Stage 1:

```bash
# Download the pre-generated dataset
python -c "
from datasets import load_dataset
ds = load_dataset('nvidia/Retrieval-Synthetic-NVDocs-v1', split='train')
ds.to_json('./output/embed/stage0_sdg/nv_docs_sdg.json')
"

# Start from Stage 1 (data preparation) using the downloaded data
nemotron embed prep -c default sdg_input_path=./output/embed/stage0_sdg

# Continue with the rest of the pipeline
nemotron embed finetune -c default
nemotron embed eval -c default
```

This is useful for quickly getting started with the recipe or benchmarking the pipeline without needing an NVIDIA API key for synthetic data generation.

## Execution Modes

The embed recipe supports multiple execution modes for flexibility between local development and production cluster runs.

### Local Execution (Default)

Run directly on your local machine with GPU:

```bash
nemotron embed finetune -c default
nemotron embed eval -c default
```

### Docker Execution

Run inside a Docker container with GPU passthrough using `--run local-docker`:

```bash
# Runs the command inside a Docker container with GPU access
nemotron embed finetune -c default --run local-docker

# All stages support Docker execution
nemotron embed sdg -c default --run local-docker
nemotron embed prep -c default --run local-docker
nemotron embed eval -c default --run local-docker
```

> **Note**: Requires `local-docker` profile in `env.toml` (see [Execution Profiles](#execution-profiles) below)

### Slurm Batch Execution

Submit jobs to a Slurm cluster for production workloads:

```bash
# Attached execution (waits for completion, streams logs via SSH)
nemotron embed finetune -c default --run my-cluster

# Detached execution (submits job and exits immediately)
nemotron embed finetune -c default --batch my-cluster

# Run full pipeline on cluster
nemotron embed sdg -c default --batch my-cluster
nemotron embed prep -c default --batch my-cluster
nemotron embed finetune -c default --batch my-cluster
nemotron embed eval -c default --batch my-cluster
```

### Execution Profiles

Execution profiles are defined in `env.toml` in the **repository root directory**.

**Example `env.toml` for local and cluster execution:**

```toml
# Weights & Biases configuration (optional but recommended)
[wandb]
project = "my-embedding-project"
entity = "my-team"

# Local Docker execution profile
[local-docker]
executor = "docker"
container_image = "nvcr.io/nvidia/pytorch:25.01-py3"
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
container_image = "nvcr.io/nvidia/pytorch:25.01-py3"
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
nemotron embed finetune -c default --run my-cluster run.env.gpus_per_node=4

# Use different partition
nemotron embed finetune -c default --batch my-cluster run.env.partition=batch

# Override time limit
nemotron embed finetune -c default --batch my-cluster run.env.time=08:00:00
```

### Interactive Debugging

Stage files to the cluster for interactive debugging:

```bash
# Stage files without executing
nemotron embed finetune -c default --run my-cluster --stage

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
output_dir: ./output/embed/stage0_sdg  # Path to save the generated data
artifact_extraction_model: nvidia/nemotron-3-nano-30b-a3b  # LLM Model name for document artifacts extraction
qa_generation_model: nvidia/nemotron-3-nano-30b-a3b  # LLM Model name for QA generation
quality_judge_model: nvidia/nemotron-3-nano-30b-a3b  # LLM Model name for QA quality evaluation
max_parallel_requests_for_gen: 4  # Number of parallel requests to submit to LLMs
```

**Stage 1: Data Prep**
```yaml
base_model: nvidia/llama-nemotron-embed-1b-v2  # Model for hard negative mining
quality_threshold: 7.0         # Minimum Q&A quality score (0-10)
hard_negatives_to_mine: 5      # Number of hard negatives per query
query_max_length: 512          # Max query tokens (check your base model's max sequence length)
passage_max_length: 512        # Max passage tokens (check your base model's max sequence length)
# Adjust train/val/test split ratio based on your generated data size
# For small data (e.g. the sample data `nv_pp_random`), use 80/20 for train/test and 0 validation in order to make the most use of the limited data
# For medium/large data, use 80/10/10 or tune for your use case
train_ratio: 0.8               # Training data split (80%)
val_ratio: 0.1                 # Validation split (10%)
test_ratio: 0.1                # Test split (10%)
```

**Stage 2: Finetune**
```yaml
base_model: nvidia/llama-nemotron-embed-1b-v2
num_epochs: 3
global_batch_size: 128
learning_rate: 1.0e-5
query_max_length: 512          # Max query tokens (check your base model's max sequence length)
passage_max_length: 512        # Max passage tokens (check your base model's max sequence length)
# attn_implementation: null    # Auto-detects flash_attention_2 if available, else sdpa
train_n_passages: 5            # 1 positive + 4 hard negatives
```

> **Warning — Overfitting risk**: The default `num_epochs: 3` is set for the small example dataset shipped with this recipe, where fewer epochs may not produce a visible training signal. For most real-world datasets, **1–2 epochs is sufficient** and 3 epochs carries a high risk of overfitting. Lower this value when working with your own data (e.g., `nemotron embed finetune -c default num_epochs=1`).

**Stage 3: Eval**
```yaml
k_values: [1, 5, 10, 100]      # K values for Recall@k, nDCG@k
max_length: 512                # Max sequence length (check your base model's max sequence length)
eval_base: true                # Evaluate base model
eval_finetuned: true           # Evaluate fine-tuned model
eval_nim: false                # Evaluate NIM endpoint
```

**Stage 4: Export**
```yaml
model_path: ./output/embed/stage2_finetune/checkpoints/LATEST/model/consolidated
export_to_trt: true            # Export to TensorRT (requires nemo:25.07+ container)
quant_cfg: null                # Quantization: null, "fp8", "int8_sq"
trt_opt_batch: 16              # Optimal batch size for TRT
trt_opt_seq_len: 128           # Optimal sequence length for TRT
```

**Stage 5: Deploy**
```yaml
nim_image: nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:1.10.1
model_dir: ./output/embed/stage4_export/onnx  # Path to exported model
host_port: 8000                # Port for NIM API
detach: false                  # Run in background
```

### Customizing Sequence Length

The pipeline defaults to 512-token sequences. You can change this to match your use case, up to the base model's max sequence length (e.g., 8192 for the default `nvidia/llama-nemotron-embed-1b-v2`; check your model's documentation if using a different base model).

For example, to use 2000-token passages, override the sequence length consistently across stages:

```bash
# Stage 0: Increase sentences per chunk so passages approach the new token budget
# ~80 sentences ≈ 2000 tokens for average English text; adjust for your domain
nemotron embed sdg -c default sentences_per_chunk=80

# Stage 1: Match sequence length for hard negative mining
nemotron embed prep -c default query_max_length=2000 passage_max_length=2000

# Stage 2: Match sequence length for training
nemotron embed finetune -c default query_max_length=2000 passage_max_length=2000

# Stage 3: Match sequence length for evaluation
nemotron embed eval -c default max_length=2000

# Stage 4: Update TensorRT profile for longer sequences
nemotron embed export -c default trt_max_seq_len=2000 trt_opt_seq_len=512
```

> **Note**: Longer sequences increase GPU memory usage significantly (attention is quadratic in sequence length). You may need to reduce `global_batch_size` in Stage 2 to avoid out-of-memory errors. Ensure `sentences_per_chunk` in Stage 0 produces passages that actually use the longer token budget — the default of 5 sentences typically yields passages well under 512 tokens.

### Overriding Configuration

Override config values on the command line:

```bash
# Override training epochs
nemotron embed finetune -c default num_epochs=5

# Override learning rate
nemotron embed finetune -c default learning_rate=2e-5

# Override multiple values
nemotron embed finetune -c default num_epochs=5 learning_rate=2e-5

# Force specific attention implementation
nemotron embed finetune -c default attn_implementation=flash_attention_2
```

## CLI Commands

### Workspace Info

```bash
# Display workflow overview
nemotron embed info
```

### Data

```bash
# Generate synthetic Q&A pairs from documents
nemotron embed sdg -c default corpus_dir=/path/to/docs

# Prepare training data (convert, mine, unroll)
nemotron embed prep -c default sdg_input_path=/path/to/sdg
```

### Training

```bash
# Fine-tune the embedding model
nemotron embed finetune -c default train_data_path=/path/to/data
```

### Evaluation

```bash
# Evaluate base and fine-tuned models
nemotron embed eval -c default finetuned_model_path=/path/to/checkpoint
```

### Export

```bash
# Export model to ONNX and TensorRT
nemotron embed export -c default model_path=/path/to/checkpoint

# Export to ONNX only (skip TensorRT)
nemotron embed export -c default export_to_trt=false

# Export with FP8 quantization
nemotron embed export -c default quant_cfg=fp8
```

### Deploy

```bash
# Deploy NIM with custom TensorRT model (foreground)
nemotron embed deploy -c default

# Deploy in background (detached mode)
nemotron embed deploy -c default detach=true

# Deploy with ONNX model instead
nemotron embed deploy -c default model_dir=./output/embed/stage4_export/onnx

# Stop the NIM container
docker stop nemotron-embed-nim
```

### Verify NIM Accuracy

```bash
# Evaluate NIM endpoint against fine-tuned checkpoint
nemotron embed eval -c default eval_nim=true eval_base=false

# The output will show if NIM metrics match the checkpoint
# ✓ indicates metrics match within tolerance (0.03 for @1, 0.01 for @5+)
# ⚠️ indicates potential accuracy loss beyond ONNX/TensorRT conversion noise
```

## Output Structure

After running the full pipeline:

```
output/embed/
├── stage0_sdg/                    # Synthetic Q&A pairs
│   └── generated_batch*.json
├── stage1_data_prep/              # Training-ready data
│   ├── train.json                 # Original training data
│   ├── train_mined.automodel.json # With hard negatives
│   ├── train_mined.automodel_unrolled.json  # Final training file
│   ├── val.json                   # Validation data
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
    └── tensorrt/                  # TensorRT engine
        └── model.plan
```

## Evaluation Metrics

The evaluation stage computes standard information retrieval metrics using the BEIR framework.

| Metric | Description | Range |
|--------|-------------|-------|
| **nDCG@k** | Normalized Discounted Cumulative Gain (ranking quality) | 0.0-1.0 |
| **Recall@k** | Fraction of relevant documents in top-k results | 0.0-1.0 |
| **Precision@k** | Fraction of retrieved documents that are relevant | 0.0-1.0 |
| **MAP@k** | Mean Average Precision | 0.0-1.0 |

Higher scores indicate better retrieval performance.

### Interpreting Results

**Good fine-tuning results typically show:**
- nDCG@10 and Recall@10 improvement of **15%** over base model
- Consistent improvements across all k values

**Example successful evaluation:**

```
Model: base
- nDCG@10: 0.42
- Recall@10: 0.65
- Precision@10: 0.38

Model: fine-tuned
- nDCG@10: 0.51 (+21%) ✓
- Recall@10: 0.78 (+20%) ✓
- Precision@10: 0.45 (+18%) ✓
```

**Warning signs:**
- **No improvement**: May need more training data or higher quality Q&A pairs
- **Worse performance**: Check for data quality issues or training hyperparameters
- **Overfitting**: Good training metrics but poor validation metrics

## Key Components

| Component | Purpose | Repository |
|-----------|---------|------------|
| retriever-sdg | Synthetic data generation using NeMo Data Designer | [GitHub](https://github.com/NVIDIA/NeMo-Data-Designer) |
| Automodel | Embedding model training framework | [GitHub](https://github.com/NVIDIA/NeMo-Automodel) |
| BEIR | Evaluation framework for information retrieval | [GitHub](https://github.com/beir-cellar/beir) |
| NeMo Export-Deploy | ONNX/TensorRT export for optimized inference | [GitHub](https://github.com/NVIDIA/NeMo-Export-Deploy) |
| NVIDIA NIM | Production inference microservice with custom model support | [Developer Site](https://developer.nvidia.com/nim) |

## Base Model

| Property | Value |
|----------|-------|
| Model | nvidia/llama-nemotron-embed-1b-v2 |
| Parameters | ~1B |
| Embedding Dimension | 768 |
| Max Sequence Length | 8192 (pipeline default: 512; see [Customizing Sequence Length](#customizing-sequence-length)) |
| Pooling | Average |
| HuggingFace | [Model Card](https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2) |

## Troubleshooting

### Installation Issues

**Error: `uv: command not found`**
```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
# Add to PATH
export PATH="$HOME/.cargo/bin:$PATH"
```

**Error: `nemotron: command not found`**
```bash
# Make sure you're in the Nemotron directory
cd /path/to/Nemotron
# Run with uv
uv run nemotron embed info
```

### Stage 0: SDG Issues

**Error: `NVIDIA_API_KEY not set`**
```bash
# Set your API key
export NVIDIA_API_KEY=nvapi-your_key_here
```

**Error: API rate limiting**
- **Solution**: Reduce batch size in config or add delays between API calls
- **Alternative**: Contact NVIDIA for increased rate limits for production use

**Error: Poor Q&A quality scores**
- **Solution**: Check document quality - ensure clean, well-formatted text
- **Solution**: Adjust `sentences_per_chunk` in config (default: 5 sentences per chunk)

### Stage 1: Data Preparation Issues

**Error: `CUDA out of memory` during hard negative mining**
```bash
# Reduce mining batch size in config
nemotron embed prep -c default mining_batch_size=64
```

**Error: No valid Q&A pairs after quality filtering**
- **Solution**: Lower quality_threshold (default: 7.0)
- **Solution**: Check SDG output quality scores

**Error: Import errors for `nemo_automodel`**
```bash
# Ensure dependencies are installed
cd /path/to/Nemotron
uv sync --all-extras
```

### Stage 2: Training Issues

**Error: Train Loss not decreasing**
- **Solution**: Try adjusting learning rate (default: 1e-5; try 5e-6 or 2e-5)
- **Solution**: Lower `global_batch_size` to increase the number of gradient update steps
- **Solution**: Check training data quality

**Error: Train Loss is NaN**
- **Solution**: Reduce learning rate significantly
- **Solution**: Check for data quality issues (missing values, corrupted entries)

**Error: `CUDA out of memory` during training**
```bash
# Reduce local batch size
nemotron embed finetune -c default local_batch_size=2
```

**Error: Training very slow**
- **Check**: GPU utilization with `nvidia-smi`
- **Solution**: Increase batch size if GPU not fully utilized
- **Solution**: Enable mixed precision training (usually enabled by default)

### Stage 3: Evaluation Issues

**Error: Model checkpoint not found**
```bash
# Check checkpoint path
ls -la output/embed/stage2_finetune/checkpoints/LATEST/model/consolidated/

# Specify custom path
nemotron embed eval -c default finetuned_model_path=/path/to/checkpoint
```

**Error: BEIR evaluation fails**
- **Solution**: Ensure eval_beir data was created in Stage 1
- **Solution**: Check that corpus.jsonl and queries.jsonl exist

### Stage 4: Export Issues

**Error: TensorRT export fails**
- **Solution**: Ensure using NGC container with TensorRT (nemo:25.07+)
- **Solution**: Try ONNX-only export first: `export_to_trt=false`

**Error: ONNX export fails**
- **Solution**: Check model checkpoint is valid
- **Solution**: Ensure sufficient disk space

### Stage 5: Deployment Issues

**Error: NIM container fails to start**
```bash
# Check NGC credentials
docker login nvcr.io

# Check if port is already in use
sudo lsof -i :8000

# Use different port
nemotron embed deploy -c default host_port=8002
```

**Error: NIM accuracy differs from checkpoint**
- **Solution**: Ensure using same model format (TensorRT vs ONNX)
- **Solution**: Check quantization settings match
- **Solution**: Verify model files are complete and not corrupted

### CUDA Library Errors

**Error: `nvJitLink` or CUDA symbol errors**
```bash
# Clear LD_LIBRARY_PATH to avoid conflicts
export LD_LIBRARY_PATH=""
```

**Error: HybridCache import errors**
```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/modules/transformers_modules/nvidia/
```

### Docker Issues

**Error: Container has no GPU access**
```bash
# Verify NVIDIA runtime is installed
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check docker daemon.json includes nvidia runtime
cat /etc/docker/daemon.json
```

### Slurm Issues

**Error: Job submission fails**
```bash
# Check Slurm is configured in env.toml
cat env.toml

# Verify SSH access to cluster
ssh cluster.example.com

# Check Slurm partition exists
sinfo -p interactive
```

**Error: Job stays in pending state**
```bash
# Check job queue
squeue -u $USER

# Check job reason
squeue -j <job-id> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
```

**Debugging Slurm jobs**
```bash
# Check job status
squeue -u $USER

# View job logs
cat /path/to/job/logs/stdout.txt
cat /path/to/job/logs/stderr.txt

# Cancel stuck job
scancel <job-id>

# View job details
scontrol show job <job-id>
```

### General Debugging

**Enable verbose logging**
```bash
# Add --verbose flag (if available)
nemotron embed finetune -c default --verbose

# Check logs in output directory
cat output/embed/stage2_finetune/logs/*.log
```

**Dry run to preview**
```bash
# Preview command without executing
nemotron embed finetune -c default --dry-run
```

## Monitoring Training

### Weights & Biases Integration

Training automatically logs to Weights & Biases if configured in `env.toml`:

```toml
[wandb]
project = "my-embedding-project"
entity = "my-team"
```

Monitor training progress at: `https://wandb.ai/<entity>/<project>`

### Local Monitoring

**Check training logs:**
```bash
tail -f output/embed/stage2_finetune/logs/train.log
```

**GPU utilization:**
```bash
watch -n 1 nvidia-smi
```

**Checkpoint progress:**
```bash
ls -lh output/embed/stage2_finetune/checkpoints/
```

## Best Practices

### Data Quality
- Use clean, well-formatted documents
- Ensure documents represent your target domain
- Aim for diverse document types and topics
- Start with small corpus to test pipeline, then scale up

### Training
- Start with default hyperparameters (1-2 epochs, LR 1e-5, batch size 128)
- Monitor validation metrics to avoid overfitting
- Attention implementation is auto-detected (flash_attention_2 if available, else sdpa)
- Checkpoint frequency: if `ckpt_every_steps` is omitted, defaults to once per epoch for map-style datasets or twice during training for iterable datasets

**Key hyperparameters to tune:**

| Parameter | Default | Notes |
| :---- | :---- | :---- |
| Epochs | 3 | **Default is tuned for the small example dataset; for real-world data, 1–2 epochs is usually sufficient.** 3 epochs risks overfitting on most datasets. See [epoch guidance in FAQ](#how-many-epochs-typically-improve-accuracy-before-overfitting-becomes-a-risk-is-there-a-rule-of-thumb). |
| Learning rate | 1e-5 | Try double and half of the default value |
| Learning rate warmup steps | 5 | Set to 5-10% of total steps of finetune to have better early training stability |
| Sequence length | 512 | Set `query_max_length` / `passage_max_length` consistently across Stages 1-3 (up to your base model's max sequence length). Increase `sentences_per_chunk` in Stage 0 accordingly. Longer sequences require reducing batch size. See [Customizing Sequence Length](#customizing-sequence-length). |

### Evaluation
- Always compare against base model
- Test on held-out test set (not used in training)
- Evaluate on realistic queries from your domain
- Consider multiple metrics (nDCG, Recall, Precision)

### Deployment
- Test exported models thoroughly before production
- Verify NIM accuracy matches checkpoint
- Monitor inference latency and throughput
- Set up proper logging and monitoring

## FAQ

### Synthetic Data Generation (SDG)

**How much does SDG model quality impact final embedding accuracy, and when is it worth upgrading the SDG model?**

The SDG model directly determines the quality of synthetic queries and answers used to train the embedding model, so it has a first-order effect on final retrieval accuracy. The default model (`nvidia/nemotron-3-nano-30b-a3b`) is a practical starting point that balances cost, speed, and structured-generation reliability. Consider upgrading the SDG model when: (1) quality scores from the SDG quality judge are consistently low (median below 7/10), (2) you observe that generated queries are shallow or fail to capture the reasoning patterns your users need, or (3) your domain has highly specialized terminology that the default model handles poorly. You can swap SDG models per-task via config — e.g., use a stronger model for `qa_generation_model` while keeping the lighter one for `artifact_extraction_model` and `quality_judge_model` to control cost. Run a small pilot (50–100 docs) with each model and compare the downstream Stage 3 eval metrics to decide whether the upgrade is worth the added latency and API cost.

As of now, we’ve seen good results with the NVIDIA Nemotron Super 49B and Nemotron Ultra 253B

**How should SDG prompts be designed to reliably capture rare words and domain-specific identifiers (bug IDs, product IDs, versions) that matter for retrieval accuracy?**

The built-in SDG pipeline extracts structured artifacts from each document chunk — including entities, technical terms, key concepts, and relationships — before generating QA pairs. These artifacts are injected into the QA generation prompt, which biases the LLM toward producing queries that reference specific identifiers. To improve coverage of rare tokens: (1) ensure your source documents contain the identifiers in-context (the LLM can only reference what it sees in the chunk), (2) increase `max_artifacts_per_type` (default: 2) so more entities and technical terms are extracted per chunk, and (3) increase `num_pairs` to generate more QA pairs per document, raising the chance that niche identifiers appear in at least some queries. If identifiers span multiple chunks (e.g., a bug ID mentioned in one section and its resolution in another), enable multi-document bundling (`multi_doc: true`) so the LLM sees cross-chunk context. After SDG, spot-check a sample of generated queries for identifier coverage before proceeding to training.

In addition, prompts can be tailored to the specific document type - bug vs ticket vs technical manual. It makes sense that prompts should be tailored to the specific document type for optimal Q/A generation.

### Data Volume and Saturation

**What is the optimal number of source documents (or QA pairs) needed before embedding fine-tuning accuracy saturates?**

There is no universal threshold — saturation depends on domain complexity, vocabulary diversity, and document heterogeneity. As a rough guide:

| Corpus Size | Typical Outcome |
|-------------|-----------------|
| 100+ docs | Basic domain adaptation |
| 500-1000 docs | Good domain coverage for enterprise corpora |
| 5000+ docs | Strong and reliable adaptation |

After SDG and quality filtering (default threshold 7.0), the effective training set is typically smaller than the raw doc count. Monitor Stage 3 eval metrics (nDCG@10, Recall@10) across runs with increasing data to find your domain's saturation point.

**How can we reliably detect saturation of the embedding model as we scale data volume?**

Run the pipeline at two or three data scales (e.g., 25%, 50%, 100% of your corpus) with identical hyperparameters and compare Stage 3 eval metrics. Saturation is reached when doubling the data yields less than ~1–2 absolute points of nDCG@10 improvement. Use a fixed held-out evaluation set across all runs to ensure comparability (see the evaluation questions below).

**Should we prioritize adding more documents or generating more queries per document to improve accuracy?**

In general, more documents with diverse content have a larger impact than more queries per document, because new documents introduce new vocabulary, concepts, and retrieval patterns. More queries per document (via `num_pairs`) primarily helps the model see the same content from different query angles, which has diminishing returns once the core semantics are covered. Prioritize adding documents first; once your corpus is representative, increase `num_pairs` (default: 10) to improve query diversity for chunks that cover complex or multi-faceted topics.

### Using Existing Vector-DB Chunks

**Would using real production vector-DB chunks as positives (instead of synthetic chunks) improve embedding accuracy?**

Yes, this can improve accuracy — if your production chunks reflect the actual retrieval units users will query against, training on them aligns the embedding space more closely with your deployment setup. The recipe supports this: you can skip Stage 0 entirely and start from Stage 1 by supplying your own QA pairs with real chunks as positives (see the [Pipeline Flexibility](#pipeline-flexibility) table). Format your data as JSON with query–positive-passage pairs and feed it to `nemotron embed prep`. The main risk is that real chunks without synthetic queries may lack query diversity; consider generating synthetic queries against your real chunks (see next question) to get the best of both worlds.

**Is it recommended to generate multiple synthetic queries per real chunk to better shape the embedding space?**

Yes. Generating multiple diverse queries per chunk teaches the model that many different phrasings should map to the same passage. You can do this by running Stage 0 with your real chunks as input documents and increasing `num_pairs`. The SDG pipeline will generate varied query types (factual, relational, inferential, procedural, etc.) and complexity levels against each chunk. This is especially valuable for chunks covering dense or multi-faceted content where a single query captures only one retrieval intent.

**Should training-time chunking exactly match production chunking to maximize retrieval accuracy, or is approximate alignment sufficient?**

Exact matching is ideal but approximate alignment is usually good enough. What matters most is that training chunks and production chunks are in the same ballpark of length and boundary style — if production chunks are ~500 tokens with sentence-boundary splitting, training on ~500-token sentence-boundary chunks will transfer well even if the exact split points differ. The embedding model learns semantic similarity at the passage level, not memorized chunk boundaries. That said, large mismatches hurt: training on 5-sentence chunks (the Stage 0 default, typically ~100–150 tokens) while deploying with 2000-token chunks creates a distribution gap where the model has never seen passages of that length during training. To close the gap, either (1) feed your real production chunks directly as positives (see above), or (2) adjust `sentences_per_chunk` in Stage 0 and `passage_max_length` in Stages 1–3 to approximate your production chunk size. Also ensure `passage_max_length` is set consistently across stages so that tokenization truncation during training and evaluation matches what happens at inference time. In practice, aligning chunk length within ~2x of production is sufficient; pixel-perfect boundary matching yields negligible additional gain.

### Hard-Negative Mining

**How should hard-negative mining thresholds be tuned to improve embedding discrimination?**

Hard-negative mining uses a margin-based filter to exclude documents that are too similar to the positive. The key parameter is `hard_neg_margin` (default: 0.95 with `perc` margin type), which acts as an exclusion ceiling: any document scoring *above* `min_positive_score * margin` is eliminated, and the top-k highest-scoring survivors become the hard negatives. To tune:

- **Raise the margin** (e.g., 0.98–1.0) to narrow the exclusion zone, allowing negatives that score closer to the positive. This produces harder negatives that improve discrimination but risks including false negatives (relevant documents mislabeled as negative), especially in corpora with near-duplicate passages.
- **Lower the margin** (e.g., 0.85–0.90) to widen the exclusion zone, forcing negatives to be further from the positive score. This produces easier, safer negatives with less risk of false negatives, but provides weaker training signal.
- **Increase `hard_negatives_to_mine`** (default: 5) to give the model more contrastive examples per query. The training stage uses `train_n_passages` (default: 5, meaning 1 positive + 4 negatives), so mine at least as many as you plan to train with.

Start with defaults and only raise the margin if Stage 3 metrics plateau — aggressive hard negatives on noisy data can hurt more than help.

**What is the recommended number of hard negatives for best accuracy (e.g., 5 vs 10 vs higher)?**

The default of 4 hard negatives per query (`train_n_passages: 5` = 1 positive + 4 negatives) is a solid baseline. Increasing to 10 negatives can improve discrimination, especially for large corpora with many similar-looking passages, but the gains taper off quickly beyond that. Two parameters must be adjusted together: `hard_negatives_to_mine` in Stage 1 (how many candidates are mined) and `train_n_passages` in Stage 2 (how many are used during training), e.g., mine 10 and train with 10.

### Training Hyperparameters

**Which hyperparameters most strongly affect embedding accuracy (learning rate, epochs, batch size), and in what priority order should they be tuned?**

In order of typical impact:

1. **Learning rate** (default: 1e-5) — the single most sensitive parameter. Try 5e-6 and 2e-5 as first alternatives. Too high causes instability or NaN loss; too low undertrains.
2. **Epochs** (default: 3) — controls how many passes the model makes over the data. The default of 3 is calibrated for the small example dataset in this recipe; **for most real-world datasets, 1–2 epochs is recommended** to avoid overfitting. See the epoch table below.
3. **Learning rate warmup** (default: 5 steps) — set to 5–10% of total training steps for better early stability.
4. **Batch size** (default: 128) — determines the number of gradient update steps per epoch. Use smaller values for small datasets to get more updates; see the [batch size FAQ](#how-does-batch-size-affect-training-and-how-should-it-be-set) below.

**What learning-rate sweep strategy is recommended to maximize accuracy (e.g., halve/double defaults)?**

A simple three-point sweep around the default is the most cost-effective approach. Start with the default 1e-5, then try 5e-6 (half) and 2e-5 (double). Compare Stage 3 eval metrics (nDCG@10) across the three runs — the winner is usually obvious. If the best result is at an endpoint (e.g., 2e-5 beats both others), extend one more step in that direction (try 4e-5) to confirm you haven't undershot. Keep epochs and all other hyperparameters fixed during the sweep so that LR is the only variable.

**How many epochs typically improve accuracy before overfitting becomes a risk? Is there a rule of thumb?**

The default `num_epochs: 3` exists because the example dataset shipped with this recipe is very small and training for only 1–2 epochs may not produce a measurable signal. **For your own data, start with 1–2 epochs and only increase if evaluation metrics are still improving.**

| Dataset Size | Recommended Epochs | Notes |
|--------------|--------------------|-------|
| Small (<1K examples) | 2–3 | Use 3 only if val loss is still decreasing |
| Medium (1K–10K examples) | 1–2 | 2 epochs is usually the upper bound |
| Large (10K+ examples) | 1 | More than 1 epoch rarely helps and often hurts |

**How does batch size affect training, and how should it be set?**

This pipeline uses only hard negatives in the contrastive loss (no in-batch negatives), so batch size does not change the number of negatives per query. Instead, batch size primarily affects the **number of gradient update steps** the model takes: `steps_per_epoch = total_training_samples / global_batch_size`. A smaller `global_batch_size` means more steps and more frequent weight updates; a larger one means fewer steps and faster wall-clock time per epoch.

As a rule of thumb, use a **smaller `global_batch_size` for small datasets** and a **larger `global_batch_size` for larger datasets**.

### Loss Interpretation and Evaluation

**How should training loss and evaluation loss be interpreted to assess real accuracy gains?**

- **Training loss** (contrastive/InfoNCE loss) measures how well the model separates positives from negatives in each batch. A steadily decreasing training loss is expected; a very low floor (~0.0–0.01) suggests the model has learned the training set well.
- **Validation loss** tracks the same metric on held-out data. The gap between training and validation loss is your primary overfitting indicator. If validation loss decreases alongside training loss, the model is generalizing. If validation loss plateaus or rises while training loss keeps falling, stop training or reduce epochs.
- **Neither loss directly equals retrieval accuracy.** Always rely on Stage 3 eval metrics (nDCG@k, Recall@k) as the ground truth for actual embedding quality. Loss is a proxy — it's possible for loss to improve while retrieval metrics stagnate if the hard negatives are too easy.

**Are there target loss behaviors or patterns that indicate optimal embedding learning?**

Healthy training typically shows: (1) rapid loss decrease in the first 20–30% of training, (2) gradual flattening toward a stable floor, (3) validation loss tracking close to training loss throughout. Warning signs include loss spikes (often caused by bad data or LR too high), NaN loss (reduce LR or batch size), or loss that never decreases (LR too low, or data quality issues). There is no universal "target loss value" — the absolute number depends on batch size, number of negatives, and temperature. Focus on the trajectory and the train/val gap rather than the absolute value.

**Should a fixed evaluation dataset always be used to measure true accuracy gains across runs?**

Yes. The recipe creates a fixed held-out test split in Stage 1 (default: 10–20% of data, configurable via `test_ratio`) that is formatted into BEIR evaluation format (`eval_beir/`). This test set is never used during training and provides a consistent benchmark across experiments. When comparing runs with different hyperparameters, SDG models, or data volumes, always use the same evaluation set — otherwise metric differences may reflect data variance rather than model improvement. For the most rigorous comparison, freeze the evaluation set from your first run and reuse it across all subsequent experiments.

**How large should the fixed evaluation set be to reliably reflect embedding accuracy improvements?**

Aim for at least 100 queries in the evaluation set to get stable metric estimates; 200–500 queries is better for detecting small improvements. The pipeline warns if your evaluation set has fewer than 50 queries. With very small evaluation sets, metric variance can be high enough to mask real gains. If your total dataset is small, consider using a 80/0/20 train/val/test split (the default for small datasets) to maximize the evaluation set size at the cost of skipping validation-during-training.

**What level of accuracy improvement (relative vs absolute) is typical or expected from embedding fine-tuning in similar enterprise domains?**

Typical results on enterprise domains show **+5 to +20 absolute points of nDCG@10** over the base model, depending on how specialized the domain is. Domains with highly specialized vocabulary (legal, biomedical, internal engineering docs) tend to see larger gains because the base model has less prior exposure. Domains closer to general web text see smaller but still meaningful improvements. If you see less than 5 absolute points of nDCG@10 improvement, investigate data quality, SDG coverage, or hyperparameter settings before concluding that fine-tuning doesn't help your domain.

## Further Reading

- [NeMo Data Designer Documentation](https://github.com/NVIDIA/NeMo-Data-Designer) - Synthetic data generation framework
- [Automodel Documentation](https://github.com/NVIDIA/NeMo-Automodel) - Model training framework
- [BEIR Benchmark](https://github.com/beir-cellar/beir) - Information retrieval evaluation
- [NVIDIA NIM Documentation](https://developer.nvidia.com/nim) - Production inference microservices
- [Llama-Nemotron-Embed-1B-v2 Model Card](https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2) - Base model details
- [Retrieval-Synthetic-NVDocs-v1 Dataset](https://huggingface.co/datasets/nvidia/Retrieval-Synthetic-NVDocs-v1) - Pre-generated synthetic retrieval dataset on NVIDIA content

## Support

For issues, questions, or contributions:
- **Issues**: [GitHub Issues](https://github.com/NVIDIA/Nemotron/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NVIDIA/Nemotron/discussions)
- **Documentation**: [Nemotron Documentation](https://github.com/NVIDIA/Nemotron)
