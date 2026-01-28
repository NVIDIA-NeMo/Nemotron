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

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Install Nemotron

```bash
# Clone the repository
git clone https://github.com/NVIDIA/Nemotron.git
cd Nemotron

# Install with all dependencies for the embed recipe
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
| **Stage 1** | Q&A pairs (JSON) | Skip SDG if you have labeled data |
| **Stage 2** | Training data (Automodel format) | Skip data prep if data is ready |
| **Stage 3** | Model checkpoint | Evaluate existing checkpoint |
| **Stage 4** | Model checkpoint | Export existing model |
| **Stage 5** | Exported model (ONNX/TensorRT) | Deploy existing model |

See individual stage READMEs for input format requirements.

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
global_batch_size: 128         # Auto-scaled down for small datasets
learning_rate: 1.0e-5
# attn_implementation: null    # Auto-detects flash_attention_2 if available, else sdpa
train_n_passages: 5            # 1 positive + 4 hard negatives
```

**Stage 3: Eval**
```yaml
k_values: [1, 5, 10, 100]      # K values for Recall@k, nDCG@k
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
nemotron embed prep -c default sdg_output_dir=/path/to/sdg
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
- nDCG@10 improvement of **5-15%** over base model
- Recall@10 improvement of **10-30%**
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
| Max Sequence Length | 512 |
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
- **Solution**: Adjust chunk size in config (default: 512 tokens)

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

**Error: Loss not decreasing**
- **Solution**: Try adjusting learning rate (default: 1e-5; try 5e-6 for larger datasets, 2e-5 for very small ones)
- **Solution**: Check training data quality
- **Solution**: Increase training epochs

**Error: Loss is NaN**
- **Solution**: Reduce batch size
- **Solution**: Reduce learning rate significantly
- **Solution**: Check for data quality issues (missing values, corrupted entries)

**Error: `CUDA out of memory` during training**
```bash
# Reduce global batch size
nemotron embed finetune -c default global_batch_size=64

# Or use gradient accumulation (if supported)
nemotron embed finetune -c default global_batch_size=128 micro_batch_size=16
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
- Start with default hyperparameters (3 epochs, LR 1e-5, batch size auto-scaled)
- Monitor validation metrics to avoid overfitting
- Batch size is auto-scaled down for small datasets (<2000 examples)
- Attention implementation is auto-detected (flash_attention_2 if available, else sdpa)
- Save intermediate checkpoints (every 100 steps by default, auto-adjusted for small datasets)

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

## Further Reading

- [NeMo Data Designer Documentation](https://github.com/NVIDIA/NeMo-Data-Designer) - Synthetic data generation framework
- [Automodel Documentation](https://github.com/NVIDIA/NeMo-Automodel) - Model training framework
- [BEIR Benchmark](https://github.com/beir-cellar/beir) - Information retrieval evaluation
- [NVIDIA NIM Documentation](https://developer.nvidia.com/nim) - Production inference microservices
- [Llama-Nemotron-Embed-1B-v2 Model Card](https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2) - Base model details

## Support

For issues, questions, or contributions:
- **Issues**: [GitHub Issues](https://github.com/NVIDIA/Nemotron/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NVIDIA/Nemotron/discussions)
- **Documentation**: [Nemotron Documentation](https://github.com/NVIDIA/Nemotron)
