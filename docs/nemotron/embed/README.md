# Embedding Model Fine-Tuning Recipe

Adapt an embedding model to your domain-specific data, then evaluate and deploy it with this complete six-stage pipeline.

## Overview

This recipe fine-tunes the Nemotron 3 Embed checkpoint for domain-specific retrieval and deploys the resulting checkpoint with Retriever NIM or vLLM.

### Why Fine-Tune Embedding Models?

Pre-trained embedding models work well for general-purpose retrieval, but may underperform on specialized domains with unique terminology, document structures, or query patterns. Fine-tuning adapts the model to:

- Understand domain-specific vocabulary and concepts
- Better match the types of queries your users will ask
- Improve retrieval accuracy on your specific document corpus

## Training Pipeline

A large language model (LLM) generates synthetic question-and-answer pairs in Stage 0.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           YOUR DOCUMENT CORPUS                              │
│              (Text files: .txt, .md, and other supported formats)           │
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
│                       STAGE 4: EXPORT (WHEN REQUIRED)                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │       Export models that require a deployment artifact; otherwise skip  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             STAGE 5: DEPLOY                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │               Launch NIM or vLLM from the selected artifact             ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

| Stage | Command | Description | Output |
|-------|---------|-------------|--------|
| Stage 0: SDG | `nemotron embed sdg` | Validate corpus, generate synthetic Q&A pairs from documents | Q&A pairs with quality scores |
| Stage 1: Data Prep | `nemotron embed prep` | Convert, mine hard negatives, unroll | Training-ready data |
| Stage 2: Finetune | `nemotron embed finetune` | Fine-tune embedding model | Model checkpoint |
| Stage 3: Eval | `nemotron embed eval` | Evaluate on retrieval metrics | Metrics comparison |
| Stage 4: Export | `nemotron embed export` | Export when required by the selected profile | Exported model or skipped |
| Stage 5: Deploy | `nemotron embed deploy` | Deploy the selected model artifact with NIM or vLLM | Running inference service |

## Installation

### 1. Install the uv Package Manager

This project requires the [uv](https://docs.astral.sh/uv/) package manager. It automatically creates and manages a virtual environment under the repository root, and each pipeline stage uses an isolated environment. **Do not use `pip install`**. The project relies on uv workspaces and per-stage dependency isolation.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Install Nemotron

```bash
# Clone the repository
git clone https://github.com/NVIDIA-NeMo/Nemotron.git
cd Nemotron

# uv creates a virtual environment at .venv/ and installs all dependencies
uv sync --all-extras
```

### 3. Get Your NVIDIA API Credential

Stage 0 sends synthetic-data-generation requests to Data Designer's built-in
NVIDIA endpoint unless the generic `NVIDIA_API_BASE_URL` environment variable
is set. Use the credential for the selected endpoint:

```bash
export NVIDIA_API_KEY=your_key_here
```

### 4. Optional: Configure Execution Profiles

For Docker or Slurm execution, create `env.toml` in the **repository root directory**.

**Minimal configuration (local execution only):**
```toml
[wandb]
project = "my-embedding-project"
entity = "my-username"
```

**Full configuration with Docker and Slurm support:**
Refer to [Execution Profiles](#execution-profiles).

## Preparing Your Corpus

### Supported Formats

- Any text files, default: .txt, .md, .text, and files with no extension
- Documents should be UTF-8 encoded
- Files are processed recursively from the corpus directory

### Corpus Size Recommendations

| Corpus Size | Documents | Expected Results |
|-------------|-----------|------------------|
| **Minimum** | 50–100 documents (approximately 50,000 tokens) | Basic domain adaptation |
| **Recommended** | 500 or more documents | Good domain coverage |
| **Optimal** | 1,000 or more documents | Best performance |

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

- **Length**: Aim for 200–2,000 tokens per document
- **Content**: Ensure documents are representative of your domain
- **Diversity**: Include various document types/topics from your domain
- **Quality**: Clean, well-formatted text yields better synthetic Q&A pairs

## Prerequisites

### Hardware Requirements

- **GPU**: An NVIDIA GPU with at least 80 GB of VRAM, such as an A100 or H100
  - Stage 0 uses the NVIDIA API and does not require a GPU.
  - Stages 1–5 require a GPU for model inference and training.
- **CPU**: Modern multicore processor (16 or more cores recommended)
- **Memory**: 128 GB or more RAM recommended
- **Storage**: Approximately 50 GB of free disk space for outputs, models, and containers

### Software Requirements

- **Python**: 3.12 or later
- **uv**: Package manager (installation instructions above)
- **NVIDIA API Key**: Required for synthetic data generation
- **NVIDIA GPU Drivers**: Latest drivers for your GPU
- **Docker** (optional): For containerized execution
- **Slurm** (optional): For cluster execution

### Expected Runtime and Resources

| Stage | GPU VRAM | CPU | Notes |
|-------|----------|-----|-------|
| Stage 0 (SDG) | N/A | 8 or more cores | Uses API (no local GPU); runtime varies by dataset size |
| Stage 1 (Data Prep) | 40 GB | 16 or more cores | Hard-negative mining on GPU; runtime varies by dataset size |
| Stage 2 (Finetune) | 80 GB | 16 or more cores | Runtime varies by dataset size and epochs |
| Stage 3 (Eval) | 40 GB | 8 or more cores | Evaluation metrics computation; runtime varies by dataset size |
| Stage 4 (Export) | 40 GB | 8 or more cores | TensorRT export requires NGC container |
| Stage 5 (Deploy) | 40 GB | 4 or more cores | Inference container initialization |

**Total disk space**: Approximately 50 GB for outputs, model checkpoints, and containers

**Runtime**: Runtime depends on dataset size. Larger corpora and more training epochs take longer.

- For a small dataset, such as `nv_pp_random` with approximately 70 input files, Stage 0 (SDG) can take about 30 minutes with the default setup. Changing LLM endpoints or tuning `max_parallel_requests_for_gen` can affect runtime, rate-limit behavior, and generation quality. Stage 2 (Finetune) can take 10–20 minutes with the default setup.
- For a large dataset, such as 10,000 or more input files, Stage 0 (SDG) can take tens of hours or 1–2 days, and Stage 2 (Finetune) can take 5–10 hours. Changing model endpoints, GPU type and count, and other fine-tuning parameters can affect runtime.


### Large Language Model (LLM) API Usage (Stage 0)

Stage 0 uses LLM APIs for synthetic data generation. By default, it uses NVIDIA's hosted LLMs:

- **Default endpoint**: Data Designer built-in, optionally overridden with `NVIDIA_API_BASE_URL`
- **Default generation/judge model**: `nvidia/nemotron-3-ultra-550b-a55b`
- **Usage**: ~4 API calls per document (artifact extraction, QA generation, dedup, quality eval)
- **Access and limits**: Depend on the selected NVIDIA endpoint and credential
- **Progress**: Built-in progress logging shows completion %, records/second, and ETA per stage
- **Other providers**: NeMo Data Designer supports multiple providers, such as OpenAI and OpenRouter.
  - Customize provider settings in the config file
  - See [default provider settings](https://docs.nvidia.com/nemo/datadesigner/concepts/models/default-model-settings) for configuration options

## Quick Start

### Default Profile

The default profile loads `nvidia/Nemotron-3-Embed-1B-BF16` from
Hugging Face for mining, fine-tuning, and base-model evaluation. Stage 5 can
mount the consolidated Stage 2 checkpoint directly into Retriever NIM 2.x or
serve it with vLLM:

```bash
export LD_LIBRARY_PATH=""
export NVIDIA_API_KEY=your_endpoint_credential
# Optional: override Data Designer's built-in NVIDIA endpoint.
export NVIDIA_API_BASE_URL=https://your-authorized-endpoint.example/v1

nemotron embed sdg -c default corpus_dir=/path/to/your/docs
nemotron embed prep -c default
nemotron embed finetune -c default
nemotron embed eval -c default

# Stage 4 is an intentional no-op for this profile.
nemotron embed export -c default

# Option A: deploy with a compatible Retriever NIM image.
export NEMOTRON3_EMBED_NIM_IMAGE=nvcr.io/your-compatible-embedding-nim:tag
nemotron embed deploy -c default
nemotron embed eval -c default eval_nim=true eval_base=false eval_finetuned=true \
  output_dir=./output/embed/nemotron-3-1b/stage3_eval_nim_comparison

# Option B: deploy with the checked-in vLLM backend.
nemotron embed deploy -c default backend=vllm
nemotron embed eval -c default eval_nim=true eval_base=false eval_finetuned=true \
  embedding_api_backend=vllm \
  output_dir=./output/embed/nemotron-3-1b/stage3_eval_vllm_comparison
```

Stage 0 uses `nvidia/nemotron-3-ultra-550b-a55b` through Data
Designer's built-in endpoint or the optional `NVIDIA_API_BASE_URL` override.
Model-dependent artifacts are isolated
under `output/embed/nemotron-3-1b/`. Both deploy backends mount
`stage2_finetune/checkpoints/LATEST/model/consolidated` read-only at `/model`
without ONNX or TensorRT conversion. NIM selects it through
`NIM_ENGINE_MODEL_PATH=/model`; vLLM runs `vllm serve /model` and detects the
checkpoint's embedding configuration automatically.
Because the artifact is already local, neither path forwards `NGC_API_KEY`
into the container.

The deploy preflight requires a checkpoint with this supported fingerprint:
hidden size 2,048; 16 layers; 24 attention heads; 8 key/value heads;
intermediate size 6,144; and vocabulary size 131,072.

NIM uses a 512-token runtime limit and defaults to `padded-naive-fp16`. You can
override this NIM setting with `NEMOTRON3_EMBED_NIM_PIPELINE_ID`. vLLM derives
the checkpoint's sequence length, pooling, activation, and prompt behavior
automatically. The evaluator uses vLLM's `/v2/embed` endpoint and passes
`input_type` (`query` or `document`) without adding text prefixes. For null or
nonfinite NIM responses, the evaluator retries up to 32 times per affected
input. Treat every retry warning as a serving-reliability defect.

Stage 2 uses a commit-pinned Automodel source with Transformers 5.12.1 to write
the deployable checkpoint. Stages 1 and 3 retain Transformers 5.1 through 5.5
for the original checkpoint path.

The preceding served-endpoint evaluation commands evaluate the fine-tuned
checkpoint and served endpoint in one process. They record endpoint, model, and
dimension diagnostics and write results to a separate output directory. The
reported metric deltas compare aggregate retrieval behavior; they do not prove
artifact identity. The deploy mount and fingerprint checks establish which local
artifact was selected. Set `fail_on_nim_metric_drift=true` only when the
configured tolerances should gate the run.

Use `NEMOTRON3_EMBED_DEPLOY_CHECKPOINT` to override the default checkpoint
directory for either backend. Use `NEMOTRON3_EMBED_NIM_MODEL` to set the model
alias advertised by NIM or passed to vLLM as `--served-model-name`.

### Dry Run

```bash
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
| **Stage 4** | Model checkpoint | Export existing model when the selected profile requires it |
| **Stage 5** | Model checkpoint or exported artifact | Deploy existing model |

The default profile loads the Stage 2 Hugging Face-style PyTorch checkpoint
directly through `NIM_ENGINE_MODEL_PATH`, so Stage 4 is an explicit no-op.

For stage-specific input requirements and options, run `nemotron embed <stage> --help`. In a source checkout, inspect the corresponding `config/default.yaml` file for default paths and settings.

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

# `datasets.to_json()` writes JSONL; Stage 1 accepts that file when given its directory.
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

> **Note**: Requires the `local-docker` profile in `env.toml` (refer to [Execution Profiles](#execution-profiles)).

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

Each stage's `default.yaml` config derives its paths from the profile-wide
`artifact_root`, which defaults to `./output/embed/nemotron-3-1b`. Override it
once for an entire pipeline run:

```bash
nemotron embed run -c default --to eval artifact_root=./output/embed/experiments/domain-a
```

Changing `artifact_root` only relocates artifacts. A future model profile such
as `nemotron-3-8b` must also set its own base model, runtime settings, NIM
identity and fingerprint, and any model-specific dependencies.

### Default Model Settings

```yaml
# Stage 0
# Set NVIDIA_API_BASE_URL to override Data Designer's built-in endpoint.

# Stage 1-3
base_model: nvidia/Nemotron-3-Embed-1B-BF16
trust_remote_code: true

# Stage 4
enabled: false

# Stage 5
backend: nim  # Override with backend=vllm
vllm_image: nvcr.io/nvidia/vllm:26.06-py3
model_dir: ./output/embed/nemotron-3-1b/stage2_finetune/checkpoints/LATEST/model/consolidated
model_path_env: NIM_ENGINE_MODEL_PATH
container_model_path: /model
max_seq_len: 512  # NIM runtime setting; vLLM reads checkpoint metadata
```

All ordinary fields can still be overridden on the command line:

```bash
nemotron embed finetune -c default num_epochs=1 learning_rate=2e-5
```

### Sequence length

Mining, training, evaluation, and the default NIM profile use 512 tokens. Keep
those settings aligned. The default NIM profile intentionally fixes
`max_seq_len: 512`; changing local training lengths alone does not expand the
served limit.

Longer sequences increase GPU memory use substantially; reduce batch sizes when
necessary.

## CLI Commands

```bash
# Inspect the recipe
nemotron embed info

# Default direct-checkpoint path
nemotron embed run -c default --to eval
nemotron embed deploy -c default detach=true
nemotron embed eval -c default eval_nim=true eval_base=false eval_finetuned=true \
  output_dir=./output/embed/nemotron-3-1b/stage3_eval_nim_comparison

# Stop the default container
docker stop nemotron-embed-nim
```

## Output Structure

```text
output/embed/
└── nemotron-3-1b/
│   ├── stage0_sdg/
│   ├── stage1_data_prep/
│   ├── stage2_finetune/checkpoints/LATEST/model/consolidated/
│   └── stage3_eval/eval_results.json
```

## Evaluation Metrics

The evaluation stage computes standard information retrieval metrics using the BEIR framework.

| Metric | Description | Range |
|--------|-------------|-------|
| **nDCG@k** | Normalized Discounted Cumulative Gain (ranking quality) | 0.0–1.0 |
| **Recall@k** | Fraction of relevant documents in top-k results | 0.0–1.0 |
| **Precision@k** | Fraction of retrieved documents that are relevant | 0.0–1.0 |
| **MAP@k** | Mean Average Precision | 0.0–1.0 |

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
| retriever-sdg | Synthetic data generation using NeMo Data Designer | [GitHub](https://github.com/NVIDIA-NeMo/DataDesigner) |
| Automodel | Embedding model training framework | [GitHub](https://github.com/NVIDIA/NeMo-Automodel) |
| BEIR | Evaluation framework for information retrieval | [GitHub](https://github.com/beir-cellar/beir) |
| NeMo Export-Deploy | ONNX/TensorRT export for optimized inference | [GitHub](https://github.com/NVIDIA/NeMo-Export-Deploy) |
| NVIDIA NIM | Production inference microservice with custom model support | [Developer Site](https://developer.nvidia.com/nim) |

## Default Model Profile

| Property | Value |
|----------|-------|
| Model | Nemotron 3 Embed |
| Model locator | `nvidia/Nemotron-3-Embed-1B-BF16` (Hugging Face) |
| Embedding dimension | 2048 |
| Recipe sequence length | 512 |
| Deployment backend | Retriever NIM or vLLM |
| Serving input artifact | Hugging Face PyTorch/safetensors checkpoint |
| NIM selector | `NIM_ENGINE_MODEL_PATH` when using NIM |
| Stage 4 | Skipped for the default profile |

## Troubleshooting

### Installation Issues

**Error: `uv: command not found`**
```bash
# Install the uv package manager
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
# Use the credential for the built-in endpoint or NVIDIA_API_BASE_URL.
export NVIDIA_API_KEY=your_key_here
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
ls -la output/embed/nemotron-3-1b/stage2_finetune/checkpoints/LATEST/model/consolidated/

# Specify custom path
nemotron embed eval -c default finetuned_model_path=/path/to/checkpoint
```

**Error: BEIR evaluation fails**
- **Solution**: Ensure eval_beir data was created in Stage 1
- **Solution**: Check that corpus.jsonl and queries.jsonl exist

### Stage 5: Deployment Issues

**Error: NIM image is not configured**
```bash
export NEMOTRON3_EMBED_NIM_IMAGE=nvcr.io/your-compatible-embedding-nim:tag
nemotron embed deploy -c default
```

Alternatively, use the checked-in vLLM backend:
```bash
nemotron embed deploy -c default backend=vllm
```

**Error: Serving container fails to start**
```bash
# Authenticate to NGC when required for the selected image.
docker login nvcr.io

# Check if port is already in use
sudo lsof -i :8000

# Use different port
nemotron embed deploy -c default backend=vllm host_port=8002
```

**Error: Served accuracy differs from checkpoint**
- **Solution**: For vLLM evaluation, set `embedding_api_backend=vllm`; it uses
  `/v2/embed` and sends `input_type=query` or `input_type=passage` without
  manually prefixing text.
- **Solution**: Let vLLM detect pooling, activation, normalization, and sequence
  length from the checkpoint. Do not add overrides unless you are deliberately
  testing a different serving contract.
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

AutoModel creates one W&B run from distributed rank zero and records training
loss, learning rate, gradient norm, memory, step timing, and validation metrics.
Slurm run names include the Slurm job ID. Sensitive Slurm environment variables
are delivered through a private mode-0600 file rather than being embedded in
generated executor or sbatch files.

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
| Epochs | 3 | **Default is tuned for the small example dataset; for real-world data, 1–2 epochs is usually sufficient.** 3 epochs risks overfitting on most datasets. Refer to [epoch guidance in the FAQ](#how-many-epochs-typically-improve-accuracy-before-overfitting-becomes-a-risk-is-there-a-rule-of-thumb). |
| Learning rate | 1e-5 | Try double and half of the default value |
| Learning rate warmup steps | 5 | Set to 5-10% of total steps of finetune to have better early training stability |
| Sequence length | 512 | Set `query_max_length` / `passage_max_length` consistently across Stages 1–3 (up to your base model's maximum sequence length). Increase `sentences_per_chunk` in Stage 0 accordingly. Longer sequences require reducing batch size. Refer to [Sequence Length](#sequence-length). |

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

The SDG model directly determines the quality of synthetic queries and answers used to train the embedding model, so it has a first-order effect on final retrieval accuracy. The recipe uses `nvidia/nemotron-3-ultra-550b-a55b` to prioritize generation and judging quality. Consider changing the SDG model when: (1) quality scores from the SDG quality judge are consistently low (median below 7/10), (2) generated queries are shallow or miss the reasoning patterns users need, or (3) cost and latency are too high for the corpus size. You can select models per task—for example, keep a stronger `qa_generation_model` while using a smaller approved model for artifact extraction and judging. Run a small pilot (50–100 docs) and compare downstream Stage 3 metrics on the same held-out set before adopting a different model.

**How should SDG prompts be designed to reliably capture rare words and domain-specific identifiers (bug IDs, product IDs, versions) that matter for retrieval accuracy?**

The built-in SDG pipeline extracts structured artifacts from each document chunk — including entities, technical terms, key concepts, and relationships — before generating QA pairs. These artifacts are injected into the QA generation prompt, which biases the LLM toward producing queries that reference specific identifiers. To improve coverage of rare tokens: (1) ensure your source documents contain the identifiers in context (the LLM can only reference what it sees in the chunk), (2) increase `max_artifacts_per_type` (default: 2) so more entities and technical terms are extracted per chunk, and (3) increase `num_pairs` to generate more QA pairs per document, raising the chance that niche identifiers appear in at least some queries. If identifiers span multiple chunks, for example, a bug ID mentioned in one section and its resolution in another, enable multi-document bundling (`multi_doc: true`) so the LLM sees cross-chunk context. After SDG, spot-check a sample of generated queries for identifier coverage before proceeding to training.

Tailor prompts to the document type, such as a bug report, ticket, or technical manual, to optimize Q&A generation.

### Data Volume and Saturation

**What is the optimal number of source documents (or QA pairs) needed before embedding fine-tuning accuracy saturates?**

There is no universal threshold — saturation depends on domain complexity, vocabulary diversity, and document heterogeneity. As a rough guide:

| Corpus Size | Typical Outcome |
|-------------|-----------------|
| 100 or more documents | Basic domain adaptation |
| 500–1,000 documents | Good domain coverage for enterprise corpora |
| 5,000 or more documents | Strong and reliable adaptation |

After SDG and quality filtering (default threshold 7.0), the effective training set is typically smaller than the raw document count. Monitor Stage 3 eval metrics (nDCG@10, Recall@10) across runs with increasing data to find your domain's saturation point.

**How can you reliably detect saturation of the embedding model as you scale data volume?**

Run the pipeline at two or three data scales, for example, 25%, 50%, and 100% of your corpus, with identical hyperparameters and compare Stage 3 eval metrics. Saturation is reached when doubling the data yields less than approximately 1–2 absolute points of nDCG@10 improvement. Use a fixed held-out evaluation set across all runs to ensure comparability. Refer to the evaluation questions below.

**Should you prioritize adding more documents or generating more queries per document to improve accuracy?**

In general, more documents with diverse content have a larger impact than more queries per document because new documents introduce new vocabulary, concepts, and retrieval patterns. More queries per document (using `num_pairs`) primarily help the model see the same content from different query angles, which has diminishing returns after the core semantics are covered. Prioritize adding documents first. After your corpus is representative, increase `num_pairs` (default: 10) to improve query diversity for chunks that cover complex or multifaceted topics.

### Using Existing Vector-DB Chunks

**Would using real production vector-DB chunks as positives (instead of synthetic chunks) improve embedding accuracy?**

Yes, this can improve accuracy. If your production chunks reflect the actual retrieval units that you query against, training on them aligns the embedding space more closely with your deployment setup. The recipe supports this: you can skip Stage 0 entirely and start from Stage 1 by supplying your own QA pairs with real chunks as positives. Refer to the [Pipeline Flexibility](#pipeline-flexibility) table. Format your data as JSON with query–positive-passage pairs and feed it to `nemotron embed prep`. The main risk is that real chunks without synthetic queries may lack query diversity. Consider generating synthetic queries against your real chunks, as described in the next question, to get the best of both approaches.

**Is it recommended to generate multiple synthetic queries per real chunk to better shape the embedding space?**

Yes. Generating multiple diverse queries per chunk teaches the model that many different phrasings should map to the same passage. You can do this by running Stage 0 with your real chunks as input documents and increasing `num_pairs`. The SDG pipeline generates varied query types, such as factual, relational, inferential, and procedural queries, with different complexity levels for each chunk. This is especially valuable for chunks that cover dense or multifaceted content where a single query captures only one retrieval intent.

**Should training-time chunking exactly match production chunking to maximize retrieval accuracy, or is approximate alignment sufficient?**

Exact matching is ideal, but approximate alignment is usually sufficient. Keep training and production chunks similar in length and boundary style. For example, if production chunks contain about 500 tokens and use sentence-boundary splitting, train on chunks with a similar length and boundary style. The embedding model learns semantic similarity at the passage level, not memorized chunk boundaries.

Large mismatches can reduce accuracy. For example, training on five-sentence chunks (the Stage 0 default, typically about 100–150 tokens) while deploying with 2,000-token chunks creates a distribution gap. To reduce the gap, either feed your real production chunks directly as positives or adjust `sentences_per_chunk` in Stage 0 and `passage_max_length` in Stages 1–3 to approximate your production chunk size. Set `passage_max_length` consistently across stages so tokenization truncation during training and evaluation matches inference-time behavior. In practice, aligning chunk length within about a factor of two of the production length is sufficient. Exact boundary matching adds little benefit.

### Hard-Negative Mining

**How should hard-negative mining thresholds be tuned to improve embedding discrimination?**

Hard-negative mining uses a margin-based filter to exclude documents that are too similar to the positive. The key parameter is `hard_neg_margin` (default: 0.95 with `perc` margin type), which acts as an exclusion ceiling: any document scoring *above* `min_positive_score * margin` is eliminated, and the top-k highest-scoring survivors become the hard negatives. To tune:

- **Raise the margin** (for example, 0.98–1.0) to narrow the exclusion zone, allowing negatives that score closer to the positive. This produces harder negatives that improve discrimination but risks including false negatives (relevant documents mislabeled as negative), especially in corpora with near-duplicate passages.
- **Lower the margin** (for example, 0.85–0.90) to widen the exclusion zone, forcing negatives to be further from the positive score. This produces easier, safer negatives with less risk of false negatives, but provides weaker training signal.
- **Increase `hard_negatives_to_mine`** (default: 5) to give the model more contrastive examples per query. The training stage uses `train_n_passages` (default: 5, meaning 1 positive + 4 negatives), so mine at least as many as you plan to train with.

Start with defaults and only raise the margin if Stage 3 metrics plateau — aggressive hard negatives on noisy data can hurt more than help.

**What is the recommended number of hard negatives for best accuracy, such as 5, 10, or more?**

The default of 4 hard negatives per query (`train_n_passages: 5` = 1 positive + 4 negatives) is a solid baseline. Increasing to 10 negatives can improve discrimination, especially for large corpora with many similar-looking passages, but the gains taper off quickly beyond that. Adjust two parameters together: `hard_negatives_to_mine` in Stage 1 (how many candidates are mined) and `train_n_passages` in Stage 2 (how many are used during training). For example, mine 10 and train with 10.

### Training Hyperparameters

**Which hyperparameters most strongly affect embedding accuracy (learning rate, epochs, batch size), and in what priority order should they be tuned?**

In order of typical impact:

1. **Learning rate** (default: 1e-5) — the single most sensitive parameter. Try 5e-6 and 2e-5 as first alternatives. Too high causes instability or NaN loss; too low undertrains.
2. **Epochs** (default: 3) — controls how many passes the model makes over the data. The default of 3 is calibrated for the small example dataset in this recipe; **for most real-world datasets, 1–2 epochs is recommended** to avoid overfitting. See the epoch table below.
3. **Learning rate warmup** (default: 5 steps) — set to 5–10% of total training steps for better early stability.
4. **Batch size** (default: 128) — determines the number of gradient update steps per epoch. Use smaller values for small datasets to get more updates. Refer to the [batch size FAQ](#how-does-batch-size-affect-training-and-how-should-it-be-set).

**What learning-rate sweep strategy is recommended to maximize accuracy, such as halving or doubling defaults?**

Use a simple three-point sweep around the default. Start with the default 1e-5, then try 5e-6 (half) and 2e-5 (double). Compare Stage 3 eval metrics (nDCG@10) across the three runs. The best result is usually clear. If the best result is at an endpoint, for example, 2e-5 beats both others, extend one more step in that direction and try 4e-5 to confirm that you have not undershot. Keep epochs and all other hyperparameters fixed during the sweep so that the learning rate is the only variable.

#### How many epochs typically improve accuracy before overfitting becomes a risk? Is there a rule of thumb?

The default `num_epochs: 3` exists because the example dataset shipped with this recipe is very small and training for only 1–2 epochs may not produce a measurable signal. **For your own data, start with 1–2 epochs and only increase if evaluation metrics are still improving.**

| Dataset Size | Recommended Epochs | Notes |
|--------------|--------------------|-------|
| Small (fewer than 1,000 examples) | 2–3 | Use 3 only if validation loss is still decreasing |
| Medium (1,000–10,000 examples) | 1–2 | 2 epochs is usually the upper bound |
| Large (10,000 or more examples) | 1 | More than 1 epoch rarely helps and often hurts |

#### How does batch size affect training, and how should it be set?

This pipeline uses only hard negatives in the contrastive loss (no in-batch negatives), so batch size does not change the number of negatives per query. Instead, batch size primarily affects the **number of gradient update steps** the model takes: `steps_per_epoch = total_training_samples / global_batch_size`. A smaller `global_batch_size` means more steps and more frequent weight updates; a larger one means fewer steps and faster wall-clock time per epoch.

As a rule of thumb, use a **smaller `global_batch_size` for small datasets** and a **larger `global_batch_size` for larger datasets**.

### Loss Interpretation and Evaluation

**How should training loss and evaluation loss be interpreted to assess real accuracy gains?**

- **Training loss** (contrastive/InfoNCE loss) measures how well the model separates positives from negatives in each batch. A steadily decreasing training loss is expected; a very low floor (~0.0–0.01) suggests the model has learned the training set well.
- **Validation loss** tracks the same metric on held-out data. The gap between training and validation loss is your primary overfitting indicator. If validation loss decreases alongside training loss, the model is generalizing. If validation loss plateaus or rises while training loss keeps falling, stop training or reduce epochs.
- **Neither loss directly equals retrieval accuracy.** Always rely on Stage 3 eval metrics (nDCG@k, Recall@k) as the ground truth for actual embedding quality. Loss is a proxy — it is possible for loss to improve while retrieval metrics stagnate if the hard negatives are too easy.

**Are there target loss behaviors or patterns that indicate optimal embedding learning?**

Healthy training typically shows: (1) rapid loss decrease in the first 20–30% of training, (2) gradual flattening toward a stable floor, (3) validation loss tracking close to training loss throughout. Warning signs include loss spikes (often caused by bad data or LR too high), NaN loss (reduce LR or batch size), or loss that never decreases (LR too low, or data quality issues). There is no universal "target loss value" — the absolute number depends on batch size, number of negatives, and temperature. Focus on the trajectory and the train/val gap rather than the absolute value.

**Should a fixed evaluation dataset always be used to measure true accuracy gains across runs?**

Yes. The recipe creates a fixed held-out test split in Stage 1 (default: 10–20% of data, configurable via `test_ratio`) that is formatted into BEIR evaluation format (`eval_beir/`). This test set is never used during training and provides a consistent benchmark across experiments. When comparing runs with different hyperparameters, SDG models, or data volumes, always use the same evaluation set — otherwise metric differences may reflect data variance rather than model improvement. For the most rigorous comparison, freeze the evaluation set from your first run and reuse it across all subsequent experiments.

**How large should the fixed evaluation set be to reliably reflect embedding accuracy improvements?**

Aim for at least 100 queries in the evaluation set to get stable metric estimates; 200–500 queries is better for detecting small improvements. The pipeline warns if your evaluation set has fewer than 50 queries. With very small evaluation sets, metric variance can be high enough to mask real gains. If your total dataset is small, consider using a 80/0/20 train/val/test split (the default for small datasets) to maximize the evaluation set size at the cost of skipping validation-during-training.

**What level of accuracy improvement (relative vs absolute) is typical or expected from embedding fine-tuning in similar enterprise domains?**

Typical results on enterprise domains show **+5 to +20 absolute points of nDCG@10** over the base model, depending on how specialized the domain is. Domains with highly specialized vocabulary (legal, biomedical, internal engineering docs) tend to see larger gains because the base model has less prior exposure. Domains closer to general web text see smaller but still meaningful improvements. If you see less than 5 absolute points of nDCG@10 improvement, investigate data quality, SDG coverage, or hyperparameter settings before concluding that fine-tuning doesn't help your domain.

## Further Reading

- [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner) - Synthetic data generation framework
- [Automodel](https://github.com/NVIDIA/NeMo-Automodel) - Model training framework
- [BEIR Benchmark](https://github.com/beir-cellar/beir) - Information retrieval evaluation
- [NVIDIA NIM Documentation](https://developer.nvidia.com/nim) - Production inference microservices
- [Retrieval-Synthetic-NVDocs-v1 Dataset](https://huggingface.co/datasets/nvidia/Retrieval-Synthetic-NVDocs-v1) - Pre-generated synthetic retrieval dataset on NVIDIA content

## Support

For issues, questions, or contributions:
- **Issues**: [GitHub Issues](https://github.com/NVIDIA-NeMo/Nemotron/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NVIDIA-NeMo/Nemotron/discussions)
- **Documentation**: [Nemotron Documentation](https://docs.nvidia.com/nemotron/latest)
