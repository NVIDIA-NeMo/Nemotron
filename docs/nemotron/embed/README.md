# Embedding Model Fine-Tuning Recipe

A complete 6-stage pipeline for fine-tuning and deploying embedding models on domain-specific data using synthetic data generation.

## Quick Start

### Prerequisites

- **NVIDIA GPU** with at least 80GB VRAM (e.g., A100, H100) for Stages 1–5
- **NVIDIA API Key** for synthetic data generation (Stage 0) — free tier at [build.nvidia.com](https://build.nvidia.com)
- **Python 3.12+** and [UV](https://docs.astral.sh/uv/) package manager

### Installation

```bash
git clone https://github.com/NVIDIA/nemotron
cd nemotron
uv sync --all-extras
```

### Configuration

Set your NVIDIA API key for synthetic data generation:

```bash
export NVIDIA_API_KEY=nvapi-your_key_here
```

Optionally, create an `env.toml` file for Docker or Slurm execution (see [Execution through NeMo-Run](../../nemo_runspec/nemo-run.md) for details):

```toml
[wandb]
project = "my-embedding-project"
entity = "my-username"

[local-docker]
executor = "docker"
container_image = "nvcr.io/nvidia/pytorch:25.01-py3"
runtime = "nvidia"
ipc_mode = "host"
shm_size = "16g"

[my-cluster]
executor = "slurm"
account = "my-account"
partition = "interactive"
container_image = "nvcr.io/nvidia/pytorch:25.01-py3"
```

### Run the Pipeline

<div class="termy">

```console
// Set environment
$ export LD_LIBRARY_PATH=""
$ export NVIDIA_API_KEY=nvapi-your_key_here

// Stage 0: Generate synthetic Q&A pairs (sample corpus auto-downloaded from HuggingFace)
$ nemotron embed sdg -c default

// Or use your own documents:
// nemotron embed sdg -c default corpus_dir=/path/to/your/docs

// Stage 1: Prepare training data (convert, mine hard negatives, unroll)
$ nemotron embed prep -c default

// Stage 2: Fine-tune the embedding model
$ nemotron embed finetune -c default

// Stage 3: Evaluate base vs fine-tuned model
$ nemotron embed eval -c default

// Stage 4: Export to ONNX/TensorRT for deployment
$ nemotron embed export -c default

// Stage 5: Deploy NIM with custom model
$ nemotron embed deploy -c default
```

</div>

## Resources

- **Base Model:** [Llama-Nemotron-Embed-1B-v2](https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2) on HuggingFace
- **Key Components:**
  - [NeMo Data Designer](https://github.com/NVIDIA/NeMo-Data-Designer) — Synthetic data generation
  - [NeMo Automodel](https://github.com/NVIDIA/NeMo-Automodel) — Embedding model training
  - [BEIR](https://github.com/beir-cellar/beir) — Information retrieval evaluation
  - [NeMo Export-Deploy](https://github.com/NVIDIA/NeMo-Export-Deploy) — ONNX/TensorRT export
  - [NVIDIA NIM](https://developer.nvidia.com/nim) — Production inference microservices

## Training Pipeline

| Stage | Command | Description | Output |
|-------|---------|-------------|--------|
| 0 | `nemotron embed sdg` | Generate synthetic Q&A pairs from documents | Q&A pairs with quality scores |
| 1 | `nemotron embed prep` | Convert, mine hard negatives, unroll | Training-ready data |
| 2 | `nemotron embed finetune` | Fine-tune embedding model | Model checkpoint |
| 3 | `nemotron embed eval` | Evaluate on retrieval metrics | Metrics comparison |
| 4 | `nemotron embed export` | Export to ONNX/TensorRT | Optimized inference models |
| 5 | `nemotron embed deploy` | Deploy NIM with custom model | Running inference service |

## Base Model

| Property | Value |
|----------|-------|
| **Model** | nvidia/llama-nemotron-embed-1b-v2 |
| **Parameters** | ~1B |
| **Embedding Dimension** | 2048 |
| **Max Sequence Length** | 8192 |
| **Pooling** | Average |

## Why Fine-Tune Embedding Models?

Pre-trained embedding models work well for general-purpose retrieval, but may underperform on specialized domains with unique terminology, document structures, or query patterns. Fine-tuning adapts the model to:

- Understand domain-specific vocabulary and concepts
- Better match the types of queries your users will ask
- Improve retrieval accuracy on your specific document corpus

## Stage Summaries

### Stage 0: Synthetic Data Generation

Validates your document corpus, chunks documents, and uses NVIDIA's hosted LLMs to generate synthetic question-answer pairs with quality scoring. Supports configurable LLM providers and parallel request batching.

### Stage 1: Training Data Preparation

Splits data into train/val/test sets, mines hard negatives using the base embedding model, and unrolls multi-hop examples into the format expected by Automodel training.

### Stage 2: Model Fine-Tuning

Fine-tunes the Llama-Nemotron-Embed-1B-v2 model using contrastive learning — queries are trained to be closer to their positive documents than to hard negatives.

### Stage 3: Evaluation

Evaluates both the base and fine-tuned models on standard information retrieval metrics (nDCG, Recall, Precision, MAP) using the BEIR framework on your held-out test set.

### Stage 4: Export

Exports the fine-tuned model to ONNX and optionally TensorRT for optimized inference. Supports FP8 and INT8 quantization.

### Stage 5: Deployment

Deploys the exported model as an NVIDIA NIM container for production inference, with built-in accuracy verification against the original checkpoint.

## Pipeline Flexibility

Stages run sequentially, but you can start from any stage if you have the required inputs:

| Start From | Requirement | Use Case |
|------------|-------------|----------|
| **Stage 0** | Document corpus | Full pipeline from scratch |
| **Stage 1** | Q&A pairs (JSON) | Skip SDG if you have labeled data |
| **Stage 2** | Training data (Automodel format) | Skip data prep if data is ready |
| **Stage 3** | Model checkpoint | Evaluate existing checkpoint |
| **Stage 4** | Model checkpoint | Export existing model |
| **Stage 5** | Exported model (ONNX/TensorRT) | Deploy existing model |

## Preparing Your Corpus

### Default Formats

By default, the pipeline processes `.txt`, `.md`, and files with no extension. You can configure which extensions to include via the `file_extensions` option:

```bash
nemotron embed sdg -c default file_extensions=".txt,.md,.rst,.html"
```

- Documents should be UTF-8 encoded
- Files are processed recursively from the corpus directory

### Corpus Size Recommendations

| Corpus Size | Documents | Expected Results |
|-------------|-----------|------------------|
| **Minimum** | 50–100 docs (~50K tokens) | Basic domain adaptation |
| **Recommended** | 500+ docs | Good domain coverage |
| **Optimal** | 1000+ docs | Best performance |

### Document Quality Tips

- **Length**: Aim for 200–2000 tokens per document
- **Content**: Ensure documents are representative of your domain
- **Diversity**: Include various document types and topics
- **Quality**: Clean, well-formatted text yields better synthetic Q&A pairs

## Execution Options

The embed recipe supports multiple execution modes:

| Mode | Command | Use Case |
|------|---------|----------|
| **Local** (default) | `nemotron embed finetune -c default` | Development, single GPU |
| **Docker** | `nemotron embed finetune -c default --run local-docker` | Containerized with GPU passthrough |
| **Slurm (attached)** | `nemotron embed finetune -c default --run my-cluster` | Cluster, streams logs |
| **Slurm (detached)** | `nemotron embed finetune -c default --batch my-cluster` | Cluster, submits and exits |

All stages also support `--dry-run` to preview execution and `--stage` for interactive debugging on a cluster.

See [Execution through NeMo-Run](../../nemo_runspec/nemo-run.md) for profile configuration.

## Configuration

Each stage has a `config/` directory with YAML configuration files. Override values on the command line:

```bash
nemotron embed finetune -c default num_epochs=5 learning_rate=2e-5
```

### Key Options by Stage

**Stage 0 (SDG):**

| Option | Default | Description |
|--------|---------|-------------|
| `corpus_dir` | `hf://nvidia/Retrieval-Synthetic-NVDocs-v1/.../nv_pp_random` | Path to your documents (sample auto-downloaded from HuggingFace) |
| `file_extensions` | `.txt,.md` | File types to process |
| `artifact_extraction_model` | `nvidia/nemotron-3-nano-30b-a3b` | LLM for document extraction |
| `max_parallel_requests_for_gen` | `4` | Parallel API requests |

**Stage 1 (Data Prep):**

| Option | Default | Description |
|--------|---------|-------------|
| `quality_threshold` | `7.0` | Minimum Q&A quality score (0–10) |
| `hard_negatives_to_mine` | `5` | Hard negatives per query |
| `train_ratio` | `0.8` | Training data split |

**Stage 2 (Finetune):**

| Option | Default | Description |
|--------|---------|-------------|
| `num_epochs` | `3` | Training epochs |
| `global_batch_size` | `128` | Auto-scaled down for small datasets |
| `learning_rate` | `1.0e-5` | Learning rate |
| `train_n_passages` | `5` | 1 positive + 4 hard negatives |

**Stage 3 (Eval):**

| Option | Default | Description |
|--------|---------|-------------|
| `k_values` | `[1, 5, 10, 100]` | K values for Recall@k, nDCG@k |
| `eval_base` | `true` | Evaluate base model |
| `eval_finetuned` | `true` | Evaluate fine-tuned model |
| `eval_nim` | `false` | Evaluate NIM endpoint |

**Stage 4 (Export):**

| Option | Default | Description |
|--------|---------|-------------|
| `export_to_trt` | `true` | Also export to TensorRT |
| `quant_cfg` | `null` | Quantization: `null`, `fp8`, `int8_sq` |

**Stage 5 (Deploy):**

| Option | Default | Description |
|--------|---------|-------------|
| `nim_image` | `nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:1.10.1` | NIM container image |
| `host_port` | `8000` | Port for NIM API |
| `detach` | `false` | Run in background |

## Hardware Requirements

| Stage | GPU VRAM | CPU | Notes |
|-------|----------|-----|-------|
| Stage 0 (SDG) | N/A | 8+ cores | Uses API (no local GPU) |
| Stage 1 (Data Prep) | 40GB | 16+ cores | Hard negative mining on GPU |
| Stage 2 (Finetune) | 80GB | 16+ cores | Contrastive training |
| Stage 3 (Eval) | 40GB | 8+ cores | Evaluation metrics computation |
| Stage 4 (Export) | 40GB | 8+ cores | TensorRT export requires NGC container |
| Stage 5 (Deploy) | 40GB | 4+ cores | NIM container initialization |

**Total disk space**: ~50GB for outputs, model checkpoints, and containers.

## Evaluation Metrics

The evaluation stage computes standard information retrieval metrics using the BEIR framework:

| Metric | Description | Range |
|--------|-------------|-------|
| **nDCG@k** | Normalized Discounted Cumulative Gain (ranking quality) | 0.0–1.0 |
| **Recall@k** | Fraction of relevant documents in top-k results | 0.0–1.0 |
| **Precision@k** | Fraction of retrieved documents that are relevant | 0.0–1.0 |
| **MAP@k** | Mean Average Precision | 0.0–1.0 |

**Good fine-tuning results typically show** nDCG@10 and Recall@10 improvement of 15%+ over the base model.

### Interpreting Results

- **No improvement**: May need more training data or higher quality Q&A pairs
- **Worse performance**: Check data quality issues or training hyperparameters
- **Overfitting**: Good training metrics but poor validation metrics

## Output Structure

```
output/embed/
├── stage0_sdg/                    # Synthetic Q&A pairs
├── stage1_data_prep/              # Training-ready data
│   ├── train_mined.automodel_unrolled.json  # Final training file
│   ├── eval_beir/                 # BEIR-format evaluation data
│   └── corpus/                    # Document corpus
├── stage2_finetune/               # Model checkpoints
│   └── checkpoints/LATEST/model/consolidated/
├── stage3_eval/                   # Evaluation results
│   └── eval_results.json
└── stage4_export/                 # Exported models
    ├── onnx/                      # ONNX model files
    └── tensorrt/                  # TensorRT engine
```

## Troubleshooting

**`NVIDIA_API_KEY not set`**: Set your API key with `export NVIDIA_API_KEY=nvapi-your_key_here`.

**`CUDA out of memory` during training**: Reduce batch size (`global_batch_size=64`) or use gradient accumulation.

**`nvJitLink` or CUDA symbol errors**: Clear `LD_LIBRARY_PATH` with `export LD_LIBRARY_PATH=""`.

**HybridCache import errors**: Clear HuggingFace cache with `rm -rf ~/.cache/huggingface/modules/transformers_modules/nvidia/`.

**No valid Q&A pairs after filtering**: Lower `quality_threshold` (default: 7.0) or check SDG output quality.

**TensorRT export fails**: Ensure using NGC container with TensorRT (nemo:25.07+), or try ONNX-only: `export_to_trt=false`.

**NIM container fails to start**: Verify NGC credentials (`docker login nvcr.io`), check port availability, or try a different port (`host_port=8002`).

**NIM accuracy differs from checkpoint**: Ensure same model format (TensorRT vs ONNX), check quantization settings, verify model files are complete.

## Best Practices

### Data Quality
- Start with a small corpus to test the pipeline, then scale up
- Use clean, well-formatted documents representative of your target domain
- Include diverse document types and topics

### Training
- Start with default hyperparameters (3 epochs, LR 1e-5, batch size auto-scaled)
- Monitor validation metrics to avoid overfitting
- Key parameters to tune: epochs, learning rate, and warmup steps

### Evaluation
- Always compare against the base model
- Test on a held-out test set (not used in training)
- Consider multiple metrics (nDCG, Recall, Precision)

### Deployment
- Test exported models before production use
- Verify NIM accuracy matches the checkpoint: `nemotron embed eval -c default eval_nim=true eval_base=false`
- Monitor inference latency and throughput

## CLI Reference

<div class="termy">

```console
// Show available commands
$ nemotron embed --help

// Display workflow overview
$ nemotron embed info

// Preview any command without executing
$ nemotron embed finetune -c default --dry-run
```

</div>

## Further Reading

- [NeMo Data Designer](https://github.com/NVIDIA/NeMo-Data-Designer) — Synthetic data generation framework
- [NeMo Automodel](https://github.com/NVIDIA/NeMo-Automodel) — Model training framework
- [BEIR Benchmark](https://github.com/beir-cellar/beir) — Information retrieval evaluation
- [NVIDIA NIM](https://developer.nvidia.com/nim) — Production inference microservices
- [Llama-Nemotron-Embed-1B-v2 Model Card](https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2) — Base model details
- [Execution through NeMo-Run](../../nemo_runspec/nemo-run.md) — Cluster and Docker execution profiles
