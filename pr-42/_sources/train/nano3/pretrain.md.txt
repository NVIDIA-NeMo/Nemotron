# Stage 0: Pretraining

This stage trains the base Nemotron 3 Nano model from scratch on 25 trillion tokens using [Megatron-Bridge](../nvidia-stack.md#megatron-bridge).

> **Open-Source Data Only**: This recipe uses exclusively open-sourced training data from the [Nemotron Pre-training Datasets](https://huggingface.co/collections/nvidia/nemotron-pre-training-datasets) collection, which is a subset of the full data used to train the released model. The recipe includes datasets from Nemotron-CC-Math-v1, Nemotron-CC-v2, Nemotron-CC-v2.1, and Nemotron-Pretraining-Specialized-v1. Results will differ from the benchmarks in the [tech report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf). Use this recipe as a reference implementation to apply the methodology with your own data.

## Quick Start

<div class="termy">

```console
// 1. Prepare data (tokenize to bin/idx format)
$ uv run nemotron nano3 data prep pretrain --run YOUR-CLUSTER

// 2. Run pretraining
$ uv run nemotron nano3 pretrain --run YOUR-CLUSTER
```

</div>

> **Note**: The `--run YOUR-CLUSTER` flag submits jobs via [NeMo-Run](../nemo-run.md). See [Execution through NeMo-Run](../nemo-run.md) for setup.

### Direct Script Execution

Inside a container on a compute node:

```bash
# Data preparation
uv run python data_prep.py --config config/data_prep.yaml

# Training (single node)
uv run python train.py --config config/default.yaml

# Training (distributed)
uv run torchrun --nproc_per_node=8 train.py --config config/default.yaml
```

## Configuration

| File | Purpose |
|------|---------|
| `config/default.yaml` | Production configuration |
| `config/data_prep.yaml` | Data preparation settings |
| `config/data_blend_raw.json` | Dataset blend definition |

## Data Preparation

The `data_prep.py` script tokenizes raw text datasets into Megatron's binary format. See [Data Preparation Module](../data-prep.md) for detailed documentation.

### CLI Command

```bash
uv run nemotron nano3 data prep pretrain [options]
```

| Option | Description |
|--------|-------------|
| `--run <profile>` | Execute on Slurm via [NeMo-Run](../nemo-run.md) |
| `--sample N` | Limit rows per dataset (for testing) |
| `--force` | Force re-run, ignoring cache |

### Output

```
output/nano3/stage0_pretrain/
├── train/
│   ├── data_00000.bin
│   ├── data_00000.idx
│   └── ...
├── valid/
├── test/
└── blend.json
```

The output is registered as a [W&B Artifact](../artifacts.md) (`DataBlendsArtifact-pretrain`) for lineage tracking.

## Training

### CLI Command

```bash
uv run nemotron nano3 pretrain [options] [overrides...]
```

| Option | Description |
|--------|-------------|
| `--run <profile>` | Attached—submits and waits, streaming logs ([NeMo-Run](../nemo-run.md)) |
| `--batch <profile>` | Detached—submits and exits immediately ([NeMo-Run](../nemo-run.md)) |
| `--dry-run` | Preview execution plan |
| `key=value` | Override config values ([CLI Framework](../cli.md#dotlist-overrides)) |

### Override Examples

```bash
# More training iterations
uv run nemotron nano3 pretrain train.train_iters=5000

# Larger batch size
uv run nemotron nano3 pretrain train.global_batch_size=64

# Different checkpoint location
uv run nemotron nano3 pretrain checkpoint.save=/path/to/checkpoints
```

## Running with NeMo-Run

Configure execution profiles in `env.toml`:

```toml
[wandb]
project = "nemotron"
entity = "YOUR-TEAM"

[YOUR-CLUSTER]
executor = "slurm"
account = "YOUR-ACCOUNT"
partition = "batch"
nodes = 2
ntasks_per_node = 8
gpus_per_node = 8
mounts = ["/lustre:/lustre"]
```

See [Execution through NeMo-Run](../nemo-run.md) for complete configuration options.

## Artifact Lineage

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryBorderColor': '#333333', 'lineColor': '#333333', 'primaryTextColor': '#333333'}}}%%
flowchart TB
    raw["Raw Text Data"] --> dp["data_prep.py"]
    dp --> data["DataBlendsArtifact-pretrain<br/>(bin/idx files + blend.json)"]
    data --> train["train.py"]
    train --> model["ModelArtifact-pretrain<br/>(checkpoint)"]
    model --> next["Stage 1: SFT"]

    style raw fill:#e1f5fe,stroke:#2196f3
    style dp fill:#e1f5fe,stroke:#2196f3
    style data fill:#e1f5fe,stroke:#2196f3
    style train fill:#e1f5fe,stroke:#2196f3
    style model fill:#e1f5fe,stroke:#2196f3
    style next fill:#f3e5f5,stroke:#9c27b0
```

## Methodology

> For complete methodology, see [Tech Report Section 2](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf).

### Pretraining Data

The pretraining corpus comprises four main dataset families:

| Dataset Family | Description |
|----------------|-------------|
| **Nemotron-CC-Code-v1** | High-quality code from Common Crawl |
| **Nemotron-Pretraining-Code-v2** | GitHub code with student-teacher generation |
| **Nemotron-CC-v2.1** | General English web crawl with synthetic rephrasing |
| **Nemotron-Pretrain-Specialized-v1** | Synthetic STEM, math textbooks, scientific coding |

Data spans 15 categories including web crawl (various quality tiers), code, math, academic, and multilingual content.

> For dataset details, see [Tech Report Section 2.2](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf).

### Data Mixture

Two-phase curriculum approach:

| Phase | Tokens | Focus |
|-------|--------|-------|
| Phase 1 | 23.5T | High diversity across web, code, math, multilingual |
| Phase 2 | 1.5T | High-quality data with curated sources |

> For mixture strategy, see [Tech Report Section 2.3](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf).

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Total Tokens** | 25 trillion |
| **Batch Size** | 8192 sequences |
| **Sequence Length** | 4096 tokens |
| **Learning Rate** | 1e-4 (stable) → 1e-5 (decay) |
| **Warmup** | 80% of training (20T tokens) |
| **Optimizer** | AdamW (β₁=0.9, β₂=0.95) |
| **Weight Decay** | 0.1 |
| **MoE Load Balancing** | DeepSeek aux-loss-free strategy |

> For hyperparameter rationale, see [Tech Report Section 2.4](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf).

### Long-Context Extension

The LC-Phase extends context to 1M tokens after main pretraining:

| Parameter | Value |
|-----------|-------|
| **Duration** | 121 billion tokens |
| **Learning Rate** | 1e-5 (constant) |
| **Global Batch Size** | 48 |
| **Parallelism** | 8-way context/tensor/expert, 4-way pipeline |

> For long-context methodology, see [Tech Report Section 2.5](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf).

## Open-Source Data

> **Note**: This recipe trains exclusively on the open-sourced subset of pretraining data. Results will differ from the tech report benchmarks, which used additional proprietary data.

## NVIDIA AI Stack

This stage uses the following components from the [NVIDIA AI Stack](../nvidia-stack.md):

| Component | Role | Documentation |
|-----------|------|---------------|
| [Megatron-Core](../nvidia-stack.md#megatron-core) | Distributed training primitives (TP, PP, DP, EP, CP, SP) | [GitHub](https://github.com/NVIDIA/Megatron-LM) |
| [Megatron-Bridge](../nvidia-stack.md#megatron-bridge) | Model definitions, training loop, checkpoint management | [Docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/) |

### Parallelism Configuration

Pretraining uses multiple parallelism strategies for efficient scaling:

| Parallelism | Config Key | Description |
|-------------|------------|-------------|
| Tensor (TP) | `model.tensor_model_parallel_size` | Split weight matrices across GPUs |
| Pipeline (PP) | `model.pipeline_model_parallel_size` | Split layers into pipeline stages |
| Data (DP) | Automatic | Replicate model, distribute batches |
| Expert (EP) | `model.expert_model_parallel_size` | Distribute MoE experts across GPUs |
| Context (CP) | `model.context_parallel_size` | Distribute long sequences |
| Sequence (SP) | `model.sequence_parallel` | Distribute LayerNorm/Dropout activations |

### Container

```
nvcr.io/nvidia/nemo:25.11.nemotron_3_nano
```

## Next Steps

After pretraining completes, proceed to [Stage 1: SFT](./sft.md) for instruction tuning.

## Reference

- [Tech Report Section 2](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf) — Pretraining methodology
- [NVIDIA AI Stack](../nvidia-stack.md) — Megatron-Core, Megatron-Bridge documentation
- [Artifact Lineage](../artifacts.md) — W&B artifact system
- **Recipe Source**: `src/nemotron/recipes/nano3/stage0_pretrain/` — Implementation details
- [Back to Overview](./README.md)
