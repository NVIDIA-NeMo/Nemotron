# Stage 0: Pretraining

Pretrain Nemotron 3 Nano on a large text corpus using Megatron-Bridge.

## Overview

This stage tokenizes raw text data and trains the base language model from scratch. It produces a pretrained checkpoint that serves as the foundation for subsequent instruction tuning (SFT) and alignment (RL) stages.

Nemotron-3-Nano-30B-A3B-Base was pretrained using the Warmup-Stable-Decay learning rate schedule on 25 trillion tokens spanning 15 categories (see [Tech Report Section 2.2](https://arxiv.org/abs/2506.XXXXX)). Training was divided into 2 phases:
- **Phase 1**: 23.5 trillion tokens of diverse data
- **Phase 2**: 1.5 trillion tokens of high-quality data

The base model achieves better accuracy than equivalent-sized Qwen3-30B-A3B-Base on most academic benchmarks across Code, Math, Long Context, General Knowledge, and Commonsense Understanding categories.

> **Note**: This recipe uses only the **open-sourced subset** of pretraining data. Results are not expected to match the full tech report benchmarks. This serves as a **reference implementation** for the pretraining methodology.

| Component | Description |
|-----------|-------------|
| `data_prep.py` | Tokenizes raw text into Megatron bin/idx format |
| `train.py` | Runs pretraining using Megatron-Bridge |
| `config/` | Configuration files for data prep and training |

## Pretraining Data

The pretraining corpus spans fifteen data categories. The largest component is web crawl data, which we subdivide into five quality-based groups following the Nemotron-CC taxonomy:
- `crawl-medium`, `crawl-medium-high`, `sys-crawl-medium-high`, `crawl-high`, and `sys-crawl-high`

Beyond web crawl, the data mixture also includes:
- **math** - Mathematical content
- **Wikipedia** - Encyclopedia articles
- **code** - Programming code
- **semantic-code** - Code with semantic annotations
- **academic** - Academic papers
- **Crawl++** - Enhanced crawl data (OpenWebText, BigScience and Reddit datasets)
- **multilingual data** - Content across 15 languages: Arabic, Chinese, Czech, Danish, Dutch, Finnish, French, German, Hebrew, Hindi, Italian, Japanese, Korean, Portuguese, Polish, Russian, Spanish, Swedish, and Thai

We design our data mixtures to balance coverage and quality by employing composite weight to ensure of similar estimated quality. Higher-quality datasets are prioritized accordingly, receiving greater weight in the blend.

### Data Mixture and Ordering

A curriculum-based approach is used in two phases:
1. **Phase 1**: A data mixture that promotes diversity in data
2. **Phase 2**: We switched to the second phase at the 94% point of training, using a data mixture emphasizing high-quality datasets

### Hyperparameters

- **Total Tokens**: 25 trillion
- **Warmup**: We maintained the warm-up till 80% of the training (20 trillion tokens) and then finally decayed to a minimum of 1e-5 during the last 20% of training (5 trillion tokens)
- **Optimizer**: AdamW optimizer with weight decay of 0.1, β₁ = 0.9, and β₂ = 0.95
- **Sequence Length**: 4096 tokens with a batch size of 8192
- **MoE Load Balancing**: DeepSeek's aux-loss-free load balancing strategy with load balancing coefficient of 1e-4

### Long-Context Extension

Similar to Nemotron Nano 2, a long-context phase (LC-Phase) was added at the end of pretraining to equip the base model with long-context ability:
- **Learning Rate**: Constant learning rate of 1e-5
- **Global Batch Size**: 48
- **Parallelism**: 8-way context parallelism, 8-way tensor parallelism, 8-way expert parallelism, and 4-way pipeline parallelism on H100 GPUs
- **Data**: Long-context document QA dataset from Nemotron Nano 2 (further scaled 3x), plus RULER-style data for CPT data blend

The Phase LC lasted for 121 billion tokens.

## Quick Start

### Using nemotron CLI (Recommended)

```bash
# 1. Prepare data (tokenize to bin/idx format)
nemotron nano3 data prep pretrain --run YOUR-CLUSTER

# 2. Run pretraining
nemotron nano3 pretrain --run YOUR-CLUSTER

# Quick test with tiny config
nemotron nano3 pretrain -c tiny --run YOUR-CLUSTER
```

### Direct Script Execution

Inside a container on a compute node:

```bash
# Data preparation
python data_prep.py --config config/data_prep.yaml

# Training (single node)
python train.py --config config/tiny.yaml

# Training (distributed)
torchrun --nproc_per_node=8 train.py --config config/tiny.yaml
```

## Data Preparation

The `data_prep.py` script tokenizes raw text datasets into Megatron's binary format.

### CLI Command

```bash
nemotron nano3 data prep pretrain [options]
```

| Option | Description |
|--------|-------------|
| `--run <profile>` | Execute on Slurm via NeMo-Run |
| `--sample N` | Limit rows per dataset (for testing) |
| `--force` | Force re-run, ignoring cache |

### Input

Dataset blend defined in `config/data_blend_raw.json`:

```json
{
  "datasets": [
    {"name": "dataset-name", "weight": 1.0, "split": "train"},
    ...
  ]
}
```

### Output

```
output/nano3/stage0_pretrain/
├── train/
│   ├── data_00000.bin
│   ├── data_00000.idx
│   └── ...
├── valid/
│   └── ...
├── test/
│   └── ...
└── blend.json          # Per-split data paths for Megatron-Bridge
```

The output is registered as a W&B Artifact (`DataBlendsArtifact-pretrain`) for lineage tracking.

### Configuration

`config/data_prep.yaml`:

```yaml
blend_path: config/data_blend_raw.json
output_dir: output/nano3/stage0_pretrain
num_shards: 128
tokenizer_model: nvidia/NVIDIA-Nemotron-Nano-9B-v2
add_bos: false
add_eos: true
```

## Training

The `train.py` script runs pretraining using Megatron-Bridge.

### CLI Command

```bash
nemotron nano3 pretrain [options] [overrides...]
```

| Option | Description |
|--------|-------------|
| `--run <profile>` | **Attached** - submits and waits, streaming logs |
| `--batch <profile>` | **Detached** - submits and exits immediately |
| `-c <config>` | Config file (e.g., `-c tiny` for testing) |
| `--dry-run` | Preview execution plan |
| `key=value` | Override config values (Hydra-style) |

#### `--run` vs `--batch`

- **`--run`**: Use for interactive development, debugging, and short test runs where you want to see logs in real-time
- **`--batch`**: Use for long training runs (hours/days), job queues, and overnight/unattended runs

### Input

- **Data**: `DataBlendsArtifact-pretrain` (from data prep)
- **Config**: `config/default.yaml` or `config/tiny.yaml`

### Output

- Model checkpoints saved to configured `checkpoint.save` path
- Registered as W&B Artifact for downstream stages

### Configuration Files

| File | Purpose |
|------|---------|
| `config/default.yaml` | Production configuration |
| `config/tiny.yaml` | Testing (small model, 1700 iterations) |
| `config/data_blend_raw.json` | Full dataset blend |
| `config/data_blend_raw_small.json` | Small blend (math-only) for testing |

#### config/tiny.yaml

```yaml
run:
  data: DataBlendsArtifact-pretrain:latest

recipe:
  _target_: megatron.bridge.recipes.qwen.qwen3.qwen3_8b_pretrain_config
  per_split_data_args_path: ${art:data,path}/blend.json

train:
  train_iters: 1700
  global_batch_size: 32

scheduler:
  lr_warmup_iters: 32

logger:
  log_interval: 10
  wandb_project: ${run.wandb.project}
  wandb_entity: ${run.wandb.entity}

checkpoint:
  save: /nemo_run/pretrain
  save_interval: 20
```

### Override Examples

```bash
# More training iterations
nemotron nano3 pretrain -c tiny train.train_iters=5000

# Larger batch size
nemotron nano3 pretrain -c tiny train.global_batch_size=64

# Different checkpoint location
nemotron nano3 pretrain -c tiny checkpoint.save=/path/to/checkpoints

# Multiple overrides
nemotron nano3 pretrain -c tiny \
    train.train_iters=5000 \
    scheduler.lr_warmup_iters=100 \
    checkpoint.save_interval=50
```

## Running with NeMo-Run

The nemotron CLI uses [NeMo-Run](https://github.com/NVIDIA-NeMo/Run) for job orchestration.

### env.toml Setup

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

> **Note**: Container images are specified in the recipe config files (e.g., `config/tiny.yaml`), not in env.toml.

### Execution Examples

```bash
# Attached (wait for completion, stream logs)
nemotron nano3 pretrain -c tiny --run YOUR-CLUSTER

# Detached (submit and exit immediately)
nemotron nano3 pretrain -c tiny --batch YOUR-CLUSTER

# Preview without executing
nemotron nano3 pretrain -c tiny --run YOUR-CLUSTER --dry-run
```

See [nemo-run.md](../../nemo-run.md) for complete configuration options.

## Artifact Lineage

```
Raw Text Data
     ↓
data_prep.py
     ↓
DataBlendsArtifact-pretrain (bin/idx files + blend.json)
     ↓
train.py
     ↓
ModelArtifact-pretrain (checkpoint)
     ↓
[Stage 1: SFT]
```

## Next Steps

After pretraining completes, proceed to [Stage 1: SFT](./sft.md) for instruction tuning.

## Reference

- [Recipe Source](../../../src/nemotron/recipes/nano3/stage0_pretrain/) - Implementation details
- [Back to Overview](./README.md)
