# Nemotron Nano 3 Training Guide

This guide walks you through training Nemotron Nano 3 using the nemotron CLI and [NeMo-Run](https://github.com/NVIDIA-NeMo/Run).

## NVIDIA AI Stack

This training pipeline leverages multiple components from the NVIDIA AI Stack:

| Component | Purpose | Used In |
|-----------|---------|---------|
| **[NeMo Curator](https://github.com/NVIDIA/NeMo-Curator)** | Data Processing - 100TB+ curation, filtering, dedup | Pretraining data preparation *(coming soon)* |
| **[Megatron-Core + Bridge](https://github.com/NVIDIA/Megatron-LM)** | Pre-training - Distributed training, parallelism | Stage 0 (Pretrain), Stage 1 (SFT) |
| **[NeMo-RL](https://github.com/NVIDIA/NeMo-RL)** | Alignment - SFT, DPO, PPO, RLHF | Stage 2 (RL) |
| **[ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer)** | Compression - Pruning, quant, distillation | Model optimization *(coming soon)* |
| **[NeMo-Eval](https://github.com/NVIDIA/NeMo-Eval)** | Comprehensive benchmark evaluation suite | Model evaluation *(coming soon)* |
| **[NeMo-Run](https://github.com/NVIDIA/NeMo-Run)** | Job orchestration - each recipe step can be executed through NeMo-Run | All stages |

## About Nemotron 3 Nano

Nemotron 3 Nano is an open, efficient **Mixture-of-Experts (MoE) hybrid Mamba-Transformer** language model optimized for agentic reasoning. Key characteristics:

| Specification | Value |
|---------------|-------|
| **Total Parameters** | 31.6B |
| **Active Parameters** | 3.6B (per forward pass) |
| **Pretraining Tokens** | 25 trillion |
| **Context Length** | Up to 1M tokens |
| **Architecture** | Hybrid Mamba-Transformer with sparse MoE |

### Key Features

- **Efficient Inference**: Achieves up to 3.3x higher inference throughput than similarly sized open models like GPT-OSS-20B and Qwen3-30B-A3B while being more accurate on popular benchmarks
- **Agentic Capabilities**: Demonstrates enhanced agentic, reasoning, and chat abilities
- **Long Context**: Supports extended context length of up to 1M tokens
- **MoE Architecture**: Uses sparse Mixture-of-Experts with granular MoE architectures and shared experts for better accuracy at a fraction of the active parameter count

### Model Architecture

Nemotron 3 Nano uses a hybrid Mamba-Transformer architecture similar to previous Nemotron models, with MoE layers instead of standard FFN layers:

| Component | Value |
|-----------|-------|
| Num Layers | 32 |
| Model Dimension | 3008 |
| Q-heads | 32 |
| K/V-heads | 8 |
| Head Dimension | 128 |
| Mamba State Dimension | 128 |
| Mamba Groups | 8 |
| Mamba Heads | 64 |
| Mamba Head Dimension | 64 |
| Expert Dimension | 1856 |
| Total Routable Experts | 128 |
| Num Activated Experts | 6 |
| Shared Experts | 2 |

**Tech Report**: [Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning](https://arxiv.org/abs/2506.XXXXX)

> **Note**: The training recipes in this repository use only the **open-sourced subset** of the training data. Results from these recipes are not expected to match the full tech report benchmarks. These recipes serve as a **reference implementation** for reproducing the training methodology with your own data.

## Training Pipeline Overview

Nemotron 3 Nano is trained in three stages:

| Stage | Name | Purpose | Framework | Guide |
|-------|------|---------|-----------|-------|
| 0 | [Pretraining](./pretrain.md) | Train on large text corpus | Megatron-Bridge | [pretrain.md](./pretrain.md) |
| 1 | [SFT](./sft.md) | Instruction tuning | Megatron-Bridge | [sft.md](./sft.md) |
| 2 | [RL](./rl.md) | Alignment with GRPO | NeMo-RL | [rl.md](./rl.md) |

Each stage builds on the previous one, with full lineage tracking via W&B Artifacts.

### Training Summary

The post-training methodology includes:

1. **Supervised Fine-Tuning (SFT)** - Multi-domain instruction following with chat templates and role-based loss masking
2. **Multi-Environment Reinforcement Learning (RLVR)** - Training on all environments simultaneously using GRPO with curriculum sampling
3. **Reinforcement Learning from Human Feedback (RLHF)** - Using GenRM for generative reward model-based training

## Prerequisites

### v0 Requirements

The current version is built for Slurm cluster execution:

- **Slurm cluster** with GPU nodes
- **Weights & Biases account** for experiment tracking and artifact lineage
- **Container images**:
  - Data prep: `anyscale/ray:2.49.2-py311`
  - Training: `nvcr.io/nvidian/nemo:25.11-nano-v3.rc2`
  - RL: NeMo-RL container

> **Note**: Future versions will make the artifact backend configurable, removing the W&B requirement.

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd nemotron

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Configuration

### env.toml Setup

Create an `env.toml` file in your project root to configure execution profiles:

```toml
# Weights & Biases configuration (required for v0)
[wandb]
project = "nemotron"
entity = "YOUR-TEAM"

# CLI display settings
[cli]
theme = "github-light"

# Cluster execution profile
[YOUR-CLUSTER]
executor = "slurm"
account = "YOUR-ACCOUNT"
partition = "batch"
nodes = 2
ntasks_per_node = 8
gpus_per_node = 8
mem = "0"
exclusive = true
mounts = ["/lustre:/lustre"]
```

> **Note**: Container images are specified in the recipe config files (e.g., `config/tiny.yaml`), not in env.toml.

See [nemo-run.md](../../nemo-run.md) for complete configuration options.

## Quick Start

### Testing with Tiny Config

Before running the full pipeline, test with the `tiny` config variant:

```bash
# Test data preparation
nemotron nano3 data prep pretrain --run YOUR-CLUSTER --sample 1000

# Test training (small model, few iterations)
nemotron nano3 pretrain -c tiny --run YOUR-CLUSTER
```

### Full Training Pipeline

```bash
# Stage 0: Pretraining
nemotron nano3 data prep pretrain --run YOUR-CLUSTER
nemotron nano3 pretrain --run YOUR-CLUSTER

# Stage 1: Supervised Fine-Tuning
nemotron nano3 data prep sft --run YOUR-CLUSTER
nemotron nano3 sft --run YOUR-CLUSTER

# Stage 2: Reinforcement Learning
nemotron nano3 data prep rl --run YOUR-CLUSTER
nemotron nano3 rl --run YOUR-CLUSTER
```

## Execution Methods

### nemotron CLI (Recommended)

The main entrypoint integrates with NeMo-Run for streamlined execution across Slurm, local, Docker, and cloud backends.

#### `--run` vs `--batch`

| Option | Behavior | Use Case |
|--------|----------|----------|
| `--run <profile>` | **Attached** - submits job and waits for completion, streaming logs to terminal | Interactive development, debugging, monitoring output in real-time |
| `--batch <profile>` | **Detached** - submits job and exits immediately | Long-running training jobs, job queues, overnight runs |

```bash
# Attached execution: waits for job, streams logs
nemotron nano3 pretrain -c tiny --run YOUR-CLUSTER

# Detached execution: submits and returns immediately
nemotron nano3 pretrain -c tiny --batch YOUR-CLUSTER

# Preview execution plan (no submission)
nemotron nano3 pretrain -c tiny --run YOUR-CLUSTER --dry-run
```

**When to use `--run`**:
- You want to see logs in real-time
- Debugging or iterating on configuration
- Short test runs where you want immediate feedback

**When to use `--batch`**:
- Long training runs (hours/days)
- Submitting multiple jobs to a queue
- Running overnight or unattended

### Direct Script Execution

Scripts can be executed directly inside a container on a compute node (useful for debugging):

```bash
# Inside container on compute node
cd src/nemotron/recipes/nano3/stage0_pretrain

# Data prep
python data_prep.py --config config/data_prep.yaml

# Training
python train.py --config config/tiny.yaml

# Distributed training
torchrun --nproc_per_node=8 train.py --config config/tiny.yaml
```

## Artifact Lineage

The pipeline uses W&B Artifacts to track full lineage:

```
Raw Text Data
     ↓
DataBlendsArtifact-pretrain (bin/idx)
     ↓
nemotron nano3 pretrain
     ↓
ModelArtifact-pretrain
     ↓
DataBlendsArtifact-sft (.npy)
     ↓
nemotron nano3 sft
     ↓
ModelArtifact-sft
     ↓
DataBlendsArtifact-rl (JSONL)
     ↓
nemotron nano3 rl
     ↓
ModelArtifact-rl (Final Model)
```

Each artifact is automatically linked when running stages in sequence, providing full traceability from raw data to final model.

## CLI Reference

### Data Preparation Commands

```bash
# Pretrain data (bin/idx format)
nemotron nano3 data prep pretrain [--run <profile>] [--sample N] [--force]

# SFT data (packed .npy format)
nemotron nano3 data prep sft [--run <profile>] [--sample N] [--force]

# RL data (JSONL format)
nemotron nano3 data prep rl [--run <profile>] [--sample N] [--force]
```

### Training Commands

```bash
# Pretraining
nemotron nano3 pretrain [--run|--batch <profile>] [-c <config>] [overrides...]

# Supervised Fine-Tuning
nemotron nano3 sft [--run|--batch <profile>] [-c <config>] [overrides...]

# Reinforcement Learning
nemotron nano3 rl [--run|--batch <profile>] [-c <config>] [overrides...]
```

### Global Options

| Option | Description |
|--------|-------------|
| `--run <profile>` | **Attached** execution - submits job and waits, streaming logs to terminal |
| `--batch <profile>` | **Detached** execution - submits job and exits immediately |
| `-c <config>` | Select config file from stage's config/ directory |
| `--dry-run` | Preview execution without running |

> **Tip**: Use `--run` for interactive development and `--batch` for production training runs.

## Troubleshooting

### Common Issues

**W&B authentication**:
```bash
wandb login
```

**Container not found**:
- Verify container image path in env.toml
- For SSH tunnels, ensure squashed images exist on remote

**Job submission fails**:
- Check Slurm account and partition in env.toml
- Verify SSH tunnel configuration for remote clusters

### Getting Help

```bash
# Show available commands
nemotron nano3 --help

# Show options for a specific command
nemotron nano3 pretrain --help
```

## Stage Guides

- [Stage 0: Pretraining](./pretrain.md) - Train the base model on large text corpus
- [Stage 1: SFT](./sft.md) - Supervised fine-tuning for instruction following
- [Stage 2: RL](./rl.md) - Reinforcement learning for alignment
- [Importing Models & Data](./import.md) - Import existing checkpoints and data as W&B artifacts

## Further Reading

- [NeMo-Run Configuration](../../nemo-run.md) - Complete guide to env.toml and execution profiles
- [Data Preparation](../../data_prep.md) - Detailed data preparation documentation
- [Recipe Source](../../../src/nemotron/recipes/nano3/) - Stage-specific README files with implementation details
