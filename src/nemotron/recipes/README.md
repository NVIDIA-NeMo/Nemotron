# Nemotron Recipes

Reproducible training pipelines for NVIDIA Nemotron models.

## Overview

Recipes are complete training pipelines that take you from raw data to a fully trained model. Each recipe includes:

- **Data preparation**: Tokenization and formatting for each training stage
- **Training scripts**: Pre-configured for distributed training with Megatron-Bridge or NeMo-RL
- **Configuration files**: Production-ready defaults with testing variants
- **Artifact tracking**: Full lineage from raw data → tokenized data → model checkpoints via W&B Artifacts API

## Available Recipes

| Recipe | Description | Status |
|--------|-------------|--------|
| [nano3](./nano3/) | Nemotron Nano 3 (31.6B total / 3.6B active params) - 3-stage training pipeline | Available |
| [super3](./super3/) | Nemotron Super 3 (Hybrid Mamba-Transformer-MoE with MTP and DeepEP) - 4-stage training pipeline | Available |
| chipnemo | ChipNeMo/ScaleRTL (Domain-adapted for RTL code generation) | Planned |

## Prerequisites (v0)

> **Slurm Only**: This initial release has been tested exclusively with Slurm execution. Support for additional NeMo-Run executors (local, Docker, SkyPilot, DGX Cloud) is planned for future releases.

- **Slurm cluster**: GPU nodes (H100 recommended)
- **Weights & Biases**: Required for experiment tracking and artifact lineage
- **Container images**: NeMo containers (e.g., `nvcr.io/nvidian/nemo:25.11-nano-v3.rc2`)

> Future versions will also make the artifact backend configurable, removing the W&B requirement.

## Quick Start

### 1. Set up env.toml

Create an `env.toml` file in your project root with your execution profiles and W&B configuration:

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

See [docs/nemo_runspec/nemo-run.md](../../docs/nemo_runspec/nemo-run.md) for complete profile configuration options.

### 2. Run a Recipe

```bash
# Prepare data for pretraining
uv run nemotron nano3 data prep pretrain --run YOUR-CLUSTER

# Run pretraining on Slurm
uv run nemotron nano3 pretrain -c tiny --run YOUR-CLUSTER

# Check what would be executed (dry-run)
uv run nemotron nano3 pretrain -c tiny --run YOUR-CLUSTER --dry-run
```

## Execution Methods

### nemotron CLI (Recommended)

The main entrypoint for running recipes. It integrates natively with [NeMo-Run](https://github.com/NVIDIA-NeMo/Run), providing:

- Automatic code packaging and syncing to remote clusters
- Environment setup and container management
- Job submission to Slurm, local execution, Docker, or cloud backends

#### `--run` vs `--batch`

| Option | Behavior | Use Case |
|--------|----------|----------|
| `--run <profile>` | **Attached** - submits and waits, streaming logs | Interactive development, debugging |
| `--batch <profile>` | **Detached** - submits and exits immediately | Long training runs, job queues |

```bash
# Attached: waits for job, streams logs to terminal
uv run nemotron nano3 pretrain -c tiny --run YOUR-CLUSTER

# Detached: submits job and returns immediately
uv run nemotron nano3 pretrain -c tiny --batch YOUR-CLUSTER

# Preview execution plan (no submission)
uv run nemotron nano3 pretrain -c tiny --run YOUR-CLUSTER --dry-run
```

### Direct Script Execution

Scripts can also be executed directly inside a container on a compute node. This is useful for debugging or interactive development:

```bash
# Inside container on compute node
python train.py --config /path/to/config.yaml

# With torchrun for distributed training
torchrun --nproc_per_node=8 train.py --config /path/to/config.yaml
```

## CLI Command Structure

```bash
uv run nemotron <recipe> <command> [options]

# Data preparation commands
uv run nemotron nano3 data prep pretrain [--run <profile>] [options]
uv run nemotron nano3 data prep sft [--run <profile>] [options]
uv run nemotron nano3 data prep rl [--run <profile>] [options]

# Training commands
uv run nemotron nano3 pretrain [--run <profile>] [-c <config>] [overrides]
uv run nemotron nano3 sft [--run <profile>] [-c <config>] [overrides]
uv run nemotron nano3 rl [--run <profile>] [-c <config>] [overrides]
```

## Artifact Lineage

Recipes use the W&B Artifacts API to track the full lineage of your training pipeline:

```
Raw Data → Data Prep → Tokenized Data (Artifact)
                            ↓
                       Pretraining → Checkpoint (Artifact)
                            ↓
                          SFT → Checkpoint (Artifact)
                            ↓
                          RL → Final Model (Artifact)
```

Each stage can reference artifacts from previous stages, ensuring reproducibility and traceability.

## Further Reading

- [NeMo-Run Configuration](../../docs/nemo_runspec/nemo-run.md) - Detailed guide on execution profiles and env.toml setup
- [Nano3 Recipe](./nano3/) - Complete documentation for the Nemotron Nano 3 training pipeline
- [Super3 Recipe](./super3/) - Complete documentation for the Nemotron Super 3 training pipeline
