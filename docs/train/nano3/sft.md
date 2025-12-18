# Stage 1: Supervised Fine-Tuning (SFT)

This stage fine-tunes the pretrained model for instruction following using [Megatron-Bridge](../nvidia-stack.md#megatron-bridge).

> **Open-Source Data Only**: This recipe uses exclusively open-sourced SFT data from the [Nemotron Post-training Datasets](https://huggingface.co/collections/nvidia/nemotron-post-training-v3) collection, which is a subset of the full data used to train the released model. The recipe includes datasets from Nemotron-Science-v1, Nemotron-Instruction-Following-Chat-v1, Nemotron-Math-Proofs-v1, Nemotron-SWE-v1, Nemotron-Agentic-v1, and Nemotron-Competitive-Programming-v1. Results will differ from the benchmarks in the [tech report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf). Use this recipe as a reference implementation to apply the methodology with your own data.

## Quick Start

<div class="termy">

```console
// 1. Prepare data (apply chat templates, tokenize to .npy)
$ uv run nemotron nano3 data prep sft --run YOUR-CLUSTER

// 2. Run SFT
$ uv run nemotron nano3 sft --run YOUR-CLUSTER
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

The `data_prep.py` script processes OpenAI-format chat data into packed sequences with role-based loss masking. See [Data Preparation Module](../data-prep.md) for detailed documentation.

### CLI Command

```bash
uv run nemotron nano3 data prep sft [options]
```

| Option | Description |
|--------|-------------|
| `--run <profile>` | Execute on Slurm via [NeMo-Run](../nemo-run.md) |
| `--sample N` | Limit rows per dataset (for testing) |
| `--force` | Force re-run, ignoring cache |

### Output

```
output/stage1_sft/
├── training.npy
├── validation.npy
├── test.npy
└── metadata.json
```

The output is registered as a [W&B Artifact](../artifacts.md) (`DataBlendsArtifact-sft`) for lineage tracking.

## Training

### CLI Command

```bash
uv run nemotron nano3 sft [options] [overrides...]
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
uv run nemotron nano3 sft train.train_iters=5000

# Different learning rate
uv run nemotron nano3 sft optimizer.lr=1e-5

# Load specific pretrained checkpoint
uv run nemotron nano3 sft checkpoint.load=/path/to/pretrain/checkpoint
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
    prev["ModelArtifact-pretrain<br/>(from Stage 0)"] --> train
    inst["Instruction Datasets<br/>(OpenAI chat format)"] --> dp["data_prep.py"]
    dp --> data["DataBlendsArtifact-sft<br/>(packed .npy files)"]
    data --> train["train.py"]
    train --> model["ModelArtifact-sft<br/>(fine-tuned checkpoint)"]
    model --> next["Stage 2: RL"]

    style prev fill:#e1f5fe,stroke:#2196f3
    style inst fill:#f3e5f5,stroke:#9c27b0
    style dp fill:#f3e5f5,stroke:#9c27b0
    style data fill:#f3e5f5,stroke:#9c27b0
    style train fill:#f3e5f5,stroke:#9c27b0
    style model fill:#f3e5f5,stroke:#9c27b0
    style next fill:#e8f5e9,stroke:#4caf50
```

## Methodology

> For complete methodology, see [Tech Report Section 3.1](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf).

### Chat Template

Nemotron 3 Nano supports both reasoning and non-reasoning modes:

- **Multi-Step**: Existing reasoning tokens preserved for reuse in subsequent steps
- **Multi-Turn**: Reasoning from previous turns dropped when user message introduced
- **Tool Calling**: Uses XML-style special tags to reduce character escaping

### SFT Data Domains

| Domain | Description |
|--------|-------------|
| **Competition Math** | Tool-integrated reasoning with GPT-OSS teachers |
| **Competition Code** | OpenCodeReasoning solutions with obfuscation/complication |
| **InfinityByte** | Cross-domain code synthesis at model capability boundaries |
| **STEM Reasoning (RQA)** | Reasoning Q&A from undergraduate/graduate STEM content |
| **Conversational Tool Use** | Multi-turn trajectories with simulated tool execution |
| **Long Context** | 128k mean token length, 256k hard limit |
| **Formal Proofs** | Lean theorem proving with 300k examples |
| **Multilingual** | French, Spanish, Italian, German, Japanese |
| **Terminal Use** | Terminal operations from Terminal Bench |
| **General Chat** | Multi-turn responses from LMSYS and WildChat |
| **Instruction Following** | Tülu 3 methodology with verifier filtering |
| **Safety** | Refusal behaviors from safety datasets |
| **Software Engineering** | GitHub issue resolution trajectories |
| **Science** | Physics, chemistry, biology via NeMo Data Designer |

> For detailed data generation pipelines, see [Tech Report Section 3.1](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf).

### Data Filtering

The pipeline applies:
- **Structural checks**: Discard malformed examples
- **Pathological repetition filtering**: Remove repeated n-grams
- **Consistency filtering**: Judge-based action consistency verification
- **Narrative filtering**: Remove political/nationalistic narratives

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Learning Rate** | 1e-5 |
| **Sequence Length** | 4096 tokens (pack_size) |
| **Loss Masking** | Role-based (assistant tokens only) |
| **Optimizer** | AdamW |
| **Total Samples** | 18M+ |

## Open-Source Data

> **Note**: This recipe trains exclusively on the open-sourced subset of SFT data. Results will differ from the tech report benchmarks, which used additional proprietary data.

## NVIDIA AI Stack

This stage uses the following components from the [NVIDIA AI Stack](../nvidia-stack.md):

| Component | Role | Documentation |
|-----------|------|---------------|
| [Megatron-Core](../nvidia-stack.md#megatron-core) | Distributed training primitives (TP, PP, DP, EP) | [GitHub](https://github.com/NVIDIA/Megatron-LM) |
| [Megatron-Bridge](../nvidia-stack.md#megatron-bridge) | Fine-tuning loop, checkpoint loading, loss masking | [Docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/) |

### Key Features Used

| Feature | Purpose |
|---------|---------|
| `finetune()` entry point | SFT training with pre-loaded checkpoint |
| Role-based loss masking | Only compute loss on assistant tokens |
| Mixed precision (BF16) | Memory-efficient training |
| Gradient checkpointing | Reduce memory footprint |

### Container

```
nvcr.io/nvidia/nemo:25.11.nemotron_3_nano
```

## Next Steps

After SFT completes, proceed to [Stage 2: RL](./rl.md) for alignment training.

## Reference

- [Tech Report Section 3.1](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf) — SFT methodology
- [NVIDIA AI Stack](../nvidia-stack.md) — Megatron-Core, Megatron-Bridge documentation
- [Artifact Lineage](../artifacts.md) — W&B artifact system
- [Stage 0: Pretraining](./pretrain.md) — Pretrain the base model
- **Recipe Source**: `src/nemotron/recipes/nano3/stage1_sft/` — Implementation details
- [Back to Overview](./README.md)
