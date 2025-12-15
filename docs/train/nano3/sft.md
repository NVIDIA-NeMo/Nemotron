# Stage 1: Supervised Fine-Tuning (SFT)

Fine-tune the pretrained model to follow instructions using Megatron-Bridge.

## Overview

This stage takes instruction-following datasets in OpenAI chat format, applies chat templates with role-based loss masking, and fine-tunes the pretrained model. The output is an instruction-following model ready for alignment training.

Since the release of Nemotron Nano 2, we have significantly improved upon SFT strategy. We increased dataset quality and diversity, adding a wide variety of new data with an emphasis on multi-step and multi-turn agentic tasks. We release the majority of our training data and open source our SFT codebase.

> **Note**: This recipe uses only the **open-sourced subset** of SFT data. Results are not expected to match the full tech report benchmarks. This serves as a **reference implementation** for the SFT methodology.

| Component | Description |
|-----------|-------------|
| `data_prep.py` | Applies chat templates, tokenizes to packed .npy format |
| `train.py` | Runs supervised fine-tuning using Megatron-Bridge |
| `config/` | Configuration files for data prep and training |

## SFT Data Categories

The SFT training data covers diverse domains (see [Tech Report Section 3.1](https://arxiv.org/abs/2506.XXXXX)):

### Chat Template

We allow Nemotron 3 Nano in reasoning or non-reasoning mode through the chat template:
- **Multi-Step**: In a series of assistant model calls, the existing reasoning tokens are preserved to allow the model to reuse existing reasoning for subsequent steps
- **Multi-Turn**: When a user message is introduced, any reasoning from previous turns are dropped

For tool calling, we use XML-style special tags to reduce character escaping.

### Data Domains

#### Competition Math
We use a similar strategy to Nemotron Nano 2. However, we refresh the responses with GPT-OSS-120B. In addition, we created tool-integrated reasoning traces using problem tools and GPT-OSS-20B as the teacher model.

#### Competition Code
For code we use the same data from Nemotron Nano 2, which is made up of prompts from competitive coding problems from six groundbreaking OpenCodeReasoning datasets, physics, chemistry, and other sciences.

#### Synthetic Cross-Domain Code Data
We develop a novel approach called **InfinityByte** that cross-breeds multiple datasets together. When applied to code, InfinityByte creates entirely new programming problems by bringing together concepts from different fields to pose never before seen questions.

#### Synthetic STEM Reasoning
To reinforce complex reasoning capabilities within STEM domains, we built the Reasoning Question-Answer (RQA) dataset with goals to:
1. Demonstrate advanced scientific reasoning and instruction following that can be further reinforced in post-training
2. Reinforce correlations between advanced topics that are otherwise rarely observed in web-scale data

#### Conversational Tool Use
We generate synthetic multi-turn trajectories to demonstrate conversational tool use. The generation of these trajectories involves a user that is given a task to accomplish, an agent that is instructed to help the user accomplish their task, and a tool execution environment.

#### Long Context
We generate synthetic data aiming to improve a subset of RULER tasks, with the mean token length of 128k tokens with a hard limit of 256k tokens.

#### Formal Proofs
For Lean theorem proving, we curated SFT data by first autoformalizing 580k natural language theorems from online mathematics communities (AoPS, Math StackExchange, MathOverflow).

#### Multilingual
We use multilingual data in a similar manner to Nemotron Nano 2. We used Qwen2.5-Instruct to translate our existing English post-training data into 5 target languages: French, Spanish, Italian, German and Japanese.

#### Terminal Use
To teach Nemotron 3 Nano to complete tasks on the terminal, we generate a variety of verifiable tasks based on Terminal Bench. We combine such synthetic tasks with a set of vendor collected terminal tasks.

#### General Chat
We create SFT data by generating responses to the LMSYS and WildChat datasets. The data is extended to multi-turn by having the same language model simulate the user and further continue the conversation.

#### Instruction Following
We create targeted instruction following data with the methodology used in Tülu 3. We stimulate users in a conversation using language models seeded with a user persona from Nemotron-Personas-USA.

#### Safety
We compile a diverse set of unsafe prompts sourced from the Nemotron Content Safety v2, the Gretel Safety Alignment v1, and Red-Team-2K datasets to target content safety risks and Harmful Tasks.

#### Software Engineering
To train Nemotron 3 Nano for autonomous software engineering capabilities including code exploration, issue reproduction and bug fixing, we curate a dataset of coding tasks derived from real-world GitHub issues.

#### Science
The science dataset spans physics, chemistry, and biology, and is produced through a unified pipeline that integrates synthetic, real, and document-based seed sources.

### Data Filtering

For all domains, we apply a unified data filtering pipeline to ensure that only high-quality, license-compliant, and verifiable samples are used for training. This includes:
- Discarding malformed examples using structural checks
- Filtering reasoning traces exhibiting pathological repetition
- Removing trajectories where the judge considers an action of an entity to be inconsistent with its goals

### Data Mixture

The exact data blend can be found in Table 5 of the tech report (all datasets not listed make up less than 1% of the blend). Training uses over 18M total samples with dynamic sampling approach for different dataset sizes.

## Quick Start

### Using nemotron CLI (Recommended)

```bash
# 1. Prepare data (apply chat templates, tokenize to .npy)
nemotron nano3 data prep sft --run YOUR-CLUSTER

# 2. Run SFT
nemotron nano3 sft --run YOUR-CLUSTER

# Quick test with tiny config
nemotron nano3 sft -c tiny --run YOUR-CLUSTER
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

The `data_prep.py` script processes OpenAI-format chat data into packed sequences with role-based loss masking.

### Pipeline

1. **Apply chat template** → Role-labeled chunks (system, user, assistant)
2. **Tokenize** → input_ids with role boundaries
3. **Build loss_mask** → 0 for system/user tokens, 1 for assistant tokens
4. **Pack sequences** → Efficient batching up to `pack_size` tokens
5. **Split by ratio** → training.npy, validation.npy, test.npy

### CLI Command

```bash
nemotron nano3 data prep sft [options]
```

| Option | Description |
|--------|-------------|
| `--run <profile>` | Execute on Slurm via NeMo-Run |
| `--sample N` | Limit rows per dataset (for testing) |
| `--force` | Force re-run, ignoring cache |

### Input

OpenAI chat format datasets defined in `config/data_blend_raw.json`:

```json
{
  "datasets": [
    {
      "name": "dataset-name",
      "path": "hf-org/dataset",
      "split": "train",
      "weight": 1.0
    }
  ]
}
```

Expected record format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"}
  ]
}
```

### Output

```
output/stage1_sft/
├── training.npy      # Packed training sequences
├── validation.npy    # Packed validation sequences
├── test.npy          # Packed test sequences
└── metadata.json     # Split statistics and packing info
```

Each .npy file contains packed sequences with `input_ids` and `loss_mask` arrays.

The output is registered as a W&B Artifact (`DataBlendsArtifact-sft`) for lineage tracking.

### Configuration

`config/data_prep.yaml`:

```yaml
blend_path: config/data_blend_raw.json
output_dir: output/stage1_sft
tokenizer_model: nvidia/NVIDIA-Nemotron-Nano-9B-v2
pack_size: 4096
chat_template: nano3
messages_field: messages
train_ratio: 0.98
valid_ratio: 0.01
test_ratio: 0.01
```

| Parameter | Description |
|-----------|-------------|
| `pack_size` | Maximum tokens per packed sequence |
| `chat_template` | Template name (`nano3`) or path to .jinja file |
| `messages_field` | Field containing OpenAI-format messages |
| `train_ratio` | Fraction for training split (default 0.98) |

## Training

The `train.py` script runs supervised fine-tuning using Megatron-Bridge.

### CLI Command

```bash
nemotron nano3 sft [options] [overrides...]
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

- **Model**: Pretrained checkpoint from Stage 0 (`ModelArtifact-pretrain`)
- **Data**: `DataBlendsArtifact-sft` (from data prep)
- **Config**: `config/default.yaml` or `config/tiny.yaml`

### Output

- Fine-tuned model checkpoints
- Registered as W&B Artifact (`ModelArtifact-sft`) for downstream RL stage

### Configuration Files

| File | Purpose |
|------|---------|
| `config/default.yaml` | Production configuration |
| `config/tiny.yaml` | Testing variant |
| `config/data_blend_raw.json` | Full dataset blend |
| `config/data_blend_tiny.json` | Small blend for testing |

### Override Examples

```bash
# More training iterations
nemotron nano3 sft -c tiny train.train_iters=5000

# Different learning rate
nemotron nano3 sft -c tiny optimizer.lr=1e-5

# Load specific pretrained checkpoint
nemotron nano3 sft -c tiny checkpoint.load=/path/to/pretrain/checkpoint

# Multiple overrides
nemotron nano3 sft -c tiny \
    train.train_iters=5000 \
    optimizer.lr=1e-5 \
    checkpoint.save_interval=100
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
nemotron nano3 sft -c tiny --run YOUR-CLUSTER

# Detached (submit and exit immediately)
nemotron nano3 sft -c tiny --batch YOUR-CLUSTER

# Preview without executing
nemotron nano3 sft -c tiny --run YOUR-CLUSTER --dry-run
```

See [nemo-run.md](../../nemo-run.md) for complete configuration options.

## Artifact Lineage

```
ModelArtifact-pretrain (from Stage 0)
     ↓
Instruction Datasets (OpenAI chat format)
     ↓
data_prep.py
     ↓
DataBlendsArtifact-sft (packed .npy files)
     ↓
train.py
     ↓
ModelArtifact-sft (fine-tuned checkpoint)
     ↓
[Stage 2: RL]
```

## Next Steps

After SFT completes, proceed to [Stage 2: RL](./rl.md) for alignment training.

## Previous Stage

- [Stage 0: Pretraining](./pretrain.md) - Pretrain the base model

## Reference

- [Recipe Source](../../../src/nemotron/recipes/nano3/stage1_sft/) - Implementation details
- [Back to Overview](./README.md)
