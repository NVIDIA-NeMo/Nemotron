# Stage 2: Reinforcement Learning (RL)

Align the instruction-tuned model using GRPO (Group Relative Policy Optimization) with NeMo-RL.

## Overview

This stage takes the SFT model and further aligns it using reinforcement learning. The GRPO algorithm optimizes the policy based on reward signals from NeMo-Gym environments, producing a final aligned model.

Nemotron 3 Nano is trained in three stages: supervised fine tuning (SFT), multi-environment reinforcement learning (RLVR), and reinforcement learning from human feedback (RLHF) (see [Tech Report Section 3.2](https://arxiv.org/abs/2506.XXXXX)). During SFT, we trained Nemotron 3 Nano on a diverse set of chat, agentic, and reasoning tasks to imbue the model with reasoning budget control, reasoning on/off control, and tool-integrated reasoning capabilities. Following SFT, we used multi-environment RL to strengthen model capabilities. We trained on all environments simultaneously, resulting in a smooth and uniform improvement in model capabilities. During RLHF, we utilized a large and accurate generative reward model (GenRM) to enhance the performance of Nemotron 3 Nano on key chat benchmarks.

> **Note**: This recipe uses only the **open-sourced subset** of RL data. Results are not expected to match the full tech report benchmarks. This serves as a **reference implementation** for the RL methodology.

| Component | Description |
|-----------|-------------|
| `data_prep.py` | Converts datasets to JSONL format for NeMo-RL |
| `train.py` | Runs GRPO training using NeMo-RL with Ray |
| `config/` | Configuration files for data prep and training |

## Multi-Environment Reinforcement Learning from Verifiable Rewards (RLVR)

We employ a unified RLVR stage, training on all environments simultaneously. This results in stable gains across all benchmarks throughout training, while single environment training often results in co-reward degradation of other benchmarks. We do two stages of each RLVR: one immediately after SFT and one after RLHF.

### Environments

We train on five different reward environments using NeMo-Gym:

| Environment | Datasets | Description |
|-------------|----------|-------------|
| **Competition Math** | DAPO, SkyWorks | Mathematical reasoning with verifiable solutions |
| **Competition Coding** | Competitive coding problems | Code correctness verification with test cases |
| **Question Answering** | Multiple choice datasets | STEM-focused QA with verifiable answers |
| **Structured Outputs** | JSON schema adherence | Strong JSON schema compliance with syntactic validity |
| **Instruction Following** | IFEval-style environments | Multi-constraint instruction compliance verification |
| **Long Context** | Challenging long-context QA pairs | Multi-document synthesis with reference verification |
| **Agentic Tool Use** | Workplace Assistant, Multi-Turn Agent | Tool call correctness and sandbox task completion |

### Infrastructure

RL at the frontier of model post-training is currently defined by scaling up to an increasing diversity of tasks or environments designed for the model to learn increasingly general capabilities. Scaling RL to many environments requires a high-performance, extensible, and standardized interface for coordinating between rollouts and training.

We use **NeMo-Gym** based on the abstraction of servers. There are three core varieties of servers in Gym:
1. **Agents** (2) models, and (3) resources
2. An agent server implements the rollout kernel at a RL environment
3. A model server wraps an inference engine such as vLLM to provide a prompt-response API

### GRPO Algorithm Details

We train Nemotron 3 Nano using synchronous GRPO with masked importance sampling to mitigate training-inference alignment mismatch. Key hyperparameters:
- **Prompts per step**: 128
- **Generations per prompt**: 16
- **Batch Size**: 128 prompts per batch, 16 generations per prompt
- **MoE Load Balancing**: DeepSeek's aux-loss-free load balancing strategy with keep updating our update-on-policy
- **Cosine Annealing**: Epsilon filtering with 4% limit

Our entire training run is done with a maximum generation length of 49K tokens. We use cosine filtering, which we find boosts performance on reasoning-intensive benchmarks.

### Curriculum Sampling

We compare curriculum sampling against random sampling using an intermediate SFT checkpoint, maintaining identical domain ratios in both cases. As shown in the tech report, curriculum sampling ensures stable learning across multiple domains throughout training. In contrast, random sampling biases the model toward easier tasks, preventing it from effectively learning more challenging ones.

For each domain, we model the target pass-rate distribution as a Gaussian function, shifting from high pass-rate (easier) samples early in training to low pass-rate (harder) samples later. The target mean of Gaussian distribution decreases linearly throughout training steps. Within each batch, samples from different domains are shuffled. This Gaussian sampling strategy prevents overfitting to either overly easy or overly difficult examples, ensuring a balanced learning progression.

## Reinforcement Learning from Human Feedback (RLHF)

### GenRM: Generative Reward Model Training

Many recent works have demonstrated that generative reward models (GenRM) generalize better than traditional Bradley-Terry models, reducing the risk of reward hacking during RLHF. Building on the methodology of GenRM, we train Qwen3-253B-A22B-Thinking-2507 using Qwen3-253B-A22B-Thinking-2507 to become a GenRM with GRPO algorithm.

Given the conversation history, a new user request, and two candidate assistant responses, the GenRM first reasons through the strengths and weaknesses of both responses, then produces an individual helpfulness score for each response as well as a ranking score. For GenRM training, we use 128 prompts per batch, 8 generations per prompt, and do one gradient step on the full batch.

### RLHF with Group Relative Length Control

With a trained GenRM, we conduct RLHF on the same set of prompts. Same as RLVR, we use a batch of 128 prompts and 16 responses per prompt. Notably competing all pairs of N responses would require O(N) comparisons per prompt, which scales quadratically and becomes prohibitively expensive for large N. Instead, we adopt a circular comparison strategy where each response is compared only with its neighbor: `r₁→r₂→r₃→...→rₙ→r₁`, yielding exactly N comparisons.

Key RLHF training details:
- **Length-Normalized Reward Adjustment**: We compute a zero-mean, group-relative length bonus that encourages shorter responses within a group
- **Quality-Gated Conciseness Bonus**: We introduce optimal bonuses for the shortest responses without sacrificing quality

### Reasoning Control

Nemotron 3 Nano allows for two different forms of reasoning control: reasoning on/off control and token budget control. Similar to NVIDIA (2025), to enable reasoning on/off control we strip the reasoning traces from a random 10% of samples, and to enable budget control, we randomly truncate 3% of reasoning traces to different reasoning budgets, before continuing with the original post-reasoning response.

## DPO for Reducing Tool Hallucination

Reducing hallucinated tool usage is one of the key objectives of our alignment experiments. Although our released model does not rely on DPO, because reinforcement learning (RL) already achieved comparable performance, we nevertheless explored DPO as an additional technique due to its simplicity and minimal computational overhead. As shown in the tech report, a very small amount of DPO training yields meaningful reductions in hallucinated tool calls and improves reasoning stability.

### Training Setup for DPO
- **Learning Rate**: 3e-6
- **Batch Size**: 128
- **Training Steps**: 50
- **SFT Loss Coefficient**: 0.2
- **DPO Loss Coefficient**: 1.0
- **KL Loss Coefficient**: 0.05

For AIME25, accuracy increases from 80.88% to 84.58%, indicating that DPO not only suppresses undesirable tool-related behaviors but also enhances overall solution quality.

## Quick Start

### Using nemotron CLI (Recommended)

```bash
# 1. Prepare data (convert to JSONL format)
nemotron nano3 data prep rl --run YOUR-CLUSTER

# 2. Run RL training
nemotron nano3 rl --run YOUR-CLUSTER

# Quick test with tiny config
nemotron nano3 rl -c tiny --run YOUR-CLUSTER
```

### Direct Script Execution

Inside a container on a compute node (requires NeMo-RL and Ray):

```bash
# Data preparation
python data_prep.py --config config/data_prep.yaml

# Training (Ray will be initialized internally)
python train.py --config config/grpo_nanov3.yaml
```

## Data Preparation

The `data_prep.py` script converts datasets to JSONL format compatible with NeMo-RL's NeMo-Gym interface.

### CLI Command

```bash
nemotron nano3 data prep rl [options]
```

| Option | Description |
|--------|-------------|
| `--run <profile>` | Execute on Slurm via NeMo-Run |
| `--sample N` | Limit rows per dataset (for testing) |
| `--force` | Force re-run, ignoring cache |

### Input

RL datasets defined in `config/data_blend_raw.json`. The data is transformed using the `nemotron_rl` transform which extracts from `responses_create_params.input`.

### Output

```
output/nano3/stage2_rl/
├── train/
│   └── data.jsonl       # Training data in NeMo-Gym format
├── val/
│   └── data.jsonl       # Validation data
├── test/
│   └── data.jsonl       # Test data
└── manifest.json        # Split paths and ratios
```

The output is registered as a W&B Artifact (`DataBlendsArtifact-rl`) for lineage tracking.

### Configuration

`config/data_prep.yaml`:

```yaml
blend_path: config/data_blend_raw.json
output_dir: output/nano3/stage2_rl
shard_size: 256MB
split_output: train_val_test
train_ratio: 0.98
val_ratio: 0.01
```

| Parameter | Description |
|-----------|-------------|
| `split_output` | `train_val_test` for separate splits, `none` for single output |
| `train_ratio` | Fraction for training split (default 0.98) |
| `val_ratio` | Fraction for validation split (default 0.01) |

## Training

The `train.py` script runs GRPO training using NeMo-RL with Ray for distributed execution.

### CLI Command

```bash
nemotron nano3 rl [options] [overrides...]
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

- **Model**: SFT checkpoint from Stage 1 (`ModelArtifact-sft`)
- **Data**: `DataBlendsArtifact-rl` (from data prep)
- **Config**: `config/grpo_nanov3.yaml` or `config/tiny.yaml`

### Output

- Aligned model checkpoints
- Training logs and metrics
- Registered as W&B Artifact (`ModelArtifact-rl`)

### Configuration Files

| File | Purpose |
|------|---------|
| `config/grpo_nanov3.yaml` | Production GRPO configuration |
| `config/tiny.yaml` | Testing variant |
| `config/data_blend_raw.json` | RL dataset blend (6 datasets) |

### Key Configuration Sections

```yaml
policy:
  model_name: "path/to/sft/checkpoint"
  tokenizer: "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
  generation:
    temperature: 0.7
    max_new_tokens: 1024

grpo:
  num_iterations: 100
  batch_size: 32
  learning_rate: 1e-6

data:
  train_jsonl_fpath: "/path/to/train/data.jsonl"
  validation_jsonl_fpath: "/path/to/val/data.jsonl"

env:
  nemo_gym:
    # NeMo-Gym environment configuration
```

### Override Examples

```bash
# More iterations
nemotron nano3 rl -c tiny grpo.num_iterations=200

# Different temperature
nemotron nano3 rl -c tiny policy.generation.temperature=0.8

# Different learning rate
nemotron nano3 rl -c tiny grpo.learning_rate=5e-7

# Multiple overrides
nemotron nano3 rl -c tiny \
    grpo.num_iterations=200 \
    policy.generation.temperature=0.8 \
    grpo.learning_rate=5e-7
```

## Running with NeMo-Run

The nemotron CLI uses [NeMo-Run](https://github.com/NVIDIA-NeMo/Run) for job orchestration. RL training uses Ray internally for distributed execution.

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
mem = "0"
exclusive = true
mounts = ["/lustre:/lustre"]
```

> **Note**: Container images are specified in the recipe config files (e.g., `config/tiny.yaml`), not in env.toml.

### Execution Examples

```bash
# Attached (wait for completion, stream logs)
nemotron nano3 rl -c tiny --run YOUR-CLUSTER

# Detached (submit and exit immediately)
nemotron nano3 rl -c tiny --batch YOUR-CLUSTER

# Preview without executing
nemotron nano3 rl -c tiny --run YOUR-CLUSTER --dry-run
```

See [nemo-run.md](../../nemo-run.md) for complete configuration options.

## GRPO Algorithm

GRPO (Group Relative Policy Optimization) is a reinforcement learning algorithm that:

1. **Generates responses** from the current policy
2. **Evaluates** responses using NeMo-Gym reward environments
3. **Computes group-relative advantages** across response groups
4. **Updates the policy** to favor higher-reward responses

Key features:
- Efficient batched generation and evaluation
- Ray-based distributed training
- Integration with NeMo-Gym for flexible reward computation

## Artifact Lineage

```
ModelArtifact-sft (from Stage 1)
     ↓
RL Datasets (preference/reward data)
     ↓
data_prep.py
     ↓
DataBlendsArtifact-rl (JSONL files)
     ↓
train.py (GRPO with NeMo-RL)
     ↓
ModelArtifact-rl (final aligned model)
```

## Requirements

- **NeMo-RL**: Required for GRPO training
- **Ray**: Automatically initialized for distributed execution
- **NeMo-Gym**: Provides reward environments
- **GPU nodes**: Recommended 8 GPUs per node

## Previous Stages

- [Stage 0: Pretraining](./pretrain.md) - Pretrain the base model
- [Stage 1: SFT](./sft.md) - Instruction tuning

## Reference

- [Recipe Source](../../../src/nemotron/recipes/nano3/stage2_rl/) - Implementation details
- [Back to Overview](./README.md)
