# Stage 2: Reinforcement Learning (RL)

This stage aligns the instruction-tuned model using GRPO (Group Relative Policy Optimization) with [NeMo-RL](../nvidia-stack.md#nemo-rl).

> **Open-Source Data Only**: This recipe uses exclusively open-sourced RL data from the [Nemotron Post-training Datasets](https://huggingface.co/collections/nvidia/nemotron-post-training-v3) collection, which is a subset of the full data used to train the released model. The recipe uses the [Nemotron-3-Nano-RL-Training-Blend](https://huggingface.co/datasets/nvidia/Nemotron-3-Nano-RL-Training-Blend) dataset. Results will differ from the benchmarks in the [tech report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf). Use this recipe as a reference implementation to apply the methodology with your own data.

## Quick Start

<div class="termy">

```console
// 1. Prepare data (convert to JSONL format)
$ uv run nemotron nano3 data prep rl --run YOUR-CLUSTER

// 2. Run RL training
$ uv run nemotron nano3 rl --run YOUR-CLUSTER
```

</div>

> **Note**: The `--run YOUR-CLUSTER` flag submits jobs via [NeMo-Run](../nemo-run.md). See [Execution through NeMo-Run](../nemo-run.md) for setup.

### Direct Script Execution

Inside a container on a compute node (requires [NeMo-RL](../nvidia-stack.md#nemo-rl) and Ray):

```bash
# Data preparation
uv run python data_prep.py --config config/data_prep.yaml

# Training (Ray initialized internally)
uv run python train.py --config config/grpo_nanov3.yaml
```

## Configuration

| File | Purpose |
|------|---------|
| `config/grpo_nanov3.yaml` | Production GRPO configuration |
| `config/data_prep.yaml` | Data preparation settings |
| `config/data_blend_raw.json` | RL dataset blend |

## Data Preparation

The `data_prep.py` script converts datasets to JSONL format compatible with [NeMo-RL](../nvidia-stack.md#nemo-rl)'s NeMo-Gym interface. See [Data Preparation Module](../data-prep.md) for detailed documentation.

### CLI Command

```bash
uv run nemotron nano3 data prep rl [options]
```

| Option | Description |
|--------|-------------|
| `--run <profile>` | Execute on Slurm via [NeMo-Run](../nemo-run.md) |
| `--sample N` | Limit rows per dataset (for testing) |
| `--force` | Force re-run, ignoring cache |

### Output

```
output/nano3/stage2_rl/
├── train/
│   └── data.jsonl
├── val/
│   └── data.jsonl
├── test/
│   └── data.jsonl
└── manifest.json
```

The output is registered as a [W&B Artifact](../artifacts.md) (`DataBlendsArtifact-rl`) for lineage tracking.

## Training

### CLI Command

```bash
uv run nemotron nano3 rl [options] [overrides...]
```

| Option | Description |
|--------|-------------|
| `--run <profile>` | Attached—submits and waits, streaming logs ([NeMo-Run](../nemo-run.md)) |
| `--batch <profile>` | Detached—submits and exits immediately ([NeMo-Run](../nemo-run.md)) |
| `--dry-run` | Preview execution plan |
| `key=value` | Override config values ([CLI Framework](../cli.md#dotlist-overrides)) |

### Override Examples

```bash
# More iterations
uv run nemotron nano3 rl grpo.num_iterations=200

# Different temperature
uv run nemotron nano3 rl policy.generation.temperature=0.8

# Different learning rate
uv run nemotron nano3 rl grpo.learning_rate=5e-7
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
mem = "0"
exclusive = true
mounts = ["/lustre:/lustre"]
```

See [Execution through NeMo-Run](../nemo-run.md) for complete configuration options.

## Artifact Lineage

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryBorderColor': '#333333', 'lineColor': '#333333', 'primaryTextColor': '#333333'}}}%%
flowchart TB
    prev["ModelArtifact-sft<br/>(from Stage 1)"] --> train
    rl["RL Datasets<br/>(preference/reward data)"] --> dp["data_prep.py"]
    dp --> data["DataBlendsArtifact-rl<br/>(JSONL files)"]
    data --> train["train.py<br/>(GRPO with NeMo-RL)"]
    train --> model["ModelArtifact-rl<br/>(final aligned model)"]

    style prev fill:#f3e5f5,stroke:#9c27b0
    style rl fill:#e8f5e9,stroke:#4caf50
    style dp fill:#e8f5e9,stroke:#4caf50
    style data fill:#e8f5e9,stroke:#4caf50
    style train fill:#e8f5e9,stroke:#4caf50
    style model fill:#e8f5e9,stroke:#4caf50
```

## Methodology

> For complete methodology, see [Tech Report Section 3.2](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf).

The RL pipeline consists of three components:
1. **RLVR** — Multi-environment training with verifiable rewards
2. **RLHF with GenRM** — Generative reward model-based alignment
3. **DPO** — Preference learning to reduce tool hallucination

### Multi-Environment RLVR

Training uses 7 reward environments through NeMo-Gym:

| Environment | Description |
|-------------|-------------|
| **Competition Math** | Mathematical reasoning (DAPO, SkyWorks math) |
| **Competition Coding** | Code correctness with test case execution |
| **Question Answering** | STEM multiple choice verification |
| **Structured Outputs** | JSON schema adherence |
| **Instruction Following** | IFEval, Multi-Challenge compliance |
| **Long Context** | 256k token multi-document synthesis |
| **Agentic Tool Use** | Workplace Assistant, Multi-Turn Agent |

Training on all environments simultaneously provides stable gains without co-reward degradation.

> For GRPO algorithm details, see [Tech Report Section 3.2](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf).

### GenRM (RLHF)

Generative reward models use circular comparison strategy (N comparisons instead of O(N²)) with length-normalized reward adjustment.

| Parameter | Value |
|-----------|-------|
| **Prompts per batch** | 128 |
| **Responses per prompt** | 16 |
| **Comparison strategy** | Circular |
| **Length bonus α** | 0.5 |

> For GenRM training details, see [Tech Report Section 3.2](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf).

### DPO for Tool Hallucination

DPO reduces hallucinated tool usage with minimal computational overhead:

| Metric | Before DPO | After DPO |
|--------|------------|-----------|
| **AIME25 Accuracy** | 80.88% | 84.58% |
| **Hallucination Rate** | 8.33% | 0.7% |

> For DPO methodology, see [Tech Report Appendix C](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf).

### GRPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Prompts per step** | 128 |
| **Generations per prompt** | 16 |
| **Max Generation Length** | 49K tokens |
| **Epsilon Filtering** | Cosine annealing with 4% limit |
| **MoE Load Balancing** | DeepSeek aux-loss-free strategy |

### Reasoning Control

The model supports:
- **Reasoning on/off control** — Strip reasoning from 10% of samples
- **Token budget control** — Truncate 3% of reasoning traces to different budgets

## Requirements

- **GPU nodes**: Recommended 8 GPUs per node (H100)
- **Ray cluster**: Automatically initialized for distributed execution

## NVIDIA AI Stack

This stage uses the following components from the [NVIDIA AI Stack](../nvidia-stack.md):

| Component | Role | Documentation |
|-----------|------|---------------|
| [NeMo-RL](../nvidia-stack.md#nemo-rl) | GRPO algorithm, policy training, reward computation | [Docs](https://docs.nvidia.com/nemo/rl/latest/) |
| [Ray](https://ray.io/) | Distributed actor coordination | [Docs](https://docs.ray.io/) |
| vLLM | Fast rollout generation | [GitHub](https://github.com/vllm-project/vllm) |

### Key Features Used

| Feature | Purpose |
|---------|---------|
| GRPO algorithm | Group Relative Policy Optimization with clipped gradients |
| Multi-environment training | Simultaneous training across 7 reward environments |
| NeMo-Gym | Reward environments (math, code, tool-use) |
| DTensor backend | FSDP2-based distributed training |

### Architecture

NeMo-RL uses a Ray-based actor model:

| Actor | Function |
|-------|----------|
| Policy Model | Trainable policy weights |
| Generator | vLLM-backed rollout generation |
| Reward Model | Environment-specific reward computation |
| Reference Model | KL divergence regularization |

### Container

```
nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano
```

## Open-Source Data

> **Note**: This recipe trains exclusively on the open-sourced subset of RL data. Results will differ from the tech report benchmarks, which used additional proprietary data.

## Reference

- [Tech Report Section 3.2](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf) — RL methodology
- [NVIDIA AI Stack](../nvidia-stack.md) — NeMo-RL documentation
- [Artifact Lineage](../artifacts.md) — W&B artifact system
- [Stage 0: Pretraining](./pretrain.md) — Pretrain the base model
- [Stage 1: SFT](./sft.md) — Instruction tuning
- **Recipe Source**: `src/nemotron/recipes/nano3/stage2_rl/` — Implementation details
- [Back to Overview](./README.md)
