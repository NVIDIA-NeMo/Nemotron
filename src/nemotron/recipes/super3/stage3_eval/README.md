# Stage 3: Evaluation

Evaluate the trained model on standard benchmarks using NeMo-Evaluator.

## Overview

This stage evaluates the aligned model from Stage 2 (RL) on standard NLP benchmarks. It integrates with the `nemo-evaluator-launcher` and supports vLLM-based deployment for inference during evaluation.

| Component | Description |
|-----------|-------------|
| `config/default.yaml` | Evaluation configuration for NeMo-Evaluator |

## Quick Start

### Using nemotron CLI (Recommended)

```bash
# Run evaluation with default tasks
uv run nemotron super3 eval --run YOUR-CLUSTER

# Evaluate a specific model version
uv run nemotron super3 eval --run YOUR-CLUSTER run.model=sft:v2

# Select specific tasks
uv run nemotron super3 eval --run YOUR-CLUSTER -t adlr_mmlu -t hellaswag

# Preview without executing
uv run nemotron super3 eval --dry-run
```

## Configuration

### Default Tasks

The default evaluation runs these benchmarks:

| Task | Description |
|------|-------------|
| `adlr_mmlu` | Massive Multitask Language Understanding |
| `hellaswag` | Commonsense NLI |
| `arc_challenge` | AI2 Reasoning Challenge |

### config/default.yaml

```yaml
run:
  model: rl:latest
  env:
    executor: local
    container: nvcr.io/nvidia/nemo-evaluator:latest

deployment:
  type: vllm
  checkpoint_path: ${art:model,path}
  tensor_parallel_size: 4
  data_parallel_size: 1
  extra_args: "--max-model-len 32768"

evaluation:
  tasks:
    - name: adlr_mmlu
    - name: hellaswag
    - name: arc_challenge

export:
  wandb:
    entity: ${run.wandb.entity}
    project: ${run.wandb.project}
```

### Override Examples

```bash
# Evaluate with more tasks
uv run nemotron super3 eval --run YOUR-CLUSTER -t adlr_mmlu -t hellaswag -t arc_challenge -t winogrande

# Evaluate SFT model (skip RL)
uv run nemotron super3 eval --run YOUR-CLUSTER run.model=sft:latest

# Quick test with limited samples
uv run nemotron super3 eval --run YOUR-CLUSTER evaluation.nemo_evaluator_config.config.params.limit_samples=10

# Override parallelism
uv run nemotron super3 eval --run YOUR-CLUSTER deployment.tensor_parallel_size=8
```

## Running with NeMo-Run

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
nodes = 1
ntasks_per_node = 8
gpus_per_node = 8
mounts = ["/lustre:/lustre"]
```

### Execution Modes

```bash
# Attached (wait for completion)
uv run nemotron super3 eval --run YOUR-CLUSTER

# Preview without executing
uv run nemotron super3 eval --dry-run
```

See [docs/nemo_runspec/nemo-run.md](../../../../docs/nemo_runspec/nemo-run.md) for complete configuration options.

## Results

Evaluation results are automatically exported to W&B when `auto_export` is enabled (default). Results include per-task accuracy scores and can be compared across training runs.

## Previous Stages

- [Stage 0: Pretraining](../stage0_pretrain/README.md) - Pretrain the base model
- [Stage 1: SFT](../stage1_sft/README.md) - Instruction tuning
- [Stage 2: RL](../stage2_rl/README.md) - Reinforcement learning for alignment
