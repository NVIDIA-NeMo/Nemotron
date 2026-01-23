# Stage 3: Evaluation

Evaluate trained models using NeMo-Evaluator, supporting multiple benchmark tasks and automatic results export.

## Overview

The evaluation stage integrates with `nemo-evaluator-launcher` to:
- Deploy your trained model using vLLM
- Run standardized benchmark tasks (MMLU, HellaSwag, etc.)
- Export results to W&B for tracking

## Prerequisites

Install the evaluator dependency:

```bash
pip install "nemotron[evaluator]"
```

## Quick Start

```bash
# Evaluate the RL model (default)
uv run nemotron nano3 eval --run YOUR-CLUSTER

# Evaluate a specific model
uv run nemotron nano3 eval --run YOUR-CLUSTER run.model=sft:v2

# Run specific tasks only
uv run nemotron nano3 eval --run YOUR-CLUSTER -t adlr_mmlu -t hellaswag

# Preview config without running
uv run nemotron nano3 eval --run YOUR-CLUSTER --dry-run
```

## Configuration

### Default Config

The default configuration (`config/default.yaml`) includes:

- **Model**: `run.model=rl:latest` (last RL checkpoint)
- **Deployment**: vLLM with TP=4 for Nano3 (30B MoE)
- **Tasks**: MMLU, HellaSwag, ARC-Challenge

### Config Structure

```yaml
# Nemotron artifact resolution
run:
  model: rl:latest  # Model artifact to evaluate
  env: {...}        # Populated from env.toml
  wandb: {...}      # Populated from env.toml [wandb]

# Evaluator launcher config
execution:
  type: slurm       # Execution backend (local/slurm)
  hostname: ...     # From ${run.env.host}
  
deployment:
  type: vllm
  checkpoint_path: ${art:model,path}  # Resolved from artifact
  tensor_parallel_size: 4
  
evaluation:
  tasks:
    - name: adlr_mmlu
    - name: hellaswag

export:
  wandb:
    entity: ${run.wandb.entity}
    project: ${run.wandb.project}
```

### Task Filtering

Use `-t/--task` to run specific tasks:

```bash
# Single task
uv run nemotron nano3 eval --run CLUSTER -t adlr_mmlu

# Multiple tasks
uv run nemotron nano3 eval --run CLUSTER -t adlr_mmlu -t hellaswag -t arc_challenge
```

## env.toml Integration

Evaluation uses the same `env.toml` profile as training:

```toml
[YOUR-CLUSTER]
executor = "slurm"
host = "login.cluster.com"
user = "myuser"
account = "my-account"
partition = "batch"
remote_job_dir = "/lustre/jobs"
time = "04:00:00"

[wandb]
entity = "my-org"
project = "nano3-evals"
```

The env.toml fields map to evaluator config:
- `host` → `execution.hostname`
- `user` → `execution.username`
- `account` → `execution.account`
- `partition` → `execution.partition`
- `remote_job_dir` → `execution.output_dir` base
- `time` → `execution.walltime`
- `[wandb]` → `export.wandb.*`

## Artifacts

### Input Artifact

By default, evaluates the RL stage output. Override with:

```bash
# Evaluate SFT checkpoint
uv run nemotron nano3 eval --run CLUSTER run.model=sft:latest

# Evaluate specific version
uv run nemotron nano3 eval --run CLUSTER run.model=sft:v2
```

### Output

Results are exported to W&B as specified in `export.wandb`. Check status:

```bash
nemo-evaluator-launcher status <invocation_id>
nemo-evaluator-launcher logs <invocation_id>
```

## Local Execution

For local testing without Slurm:

```bash
# Set execution type to local
uv run nemotron nano3 eval execution.type=local
```

## Generic Evaluate Command

For custom evaluation configs not tied to nano3:

```bash
# Requires explicit config path
uv run nemotron evaluate -c /path/to/eval.yaml --run YOUR-CLUSTER
```

## Troubleshooting

### Missing Evaluator Package

```
Error: nemo-evaluator-launcher is required for evaluation
Install with: pip install "nemotron[evaluator]"
```

### Task Not Found

```
Error: Requested task(s) not found in config: ['missing_task']
Available tasks: ['adlr_mmlu', 'hellaswag', 'arc_challenge']
```

Check available tasks in your config or use `nemo-evaluator-launcher tasks` to list all available tasks.

## Further Reading

- [NeMo-Evaluator Documentation](https://github.com/NVIDIA-NeMo/Evaluator)
- [env.toml Configuration](../../../../docs/train/nemo-run.md)
