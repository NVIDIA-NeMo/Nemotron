# Evaluation Framework

The Nemotron evaluation framework provides model evaluation capabilities using [NeMo Evaluator](https://github.com/NVIDIA/nemo-evaluator-launcher), enabling benchmark testing of trained models on standard NLP tasks.

<div class="termy">

```console
$ uv run nemotron evaluate -c nemotron-3-nano-nemo-ray --run MY-CLUSTER
Compiled Configuration
╭──────────────────────────────────── run ─────────────────────────────────────╮
│ wandb:                                                                       │
│   project: nemotron                                                          │
│   entity: my-team                                                            │
╰──────────────────────────────────────────────────────────────────────────────╯

[info] Detected W&B login, setting WANDB_API_KEY

Starting evaluation...
✓ Evaluation submitted: 480d3c89bfe4a55c
Check status: nemo-evaluator-launcher status 480d3c89bfe4a55c
```

</div>

## Overview

The evaluation framework enables:

- **Benchmark Testing** — Run standard benchmarks (MMLU, ARC, HellaSwag, etc.) on your models
- **W&B Integration** — Auto-export results to Weights & Biases for tracking
- **Slurm Execution** — Submit evaluation jobs to HPC clusters
- **Auto-Squash** — Automatically converts Docker images to squashfs for Slurm clusters
- **Credential Auto-Propagation** — Automatically passes W&B tokens to remote jobs

The evaluator uses the same `env.toml` execution profiles as training recipes, providing a unified experience across all stages.

## Quick Start

```bash
# Run evaluation on a cluster
uv run nemotron evaluate -c nemotron-3-nano-nemo-ray --run MY-CLUSTER

# Preview config without executing
uv run nemotron evaluate -c nemotron-3-nano-nemo-ray --dry-run

# Filter to specific tasks
uv run nemotron evaluate -c nemotron-3-nano-nemo-ray --run MY-CLUSTER -t adlr_mmlu

# Override checkpoint path
uv run nemotron evaluate -c nemotron-3-nano-nemo-ray --run MY-CLUSTER \
    deployment.checkpoint_path=/path/to/your/checkpoint
```

## CLI Options

| Option | Short | Description |
|--------|-------|-------------|
| `--config` | `-c` | Config name or path |
| `--run` | `-r` | Submit to cluster (attached, streams logs) |
| `--batch` | `-b` | Submit to cluster (detached, exits immediately) |
| `--dry-run` | `-d` | Preview config without executing |
| `--task` | `-t` | Filter to specific task(s), can be repeated |
| `--force-squash` | | Force re-squash even if cached |

### Task Filtering

Run specific benchmarks using the `-t` flag:

```bash
# Single task
uv run nemotron evaluate -c config --run MY-CLUSTER -t adlr_mmlu

# Multiple tasks
uv run nemotron evaluate -c config --run MY-CLUSTER -t adlr_mmlu -t hellaswag
```

### Available Tasks

Common evaluation tasks include:

| Task | Description |
|------|-------------|
| `adlr_mmlu` | Massive Multitask Language Understanding |
| `adlr_arc_challenge_llama_25_shot` | AI2 Reasoning Challenge |
| `adlr_winogrande_5_shot` | Winograd Schema Challenge |
| `hellaswag` | Commonsense reasoning |
| `openbookqa` | Open-domain question answering |

## Execution Profiles

The evaluator uses the same `env.toml` profiles as training recipes. See [Execution through NeMo-Run](./nemo-run.md) for full documentation.

### Basic Profile

```toml
# env.toml

[wandb]
project = "nemotron"
entity = "my-team"

[MY-CLUSTER]
executor = "slurm"
account = "my-account"
partition = "batch"
tunnel = "ssh"
host = "cluster.example.com"
user = "myuser"
remote_job_dir = "/lustre/fsw/users/myuser/.nemotron"
```

### Profile with Auto-Squash

Slurm clusters use [Pyxis](https://github.com/NVIDIA/pyxis) with enroot for container execution. While you can use Docker references directly, pre-squashed `.sqsh` files significantly speed up job startup by avoiding container pulls on each run.

With SSH tunnel settings, the CLI can automatically create squash files from Docker references:

```toml
[MY-CLUSTER]
executor = "slurm"
account = "my-account"
partition = "batch"

# SSH settings (enables auto-squash)
tunnel = "ssh"
host = "cluster.example.com"
user = "myuser"
remote_job_dir = "/lustre/fsw/users/myuser/.nemotron"

# Container settings - use Docker ref, auto-squashed on first run
container_image = "nvcr.io/nvidia/nemo:25.01"
```

When you run with `--run MY-CLUSTER`, the CLI will:
1. Detect that `deployment.image` is a Docker reference (not a `.sqsh` path)
2. SSH to the cluster and run `enroot import` on a compute node
3. Cache the `.sqsh` file in `${remote_job_dir}/containers/` for reuse
4. Update the config to use the squashed path

Subsequent runs reuse the cached squash file, eliminating container pull overhead.

## Configuration

Evaluation configs define how to deploy your model and which benchmarks to run.

### Example Config

```yaml
# Execution (Slurm settings)
execution:
  type: slurm
  hostname: ${run.env.host}
  account: ${run.env.account}
  partition: ${run.env.partition}
  num_nodes: 1
  gres: gpu:8

  # Auto-export to W&B after evaluation
  auto_export:
    enabled: true
    destinations:
      - wandb

# Deployment (Model serving)
deployment:
  type: generic
  image: ${run.env.container}  # Docker image or .sqsh path
  checkpoint_path: /path/to/checkpoint
  command: >-
    python deploy_ray_inframework.py
    --megatron_checkpoint /checkpoint/
    --num_gpus 8

# Evaluation (Tasks to run)
evaluation:
  tasks:
    - name: adlr_mmlu
    - name: hellaswag
    - name: openbookqa

# Export (W&B settings)
export:
  wandb:
    entity: ${run.wandb.entity}
    project: ${run.wandb.project}
```

### Key Sections

| Section | Purpose |
|---------|---------|
| `run.env` | Environment settings from env.toml (cluster, container) |
| `run.wandb` | W&B settings from env.toml `[wandb]` section |
| `execution` | Slurm executor configuration (nodes, GPUs, account) |
| `deployment` | Model deployment (container, checkpoint, command) |
| `evaluation` | Tasks and evaluation parameters |
| `export` | Result export destinations (W&B) |

## Auto-Squash

For Slurm clusters that require squashfs containers, the evaluator automatically converts Docker images to `.sqsh` files—the same behavior as training recipes.

### How It Works

1. **Detection** — CLI checks if `deployment.image` is a Docker reference (not already `.sqsh`)
2. **SSH Connection** — Connects to cluster via SSH tunnel (using `host` and `user` from env.toml)
3. **Squash** — Runs `enroot import` on a compute node to create the `.sqsh` file
4. **Cache** — Stores the squash file in `${remote_job_dir}/containers/` for reuse
5. **Config Update** — Rewrites `deployment.image` to use the squashed path

### Usage

```bash
# Auto-squash happens automatically for Docker refs
uv run nemotron evaluate -c config --run MY-CLUSTER

# Force re-squash (ignores cache)
uv run nemotron evaluate -c config --run MY-CLUSTER --force-squash

# Already-squashed paths skip the step
# (if deployment.image ends in .sqsh, no squashing needed)
```

### Requirements

Auto-squash requires these settings in your `env.toml` profile:

| Field | Required | Description |
|-------|----------|-------------|
| `executor` | Yes | Must be `"slurm"` |
| `tunnel` | Yes | Must be `"ssh"` |
| `host` | Yes | SSH hostname (e.g., `cluster.example.com`) |
| `user` | No | SSH username (defaults to current user) |
| `remote_job_dir` | Yes | Remote directory for job files and squash cache |

## W&B Integration

The evaluator automatically propagates W&B credentials when you're logged in locally—the same behavior as training recipes.

### Setup

1. **Login to W&B locally:**
   ```bash
   wandb login
   ```

2. **Configure env.toml** (same `[wandb]` section used by all recipes):
   ```toml
   [wandb]
   project = "nemotron"
   entity = "my-team"
   ```

3. **Run evaluation** — credentials are automatically passed:
   ```bash
   uv run nemotron evaluate -c config --run MY-CLUSTER
   # [info] Detected W&B login, setting WANDB_API_KEY
   ```

### What Gets Propagated

| Variable | Source | Description |
|----------|--------|-------------|
| `WANDB_API_KEY` | Local wandb login | Auto-detected via `wandb.api.api_key` |
| `WANDB_PROJECT` | `env.toml [wandb]` | Project name for result tracking |
| `WANDB_ENTITY` | `env.toml [wandb]` | Team/user entity |

## Monitoring Jobs

### Check Status

```bash
# Using nemo-evaluator-launcher directly
nemo-evaluator-launcher status INVOCATION_ID

# Check Slurm queue
ssh cluster squeue -u $USER
```

### Stream Logs

```bash
nemo-evaluator-launcher logs INVOCATION_ID
```

### Cancel Jobs

```bash
# Cancel via Slurm
ssh cluster scancel JOB_ID

# Or multiple jobs
ssh cluster "scancel JOB_ID1 JOB_ID2 JOB_ID3"
```

## Creating Custom Configs

### Step 1: Create Config File

```yaml
# src/nemotron/recipes/evaluator/config/my-model.yaml

defaults:
  - execution: slurm/default
  - deployment: generic
  - _self_

run:
  env:
    container: nvcr.io/nvidia/nemo:25.01  # Docker ref (auto-squashed)
    # OR: container: /path/to/container.sqsh  # Pre-squashed
  wandb:
    entity: null  # Populated from env.toml
    project: null

execution:
  type: slurm
  hostname: ${run.env.host}
  account: ${run.env.account}
  num_nodes: 1
  gres: gpu:8

  auto_export:
    enabled: true
    destinations:
      - wandb

deployment:
  type: generic
  image: ${run.env.container}
  checkpoint_path: /path/to/your/model/checkpoint
  command: >-
    python deploy_script.py --checkpoint /checkpoint/

evaluation:
  tasks:
    - name: adlr_mmlu
    - name: hellaswag

export:
  wandb:
    entity: ${run.wandb.entity}
    project: ${run.wandb.project}
```

### Step 2: Run Evaluation

```bash
uv run nemotron evaluate -c my-model --run MY-CLUSTER
```

## Troubleshooting

### "Missing key type" Error

Ensure your config has all required Slurm fields:

```yaml
execution:
  type: slurm  # Required
  ntasks_per_node: 1  # Required
  gres: gpu:8  # Required
```

### W&B Credentials Not Detected

1. Verify you're logged in: `wandb login`
2. Check env.toml has `[wandb]` section
3. Look for `[info] Detected W&B login` message

### Auto-Squash Not Working

1. Verify `tunnel = "ssh"` in your env.toml profile
2. Check `host` and `remote_job_dir` are set
3. Ensure `nemo-run` is installed: `pip install nemo-run`

### Jobs Stuck in PENDING

Check queue status:
```bash
ssh cluster "squeue -p batch | head"
```

Common reasons:
- `(Priority)` — Waiting for resources
- `(Resources)` — Insufficient available nodes
- `(QOSMaxJobsPerUserLimit)` — User job limit reached

## Further Reading

- [Execution through NeMo-Run](./nemo-run.md) — Execution profiles and env.toml
- [W&B Integration](./wandb.md) — Credentials and artifact tracking
- [NeMo Evaluator Documentation](https://github.com/NVIDIA/nemo-evaluator-launcher) — Launcher reference
