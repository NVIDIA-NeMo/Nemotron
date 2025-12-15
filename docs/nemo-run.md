# Running Recipes with NeMo-Run

Nemotron recipes work out of the box without any additional dependencies. For distributed execution on clusters or cloud, you can optionally use [NeMo-Run](https://github.com/NVIDIA-NeMo/Run) by adding `--run <profile>` to any recipe command.

## Quick Start

```bash
# Local execution (works without nemo-run)
python -m nemotron.recipes.nano3.stage0_pretrain.train --config.data.mock

# Optional: Execute on a Slurm cluster (requires nemo-run)
python -m nemotron.recipes.nano3.stage0_pretrain.train --run draco --config.data.mock

# Optional: Execute in Docker (requires nemo-run)
python -m nemotron.recipes.nano3.data_prep --run docker

# Optional: Execute on AWS via SkyPilot (requires nemo-run)
python -m nemotron.recipes.nano3.stage0_pretrain.train --run aws
```

## Installation

NeMo-Run is **optional**. Install it only if you need distributed execution:

```bash
pip install nemo-run
```

Without nemo-run installed, all recipes run locally using standard Python/torchrun.

## Setting Up Run Profiles

Create a `run.toml` (or `run.yaml` / `run.json`) in your project root. Each section defines a named execution profile:

```toml
# run.toml

[local]
executor = "local"
nproc_per_node = 8

[docker]
executor = "docker"
container_image = "nvcr.io/nvidia/nemo:24.01"
nproc_per_node = 8
runtime = "nvidia"
mounts = ["/data:/data"]

[draco]
executor = "slurm"
account = "my-account"
partition = "gpu"
nodes = 4
nproc_per_node = 8
time = "04:00:00"
container_image = "nvcr.io/nvidia/nemo:24.01"
mounts = ["/data:/data", "/models:/models"]

[aws]
executor = "skypilot"
cloud = "aws"
gpus = "A100:8"
nodes = 2
cluster_name = "nemotron-training"
```

## Running Recipes

### Data Preparation

```bash
# Local
python -m nemotron.recipes.nano3.data_prep --sample 1000

# On Slurm cluster
python -m nemotron.recipes.nano3.data_prep --run draco --sample 1000

# In Docker
python -m nemotron.recipes.nano3.data_prep --run docker --sample 1000
```

### Pretraining

```bash
# Local with mock data
python -m nemotron.recipes.nano3.stage0_pretrain.train --config.data.mock

# On Slurm with 4 nodes
python -m nemotron.recipes.nano3.stage0_pretrain.train --run draco

# On Slurm with 8 nodes (override profile)
python -m nemotron.recipes.nano3.stage0_pretrain.train --run draco --run.nodes 8
```

### Supervised Fine-Tuning

```bash
# Local
python -m nemotron.recipes.nano3.stage1_sft.train

# On Slurm
python -m nemotron.recipes.nano3.stage1_sft.train --run draco
```

### RL Training

```bash
# Local (requires Ray)
python -m nemotron.recipes.nano3.stage2_rl.train

# On Slurm (Ray cluster started automatically)
python -m nemotron.recipes.nano3.stage2_rl.train --run draco
```

## CLI Options

```bash
# Select a profile (attached execution - waits for completion)
python train.py --run <profile-name>

# Batch mode (detached execution - submits and exits immediately)
python train.py --batch <profile-name>

# Override profile settings
python train.py --run draco --run.nodes 8 --run.time 08:00:00
python train.py --batch draco --batch.nodes 8 --batch.time 08:00:00

# Dry-run (preview what would be executed)
python train.py --run draco --run.dry-run

# Detached mode (submit and exit) - same as using --batch
python train.py --run draco --run.detach
```

### `--run` vs `--batch`

| Option | Behavior | Use Case |
|--------|----------|----------|
| `--run` | Attached execution, waits for job to complete | Interactive development, monitoring output |
| `--batch` | Detached execution, submits and exits immediately | Long-running training jobs, job queues |

The `--batch` option automatically sets `detach=True` and `ray_mode="job"` (ensuring Ray clusters terminate after the job completes).

## Supported Executors

### Local

Runs locally using torchrun. Good for development and testing.

```toml
[local]
executor = "local"
nproc_per_node = 8
env_vars = ["NCCL_DEBUG=INFO"]
```

### Docker

Runs in a Docker container with GPU support.

```toml
[docker]
executor = "docker"
container_image = "nvcr.io/nvidia/nemo:24.01"
nproc_per_node = 8
runtime = "nvidia"
ipc_mode = "host"
shm_size = "16g"
mounts = ["/data:/data"]
```

### Slurm

Submits jobs to a Slurm cluster. Supports both local and SSH submission.

```toml
[slurm-local]
executor = "slurm"
account = "my-account"
partition = "gpu"
nodes = 4
nproc_per_node = 8
time = "04:00:00"
container_image = "nvcr.io/nvidia/nemo:24.01"
mounts = ["/data:/data"]

[slurm-ssh]
executor = "slurm"
account = "my-account"
partition = "gpu"
nodes = 4
tunnel = "ssh"
host = "cluster.example.com"
user = "username"
identity = "~/.ssh/id_rsa"
```

#### Partition Overrides

You can specify different partitions for `--run` (attached) vs `--batch` (detached) execution:

```toml
[draco]
executor = "slurm"
account = "my-account"
partition = "batch"           # Default partition
run_partition = "interactive" # Used for --run (attached)
batch_partition = "backfill"  # Used for --batch (detached)
```

This is useful when your cluster has separate partitions for interactive and batch workloads.

### SkyPilot

Launches cloud instances via SkyPilot (AWS, GCP, Azure).

```toml
[aws]
executor = "skypilot"
cloud = "aws"
gpus = "A100:8"
nodes = 2
cluster_name = "nemotron-training"
setup = "pip install -e ."
```

### DGX Cloud

Runs on NVIDIA DGX Cloud.

```toml
[dgx]
executor = "dgxcloud"
project_name = "nemotron"
nodes = 4
nproc_per_node = 8
pvcs = ["data-pvc:/data"]
```

### Lepton

Runs on Lepton AI.

```toml
[lepton]
executor = "lepton"
resource_shape = "gpu-a100-80gb"
node_group = "default"
```

## Profile Inheritance

Profiles can extend other profiles to reduce duplication:

```toml
[base-slurm]
executor = "slurm"
account = "my-account"
partition = "gpu"
time = "04:00:00"
container_image = "nvcr.io/nvidia/nemo:24.01"

[draco]
extends = "base-slurm"
nodes = 4
nproc_per_node = 8

[draco-large]
extends = "draco"
nodes = 16
time = "08:00:00"
```

## W&B Configuration

You can configure Weights & Biases tracking in the same `run.toml` file using the `[wandb]` section:

```toml
# run.toml

[wandb]
project = "my-project"
entity = "my-team"
tags = ["training", "nano3"]
notes = "Training run with optimized hyperparameters"

[draco]
executor = "slurm"
account = "my-account"
partition = "gpu"
nodes = 4
```

When a `[wandb]` section is present, W&B tracking is automatically enabled for all commands. This is equivalent to passing `--wandb.project my-project --wandb.entity my-team` on the CLI.

You can also include `[wandb]` in your recipe config files (YAML/TOML/JSON) passed via `--config-file`:

```yaml
# config.yaml
batch_size: 32
learning_rate: 1e-4

wandb:
  project: my-project
  entity: my-team
```

### W&B Configuration Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `project` | str | - | W&B project name (required to enable tracking) |
| `entity` | str | - | W&B entity/team name |
| `run_name` | str | - | W&B run name (auto-generated if not set) |
| `tags` | list | `[]` | Tags for filtering runs |
| `notes` | str | - | Notes/description for the run |

## CLI Display Settings

You can customize how the CLI displays configuration output using the `[cli]` section:

```toml
# run.toml or env.toml

[cli]
theme = "github-light"
```

The `theme` setting controls the syntax highlighting theme used when displaying compiled configurations. This applies to both `--dry-run` output and regular execution.

### Available Themes

Any Pygments theme is supported. Popular choices include:

| Theme | Description |
|-------|-------------|
| `monokai` | Dark theme (default) |
| `github-light` | Light theme matching GitHub |
| `github-dark` | Dark theme matching GitHub |
| `dracula` | Popular dark theme |
| `one-dark` | Atom One Dark theme |
| `nord` | Nord color palette |
| `solarized-dark` | Solarized dark |
| `solarized-light` | Solarized light |

## Execution Profile Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `executor` | str | `"local"` | Backend: local, docker, slurm, skypilot, dgxcloud, lepton |
| `nproc_per_node` | int | `8` | GPUs per node |
| `nodes` | int | `1` | Number of nodes |
| `container_image` | str | - | Container image |
| `mounts` | list | `[]` | Mount points (e.g., `/host:/container`) |
| `account` | str | - | Slurm account |
| `partition` | str | - | Slurm partition (default for both --run and --batch) |
| `run_partition` | str | - | Partition override for `--run` (attached execution) |
| `batch_partition` | str | - | Partition override for `--batch` (detached execution) |
| `time` | str | `"04:00:00"` | Job time limit |
| `job_name` | str | `"nemo-run"` | Job name |
| `tunnel` | str | `"local"` | Slurm tunnel: local or ssh |
| `host` | str | - | SSH host |
| `user` | str | - | SSH user |
| `cloud` | str | - | SkyPilot cloud: aws, gcp, azure |
| `gpus` | str | - | SkyPilot GPU spec (e.g., `A100:8`) |
| `env_vars` | list | `[]` | Environment variables (`KEY=VALUE`) |
| `dry_run` | bool | `false` | Preview without executing |
| `detach` | bool | `false` | Submit and exit |

## Ray-Enabled Recipes

Some recipes (like data preparation and RL training) use Ray for distributed execution. This is configured at the recipe level, not in run.toml. When you run a Ray-enabled recipe with `--run`, the Ray cluster is set up automatically on the target infrastructure.

```bash
# Data prep uses Ray internally
python -m nemotron.recipes.nano3.data_prep --run draco

# RL training uses Ray internally
python -m nemotron.recipes.nano3.stage2_rl.train --run draco
```

You can optionally specify `ray_working_dir` in your profile for Ray jobs:

```toml
[draco]
executor = "slurm"
account = "my-account"
partition = "gpu"
nodes = 4
ray_working_dir = "/workspace"
```
