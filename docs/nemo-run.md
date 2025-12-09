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
# Select a profile
python train.py --run <profile-name>

# Override profile settings
python train.py --run draco --run.nodes 8 --run.time 08:00:00

# Dry-run (preview what would be executed)
python train.py --run draco --run.dry-run

# Detached mode (submit and exit)
python train.py --run draco --run.detach
```

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

## Configuration Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `executor` | str | `"local"` | Backend: local, docker, slurm, skypilot, dgxcloud, lepton |
| `nproc_per_node` | int | `8` | GPUs per node |
| `nodes` | int | `1` | Number of nodes |
| `container_image` | str | - | Container image |
| `mounts` | list | `[]` | Mount points (e.g., `/host:/container`) |
| `account` | str | - | Slurm account |
| `partition` | str | - | Slurm partition |
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
