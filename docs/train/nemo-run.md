# Running Recipes with NeMo-Run

Nemotron recipes use [NeMo-Run](https://github.com/NVIDIA-NeMo/Run) for job orchestration. Add `--run <profile>` to any recipe command to execute on your target infrastructure.

> **Slurm Only (v0)**: This initial release has been tested exclusively with Slurm execution. Support for additional NeMo-Run executors (local, Docker, SkyPilot, DGX Cloud) is planned for future releases.

## Quick Start

```bash
# Execute on a Slurm cluster
uv run nemotron nano3 pretrain -c tiny --run YOUR-CLUSTER

# Direct script execution (inside container on compute node)
uv run python train.py --config config/tiny.yaml
```

## Setting Up Run Profiles

Create an `env.toml` in your project root. Each section defines a named execution profile:

```toml
# env.toml

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

Container images are specified in recipe config files (e.g., `config/tiny.yaml`), not in env.toml.

## Running Recipes

### Data Preparation

```bash
# On Slurm cluster
uv run nemotron nano3 data prep pretrain --run YOUR-CLUSTER --sample 1000
```

### Pretraining

```bash
# On Slurm
uv run nemotron nano3 pretrain -c tiny --run YOUR-CLUSTER

# On Slurm with node override
uv run nemotron nano3 pretrain -c tiny --run YOUR-CLUSTER run.nodes=8
```

### Supervised Fine-Tuning

```bash
# On Slurm
uv run nemotron nano3 sft -c tiny --run YOUR-CLUSTER
```

### RL Training

```bash
# On Slurm (Ray cluster started automatically)
uv run nemotron nano3 rl -c tiny --run YOUR-CLUSTER
```

## CLI Options

```bash
# Attached execution - waits for completion, streams logs
uv run nemotron nano3 pretrain -c tiny --run YOUR-CLUSTER

# Detached execution - submits and exits immediately
uv run nemotron nano3 pretrain -c tiny --batch YOUR-CLUSTER

# Override config values (Hydra-style)
uv run nemotron nano3 pretrain -c tiny --run YOUR-CLUSTER train.train_iters=5000

# Dry-run (preview what would be executed)
uv run nemotron nano3 pretrain -c tiny --run YOUR-CLUSTER --dry-run
```

### `--run` vs `--batch`

| Option | Behavior | Use Case |
|--------|----------|----------|
| `--run` | Attached execution, waits for job to complete | Interactive development, monitoring output |
| `--batch` | Detached execution, submits and exits immediately | Long-running training jobs, job queues |

The `--batch` option automatically sets `detach=True` and `ray_mode="job"` (ensuring Ray clusters terminate after the job completes).

## Slurm Configuration

Submits jobs to a Slurm cluster. Supports both local and SSH submission.

### Local Submission

Submit from a machine with direct access to the Slurm scheduler:

```toml
[YOUR-CLUSTER]
executor = "slurm"
account = "my-account"
partition = "gpu"
nodes = 4
ntasks_per_node = 8
gpus_per_node = 8
time = "04:00:00"
mounts = ["/data:/data"]
```

### SSH Tunnel Submission

Submit from a remote machine via SSH:

```toml
[YOUR-CLUSTER]
executor = "slurm"
account = "my-account"
partition = "gpu"
nodes = 4
tunnel = "ssh"
host = "cluster.example.com"
user = "username"
identity = "~/.ssh/id_rsa"
```

### Partition Overrides

You can specify different partitions for `--run` (attached) vs `--batch` (detached) execution:

```toml
[YOUR-CLUSTER]
executor = "slurm"
account = "my-account"
partition = "batch"           # Default partition
run_partition = "interactive" # Used for --run (attached)
batch_partition = "backfill"  # Used for --batch (detached)
```

This is useful when your cluster has separate partitions for interactive and batch workloads.

## Profile Inheritance

Profiles can extend other profiles to reduce duplication:

```toml
[base-slurm]
executor = "slurm"
account = "my-account"
partition = "gpu"
time = "04:00:00"

[YOUR-CLUSTER]
extends = "base-slurm"
nodes = 4
ntasks_per_node = 8
gpus_per_node = 8

[YOUR-CLUSTER-large]
extends = "YOUR-CLUSTER"
nodes = 16
time = "08:00:00"
```

## Other Executors (Coming Soon)

NeMo-Run supports additional executors that will be tested and documented in future releases:

| Executor | Description | Status |
|----------|-------------|--------|
| `local` | Local execution with torchrun | Planned |
| `docker` | Docker container with GPU support | Planned |
| `skypilot` | Cloud instances (AWS, GCP, Azure) | Planned |
| `dgxcloud` | NVIDIA DGX Cloud | Planned |
| `lepton` | Lepton AI | Planned |

See the [NeMo-Run documentation](https://github.com/NVIDIA-NeMo/Run) for configuration details.

## W&B Configuration

You can configure Weights & Biases tracking in `env.toml` using the `[wandb]` section:

```toml
# env.toml

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
# env.toml

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

Some recipes (like data preparation and RL training) use Ray for distributed execution. This is configured at the recipe level, not in env.toml. When you run a Ray-enabled recipe with `--run`, the Ray cluster is set up automatically on the target infrastructure.

```bash
# Data prep uses Ray internally
uv run nemotron nano3 data prep pretrain --run YOUR-CLUSTER

# RL training uses Ray internally
uv run nemotron nano3 rl -c tiny --run YOUR-CLUSTER
```

You can optionally specify `ray_working_dir` in your profile for Ray jobs:

```toml
[YOUR-CLUSTER]
executor = "slurm"
account = "my-account"
partition = "gpu"
nodes = 4
ray_working_dir = "/workspace"
```
