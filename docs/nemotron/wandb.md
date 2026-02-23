# Weights & Biases Integration

Nemotron provides automatic W&B configuration that seamlessly passes credentials and settings to containers running via nemo-run. `nemotron.kit.wandb` handles W&B initialization and `nemo_runspec.execution` handles credential injection into executors. This eliminates manual credential management across local, Docker, Slurm, and cloud executors.

> **Note**: The artifact system currently requires W&B. Backend-agnostic artifact tracking is in development.

## Configuration

### env.toml Setup

Add a `[wandb]` section to your `env.toml`:

```toml
[wandb]
project = "nemotron"
entity = "YOUR-TEAM"
```

| Field | Description |
|-------|-------------|
| `project` | W&B project name (required to enable tracking) |
| `entity` | W&B team/entity name |

### Authentication

Authenticate locally before running jobs:

```bash
wandb login
```

Your API key is stored in `~/.netrc` and automatically detected by the kit.

## Automatic Environment Variables

When you run jobs via nemo-run, `nemo_runspec.execution.build_env_vars()` automatically detects your W&B configuration and passes it to the container as environment variables:

| Variable | Source | Description |
|----------|--------|-------------|
| `WANDB_API_KEY` | `wandb.api.api_key` | API key from local wandb login |
| `WANDB_PROJECT` | `env.toml [wandb]` | Project name |
| `WANDB_ENTITY` | `env.toml [wandb]` | Team/entity name |

This works across all executor types:

- **Local** — Environment variables set directly
- **Docker** — Passed via container env vars
- **Slurm** — Included in job submission
- **SkyPilot** — Set in cloud instance environment
- **Ray** — Passed via `runtime_env.env_vars`

### How It Works

The `build_env_vars()` function in `nemo_runspec.execution` handles automatic detection:

```python
# Auto-detect W&B API key from local login
if "WANDB_API_KEY" not in merged_env:
    import wandb
    api_key = wandb.api.api_key
    if api_key:
        merged_env["WANDB_API_KEY"] = api_key

# Load project/entity from env.toml [wandb] section
wandb_config = load_wandb_config()
if wandb_config is not None:
    if wandb_config.project:
        merged_env["WANDB_PROJECT"] = wandb_config.project
    if wandb_config.entity:
        merged_env["WANDB_ENTITY"] = wandb_config.entity
```

## Using W&B in Training Scripts

### Initialization from Environment

Training scripts running inside containers can initialize W&B from environment variables:

```python
from nemotron.kit.train_script import init_wandb_from_env

# Reads WANDB_PROJECT and WANDB_ENTITY from environment
init_wandb_from_env()
```

### Conditional Initialization

For scripts that support optional W&B tracking:

```python
from nemotron.kit import init_wandb_if_configured
from nemotron.kit.wandb import WandbConfig

# Initialize only if WandbConfig is provided and has a project set
wandb_config = WandbConfig(project="nemotron", entity="my-team")
init_wandb_if_configured(wandb_config, job_type="training")
```

### WandbConfig Dataclass

The `WandbConfig` dataclass provides typed configuration:

```python
from nemotron.kit.wandb import WandbConfig

config = WandbConfig(
    project="nemotron",           # Required to enable tracking
    entity="my-team",             # Team/entity name
    run_name="experiment-001",    # Optional run name
    tags=("pretrain", "nano3"),   # Tags for filtering
    notes="First pretrain run",   # Run description
)

# Check if tracking is enabled
if config.enabled:
    print(f"Logging to {config.entity}/{config.project}")
```

## Artifact Lineage

W&B artifacts provide full lineage tracking. See [Artifact Lineage](../nemo_runspec/artifacts.md) for details on:

- End-to-end lineage from raw data to final model
- Semantic URIs for artifact references
- Viewing lineage in the W&B UI

## Advanced Features

### Checkpoint Logging

The kit automatically patches checkpoint saving to log artifacts to W&B:

```python
from nemotron.kit.wandb import patch_wandb_checkpoint_logging

# Patch Megatron-Bridge checkpoint saving
patch_wandb_checkpoint_logging()
```

This enables:
- Automatic artifact creation for each checkpoint
- Lineage links to training data artifacts
- Version tracking with step numbers

### NeMo-RL Checkpoint Logging

For reinforcement learning with NeMo-RL:

```python
from nemotron.kit.wandb import patch_nemo_rl_checkpoint_logging

# Patch NeMo-RL checkpoint saving
patch_nemo_rl_checkpoint_logging()
```

### Seeded Random Fix

When using seeded random states (common in RL), W&B's default run ID generation can fail. The kit provides a patch:

```python
from nemotron.kit.wandb import patch_wandb_runid_for_seeded_random

# Fix "Invalid Client ID digest" errors
patch_wandb_runid_for_seeded_random()
```

## Troubleshooting

### "WANDB_API_KEY not found"

Ensure you're logged in locally:

```bash
wandb login
```

### "Project not found"

Verify the project exists in your W&B workspace, or let W&B create it automatically on first run.

### Environment variables not passed to container

Check that your `env.toml` has a `[wandb]` section:

```toml
[wandb]
project = "nemotron"
entity = "YOUR-TEAM"
```

### Ray workers missing credentials

For Ray data prep jobs, credentials are passed via `runtime_env.env_vars`. Ensure your local wandb login is active before submitting the job.

## API Reference

### wandb.py Exports

| Export | Description |
|--------|-------------|
| `WandbConfig` | Configuration dataclass |
| `init_wandb_if_configured()` | Conditional W&B initialization |
| `patch_wandb_checkpoint_logging()` | Enable Megatron-Bridge checkpoint artifacts |
| `patch_nemo_rl_checkpoint_logging()` | Enable NeMo-RL checkpoint artifacts |
| `patch_wandb_runid_for_seeded_random()` | Fix seeded random ID generation |

### nemo_runspec Exports

| Export | Module | Description |
|--------|--------|-------------|
| `load_wandb_config()` | `nemo_runspec.env` | Load `WandbConfig` from env.toml |
| `build_env_vars()` | `nemo_runspec.execution` | Build env vars with auto W&B detection |

## Further Reading

- [OmegaConf Configuration](../nemo_runspec/omegaconf.md) — Artifact interpolations and unified logging patches
- [Artifact Lineage](../nemo_runspec/artifacts.md) — Full lineage tracking and W&B UI
- [Nemotron Kit](./kit.md) — Artifact system and lineage tracking
- [Execution through NeMo-Run](../nemo_runspec/nemo-run.md) — Execution profiles and env.toml
