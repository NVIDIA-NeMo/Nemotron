# OmegaConf Configuration System

Nemotron Kit uses [OmegaConf](https://omegaconf.readthedocs.io/) for configuration management, with custom resolvers that enable automatic artifact resolution and W&B lineage tracking. This page explains the `run` section in configs, artifact interpolations, and unified W&B logging.

## The `run` Section

The `run` section in recipe configs serves as a metadata container for execution and artifact configuration. It's separate from the training algorithm configuration.

```yaml
# config.yaml
run:
  # Artifact references - automatically resolved with W&B lineage
  data: PretrainBlendsArtifact-default:latest
  model: ModelArtifact-pretrain:v5

  # Environment configuration (container image)
  env:
    container: nvcr.io/nvidia/nemo:25.11.nemotron_3_nano

# Training configuration sections
recipe:
  _target_: megatron.bridge.recipes.nemotronh.nemotron_next_3b_v2_pretrain_config
  per_split_data_args_path: ${art:data,path}/blend.json  # Resolved from artifact

logger:
  wandb_project: ${run.wandb.project}   # Injected from env.toml
  wandb_entity: ${run.wandb.entity}     # Injected from env.toml

checkpoint:
  save: /nemo_run/pretrain
  save_interval: 20
```

### How `run.wandb` is Populated

When you run a recipe with `--run <profile>`, the CLI reads your `env.toml` and injects the `[wandb]` section into `run.wandb`:

```toml
# env.toml
[wandb]
project = "nemotron"
entity = "my-team"

[YOUR-CLUSTER]
executor = "slurm"
# ...
```

This allows configs to reference W&B settings via interpolation (`${run.wandb.project}`) without hardcoding them.

## Artifact Interpolations

The `${art:NAME,FIELD}` resolver enables automatic artifact resolution with W&B lineage tracking.

### Syntax

```yaml
# Basic path resolution
data_path: ${art:data,path}                      # /path/to/artifact

# Field resolution
model_version: ${art:model,version}              # v5
model_type: ${art:model,type}                    # "model"
checkpoint_step: ${art:model,iteration}          # 10000

# Metadata field resolution (from metadata.json)
pack_size: ${art:data,pack_size}                 # 4096
training_path: ${art:data,training_path}         # /path/to/training_4096.npy
```

### Supported Fields

| Field | Source | Description |
|-------|--------|-------------|
| `path` | Artifact info | Local filesystem path to artifact (default) |
| `version` | Artifact info | W&B artifact version (e.g., "v5") |
| `name` | Artifact info | Artifact name |
| `type` | Artifact info | Artifact type ("dataset", "model") |
| `iteration` | Metadata | Training iteration (for model checkpoints) |
| `*` | `metadata.json` | Any field from the artifact's metadata |

### Example: Pretrain Config

```yaml
run:
  data: PretrainBlendsArtifact-default:latest

recipe:
  _target_: megatron.bridge.recipes.nemotronh.nemotron_next_3b_v2_pretrain_config
  # Resolved to: /path/to/wandb/artifacts/PretrainBlendsArtifact-default-v3/blend.json
  per_split_data_args_path: ${art:data,path}/blend.json
```

### Example: SFT Config with Model Checkpoint

```yaml
run:
  data: SFTDataArtifact-default:latest
  model: pretrain:latest

recipe:
  _target_: megatron.bridge.recipes.nemotronh.nemotron_nano_9b_v2_finetune_config
  # Resolved to the pretrain checkpoint save directory
  pretrained_checkpoint: ${art:model,path}
  # Resolved to the checkpoint iteration number
  ckpt_step: ${art:model,iteration}
```

### Example: RL Config

```yaml
run:
  data: DataBlendsArtifact-rl:latest
  model: sft:latest
  env:
    container: nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano

policy:
  # Resolved to the SFT model checkpoint path
  model_name: ${art:model,path}

data:
  # Resolved from artifact metadata
  train_jsonl_fpath: ${art:data,train_path}
  validation_jsonl_fpath: ${art:data,val_path}
```

## How Artifact Resolution Works

### Resolution Modes

The artifact resolver supports two modes to handle different framework requirements:

| Mode | When Used | Description |
|------|-----------|-------------|
| `active_run` | W&B run already active | Calls `wandb.use_artifact()` directly |
| `pre_init` | Before `wandb.init()` | Uses `wandb.Api()` to resolve, defers lineage |

**Why two modes?** Megatron-Bridge owns `wandb.init()` during training. The kit resolves artifacts *before* training starts, then patches `wandb.init()` to register lineage once the run is active.

### Resolution Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. Config Loading                                                  │
│     - Load YAML with OmegaConf                                      │
│     - Detect artifact references in run section                     │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. register_resolvers_from_config()                                │
│     - Scan run section for artifact patterns                        │
│     - Download artifacts from W&B (rank 0 only in distributed)      │
│     - Store qualified_name for lineage registration                 │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. OmegaConf Resolver Registration                                 │
│     - Register ${art:NAME,FIELD} resolver                           │
│     - Fields resolve to artifact path, version, or metadata         │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. patch_wandb_init_for_lineage()                                  │
│     - Patch wandb.init() to call use_artifact() when run starts     │
│     - Registers lineage in W&B graph                                │
└─────────────────────────────────────────────────────────────────────┘
```

### Distributed Coordination

In multi-GPU training, only rank 0 downloads artifacts:

1. **Rank 0**: Downloads artifacts, writes results to shared marker file
2. **Other ranks**: Wait for marker file, read shared results
3. **All ranks**: Use identical artifact paths

This prevents redundant downloads and ensures consistency across workers.

## Unified W&B Logging

Nemotron Kit provides a unified W&B logging approach that works across Megatron-Bridge and NeMo-RL, which have different logging conventions.

> **Coming Soon**: Native support for artifact lineage and checkpoint logging is being added directly to Megatron-Bridge and NeMo-RL. Once available, the monkey patches described below will no longer be necessary.

### The Problem

| Framework | Checkpoint Logging | W&B Init |
|-----------|-------------------|----------|
| Megatron-Bridge | `on_save_checkpoint_success()` | Owns `wandb.init()` |
| NeMo-RL | `CheckpointManager.finalize_checkpoint()` | Separate init |

Both frameworks need to:
- Log checkpoints as W&B artifacts
- Track lineage from input artifacts
- Store metadata for downstream jobs

### The Solution: Monkey Patches (Temporary)

Until native support is available, the kit uses targeted monkey patches to unify logging behavior:

```python
# In train.py (pretraining/SFT)
from nemotron.kit.wandb import (
    patch_wandb_checkpoint_logging,
    patch_wandb_init_for_lineage,
)

# Resolve artifacts before wandb.init()
qualified_names = register_resolvers_from_config(config, mode="pre_init")

# Patch wandb.init to register lineage when MB initializes it
patch_wandb_init_for_lineage(artifact_qualified_names=qualified_names)

# Patch checkpoint saving to log artifacts with metadata
patch_wandb_checkpoint_logging()
```

```python
# In train.py (RL)
from nemotron.kit.wandb import patch_nemo_rl_checkpoint_logging

# Patch NeMo-RL checkpoint manager
patch_nemo_rl_checkpoint_logging()
```

### What the Patches Do

**`patch_wandb_checkpoint_logging()`** (Megatron-Bridge):
- Wraps `on_save_checkpoint_success()`
- Adds `wait()` call so artifacts appear immediately in W&B
- Stores `absolute_path` in metadata for cross-job access
- Resolves container paths (`/nemo_run/`) to actual Lustre paths

**`patch_nemo_rl_checkpoint_logging()`** (NeMo-RL):
- Wraps `CheckpointManager.finalize_checkpoint()`
- Logs checkpoints as W&B artifacts with consistent naming
- Same metadata format as Megatron-Bridge patches

**`patch_wandb_init_for_lineage()`**:
- Patches `wandb.init()` to call `use_artifact()` for resolved artifacts
- Registers lineage in W&B graph once run is active

### Container Path Resolution

When running in containers, checkpoints are saved to mount paths like `/nemo_run/`. The kit resolves these to actual filesystem paths for cross-job access:

```python
# Container path (inside job)
/nemo_run/pretrain/iter_0010000

# Resolved path (for artifact metadata)
/lustre/scratch/user/jobs/12345/pretrain/iter_0010000
```

Resolution uses:
1. `NEMO_RUN_DIR` environment variable (set by nemo-run)
2. `/proc/mounts` to find bind mount source

## Usage in Training Scripts

### Pretraining/SFT (Megatron-Bridge)

```python
from nemotron.kit.resolvers import register_resolvers_from_config
from nemotron.kit.wandb import (
    patch_wandb_checkpoint_logging,
    patch_wandb_init_for_lineage,
)

def main():
    config = OmegaConf.load("config.yaml")

    # Resolve artifacts before wandb.init()
    qualified_names = register_resolvers_from_config(
        config,
        artifacts_key="run",
        mode="pre_init",
    )

    # Patch for lineage and checkpoint logging
    patch_wandb_init_for_lineage(artifact_qualified_names=qualified_names)
    patch_wandb_checkpoint_logging()

    # Now Megatron-Bridge handles the rest
    pretrain(config=cfg, forward_step_func=forward_step)
```

### RL (NeMo-RL)

```python
from nemotron.kit.resolvers import register_resolvers_from_config
from nemotron.kit.wandb import patch_nemo_rl_checkpoint_logging

def main():
    config = load_config("grpo_config.yaml")

    # Patch before any wandb interaction
    patch_nemo_rl_checkpoint_logging()

    # Resolve artifacts
    register_resolvers_from_config(
        config,
        artifacts_key="run",
        mode="pre_init",
    )

    # Resolve config (${art:...} interpolations now work)
    config = OmegaConf.to_container(config, resolve=True)

    # NeMo-RL training
    grpo_train(...)
```

## Additional Resolvers

The kit also provides utility resolvers:

```yaml
# Multiplication resolver (NeMo-RL)
train_mb_tokens: ${mul:${policy.max_total_sequence_length}, ${policy.train_micro_batch_size}}
```

## Further Reading

- [Artifact Lineage](./artifacts.md) — W&B artifact system and lineage tracking
- [Creating Custom Artifacts](./artifacts.md#creating-custom-artifacts) — Define typed artifact classes
- [W&B Integration](./wandb.md) — Automatic credential handling
- [CLI Framework](./cli.md) — Recipe CLIs and `--run` execution
- [Execution through NeMo-Run](./nemo-run.md) — `env.toml` profiles
