# Nemotron Kit

The `nemotron.kit` module provides artifact type definitions, lineage trackers, and W&B integration for Nemotron training recipes.

> **Focused by Design**: Kit owns the artifact *types* (data classes like `PretrainBlendsArtifact`, `ModelArtifact`) and *tracking behavior* (W&B/file-based lineage). The underlying artifact *registry* and *resolution* (`art://` URIs, fsspec/wandb storage backends) live in [`nemo_runspec`](../../src/nemo_runspec/README.md). CLI, configuration, and execution also live in `nemo_runspec`. All heavy-lifting training is done by the [NVIDIA AI Stack](./nvidia-stack.md): [Megatron-Core](https://github.com/NVIDIA/Megatron-LM) for distributed training primitives, [Megatron-Bridge](https://github.com/NVIDIA/Megatron-Bridge) for model training, and [NeMo-RL](https://github.com/NVIDIA/NeMo-RL) for reinforcement learning.

## Overview

Kit handles three core responsibilities:

| Component | What kit owns | What nemo_runspec owns |
|-----------|--------------|----------------------|
| **[Artifacts](./artifacts.md)** | Type definitions (`PretrainBlendsArtifact`, `ModelArtifact`, etc.) | Registry, `art://` resolution, fsspec/wandb storage backends |
| **[Lineage Tracking](./artifacts.md)** | Trackers (`WandbTracker`, `FileTracker`) | `${art:...}` OmegaConf resolvers, distributed coordination |
| **[W&B Integration](./wandb.md)** | Init, credential handling, monkey patches, tag management | Env var injection (`build_env_vars`), `[wandb]` config loading |

For CLI infrastructure, config loading, execution, and packaging, see [`nemo_runspec`](../../src/nemo_runspec/README.md).

## Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryBorderColor': '#333333', 'lineColor': '#333333', 'primaryTextColor': '#333333', 'clusterBkg': '#ffffff', 'clusterBorder': '#333333'}}}%%
flowchart LR
    subgraph cli["CLI (nemotron.cli)"]
        direction TB
        CMD1["nemotron ... data prep"]
        CMD2["nemotron ... train/sft/rl"]
    end

    subgraph recipes["Recipe Scripts"]
        direction TB
        DataPrep["data_prep.py"]
        Train["train.py"]
    end

    subgraph runspec["nemo_runspec"]
        direction TB
        Config["Config Loading"]
        Exec["Execution"]
        Pipeline["Pipeline"]
        Registry["Artifact Registry"]
    end

    subgraph kit["nemotron.kit"]
        direction TB
        Artifact["Artifact Types"]
        Tracker["Lineage Trackers"]
        WandB["W&B Integration"]
    end

    subgraph tracking["Tracking Backends"]
        direction LR
        WandBBackend["W&B"]
        FileBackend["File-based"]
    end

    CMD1 --> DataPrep
    CMD2 --> Train
    DataPrep --> kit
    DataPrep --> runspec
    Train --> kit
    Train --> runspec
    Artifact --> Registry
    Tracker --> tracking

    style cli fill:#fce4ec,stroke:#c2185b
    style recipes fill:#e8f5e9,stroke:#388e3c
    style runspec fill:#fff3e0,stroke:#f57c00
    style kit fill:#e3f2fd,stroke:#1976d2
    style tracking fill:#f3e5f5,stroke:#7b1fa2
```

## Quick Example

```python
from nemotron.kit import PretrainBlendsArtifact, ModelArtifact
from pathlib import Path

# Load data artifact
data = PretrainBlendsArtifact.load(Path("/output/data"))
print(f"Training on {data.total_tokens:,} tokens")

# ... training code ...

# Save model artifact with lineage
model = ModelArtifact(path=Path("/output/checkpoint"), step=10000, loss=2.5)
model.save(name="ModelArtifact-pretrain")
```

## Concepts

### Artifacts

Artifacts are path-centric objects with typed metadata. The core field is always `path` -- the filesystem location of the data. See [Artifact Lineage](./artifacts.md) for details.

```python
from nemotron.kit import PretrainBlendsArtifact

# Load from semantic URI
artifact = PretrainBlendsArtifact.from_uri("art://PretrainBlendsArtifact:latest")
print(f"Path: {artifact.path}")
print(f"Tokens: {artifact.total_tokens:,}")
```

### Lineage Tracking

Kit tracks artifact lineage through pluggable backends. The `WandbTracker` logs to W&B; the `FileTracker` writes to local filesystem. See [W&B Integration](./wandb.md) for credential handling and [Artifact Lineage](./artifacts.md) for the lineage graph.

```python
from nemotron.kit import set_lineage_tracker, WandbTracker

# Use W&B for tracking
set_lineage_tracker(WandbTracker())
```

### W&B Integration

```python
from nemotron.kit import WandbConfig, init_wandb_if_configured, add_run_tags

# Initialize W&B from config
wandb_cfg = WandbConfig(entity="nvidia", project="nemotron")
init_wandb_if_configured(wandb_cfg)

# Add tags to the run
add_run_tags(["pretrain", "nano3"])
```

## Module Structure

```
src/nemotron/kit/
├── __init__.py          # Public API exports + kit.init()
├── artifact.py          # Artifact base class, ArtifactInput, display helpers
├── artifacts/           # Artifact type definitions (base, model, data blends, etc.)
├── trackers.py          # LineageTracker, WandbTracker, FileTracker, NoOpTracker
├── wandb_kit.py         # WandbConfig, init_wandb_if_configured, add_run_tags, monkey patches
├── train_script.py      # Training script utilities (init_wandb_from_env, config parsing)
├── recipe_loader.py     # Recipe loading utilities
└── megatron_stub.py     # Megatron stub for testing
```

## API Reference

### Artifacts

| Export | Description |
|--------|-------------|
| `Artifact` | Base artifact class |
| `PretrainBlendsArtifact` | Pretrain data with train/valid/test splits |
| `PretrainDataArtifact` | Raw pretrain data |
| `SFTDataArtifact` | Packed SFT sequences |
| `SplitJsonlDataArtifact` | RL JSONL data |
| `DataBlendsArtifact` | Generic data blends |
| `ModelArtifact` | Model checkpoints |
| `TrackingInfo` | Tracking metadata for artifacts |

### Tracking

| Export | Description |
|--------|-------------|
| `LineageTracker` | Abstract base for lineage tracking |
| `WandbTracker` | W&B-backed lineage tracker |
| `FileTracker` | File-based lineage tracker |
| `NoOpTracker` | No-op tracker (for testing) |
| `set_lineage_tracker()` | Set the global lineage tracker |
| `get_lineage_tracker()` | Get the current lineage tracker |
| `to_wandb_uri()` | Convert artifact to W&B URI |
| `tokenizer_to_uri()` | Convert tokenizer to URI |

### W&B

| Export | Description |
|--------|-------------|
| `WandbConfig` | W&B configuration dataclass |
| `init_wandb_if_configured()` | Conditional W&B initialization |
| `add_run_tags()` | Add tags to W&B runs |

### Kit Initialization

| Export | Description |
|--------|-------------|
| `init()` | Initialize kit with storage backend (fsspec or wandb) |
| `is_initialized()` | Check if kit has been initialized |

## Further Reading

- [`nemo_runspec` Package](../../src/nemo_runspec/README.md) – CLI toolkit, config loading, execution, packaging
- [NVIDIA AI Stack](./nvidia-stack.md) – Megatron-Core, Megatron-Bridge, NeMo-RL
- [OmegaConf Configuration](../nemo_runspec/omegaconf.md) – artifact interpolations and unified W&B logging
- [Artifact Lineage](./artifacts.md) – artifact versioning and W&B lineage
- [W&B Integration](./wandb.md) – credential handling
- [Execution through NeMo-Run](../nemo_runspec/nemo-run.md) – execution profiles and packagers
- [CLI Framework](./cli.md) – building recipe CLIs
- [Data Preparation](./data-prep.md) – data prep module
