# nemotron.kit

Training recipe framework providing artifact versioning, configuration management, and lineage tracking.

## Overview

Kit is the core infrastructure for building reproducible ML training pipelines:

- **Artifacts** — Path-centric data and model versioning with typed metadata
- **Configuration** — YAML/TOML/JSON config loading with OmegaConf and Hydra-style overrides
- **CLI Framework** — Build hierarchical CLIs with typer (see src/nemotron/cli/)
- **Execution** — Local, NeMo-Run (Slurm/Docker/cloud), and torchrun support
- **Lineage Tracking** — W&B integration for experiment tracking and artifact provenance

## Module Structure

```
src/nemotron/kit/
├── __init__.py              # Public API exports
├── artifact.py              # Artifact system (path-centric design)
├── registry.py              # Artifact registry (fsspec/W&B backends)
├── run.py                   # NeMo-Run integration
├── trackers.py              # Lineage tracking (W&B, NoOp)
├── train_script.py          # Training script utilities
├── resolvers.py             # OmegaConf custom resolvers
├── wandb.py                 # W&B configuration
├── exceptions.py            # Custom exceptions
├── filesystem.py            # fsspec for art:// URIs
├── pipeline.py              # Pipeline orchestration
├── step.py                  # Pipeline step definition
├── track.py                 # Tracking configuration
├── cli/                     # CLI submodule
│   ├── config.py            # CLI config utilities
│   ├── display.py           # Terminal display
│   ├── env.py               # Environment handling
│   ├── globals.py           # Global state
│   ├── recipe.py            # @recipe decorator, ArtifactInput
│   ├── squash.py            # Container squashing
│   └── utils.py             # Shared CLI utilities
└── packaging/               # Remote execution packaging
    ├── code_packager.py     # Code packaging for remote
    └── self_contained_packager.py
```

## Quick Start

### Creating Artifacts

```python
from nemotron.kit import PretrainBlendsArtifact, InputDatasetInfo
from pathlib import Path

artifact = PretrainBlendsArtifact(
    path=Path("/output/data"),
    total_tokens=25_000_000_000,
    source_datasets=[
        InputDatasetInfo(uri="hf://dataset/name", name="my-dataset", split="train"),
    ],
)
artifact.save(name="my-artifact")
```

### Loading Config (OmegaConf)

```python
from nemotron.kit.train_script import parse_config_and_overrides
from dataclasses import dataclass

@dataclass
class MyConfig:
    learning_rate: float = 1e-4
    batch_size: int = 32

config = parse_config_and_overrides(MyConfig, config_file="config.yaml")
```

## Public API Quick Reference

### Artifacts
- `Artifact` — Base class
- `DataBlendsArtifact`, `PretrainBlendsArtifact`, `SFTDataArtifact`, `SplitJsonlDataArtifact` — Data artifacts
- `ModelArtifact` — Model checkpoints
- `InputDatasetInfo` — Source dataset metadata

### Configuration
- `parse_config_and_overrides()`, `load_omegaconf_yaml()`, `apply_hydra_overrides()`, `omegaconf_to_dataclass()` — Training script utilities

### Execution
- `RunConfig` — NeMo-Run configuration
- `build_executor()`, `load_run_profile()` — Execution helpers

### Registry & Tracking
- `init()`, `get_config()`, `is_initialized()` — Kit initialization
- `ArtifactRegistry`, `ArtifactEntry`, `ArtifactVersion` — Registry system
- `LineageTracker`, `WandbTracker`, `NoOpTracker` — Tracking backends
- `add_wandb_tags()`, `finish_wandb()` — W&B utilities

## Full Documentation

See [docs/train/kit.md](../../../docs/train/kit.md) for complete documentation including:

- Artifact philosophy and design
- Configuration system details
- Lineage tracking
- API reference

See [docs/train/cli.md](../../../docs/train/cli.md) for CLI framework documentation including:

- Building nested CLIs with typer
- Artifact inputs and resolution
- Execution modes
- Recipe tutorial
