# nemotron.kit

Domain-specific toolkit for Nemotron training artifacts, lineage tracking, and W&B integration.

## Overview

Kit provides Nemotron-specific building blocks:

- **Artifacts** -- Path-centric data and model versioning with typed metadata
- **Lineage Tracking** -- W&B and file-based backends for experiment provenance
- **W&B Integration** -- Configuration, initialization, and tag management

For CLI infrastructure, config loading, execution, packaging, and pipeline orchestration, see the [`nemo_runspec`](../../nemo_runspec/README.md) package.

## Module Structure

```
src/nemotron/kit/
├── __init__.py          # Public API exports + kit.init()
├── artifact.py          # Artifact base class and typed artifact definitions
├── artifacts/           # Artifact type definitions
├── trackers.py          # LineageTracker, WandbTracker, FileTracker, NoOpTracker
├── wandb_kit.py         # WandbConfig, init_wandb_if_configured, add_run_tags
├── train_script.py      # Training script utilities (parse_config_and_overrides)
├── recipe_loader.py     # Recipe loading utilities
└── megatron_stub.py     # Megatron stub for testing
```

## Quick Start

### Creating Artifacts

```python
from nemotron.kit import PretrainBlendsArtifact
from pathlib import Path

artifact = PretrainBlendsArtifact(
    path=Path("/output/data"),
    total_tokens=25_000_000_000,
)
artifact.save(name="my-artifact")
```

### Lineage Tracking

```python
from nemotron.kit import set_lineage_tracker, WandbTracker

# Use W&B for lineage tracking
set_lineage_tracker(WandbTracker())
```

### Kit Initialization

```python
import nemotron.kit as kit

# Initialize with fsspec backend
kit.init(backend="fsspec", root="/data/artifacts")

# Or with W&B backend
kit.init(backend="wandb", wandb_project="my-project")
```

## Public API Quick Reference

### Artifacts
- `Artifact` -- Base class
- `DataBlendsArtifact`, `PretrainBlendsArtifact`, `PretrainDataArtifact` -- Pretrain data
- `SFTDataArtifact` -- Packed SFT sequences
- `SplitJsonlDataArtifact` -- RL JSONL data
- `ModelArtifact` -- Model checkpoints
- `TrackingInfo` -- Tracking metadata

### Tracking
- `LineageTracker`, `WandbTracker`, `FileTracker`, `NoOpTracker` -- Tracker backends
- `set_lineage_tracker()`, `get_lineage_tracker()` -- Global tracker management
- `to_wandb_uri()`, `tokenizer_to_uri()` -- URI conversion

### W&B
- `WandbConfig` -- W&B configuration dataclass
- `init_wandb_if_configured()` -- Conditional W&B init
- `add_run_tags()` -- Add tags to runs

### Kit Initialization
- `init()` -- Initialize kit with storage backend
- `is_initialized()` -- Check if kit has been initialized

## Full Documentation

See [docs/nemotron/kit.md](../../../docs/nemotron/kit.md) for complete documentation including:

- Artifact philosophy and design
- Lineage tracking
- W&B integration
- API reference

See [docs/nemo_runspec/cli.md](../../../docs/nemo_runspec/cli.md) for CLI framework documentation.
See [src/nemo_runspec/README.md](../../nemo_runspec/README.md) for the CLI toolkit.
