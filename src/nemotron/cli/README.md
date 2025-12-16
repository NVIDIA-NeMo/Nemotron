# nemotron.cli

Entry point for the `nemotron` command-line interface.

## Overview

This package provides the CLI commands for Nemotron training recipes. The CLI is built on [Typer](https://typer.tiangolo.com/) and integrates with `nemotron.kit` for configuration, artifact resolution, and NeMo-Run execution.

## Entry Point

The `nemotron` command is registered as a console script in `pyproject.toml`:

```toml
[project.scripts]
nemotron = "nemotron.cli.bin.nemotron:main"
```

## Command Structure

```
nemotron
├── nano3                    # Nano3 training recipe
│   ├── pretrain             # Stage 0: Pretraining
│   ├── sft                  # Stage 1: Supervised fine-tuning
│   ├── rl                   # Stage 2: Reinforcement learning
│   ├── data
│   │   ├── prep
│   │   │   ├── pretrain     # Prepare pretrain data
│   │   │   ├── sft          # Prepare SFT data
│   │   │   └── rl           # Prepare RL data
│   │   └── import
│   │       ├── pretrain     # Import pretrain data artifact
│   │       ├── sft          # Import SFT data artifact
│   │       └── rl           # Import RL data artifact
│   └── model
│       ├── eval             # Evaluate model
│       └── import
│           ├── pretrain     # Import pretrain checkpoint
│           ├── sft          # Import SFT checkpoint
│           └── rl           # Import RL checkpoint
└── kit                      # Kit utilities
    └── squash               # Squash container images
```

## Module Structure

```
src/nemotron/cli/
├── __init__.py              # Package marker
├── bin/
│   └── nemotron.py          # Main entry point (typer app)
├── kit/
│   ├── app.py               # Kit utility commands
│   └── squash.py            # Container squashing
└── nano3/                   # Nano3 recipe CLI
    ├── app.py               # Root nano3 group
    ├── pretrain.py          # Pretrain command
    ├── sft.py               # SFT command
    ├── rl.py                # RL command
    ├── data/
    │   ├── app.py           # Data group
    │   ├── prep/            # Data prep commands
    │   └── import_/         # Data import commands
    └── model/
        ├── app.py           # Model group
        ├── eval.py          # Model evaluation
        └── import_/         # Model import commands
```

## Global Options

All commands support these global options:

| Option | Short | Description |
|--------|-------|-------------|
| `--config` | `-c` | Config name or path |
| `--run` | `-r` | Attached execution via NeMo-Run |
| `--batch` | `-b` | Detached execution via NeMo-Run |
| `--dry-run` | `-d` | Preview config without execution |
| `--stage` | | Stage script to remote for debugging |

## Usage Examples

```bash
# Local execution with config
uv run nemotron nano3 pretrain -c tiny

# Submit to cluster (attached)
uv run nemotron nano3 pretrain -c tiny --run MY-CLUSTER

# Submit to cluster (detached)
uv run nemotron nano3 pretrain -c tiny --batch MY-CLUSTER

# Preview without execution
uv run nemotron nano3 pretrain -c tiny --dry-run

# Override config values
uv run nemotron nano3 pretrain -c tiny train.train_iters=5000

# Data preparation
uv run nemotron nano3 data prep pretrain --run MY-CLUSTER
uv run nemotron nano3 data prep sft --run MY-CLUSTER
uv run nemotron nano3 data prep rl --run MY-CLUSTER
```

## Adding New Commands

To add a new command:

1. Create command module in appropriate directory
2. Define config dataclass and handler function
3. Register with parent app using `add_typer()` or command decorator

Example:

```python
# mycommand.py
import typer
from dataclasses import dataclass

@dataclass
class MyConfig:
    param: str = "default"

def my_handler(cfg: MyConfig):
    print(f"Running with {cfg.param}")

app = typer.Typer()

@app.command()
def run(param: str = "default"):
    my_handler(MyConfig(param=param))
```

## Full Documentation

See [docs/train/cli.md](../../../docs/train/cli.md) for complete CLI framework documentation including:

- Building CLIs with App
- Artifact inputs and resolution
- Execution modes
- Recipe tutorial

See [docs/train/nemo-run.md](../../../docs/train/nemo-run.md) for execution profile configuration.
