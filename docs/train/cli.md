# CLI Framework

The `nemotron.kit` CLI framework is built on [Typer](https://typer.tiangolo.com/) and provides tools for building hierarchical command-line interfaces for training recipes, with native integration with NeMo-Run for remote execution.

<div class="termy">

```console
$ uv run nemotron nano3 sft --help
Usage: nemotron nano3 sft [OPTIONS]

 Run supervised fine-tuning with Megatron-Bridge (stage1).

╭─ Options ────────────────────────────────────────────────────────────────╮
│ --help  -h        Show this message and exit.                            │
╰──────────────────────────────────────────────────────────────────────────╯
╭─ Global Options ─────────────────────────────────────────────────────────╮
│  -c, --config NAME       Config name or path                             │
│  -r, --run PROFILE       Submit to cluster (attached)                    │
│  -b, --batch PROFILE     Submit to cluster (detached)                    │
│  -d, --dry-run           Preview config without execution                │
│  --stage                 Stage files for interactive debugging           │
╰──────────────────────────────────────────────────────────────────────────╯
╭─ Configs (-c/--config) ──────────────────────────────────────────────────╮
│ Built-in: default, tiny                                                  │
│ Custom: -c /path/to/your/config.yaml                                     │
╰──────────────────────────────────────────────────────────────────────────╯
╭─ Artifact Overrides (W&B artifact references) ───────────────────────────╮
│  run.model     Base model checkpoint artifact                            │
│  run.data      SFT data artifact (packed .npy)                           │
╰──────────────────────────────────────────────────────────────────────────╯
╭─ Run Overrides (override env.toml settings) ─────────────────────────────╮
│  run.env.nodes               Number of nodes                             │
│  run.env.nproc_per_node      GPUs per node                               │
│  run.env.partition           Slurm partition                             │
│  run.env.account             Slurm account                               │
│  run.env.time                Job time limit (e.g., 04:00:00)             │
│  run.env.container_image     Override container image                    │
╰──────────────────────────────────────────────────────────────────────────╯
╭─ env.toml Profiles ──────────────────────────────────────────────────────╮
│ Available profiles: my-cluster, my-cluster-large                         │
│ Usage: --run PROFILE or --batch PROFILE                                  │
╰──────────────────────────────────────────────────────────────────────────╯
╭─ Examples ───────────────────────────────────────────────────────────────╮
│ $ ... sft -c tiny                    Local execution                     │
│ $ ... sft -c tiny --dry-run          Preview config                      │
│ $ ... sft -c tiny --run my-cluster   Submit to cluster                   │
│ $ ... sft -c tiny -r cluster run.env.nodes=4                             │
╰──────────────────────────────────────────────────────────────────────────╯
```

</div>

## Overview

The CLI framework enables:

- **Nested Commands** — Build hierarchical CLIs like `uv run nemotron nano3 data prep pretrain`
- **Config Integration** — Automatic YAML config loading with dotlist overrides
- **[Artifact Resolution](./artifacts.md)** — Map [W&B artifacts](./wandb.md) to config fields automatically
- **[Remote Execution](./nemo-run.md)** — Submit jobs to Slurm via NeMo-Run with `--run` / `--batch`

For artifacts and configuration, see [Nemotron Kit](./kit.md). For execution profiles, see [Execution through NeMo-Run](./nemo-run.md).

## Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryBorderColor': '#333333', 'lineColor': '#333333', 'primaryTextColor': '#333333', 'clusterBkg': '#f5f5f5', 'clusterBorder': '#666666'}}}%%
flowchart LR
    subgraph cli["CLI Layer (Typer)"]
        Root["nemotron"]
        Recipe["nano3"]
        Commands["pretrain/sft/rl"]
    end

    subgraph config["Configuration"]
        YAML["YAML Config"]
        Dotlist["Dotlist Overrides"]
        Artifacts["Artifact Resolution"]
    end

    subgraph execution["Execution Modes"]
        Local["Local (torchrun)"]
        NemoRun["NeMo-Run"]
        Ray["Ray Jobs"]
    end

    Root --> Recipe --> Commands
    Commands --> config
    config --> execution
```

## The @recipe Decorator

Commands are defined using the `@recipe` decorator, which wraps Typer commands with standardized config loading and execution logic:

```python
from nemotron.kit.cli.recipe import recipe
import typer

@recipe(
    name="nano3/pretrain",
    script_path="src/nemotron/recipes/nano3/stage0_pretrain/train.py",
    config_dir="src/nemotron/recipes/nano3/stage0_pretrain/config",
    default_config="default",
    packager="self_contained",
    torchrun=True,
    ray=False,
    artifacts={
        "data": {
            "default": "PretrainBlendsArtifact-default",
            "mappings": {"path": "recipe.per_split_data_args_path"},
        },
    },
)
def pretrain(ctx: typer.Context) -> None:
    """Run pretraining with Megatron-Bridge."""
    pass  # Execution handled by decorator
```

### Decorator Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Recipe identifier (e.g., `"nano3/pretrain"`) |
| `script_path` | `str` | Path to the training script |
| `config_dir` | `str` | Directory containing YAML configs |
| `default_config` | `str` | Default config name (without `.yaml`) |
| `packager` | `str` | Code packaging strategy: `"pattern"`, `"code"`, `"self_contained"` |
| `torchrun` | `bool` | Use `torch.distributed.run` launcher |
| `ray` | `bool` | Submit as Ray job (for data prep, RL) |
| `artifacts` | `dict` | Artifact-to-config mappings |
| `run_command` | `str` | Custom command template for Ray jobs |

### Registering Commands

Commands are registered on Typer apps with specific context settings:

```python
nano3_app = typer.Typer(name="nano3", help="Nano3 training recipe")

nano3_app.command(
    name="pretrain",
    context_settings={
        "allow_extra_args": True,        # Capture dotlist overrides
        "ignore_unknown_options": True,  # Pass through unknown flags
    },
)(pretrain)
```

The `allow_extra_args=True` setting is critical—it allows commands to capture Hydra-style `key=value` overrides.

## Global Options

All recipe commands automatically receive these global options:

| Option | Short | Description |
|--------|-------|-------------|
| `--config` | `-c` | Config name or path (from `config_dir`) |
| `--run` | `-r` | Attached [NeMo-Run](./nemo-run.md) execution (waits, streams logs) |
| `--batch` | `-b` | Detached [NeMo-Run](./nemo-run.md) execution (submits, exits) |
| `--dry-run` | `-d` | Preview config without executing |
| `--stage` | | Stage files to remote for debugging |
| `key=value` | | Dotlist overrides (any position) |

### GlobalContext

Global options are captured in a `GlobalContext` dataclass:

```python
@dataclass
class GlobalContext:
    config: str | None = None      # -c/--config value
    run: str | None = None         # --run profile name
    batch: str | None = None       # --batch profile name
    dry_run: bool = False          # --dry-run flag
    stage: bool = False            # --stage flag
    dotlist: list[str]             # key=value overrides
    passthrough: list[str]         # Unknown args for script
```

Key properties:
- `mode` → `"run"`, `"batch"`, or `"local"`
- `profile` → Environment profile name (from `--run` or `--batch`)

## Configuration Pipeline

The `ConfigBuilder` class orchestrates config loading:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryBorderColor': '#333333', 'lineColor': '#333333', 'primaryTextColor': '#333333'}}}%%
flowchart LR
    Default["default.yaml"] --> Merge
    Config["--config"] --> Merge
    Dotlist["key=value"] --> Merge
    Merge --> JobConfig["job.yaml"]
    Merge --> TrainConfig["train.yaml"]
```

### Two-Config System

The CLI generates two config files:

| File | Purpose |
|------|---------|
| `job.yaml` | Full provenance: config + CLI args + env profile |
| `train.yaml` | Clean config for script (paths rewritten for remote) |

**job.yaml structure:**
```yaml
recipe:
  _target_: megatron.bridge.recipes...
  per_split_data_args_path: /data/blend.json
train:
  train_iters: 1000
run:
  mode: "run"
  profile: "my-cluster"
  env:
    executor: "slurm"
    nodes: 4
    gpus_per_node: 8
  cli:
    argv: ["nemotron", "nano3", "pretrain", "-c", "tiny", "--run", "my-cluster"]
    dotlist: ["train.train_iters=1000"]
  wandb:
    entity: "nvidia"
    project: "nemotron"
```

### Dotlist Overrides

Override any config value with `key.path=value` syntax:

```bash
# Override nested values
uv run nemotron nano3 pretrain train.train_iters=5000

# Multiple overrides
uv run nemotron nano3 pretrain \
    train.train_iters=5000 \
    train.micro_batch_size=2 \
    run.data=PretrainBlendsArtifact-v2:latest
```

## Execution Modes

### Local Execution

Without `--run` or `--batch`, commands execute locally:

```bash
# Local execution (no NeMo-Run)
uv run nemotron nano3 pretrain -c tiny

# Equivalent to:
python -m torch.distributed.run \
    --nproc_per_node=1 \
    src/nemotron/recipes/nano3/stage0_pretrain/train.py \
    --config train.yaml
```

### NeMo-Run Attached (`--run`)

Submit job and wait for completion, streaming logs:

```bash
uv run nemotron nano3 pretrain -c tiny --run MY-CLUSTER
```

### NeMo-Run Detached (`--batch`)

Submit job and exit immediately:

```bash
uv run nemotron nano3 pretrain -c tiny --batch MY-CLUSTER
```

### Ray Jobs

For recipes with `ray=True` (data prep, RL), jobs are submitted via Ray:

```bash
# Data prep uses Ray for distributed processing
uv run nemotron nano3 data prep pretrain --run MY-CLUSTER

# RL uses Ray for actor orchestration
uv run nemotron nano3 rl -c tiny --run MY-CLUSTER
```

## Artifact Inputs

Map W&B artifacts to config fields:

```python
@recipe(
    ...,
    artifacts={
        "data": {
            "default": "PretrainBlendsArtifact-default:latest",
            "mappings": {"path": "recipe.per_split_data_args_path"},
        },
        "model": {
            "default": "ModelArtifact-sft:latest",
            "mappings": {"path": "model.init_from_path"},
        },
    },
)
```

### CLI Override

Override artifacts via dotlist:

```bash
uv run nemotron nano3 sft --run MY-CLUSTER \
    run.data=PretrainBlendsArtifact-v2:latest \
    run.model=ModelArtifact-pretrain:v3
```

### Config Resolver

Use `${art:...}` in YAML configs:

```yaml
run:
  data: PretrainBlendsArtifact-default:latest

recipe:
  per_split_data_args_path: ${art:data,path}/blend.json
```

## Packager Types

Control how code is synced to remote:

| Packager | Description | Use Case |
|----------|-------------|----------|
| `pattern` | Minimal sync (`main.py` + `config.yaml`) | Default |
| `code` | Full codebase with exclusions | Ray jobs needing imports |
| `self_contained` | Inline all `nemotron.*` imports | Isolated scripts |

## CLI Examples

<div class="termy">

```console
// Preview config without executing
$ uv run nemotron nano3 pretrain -c tiny --dry-run

// Submit to cluster (attached)
$ uv run nemotron nano3 pretrain -c tiny --run MY-CLUSTER

// Submit to cluster (detached)
$ uv run nemotron nano3 pretrain -c tiny --batch MY-CLUSTER

// Override training iterations
$ uv run nemotron nano3 pretrain -c tiny --run MY-CLUSTER train.train_iters=5000

// Stage files for interactive debugging
$ uv run nemotron nano3 pretrain -c tiny --run MY-CLUSTER --stage

// Data preparation (Ray job)
$ uv run nemotron nano3 data prep pretrain --run MY-CLUSTER

// RL training (Ray job)
$ uv run nemotron nano3 rl -c tiny --run MY-CLUSTER
```

</div>

## Building a Recipe

### Step 1: Create Config Directory

```
src/nemotron/recipes/myrecipe/
├── config/
│   ├── default.yaml
│   └── tiny.yaml
├── train.py
└── data_prep.py
```

### Step 2: Define Training Script

```python
# train.py
import argparse
from pathlib import Path
from omegaconf import OmegaConf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args, unknown = parser.parse_known_args()

    # Load config
    cfg = OmegaConf.load(args.config)

    # Apply any remaining overrides
    if unknown:
        overrides = OmegaConf.from_dotlist(unknown)
        cfg = OmegaConf.merge(cfg, overrides)

    # Run training...
    print(f"Training with {cfg.train.train_iters} iterations")

if __name__ == "__main__":
    main()
```

### Step 3: Create CLI Command

```python
# src/nemotron/cli/myrecipe/train.py
from nemotron.kit.cli.recipe import recipe
import typer

@recipe(
    name="myrecipe/train",
    script_path="src/nemotron/recipes/myrecipe/train.py",
    config_dir="src/nemotron/recipes/myrecipe/config",
    default_config="default",
    torchrun=True,
)
def train(ctx: typer.Context) -> None:
    """Run training for my recipe."""
    pass
```

### Step 4: Register in CLI

```python
# src/nemotron/cli/myrecipe/__init__.py
import typer
from .train import train

app = typer.Typer(name="myrecipe", help="My training recipe")

app.command(
    name="train",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)(train)
```

### Step 5: Add to Main CLI

```python
# src/nemotron/cli/bin/nemotron.py
from nemotron.cli.myrecipe import app as myrecipe_app

main_app.add_typer(myrecipe_app, name="myrecipe")
```

### Step 6: Run

```bash
# Test locally
uv run nemotron myrecipe train -c tiny

# Run on cluster
uv run nemotron myrecipe train -c tiny --run MY-CLUSTER
```

## API Reference

### Recipe Decorator

| Export | Description |
|--------|-------------|
| `@recipe` | Decorator for training commands |
| `ConfigBuilder` | Config loading and merging |
| `GlobalContext` | Shared CLI state |
| `split_unknown_args()` | Parse dotlist vs passthrough args |

### Execution

| Export | Description |
|--------|-------------|
| `build_executor()` | Create NeMo-Run executor from profile |
| `load_env_profile()` | Load profile from `env.toml` |

## Further Reading

- [Nemotron Kit](./kit.md) — Artifacts, configuration, lineage tracking
- [Execution through NeMo-Run](./nemo-run.md) — Execution profiles and env.toml
- [Data Preparation](./data-prep.md) — Data preparation module
- [Artifact Lineage](./artifacts.md) — W&B artifact system and lineage tracking
- [W&B Integration](./wandb.md) — Credentials and configuration
- [Nano3 Recipe](./nano3/README.md) — Complete training recipe example
