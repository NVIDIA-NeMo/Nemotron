# Importing Models and Data

This guide covers how to import existing models and data as W&B artifacts using the nemotron CLI. This is useful when you want to:

- Use a pre-existing checkpoint from another training run
- Import data prepared outside of the standard pipeline
- Connect external assets to the W&B artifact lineage system

## Prerequisites

- W&B configuration in `env.toml`:
  ```toml
  [wandb]
  project = "nemotron"
  entity = "YOUR-TEAM"
  ```
- Or provide `--project` and `--entity` CLI flags

## Model Import

Import model checkpoints as W&B artifacts for use in downstream training stages.

### Commands

```bash
# Import pretrain model checkpoint
uv run nemotron nano3 model import pretrain /path/to/model_dir --step 10000

# Import SFT model checkpoint
uv run nemotron nano3 model import sft /path/to/model_dir --step 5000

# Import RL model checkpoint
uv run nemotron nano3 model import rl /path/to/model_dir --step 2000
```

### Options

| Option | Description |
|--------|-------------|
| `--step, -s` | Training step number (optional) |
| `--name, -n` | Custom artifact name (default: `nano3/<stage>/model`) |
| `--project, -p` | W&B project (overrides env.toml) |
| `--entity, -e` | W&B entity (overrides env.toml) |

### Examples

```bash
# Import with custom artifact name
uv run nemotron nano3 model import pretrain /lustre/checkpoints/model --step 50000 --name my-pretrain-model

# Import to different W&B project
uv run nemotron nano3 model import sft /path/to/sft_checkpoint --project other-project --entity my-team
```

## Data Import

Import data directories as W&B artifacts for use in training stages.

### Commands

```bash
# Import pretrain data (expects blend.json file)
uv run nemotron nano3 data import pretrain /path/to/blend.json

# Import SFT data (expects directory with blend.json)
uv run nemotron nano3 data import sft /path/to/sft_data_dir

# Import RL data (expects directory with manifest.json)
uv run nemotron nano3 data import rl /path/to/rl_data_dir
```

### Expected Directory Structures

**Pretrain**: Direct path to `blend.json` file
```
/path/to/blend.json
```

**SFT**: Directory containing `blend.json`
```
/path/to/sft_data_dir/
├── blend.json
├── train.npy
├── valid.npy
└── ...
```

**RL**: Directory containing `manifest.json`
```
/path/to/rl_data_dir/
├── manifest.json
├── train.jsonl
├── val.jsonl
└── test.jsonl
```

### Options

| Option | Description |
|--------|-------------|
| `--name, -n` | Custom artifact name (default: `nano3/<stage>/data`) |
| `--project, -p` | W&B project (overrides env.toml) |
| `--entity, -e` | W&B entity (overrides env.toml) |

### Examples

```bash
# Import SFT data with custom name
uv run nemotron nano3 data import sft /lustre/data/sft_v2 --name my-sft-data

# Import RL data to different project
uv run nemotron nano3 data import rl /path/to/rl_data --project alignment-project
```

## Model Evaluation

```bash
uv run nemotron nano3 model eval
```

> **Note**: Model evaluation is coming soon.

## Using Imported Artifacts

After importing, artifacts can be referenced in training commands via `--art.<slot>`:

```bash
# Use imported model in SFT training
uv run nemotron nano3 sft --art.model my-pretrain-model:latest --run YOUR-CLUSTER

# Use imported data in training
uv run nemotron nano3 pretrain --art.data my-pretrain-data:v1 --run YOUR-CLUSTER
```

## CLI Reference

### Model Commands

```bash
uv run nemotron nano3 model --help
uv run nemotron nano3 model eval --help
uv run nemotron nano3 model import --help
uv run nemotron nano3 model import pretrain --help
uv run nemotron nano3 model import sft --help
uv run nemotron nano3 model import rl --help
```

### Data Import Commands

```bash
uv run nemotron nano3 data import --help
uv run nemotron nano3 data import pretrain --help
uv run nemotron nano3 data import sft --help
uv run nemotron nano3 data import rl --help
```
