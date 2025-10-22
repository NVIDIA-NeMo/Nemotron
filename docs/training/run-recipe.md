# Running Training Recipes

This guide explains how to run existing training recipes for Nemotron models.

---

## Quick Start

```bash
# Run a complete stage with tiny scale (uv handles dependencies automatically)
uv run python -m nemotron.recipes.nano2.stage0_pretrain --scale tiny
```

---

## Running Individual Steps

Each training stage has three steps that can be run independently:

### 1. Data Curation

Prepare the training dataset:

```bash
# As a script
uv run python src/nemotron/recipes/nano2/stage0_pretrain/data_curation.py --scale tiny

# As a module
uv run python -m nemotron.recipes.nano2.stage0_pretrain.data_curation --scale tiny
```

### 2. Training

Train the model on the curated data:

```bash
# Using saved artifacts
uv run python src/nemotron/recipes/nano2/stage0_pretrain/training.py \
    --dataset-path /workspace/outputs/nano2/stage0/data \
    --scale tiny

# Or as a module
uv run python -m nemotron.recipes.nano2.stage0_pretrain.training --scale tiny
```

### 3. Evaluation

Evaluate the trained model:

```bash
uv run python src/nemotron/recipes/nano2/stage0_pretrain/evaluation.py \
    --checkpoint-path /workspace/outputs/nano2/stage0/training

# Or as a module
uv run python -m nemotron.recipes.nano2.stage0_pretrain.evaluation
```

---

## Running Complete Stages

Run all steps together using the stage module:

```bash
uv run python -m nemotron.recipes.nano2.stage0_pretrain --scale tiny
```

---

## Unix Piping

Steps automatically connect via JSON output:

```bash
# Pipe all three steps together
uv run python src/nemotron/recipes/nano2/stage0_pretrain/data_curation.py --scale tiny | \
    uv run python src/nemotron/recipes/nano2/stage0_pretrain/training.py --scale tiny | \
    uv run python src/nemotron/recipes/nano2/stage0_pretrain/evaluation.py
```

When piping, the output artifact from one step is automatically passed to the next via stdin.

---

## Scale Factors

All recipes support scale factors for fast iteration:

| Scale | Data Size | Use Case |
|-------|-----------|----------|
| `tiny` | 1% | Smoke tests, debugging |
| `small` | 10% | Rapid iteration, experimentation |
| `medium` | 30% | Validation before full run |
| `full` | 100% | Production training |

**Example:**

```bash
# Quick smoke test
uv run python -m nemotron.recipes.nano2.stage0_pretrain --scale tiny

# Experiment with 10% of data
uv run python -m nemotron.recipes.nano2.stage0_pretrain --scale small

# Full training run
uv run python -m nemotron.recipes.nano2.stage0_pretrain --scale full
```

---

## Configuration

All steps use [tyro](https://github.com/brentyi/tyro) for CLI generation, so you can customize any parameter:

```bash
# View all options
uv run python src/nemotron/recipes/nano2/stage0_pretrain/training.py --help

# Override parameters
uv run python src/nemotron/recipes/nano2/stage0_pretrain/training.py \
    --scale small \
    --learning-rate 1e-4 \
    --num-steps 50000 \
    --seed 123
```

---

## Artifacts

Each step creates an artifact with:
- **Metadata**: Configuration, metrics, timestamps
- **File paths**: Where data/models are stored
- **Lineage**: Which artifacts were used

Artifacts are saved to `{output_dir}/metadata.json` and can be loaded by subsequent steps.

**Example artifact structure:**

```
/workspace/outputs/nano2/stage0/
├── data/
│   ├── metadata.json          # Dataset artifact
│   ├── train.parquet
│   └── valid.parquet
├── training/
│   ├── metadata.json          # Checkpoint artifact
│   └── model.pt
└── eval/
    └── metadata.json          # Results artifact
```

---

## Tracking (Optional)

Enable W&B tracking with environment variables:

```bash
export WANDB_PROJECT="nemotron-nano2"
export WANDB_RUN_ID="stage0-pretrain-$(date +%s)"

uv run python -m nemotron.recipes.nano2.stage0_pretrain --scale full
```

When W&B is active, all artifacts and metrics are automatically logged with lineage tracking.

---

## Troubleshooting

**Import errors:**
```bash
# uv automatically handles dependencies - try running with uv run
uv run python -m nemotron.recipes.nano2.stage0_pretrain --scale tiny
```

**Permission errors:**
```bash
# Check output directory permissions
ls -ld /workspace/outputs
```

**Out of memory:**
```bash
# Start with smaller scale
uv run python -m nemotron.recipes.nano2.stage0_pretrain --scale tiny
```

---

## Next Steps

- Check individual recipe READMEs for model-specific details
- See [Add Recipe Guide](add-recipe.md) to contribute new recipes
- Explore `examples/` for framework tutorials
