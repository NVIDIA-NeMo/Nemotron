# Adding a Model Training Recipe

This guide explains how to contribute a new training recipe for a Nemotron model.

**Before you start**, read [design.md](design.md) to understand the moving parts of the a recipe.

---

## Recipe Structure

Each model recipe lives in `src/nemotron/recipes/` and follows this structure:

```
src/nemotron/recipes/
└── {model_name}/              # e.g., nano2, nemotron4_340b
    ├── README.md             # Recipe documentation
    ├── __init__.py
    ├── stage0_pretrain/
    │   ├── __init__.py
    │   ├── __main__.py      # Optional: run complete stage
    │   ├── data_curation.py # Step 1: Prepare data
    │   ├── training.py      # Step 2: Train model
    │   └── evaluation.py    # Step 3: Evaluate results
    ├── stage1_instruction_tuning/
    │   └── ... (same pattern)
    └── stage2_alignment/
        └── ... (RLHF, DPO, etc.)
```

**CLI Convention**: Each stage script uses [tyro](https://github.com/brentyi/tyro) (popularized by [torchtitan](https://github.com/pytorch/torchtitan)) for automatic CLI generation from dataclasses. This gives you type-safe configuration and auto-generated `--help` documentation for free.

---

## The Three-Step Pattern

Every training stage has three steps that follow the same pattern:

### Step 1: Data Curation

**Purpose**: Prepare high-quality training data

**Dependencies**: Typically uses [NVIDIA NeMo Curator](https://github.com/NVIDIA/NeMo-Curator) for scalable data processing, filtering, and quality enhancement.

```python
# src/nemotron/recipes/{model_name}/stage0_pretrain/data_curation.py
from dataclasses import dataclass
from pathlib import Path
import tyro
from pydantic import Field
from nemotron.artifact import Artifact, apply_scale, print_complete

@dataclass
class Config:
    output_dir: Path = Path("/workspace/outputs/{model}/stage0/data")
    sources: list[str] = field(default_factory=lambda: ["/data/pile"])
    quality_threshold: float = 0.7
    scale: str = "full"  # tiny/small/medium/full
    seed: int = 42

class Dataset(Artifact):
    """Output: Curated training dataset."""
    num_examples: int = Field(gt=0)
    num_tokens: int = Field(gt=0)
    train_path: Path
    valid_path: Path

def main(config: Config) -> Dataset:
    # 1. Setup and apply scale
    config.output_dir.mkdir(parents=True, exist_ok=True)
    actual_size = apply_scale(1_000_000, config.scale)

    # 2. Curate data (your logic here)
    train_data = curate_and_filter(config.sources, actual_size)

    # 3. Save data files
    train_path = config.output_dir / "train.parquet"
    save_data(train_data, train_path)

    # 4. Create and save artifact
    artifact = Dataset(
        path=config.output_dir,
        num_examples=len(train_data),
        num_tokens=count_tokens(train_data),
        train_path=train_path,
        valid_path=config.output_dir / "valid.parquet",
        metrics={"num_examples": len(train_data)},
    )
    artifact.save()
    print_complete({"data_curation": artifact})
    return artifact

if __name__ == "__main__":
    tyro.cli(main)
```

### Step 2: Training

**Purpose**: Train the model on curated data

**Dependencies**:
- Training frameworks (depends on model architecture and training approach):
  - [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) for Megatron-based models
  - [Automodel](https://github.com/NVIDIA-NeMo/Automodel) for HuggingFace-based models
  - [NeMo-RL](https://github.com/NVIDIA-NeMo/RL) when reinforcement learning is required
- **Last-mile data processing**: GPU-accelerated tokenization and optional sequence packing for optimal training efficiency

**Note**: Training includes a last-mile data processing step that performs tokenization and optionally sequence packing in a GPU-accelerated manner to maximize throughput and minimize preprocessing bottlenecks. This approach keeps the interface between data curation and training very clean—data curation outputs raw text data, while training handles all model-specific preprocessing (tokenization, packing) on-the-fly during training.

```python
# src/nemotron/recipes/{model_name}/stage0_pretrain/training.py
from dataclasses import dataclass
from pathlib import Path
import tyro
from pydantic import Field
from nemotron.artifact import Artifact, apply_scale, print_complete
from .data_curation import Dataset

@dataclass
class Config:
    # Inputs
    dataset_path: Path | None = None
    dataset_artifact: str | None = None

    # Outputs
    output_dir: Path = Path("/workspace/outputs/{model}/stage0/training")

    # Model hyperparameters
    hidden_size: int = 2048
    num_layers: int = 24
    learning_rate: float = 3e-4
    num_steps: int = 100_000
    scale: str = "full"
    seed: int = 42

class Checkpoint(Artifact):
    """Output: Trained model checkpoint."""
    num_steps: int = Field(gt=0)
    training_loss: float = Field(ge=0.0)
    checkpoint_path: Path

def main(config: Config) -> Checkpoint:
    # 1. Load dataset artifact
    dataset = Dataset.load(
        path=config.dataset_path,
        tracked_artifact=config.dataset_artifact
    )

    # 2. Apply scale and train
    actual_steps = apply_scale(config.num_steps, config.scale)
    model = build_model(config.hidden_size, config.num_layers)
    final_loss = train(model, dataset.train_path, actual_steps)

    # 3. Save checkpoint
    checkpoint_path = config.output_dir / "model.pt"
    save_checkpoint(model, checkpoint_path)

    # 4. Create and save artifact
    artifact = Checkpoint(
        path=config.output_dir,
        num_steps=actual_steps,
        training_loss=final_loss,
        checkpoint_path=checkpoint_path,
        metrics={"training_loss": final_loss},
    )
    artifact.save()
    print_complete({"training": artifact})
    return artifact

if __name__ == "__main__":
    tyro.cli(main)
```

### Step 3: Evaluation

**Purpose**: Benchmark the trained model

**Dependencies**: Typically uses [NVIDIA NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator) for standardized benchmark evaluation.

```python
# src/nemotron/recipes/{model_name}/stage0_pretrain/evaluation.py
from dataclasses import dataclass, field
from pathlib import Path
import tyro
from nemotron.artifact import Artifact, print_complete
from .training import Checkpoint

@dataclass
class Config:
    checkpoint_path: Path | None = None
    checkpoint_artifact: str | None = None
    output_dir: Path = Path("/workspace/outputs/{model}/stage0/eval")
    benchmarks: list[str] = field(default_factory=lambda: ["mmlu", "hellaswag"])
    seed: int = 42

class Results(Artifact):
    """Output: Evaluation results."""
    benchmarks: list[str]

def main(config: Config) -> Results:
    # 1. Load checkpoint
    checkpoint = Checkpoint.load(
        path=config.checkpoint_path,
        tracked_artifact=config.checkpoint_artifact
    )

    # 2. Run evaluation
    model = load_checkpoint(checkpoint.checkpoint_path)
    metrics = {}
    for benchmark in config.benchmarks:
        metrics[benchmark] = evaluate(model, benchmark, config.seed)

    # 3. Save results
    artifact = Results(
        path=config.output_dir,
        benchmarks=config.benchmarks,
        metrics=metrics,
    )
    artifact.save()
    print_complete({"evaluation": artifact})
    return artifact

if __name__ == "__main__":
    tyro.cli(main)
```

---

## Key Concepts

### 1. Artifacts

Artifacts are validated outputs that track:
- **Metadata**: Configuration, metrics, timestamps
- **File paths**: Where actual data/models are stored
- **Lineage**: Which artifacts were used to create this one

All artifacts inherit from `nemotron.artifact.Artifact` and use Pydantic for validation.

### 2. CLI with Tyro

Each stage uses [tyro](https://github.com/brentyi/tyro) for the command-line interface (popularized by [torchtitan](https://github.com/pytorch/torchtitan)). This allows us to:
- Define configuration using Python `@dataclass`
- Get CLI entrypoints automatically from type annotations
- Generate `--help` documentation for free from docstrings and type hints
- Maintain type safety between CLI and code

**Pattern**:
- Use `@dataclass` for configuration with typed fields
- Call `tyro.cli(main)` to auto-generate the CLI
- Always include `scale` and `seed` parameters
- Support both local paths and tracked artifacts for inputs

Example: `tyro.cli(main)` automatically converts this dataclass into a CLI with `--output-dir`, `--scale`, `--seed`, etc.

### 3. Scale Factors

Use `apply_scale()` for fast iteration:
- `tiny`: 1% (smoke tests)
- `small`: 10% (rapid iteration)
- `medium`: 30% (validation)
- `full`: 100% (production)

### 4. stdout/stderr Convention

- **stdout**: JSON output for piping (handled by `print_complete()`)
- **stderr**: Human-readable logs and progress
- Always use `file=sys.stderr` for print statements

---

## Optional: Stage Orchestrator

Create `__main__.py` to run all steps together:

```python
# src/nemotron/recipes/{model}/stage0_pretrain/__main__.py
from dataclasses import dataclass, field
import tyro
from .data_curation import main as run_data, Config as DataConfig
from .training import main as run_train, Config as TrainConfig
from .evaluation import main as run_eval, Config as EvalConfig

@dataclass
class Config:
    """Complete stage configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

def main(config: Config) -> None:
    dataset = run_data(config.data)
    config.train.dataset_path = dataset.path
    checkpoint = run_train(config.train)
    config.eval.checkpoint_path = checkpoint.path
    run_eval(config.eval)

if __name__ == "__main__":
    tyro.cli(main)
```

---

## Recipe Checklist

Before submitting a recipe, ensure:

- [ ] All three steps (data, training, eval) are implemented
- [ ] Each step has proper docstrings and type hints
- [ ] Scale factors work correctly (test with `--scale tiny`)
- [ ] Artifacts validate correctly (Pydantic models)
- [ ] Steps can be run individually or piped together
- [ ] Print statements use `file=sys.stderr` for logs
- [ ] README.md documents the recipe with:
  - Model description and size
  - Expected training time and resources
  - Example commands
  - Expected results/benchmarks
- [ ] Code follows existing style (see `examples/` for reference)

---

## Testing Your Recipe

```bash
# 1. Smoke test with tiny scale
uv run python src/nemotron/recipes/{model}/stage0_pretrain/data_curation.py --scale tiny
uv run python src/nemotron/recipes/{model}/stage0_pretrain/training.py \
    --dataset-path /workspace/outputs/{model}/stage0/data --scale tiny
uv run python src/nemotron/recipes/{model}/stage0_pretrain/evaluation.py \
    --checkpoint-path /workspace/outputs/{model}/stage0/training

# 2. Test piping
uv run python src/nemotron/recipes/{model}/stage0_pretrain/data_curation.py --scale tiny | \
    uv run python src/nemotron/recipes/{model}/stage0_pretrain/training.py --scale tiny | \
    uv run python src/nemotron/recipes/{model}/stage0_pretrain/evaluation.py

# 3. Test as module
uv run python -m nemotron.recipes.{model}.stage0_pretrain --scale tiny

# 4. Verify artifacts
ls -lh /workspace/outputs/{model}/stage0/*/metadata.json
cat /workspace/outputs/{model}/stage0/data/metadata.json
```

---

## Recipe Documentation

Create a README.md for your recipe:

```markdown
# {Model Name} Training Recipe

## Model Overview
- Architecture: ...
- Size: ... parameters
- Training data: ...

## Hardware Requirements
- GPU: ... (e.g., 8x A100 80GB)
- Memory: ... GB
- Storage: ... TB

## Expected Results
- Training time: ... hours
- Final loss: ~...
- Benchmark scores:
  - MMLU: ...%
  - HellaSwag: ...%

## Running the Recipe

\`\`\`bash
# Quick test
uv run python -m nemotron.recipes.{model}.stage0_pretrain --scale tiny

# Full training
uv run python -m nemotron.recipes.{model}.stage0_pretrain --scale full
\`\`\`

## Citation
...
```

---

## Questions?

- Check `examples/` for working examples
- Look at existing recipes in `src/nemotron/recipes/` (when available)
- See [Run Recipe Guide](run-recipe.md) for usage instructions
- Open an issue for guidance

---

**Ready to contribute?** Follow this pattern and your recipe will be consistent with the rest of the repository!
