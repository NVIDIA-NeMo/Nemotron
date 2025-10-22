# Training Framework Design

## Mission

This repository exists to show you **exactly** how NVIDIA Nemotron models are trained, from raw data to final evaluation.

Every training recipe is:
- **Readable Python code** you can understand and learn from
- **Fully reproducible** so you can run it yourself
- **Transparent** with no hidden abstractions or magic

The goal is **education and reproducibility**. When you read a Nemotron training recipe, you should understand every step of the process.

## Why This Matters

Training large language models is often treated as a black box:
- Papers describe methods at a high level
- Codebases are tangled with infrastructure
- Hyperparameters and data pipelines are hidden
- Reproducing results is nearly impossible

**We want to change that.** This repository makes Nemotron training transparent by showing you the complete pipeline in readable code.

## The Challenge

To make training recipes educational and reproducible, we need:

1. **Readable code** - You should understand what happens by reading it
2. **Reusable patterns** - Avoid repeating boilerplate across recipes
3. **Validated outputs** - Catch errors early, not after hours of training
4. **Traceable lineage** - Know exactly which data created which model
5. **Fast iteration** - Test changes quickly before full runs

We built minimal tooling to solve these challenges while keeping the code transparent.

---

## Design Principles

### 1. Code IS Documentation

Training recipes are **self-documenting** through clear, readable code:

```python
def main(config: Config) -> Dataset:
    # Step 1: Load and filter raw data
    raw_data = load_pile_dataset(config.data_path)
    filtered = filter_quality(raw_data, threshold=config.quality_threshold)

    # Step 2: Apply scaling for fast iteration
    actual_size = apply_scale(len(filtered), config.scale)
    sampled = filtered[:actual_size]

    # Step 3: Save as validated artifact
    dataset = Dataset(
        path=config.output_dir,
        num_examples=len(sampled),
        train_path=config.output_dir / "train.parquet"
    )
    dataset.save()
    return dataset
```

No framework abstractions hiding logic. Just Python you can read and understand.

### 2. Artifacts: Validated Outputs

Every step produces an **Artifact** - a validated output with metadata:

```python
class Dataset(Artifact):
    num_examples: int = Field(gt=0)
    train_path: Path
    valid_path: Path

class Checkpoint(Artifact):
    num_steps: int
    training_loss: float
    checkpoint_path: Path
    dataset_path: Path  # Reference to the dataset used
```

**Why?** Because catching errors at step boundaries is better than discovering them hours into training.

Artifacts automatically track:
- What was created (validated fields)
- When it was created (timestamps)
- How it was created (configuration, lineage)
- Where to find it (file paths)

**Artifact References**: Artifacts can reference other artifacts. For example, a Checkpoint artifact stores the path to the dataset it was trained on. This enables the evaluation step to access both the model and the data through a single artifact:

```python
# Evaluation receives the checkpoint artifact
checkpoint = Checkpoint.load()
# Can access both the model and the dataset
model = load_model(checkpoint.checkpoint_path)
dataset = Dataset.load(path=checkpoint.dataset_path)
```

This is how piping works for evaluation - the checkpoint artifact carries all necessary information.

### 3. Unix Philosophy: Composability

Steps connect like Unix tools:

```bash
# Run complete pipeline
python data_curation.py | python training.py | python evaluation.py

# Or run steps individually
python data_curation.py --scale tiny
python training.py --dataset-path /path/to/data
```

**Why?** Because you should be able to inspect, debug, and modify each step independently.

### 4. Scale Factors: Learn Fast

All steps support scale factors:

```bash
# Test with 1% of data (seconds) - max 10k rows
python data_curation.py --scale tiny

# Iterate with 10% (minutes)
python data_curation.py --scale small

# Full production run (hours/days)
python data_curation.py --scale full
```

**Why?** Because you should be able to understand and test a recipe in minutes, not days.

**Important**: `--scale tiny` is capped at 10,000 rows maximum. This ensures even full model pipelines can run in reasonable time for testing and learning.

### 5. Three Steps Per Stage

Every training stage follows the same pattern:

```
1. Data Curation → Dataset artifact
2. Training → Checkpoint artifact
3. Evaluation → Results artifact
```

**Why?** Because consistency across recipes makes them easier to understand and compare.

### 6. Orchestrators: Run Complete Pipelines

While individual steps are great for development, you often want to run complete pipelines. We provide two levels of orchestration:

**Stage Orchestrator** (`stage0_pretrain/__main__.py`) - Runs all three steps of a stage:

```python
# Run complete pretraining stage
python -m nemotron.recipes.nano2.stage0_pretrain --scale tiny
```

**Recipe Orchestrator** (`nano2/__main__.py`) - Runs all stages of a model:

```python
# Run complete model pipeline (all stages)
python -m nemotron.recipes.nano2 --scale tiny
```

**Why orchestrators?**
- **Convenience**: Run complete pipelines with one command
- **Testing**: Validate entire recipes quickly with `--scale tiny`
- **Learning**: See how stages connect together
- **Transparency**: Just Python that calls the individual steps

**Important**: Recipe orchestrators (all stages) should primarily be used with `--scale tiny` for testing, as running all stages at full scale would take days/weeks.

---

## The Minimal Tooling

We built four simple utilities to keep recipes clean:

### 1. `Artifact` - Validated Outputs

Instead of:
```python
# Scattered validation and metadata
assert num_examples > 0
metadata = {"created": datetime.now(), "examples": num_examples}
with open("metadata.json", "w") as f:
    json.dump(metadata, f)
```

Write:
```python
# Validated artifact
dataset = Dataset(path=output_dir, num_examples=num_examples)
dataset.save()
```

### 2. `apply_scale()` - Quick Iteration

Instead of:
```python
# Manual scaling logic everywhere
if scale == "tiny":
    size = int(len(data) * 0.01)
elif scale == "small":
    size = int(len(data) * 0.10)
# ...
```

Write:
```python
# Consistent scaling
size = apply_scale(len(data), config.scale)
```

### 3. `print_complete()` - Unix Piping

Instead of:
```python
# Manual JSON serialization
print(json.dumps({"path": str(output_dir)}))
print(f"Dataset: {num_examples} examples", file=sys.stderr)
```

Write:
```python
# Automatic piping support
print_complete({"dataset": dataset})
```

### 4. `LineageTracker` - Optional Tracking

Instead of:
```python
# Manual W&B integration everywhere
if wandb.run:
    wandb.log_artifact(path, type="dataset")
    wandb.log({"num_examples": num_examples})
```

Write:
```python
# Automatic tracking if configured
set_lineage_tracker(WandbTracker())  # Once at start
# All artifacts automatically tracked
```

---

## What This Enables

With these four utilities, training recipes are:

**Readable** - No boilerplate obscuring logic
```python
def main(config: Config) -> Dataset:
    data = prepare_data(config.sources)
    dataset = Dataset(path=config.output_dir, num_examples=len(data))
    dataset.save()
    return dataset
```

**Composable** - Steps connect naturally
```bash
# Each step receives the previous artifact via stdin
python data_curation.py | \
    python training.py | \
    python evaluation.py

# Training receives dataset artifact
# Evaluation receives checkpoint artifact (which references dataset)
```

**Debuggable** - Inspect any step's output
```bash
python data_curation.py > dataset.json
cat dataset.json  # See what was created
python training.py < dataset.json > checkpoint.json
cat checkpoint.json  # Checkpoint includes dataset_path reference
python evaluation.py < checkpoint.json  # Can access both model and data
```

**Fast to Test** - Iterate quickly
```bash
# Test entire pipeline in seconds
python -m nemotron.recipes.nano2.stage0_pretrain --scale tiny
```

**Reproducible** - Full lineage tracking
```python
# Artifacts track what created them
checkpoint = Checkpoint.load(path)
print(checkpoint.producer)  # Which dataset was used
print(checkpoint.created_at)  # When it was created
```

---

## Design Philosophy

### Transparency Over Abstraction

We prioritize code you can understand over clever frameworks:

❌ **Don't hide logic in abstractions:**
```python
@pipeline_step(inputs=["dataset"], outputs=["checkpoint"])
def train(dataset): ...  # Magic happens somewhere
```

✅ **Make logic explicit:**
```python
def main(config: Config) -> Checkpoint:
    dataset = Dataset.load(path=config.dataset_path)
    model = build_model(config)
    train_loop(model, dataset)
    checkpoint = Checkpoint(path=config.output_dir)
    checkpoint.save()
    return checkpoint
```

### Simplicity Over Features

We provide minimal tooling, not maximal features:

- **No workflow engine** - Just Python scripts you can run
- **No distributed scheduler** - Use any scheduler you want
- **No custom formats** - Just JSON and Parquet
- **No cloud lock-in** - Works with any filesystem

You can add these later if needed. We focus on making the recipes clear.

### Education Over Production Optimization

We optimize for learning, not enterprise deployment:

- **Readable code** over optimized but complex code
- **Clear patterns** over flexible but confusing patterns
- **Local-first** over cloud-native
- **Transparent** over efficient

Production optimizations (distributed training, cloud storage, monitoring) can be added by users who understand the recipes.

---

## What We're NOT Building

To stay focused on education and reproducibility, we explicitly avoid:

- ❌ Workflow orchestration (use Airflow, Prefect, etc.)
- ❌ Experiment management UI (use W&B, MLflow, etc.)
- ❌ Distributed training framework (use NeMo, DeepSpeed, etc.)
- ❌ Cloud infrastructure (use your preferred cloud)
- ❌ Multi-user collaboration (use git and your tools)

These are solved problems. We focus on making training recipes understandable and reproducible.

---

## For Recipe Contributors

When writing a training recipe, prioritize:

1. **Clarity** - Can someone learn from reading this?
2. **Completeness** - Does it show the full process?
3. **Reproducibility** - Can someone run this and get the same results?

Use the framework utilities to reduce boilerplate, but keep the training logic transparent and educational.

### Good Recipe Example

```python
def main(config: Config) -> Checkpoint:
    """Train Nemotron Nano 2 with supervised fine-tuning."""
    # 1. Load instruction dataset from Stage 0
    dataset = Dataset.load(path=config.dataset_path)

    # 2. Load pretrained checkpoint
    model = load_checkpoint(config.base_checkpoint_path)

    # 3. Configure SFT training
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_steps)

    # 4. Training loop (clear and explicit)
    for step, batch in enumerate(dataset.train_dataloader()):
        loss = compute_sft_loss(model, batch)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            print(f"Step {step}: loss={loss.item():.4f}", file=sys.stderr)

    # 5. Save checkpoint as artifact
    checkpoint = Checkpoint(
        path=config.output_dir,
        num_steps=config.num_steps,
        final_loss=loss.item(),
        checkpoint_path=config.output_dir / "model.pt",
        dataset_path=dataset.path  # Store reference for evaluation
    )
    checkpoint.save()
    return checkpoint
```

**Why this works:**
- Every step is explicit and readable
- Training logic is clear, not hidden
- Someone can learn how SFT works by reading this
- Framework utilities handle boilerplate (save/load)
- Checkpoint stores dataset reference so evaluation can access both model and data

### Stage Orchestrator Example

For convenience, create a `__main__.py` in each stage to run all three steps:

```python
# stage0_pretrain/__main__.py
from dataclasses import dataclass, field
import tyro
from .data_curation import main as run_data, Config as DataConfig
from .training import main as run_train, Config as TrainConfig
from .evaluation import main as run_eval, Config as EvalConfig

@dataclass
class StageConfig:
    """Configuration for complete pretraining stage."""
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

def main(config: StageConfig) -> None:
    """Run complete pretraining stage: data → training → evaluation."""
    # Step 1: Data curation
    dataset = run_data(config.data)

    # Step 2: Training (use dataset from step 1)
    config.train.dataset_path = dataset.path
    checkpoint = run_train(config.train)

    # Step 3: Evaluation (use checkpoint from step 2)
    config.eval.checkpoint_path = checkpoint.path
    run_eval(config.eval)

if __name__ == "__main__":
    tyro.cli(main)
```

**Why orchestrators are simple Python:**
- No framework magic - just function calls
- Easy to understand and modify
- Steps can still be run individually
- Artifacts automatically flow between steps

---

## Summary

This repository exists to show you how Nemotron models are trained. The framework is minimal tooling that keeps recipes:

- **Transparent** - Readable Python, no hidden abstractions
- **Reproducible** - Validated artifacts with full lineage
- **Educational** - Learn by reading the code
- **Practical** - Fast iteration with scale factors

When in doubt, choose clarity over cleverness. The goal is education and reproducibility, not building a framework.
