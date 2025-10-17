# NVIDIA Nemotron Developer Repository

Developer companion repo for working with NVIDIA's Nemotron models: inference, fine-tuning, agents, visual reasoning, deployment.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📂 Repo Layout

```
nemotron/
├── docs/                  Documentation and guides
│   ├── training/          Recipe guides (run-recipe.md, add-recipe.md)
│   ├── usage/             Usage guides (coming soon)
│   └── deployment/        Deployment guides (coming soon)
│
├── examples/              Runnable code examples
│   ├── training/          Training pipeline patterns (data prep, training, eval)
│   └── usage/             Agent workflows, RAG, tool use (coming soon)
│
├── src/nemotron/          The nemotron package
│   ├── ...                # Internals of nemotron package
│   └── recipes/           Full training reproductions (coming soon)
│       ├── nano2/         2B pretraining, instruction tuning, alignment
│       └── chipnemo/      Domain adaptation for chip design
│
└── tests/                 Training recipe tests
```

---

## What is Nemotron?

[NVIDIA Nemotron™](https://developer.nvidia.com/nemotron) is a family of open, high-efficiency models with fully transparent training data, weights, and recipes.

Nemotron models are designed for **agentic AI workflows** — they excel at coding, math, scientific reasoning, tool calling, instruction following, and visual reasoning (for the VL models).

They are optimized for deployment across a spectrum of compute tiers (edge, single GPU, data center) and support frameworks like NeMo and TensorRT-LLM, vLLM, and SGLang, with NIM microservice options for scalable serving.

---

## Quick Start

> TODO: Flesh out quick start
```bash
# Clone the repository
git clone https://github.com/NVIDIA-NeMo/Nemotron
cd nemotron
```
---

## Usage & Examples

Check out the **[examples/](examples/)** directory for runnable code examples.

### Quick Links

- **[Training Examples](examples/training/)** - Training pipeline patterns (data prep, training, eval)
- **[Usage Examples](examples/usage/)** - Agent workflows, RAG, tool use *(coming soon)*
- **[Usage Guides](docs/usage/)** - How-to guides for working with Nemotron *(coming soon)*
- **[Deployment Guides](docs/deployment/)** - Deployment strategies and best practices *(coming soon)*

### Example Training Pipeline

```bash
# Run a complete 3-stage training pipeline
uv run python examples/training/tutorial_data_prep.py | \
    uv run python examples/training/tutorial_training.py | \
    uv run python examples/training/tutorial_evaluation.py
```

---

## Training Recipes (Coming Soon)

Full, reproducible training pipelines are included in the `nemotron` package at `src/nemotron/recipes/`.

### Each Recipe Includes
- 🗂️ **Data Curation** - Scripts to prepare training data using [NVIDIA-NeMo/Curator](https://github.com/NVIDIA-NeMo/Curator)
- 🔁 **Training** - Complete training loops with hyperparameters using:
  - [NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main) for Megatron models
  - [NVIDIA-NeMo/Automodel](https://github.com/NVIDIA-NeMo/Automodel) for HuggingFace models
  - [NVIDIA-NeMo/NeMo-RL](https://github.com/NVIDIA-NeMo/RL/tree/main) when RL is needed
- 📊 **Evaluation** - Benchmark evaluation on standard suites using [NVIDIA-NeMo/Evaluator](https://github.com/NVIDIA-NeMo/Evaluator)
- 📖 **Documentation** - Detailed explanations of each stage

### Running a Recipe

```bash
# Run complete stage (uv handles dependencies automatically)
uv run python -m nemotron.recipes.nano2.stage0_pretrain --scale tiny

# Or run individual steps as scripts
uv run python -m nemotron.recipes.nano2.stage0_pretrain.data_curation --scale tiny
uv run python -m nemotron.recipes.nano2.stage0_pretrain.training \
    --dataset-path /workspace/outputs/nano2/stage0/data
uv run python -m nemotron.recipes.nano2.stage0_pretrain.evaluation \
    --checkpoint-path /workspace/outputs/nano2/stage0/training
```

See **[Running Training Recipes](docs/training/run-recipe.md)** for detailed usage instructions.

### Available Recipes

- **Nemotron Nano 2** (`nemotron.recipes.nano2`) - 2B model with pretraining, instruction tuning, and alignment ([paper](https://arxiv.org/pdf/2508.14444))
- **ChipNeMo/ScaleRTL** (`nemotron.recipes.chipnemo`) - Domain-adapted LLM for RTL code generation with reasoning data ([paper](https://arxiv.org/abs/2506.05566))

---

## Contributing

We welcome contributions! Whether it's examples, recipes, or other tools you'd find useful.

Please read our **[Contributing Guidelines](CONTRIBUTING.md)** before submitting pull requests.

### Adding a New Training Recipe

See the documentation in `docs/training/`:
- **[Running Recipes](docs/training/run-recipe.md)** - How to run existing training recipes
- **[Adding Recipes](docs/training/add-recipe.md)** - Guide for contributing new recipes

---

## Documentation

- **[Contributing Guidelines](CONTRIBUTING.md)** - How to contribute to this project
- **[Changelog](CHANGELOG.md)** - Version history and changes
- **[Framework Design](docs/training/design.md)** - Design philosophy and key concepts
- **[Running Training Recipes](docs/training/run-recipe.md)** - How to run existing recipes
- **[Adding Training Recipes](docs/training/add-recipe.md)** - Guide for contributing new recipes
- More guides coming soon...

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Repository Structure

```
nemotron/
├── docs/                      # Documentation and guides
│   ├── deployment/            # Deployment guides (coming soon)
│   ├── training/              # Training recipe documentation
│   │   ├── add-recipe.md     # How to add recipes
│   │   └── run-recipe.md     # How to run recipes
│   └── usage/                 # Usage guides (coming soon)
├── examples/                  # Runnable code examples
│   ├── training/              # Training pipeline patterns
│   └── usage/                 # Agent workflows, RAG, tool use (coming soon)
├── src/nemotron/              # The nemotron package
│   └── recipes/               # (Coming soon) Training recipes
│       ├── chipnemo/         # ChipNeMo/ScaleRTL
│       │   ├── stage0_pretrain/
│       │   ├── stage1_sft/
│       │   └── stage2_reasoning/
│       └── nano2/            # Nemotron Nano 2
│           ├── stage0_pretrain/
│           ├── stage1_instruction_tuning/
│           └── stage2_alignment/
└── tests/                     # Training recipe tests
```

---

**NVIDIA Nemotron** - Open, transparent, and reproducible.
