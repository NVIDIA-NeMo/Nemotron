# Nemotron Training Recipes

**Open and efficient models for agentic AI** — reproducible training pipelines with fully transparent data, techniques, and weights.

<div style="text-align: center; margin: 2rem 0;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/_y9SEtn1lU8" title="Nemotron Overview" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

## Quick Start

<div class="termy">

```console
// Install the Nemotron training recipes
$ git clone https://github.com/NVIDIA/nemotron
$ cd nemotron && uv sync

// Run the full Nano3 pipeline
$ uv run nemotron nano3 data prep pretrain --run YOUR-CLUSTER
$ uv run nemotron nano3 pretrain --run YOUR-CLUSTER
$ uv run nemotron nano3 data prep sft --run YOUR-CLUSTER
$ uv run nemotron nano3 sft --run YOUR-CLUSTER
$ uv run nemotron nano3 data prep rl --run YOUR-CLUSTER
$ uv run nemotron nano3 rl --run YOUR-CLUSTER
```

</div>

> **Note**: The `--run YOUR-CLUSTER` flag submits jobs to your configured Slurm cluster via [NeMo-Run](train/nemo-run.md). See [Execution through NeMo-Run](train/nemo-run.md) for setup instructions.

## Usage Cookbook & Examples

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Usage Cookbook
:link: usage-cookbook/README
:link-type: doc

Deployment guides for Nemotron models: TensorRT-LLM, vLLM, SGLang, NIM, and Hugging Face.
:::

:::{grid-item-card} Use Case Examples
:link: use-case-examples/README
:link-type: doc

End-to-end applications: RAG agents, ML agents, and multi-agent systems.
:::

::::

## Available Training Recipes

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Nemotron 3 Nano
:link: train/nano3/README
:link-type: doc

31.6B total / 3.6B active parameters, 25T tokens, up to 1M context. Hybrid Mamba-Transformer with sparse MoE.

**Stages:** Pretraining → SFT → RL
:::

::::

## Training Pipeline

The Nemotron training pipeline follows a four-stage approach with full artifact lineage tracking:

| Stage | Name | Description |
|-------|------|-------------|
| 0 | [Pretraining](train/nano3/pretrain.md) | Base model training on large text corpus |
| 1 | [SFT](train/nano3/sft.md) | Supervised fine-tuning for instruction following |
| 2 | [RL](train/nano3/rl.md) | Reinforcement learning for alignment |
| 3 | [Evaluation](train/nano3/eval.md) | Benchmark testing with NeMo Evaluator |

## Why Nemotron?

| | |
|---|---|
| **Open Models** | Transparent training data, techniques, and weights for community innovation |
| **Compute Efficiency** | Model pruning enabling higher throughput via TensorRT-LLM |
| **High Accuracy** | Built on frontier open models with human-aligned reasoning |
| **Flexible Deployment** | Deploy anywhere — edge, single GPU, or data center with NIM |

## Key Features

- **Complete Pipelines** — From raw data to deployment-ready models
- **[Artifact Lineage](train/artifacts.md)** — Full traceability via [W&B](train/wandb.md) from data to model
- **Production-Grade** — Built on [NVIDIA's NeMo stack](train/nvidia-stack.md) (Megatron-Bridge, NeMo-RL)
- **Reproducible** — Versioned configs, data blends, and checkpoints

## Resources

- [Tech Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf) — Nemotron 3 Nano methodology
- [Model Weights](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3) — Pre-trained checkpoints on HuggingFace
- [Pre-training Datasets](https://huggingface.co/collections/nvidia/nemotron-pre-training-datasets) — Open pre-training data
- [Post-training Datasets](https://huggingface.co/collections/nvidia/nemotron-post-training-v3) — SFT and RL data
- [Artifact Lineage](train/artifacts.md) — W&B integration guide

```{toctree}
:caption: Usage Cookbook
:hidden:

usage-cookbook/README.md
usage-cookbook/Nemotron-Nano2-VL/README.md
usage-cookbook/Nemotron-Parse-v1.1/README.md
```

```{toctree}
:caption: Use Case Examples
:hidden:

use-case-examples/README.md
use-case-examples/Simple Nemotron-3-Nano Usage Example/README.md
use-case-examples/Data Science ML Agent/README.md
use-case-examples/RAG Agent with Nemotron RAG Models/README.md
```

```{toctree}
:caption: Training Recipes
:hidden:

train/nano3/README.md
train/artifacts.md
```

```{toctree}
:caption: Nano3 Stages
:hidden:

train/nano3/pretrain.md
train/nano3/sft.md
train/nano3/rl.md
train/nano3/eval.md
train/nano3/import.md
```

```{toctree}
:caption: Nemotron Kit
:hidden:

train/kit.md
train/nvidia-stack.md
train/nemo-run.md
train/omegaconf.md
train/wandb.md
train/cli.md
train/data-prep.md
train/evaluator.md
```
