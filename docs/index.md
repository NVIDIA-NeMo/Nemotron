# Nemotron Training Recipes

**Open and efficient models for agentic AI.** Reproducible training pipelines with transparent data, techniques, and weights.

<div style="text-align: center; margin: 2rem 0;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/_y9SEtn1lU8" title="Nemotron Overview" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

## Quick Start

<div class="termy">

```console
// Install the Nemotron training recipes
$ git clone https://github.com/NVIDIA/nemotron
$ cd nemotron && uv sync

// Run the Nano3 pipeline stage by stage
$ uv run nemotron nano3 data prep pretrain --run YOUR-CLUSTER
$ uv run nemotron nano3 pretrain --run YOUR-CLUSTER
$ uv run nemotron nano3 data prep sft --run YOUR-CLUSTER
$ uv run nemotron nano3 sft --run YOUR-CLUSTER
$ uv run nemotron nano3 data prep rl --run YOUR-CLUSTER
$ uv run nemotron nano3 rl --run YOUR-CLUSTER
```

</div>

> **Note**: The `--run YOUR-CLUSTER` flag submits jobs to your configured Slurm cluster via [NeMo-Run](nemo_runspec/nemo-run.md). See [Execution through NeMo-Run](nemo_runspec/nemo-run.md) for setup instructions.

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
:link: nemotron/nano3/README
:link-type: doc

31.6B total / 3.6B active parameters, 25T tokens, up to 1M context. Hybrid Mamba-Transformer with sparse MoE.

**Stages:** Pretraining → SFT → RL
:::

:::{grid-item-card} Embedding Fine-Tuning
:link: nemotron/embed/README
:link-type: doc

Fine-tune Llama-Nemotron-Embed-1B-v2 on domain-specific data with synthetic data generation, evaluation, and NIM deployment.

**Stages:** SDG → Data Prep → Finetune → Eval → Export → Deploy
:::

::::

## Training Pipeline

The Nemotron training pipeline has three stages, each tracked through [artifact lineage](nemotron/artifacts.md):

| Stage | Name | Description |
|-------|------|-------------|
| 0 | [Pretraining](nemotron/nano3/pretrain.md) | Base model training on large text corpus |
| 1 | [SFT](nemotron/nano3/sft.md) | Supervised fine-tuning for instruction following |
| 2 | [RL](nemotron/nano3/rl.md) | Reinforcement learning for alignment |

## Why Nemotron?

| | |
|---|---|
| **Open Models** | Transparent training data, techniques, and weights for community innovation |
| **Compute Efficiency** | Model pruning enabling higher throughput via TensorRT-LLM |
| **High Accuracy** | Built on frontier open models with human-aligned reasoning |
| **Flexible Deployment** | Deploy anywhere: edge, single GPU, or data center with NIM |

## Features

- **End-to-end pipelines** from raw data to deployment-ready models
- **[Artifact lineage](nemotron/artifacts.md)** via [W&B](nemotron/wandb.md) from data to model
- **Built on [NVIDIA's NeMo stack](nemotron/nvidia-stack.md)** (Megatron-Bridge, NeMo-RL)
- **Reproducible** with versioned configs, data blends, and checkpoints

## Resources

- [Tech Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf) – Nemotron 3 Nano methodology
- [Model Weights](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3) – pre-trained checkpoints on HuggingFace
- [Pre-training Datasets](https://huggingface.co/collections/nvidia/nemotron-pre-training-datasets) – open pre-training data
- [Post-training Datasets](https://huggingface.co/collections/nvidia/nemotron-post-training-v3) – SFT and RL data
- [Artifact Lineage](nemotron/artifacts.md) – W&B integration guide

```{toctree}
:caption: Usage Cookbook
:hidden:

usage-cookbook/README.md
usage-cookbook/Nemotron-Nano2-VL/README.md
usage-cookbook/Nemotron-Parse-v1.1/README.md
usage-cookbook/Nemotron-3-Super/README.md
usage-cookbook/Nemotron-3-Super/grpo-dapo/README.md
usage-cookbook/Nemotron-3-Super/lora-text2sql/README.md
usage-cookbook/Nemotron-3-Super/lora-text2sql/nemo-automodel/README.md
usage-cookbook/Nemotron-3-Super/lora-text2sql/nemo-megatron-bridge/README.md
usage-cookbook/Nemotron-3-Super/AdvancedDeploymentGuide/README.md
usage-cookbook/Nemotron-3-Super/SparkDeploymentGuide/README.md
usage-cookbook/Nemotron-3-Super/OpenScaffoldingResources/README.md
```

```{toctree}
:caption: Use Case Examples
:hidden:

use-case-examples/README.md
use-case-examples/Simple Nemotron-3-Nano Usage Example/README.md
use-case-examples/Data Science ML Agent/README.md
use-case-examples/RAG Agent with Nemotron RAG Models/README.md
use-case-examples/Intelligent Document Processing with Nemotron RAG/README.md
use-case-examples/nemotron-voice-rag-agent-example/README.md
use-case-examples/sql-lora-finetuning-and-deployment/README.md
```

```{toctree}
:caption: Training Recipes
:hidden:

nemotron/nano3/README.md
nemotron/super3/README.md
nemotron/embed/README.md
nemotron/artifacts.md
```

```{toctree}
:caption: Nano3 Stages
:hidden:

nemotron/nano3/pretrain.md
nemotron/nano3/sft.md
nemotron/nano3/rl.md
nemotron/nano3/evaluate.md
nemotron/nano3/import.md
```

```{toctree}
:caption: Super3 Stages
:hidden:

nemotron/super3/README.md
nemotron/super3/pretrain.md
nemotron/super3/sft.md
nemotron/super3/rl/index.md
nemotron/super3/rl/rlvr.md
nemotron/super3/rl/swe.md
nemotron/super3/rl/rlhf.md
nemotron/super3/rl/data-prep.md
nemotron/super3/evaluate.md
nemotron/super3/quantization.md
```

```{toctree}
:caption: Nemotron Kit
:hidden:

nemotron/kit.md
nemotron/nvidia-stack.md
nemo_runspec/package-readme.md
nemo_runspec/nemo-run.md
nemo_runspec/omegaconf.md
nemo_runspec/artifacts.md
nemotron/wandb.md
nemotron/cli.md
nemotron/data-prep.md
nemotron/nemotron-cc.md
nemotron/xenna-observability.md
```

```{toctree}
:caption: Architecture
:hidden:

architecture/README.md
architecture/design-philosophy.md
architecture/cli-architecture.md
runspec/v1/spec.md
```
