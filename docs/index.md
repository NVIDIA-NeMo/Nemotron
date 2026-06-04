<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Nemotron Training Recipes

**Open and efficient models for agentic AI.** Reproducible training pipelines with transparent data, techniques, and weights.

<div style="text-align: center; margin: 2rem 0;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/_y9SEtn1lU8" title="Nemotron Overview" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

## Quick Start

<div class="termy">

```console
// Install the Nemotron training recipes
$ git clone https://github.com/NVIDIA-NeMo/Nemotron
$ cd Nemotron && uv sync

// Run a tiny SFT job on your cluster
$ uv run nemotron steps run sft/automodel -c tiny --run YOUR-CLUSTER

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

## Sample Deployments and Applications

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Deployment Guides
:link: deployment-guides
:link-type: doc

Deployment guides for Nemotron models: TensorRT-LLM, vLLM, SGLang, NIM, and Hugging Face.
:::

:::{grid-item-card} Sample Applications
:link: application-examples
:link-type: doc

End-to-end applications: RAG agents, ML agents, and multi-agent systems.
:::

::::

## Customization Workflows with Nemotron Steps

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Translation
:link: translation/index
:link-type: doc

Translate JSONL or Parquet corpora with `translate/nemo_curator`, NeMo Curator
backends, and optional FAITH quality scoring.
:::

:::{grid-item-card} Build MCQ Benchmarks
:link: build-benchmarks/index
:link-type: doc

Generate and translate custom multiple-choice benchmarks with `byob/mcq`.
:::

:::{grid-item-card} Data Curation
:link: curate/index
:link-type: doc

Filter JSONL text with `curate/nemo_curator` before translation or training data preparation.
:::

:::{grid-item-card} Synthetic Data Generation
:link: sdg/index
:link-type: doc

Use `sdg/data_designer` to produce SFT, tool-use, and preference datasets.
:::

:::{grid-item-card} Model Evaluation
:link: model-eval/index
:link-type: doc

Evaluate hosted endpoints or checkpoints with `eval/model_eval`.
:::

::::

## Training Recipes

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Nemotron 3 Nano
:link: nemotron/nano3/README
:link-type: doc

31.6B total / 3.6B active parameters, 25T tokens, up to 1M context. Hybrid Mamba-Transformer with sparse MoE.

**Stages:** Pretraining → SFT → RL
:::

:::{grid-item-card} Nemotron 3 Omni
:link: nemotron/omni3/README
:link-type: doc

GA-checkpoint multimodal post-training recipe with stage-local container builds and a three-step RL stack.

**Stages:** SFT → RL MPO → RL text → RL vision → Eval
:::

:::{grid-item-card} Embedding Fine-Tuning
:link: nemotron/embed/README
:link-type: doc

Fine-tune Llama-Nemotron-Embed-1B-v2 on domain-specific data with synthetic data generation, evaluation, and NIM deployment.

**Stages:** SDG → Data Prep → Finetune → Eval → Export → Deploy
:::

:::{grid-item-card} Reranking Fine-Tuning
:link: nemotron/rerank/README
:link-type: doc

Fine-tune Llama-Nemotron-Rerank-1B-v2 cross-encoders for domain-specific reranking with synthetic data generation, evaluation, export, and NIM deployment.

**Stages:** SDG → Data Prep → Finetune → Eval → Export → Deploy
:::

::::

## Recipe Layout

Nemotron keeps **data-producing recipes** separate from **model-family training recipes**:

| Path | Purpose | Example |
|------|---------|---------|
| `src/nemotron/recipes/data/curation/` | Filter, dedup, and curate existing corpora | [Nemotron-CC](nemotron/data/curation/nemotron-cc.md) |
| `src/nemotron/recipes/data/sdg/` | Generate synthetic datasets that can feed multiple families | [Long-document SDG](nemotron/data/sdg/long-document.md) feeding [Omni3 SFT](nemotron/omni3/sft.md) |
| `src/nemotron/recipes/<family>/` | Family-specific training, RL, evaluation, and model lifecycle commands | [Nano3](nemotron/nano3/README.md), [Omni3](nemotron/omni3/README.md) |

## Training Pipeline

Each recipe family has its own stage layout, and all of them can be tracked through [artifact lineage](nemotron/artifacts.md):

| Family | Stage layout |
|--------|--------------|
| [Nano3](nemotron/nano3/README.md) | Pretraining → SFT → RL |
| [Omni3](nemotron/omni3/README.md) | SFT → RL MPO → RL text → RL vision → Eval |
| [Super3](nemotron/super3/README.md) | Pretraining → SFT → RL → Quantization → Eval |
| [Ultra3](nemotron/ultra3/README.md) | Pretraining → SFT |
| [Embed](nemotron/embed/README.md) | SDG → Data Prep → Finetune → Eval → Export → Deploy |
| [Rerank](nemotron/rerank/README.md) | SDG → Data Prep → Finetune → Eval → Export → Deploy |

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
- [Model training steps](train-models/index.md) – SFT, PEFT, RL, and optimization with `nemotron step run`

```{toctree}
:caption: Nemotron
:hidden:

Home <self>
application-examples.md
deployment-guides.md
```

```{toctree}
:caption: Nemotron Step Basics
:hidden:

About <steps/index.md>
Basics <steps/basics.md>
Getting Started <steps/getting-started.md>
Airgap Environment <steps/airgap.md>
```

```{toctree}
:caption: Data Curation
:hidden:

About <curate/index.md>
Getting Started <curate/getting-started.md>
Tasks <curate/how-to/index.md>
Reference <curate/reference/index.md>
```

```{toctree}
:caption: Synthetic Data Generation
:hidden:

About <sdg/index>
Getting Started <sdg/getting-started>
Tips for Using Agents <sdg/using-skills>
Planning <sdg/planning>
Tasks <sdg/how-to/index>
Reference <sdg/reference/index>
```

```{toctree}
:caption: Translation
:hidden:

About <translation/index.md>
Getting Started <translation/getting-started.md>
Tips for Using Agents <translation/using-skills.md>
translation/explanation/index.md
Tasks <translation/how-to/index.md>
Reference <translation/reference/index.md>
```

```{toctree}
:caption: Build MCQ Benchmarks
:hidden:

About <build-benchmarks/index.md>
Getting Started <build-benchmarks/getting-started.md>
Concepts <build-benchmarks/explanation/index.md>
Tasks <build-benchmarks/how-to/index.md>
Reference <build-benchmarks/reference/index.md>
```

```{toctree}
:caption: Model Training
:hidden:

About <train-models/index.md>
Getting Started <train-models/getting-started.md>
Tips for Using Agents <train-models/using-skill.md>
Concepts <train-models/explanation/index.md>
Tasks <train-models/how-to/index.md>
Reference <train-models/reference/index.md>
```

```{toctree}
:caption: Model Evaluation
:hidden:

About <model-eval/index.md>
Getting Started <model-eval/getting-started.md>
Tips for Using Agents <model-eval/using-skills.md>
Concepts <model-eval/explanation/index.md>
Tasks <model-eval/how-to/index.md>
Reference <model-eval/reference/index.md>
```

```{toctree}
:caption: Training Recipes
:hidden:

Nemotron 3 Nano <nemotron/nano3/README.md>
Nemotron 3 Omni <nemotron/omni3/README.md>
Nemotron 3 Super <nemotron/super3/README.md>
Nemotron 3 Ultra <nemotron/ultra3/README.md>
Llama Nemotron Embed <nemotron/embed/README.md>
Llama Nemotron Rerank <nemotron/rerank/README.md>
nemotron/artifacts.md
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
nemotron/xenna-observability.md
```

```{toctree}
:caption: Data Recipes
:hidden:

nemotron/data/curation/nemotron-cc.md
nemotron/data/sdg/long-document.md
```

```{toctree}
:caption: Architecture
:hidden:

architecture/README.md
architecture/design-philosophy.md
architecture/cli-architecture.md
runspec/v1/spec.md
```
