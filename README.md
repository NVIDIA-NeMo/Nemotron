# NVIDIA Nemotron Developer Repository

**Open and efficient models for agentic AI** — training recipes, deployment guides, and use-case examples for the Nemotron family.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Docs](https://img.shields.io/badge/docs-dev-76B900.svg)](https://nvidia-nemo.github.io/Nemotron/dev/)

<div align="center">

[![Watch the Nemotron Overview](https://img.youtube.com/vi/_y9SEtn1lU8/hqdefault.jpg)](https://www.youtube.com/watch?v=_y9SEtn1lU8)

**[Watch: Nemotron Overview](https://www.youtube.com/watch?v=_y9SEtn1lU8)**

</div>

---

## Why Nemotron?

| | |
|---|---|
| **Open Models** | Fully transparent training data, techniques, and weights for community innovation |
| **Compute Efficiency** | Model pruning and optimization enabling higher throughput via TensorRT-LLM |
| **High Accuracy** | Built on frontier open models with human-aligned reasoning for agentic workflows |
| **Flexible Deployment** | Deploy anywhere — edge, single GPU, or data center with NIM microservices |

---

## Repository Overview

```
nemotron/
│
├── src/nemotron/recipes/    Training recipes (complete, reproducible pipelines)
│
├── usage-cookbook/          Usage cookbooks (deployment and model usage guides)
│
└── use-case-examples/       Examples of leveraging Nemotron in agentic workflows
```

---

## What is Nemotron?

[NVIDIA Nemotron](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/) is a family of open, high-efficiency multimodal models purpose-built for agentic AI.

**Model Tiers:**

- **Nano** — Optimized for edge and PC deployments
- **Super** — Single GPU deployment with highest throughput
- **Ultra** — Multi-GPU datacenter applications

Nemotron models excel at coding, math, scientific reasoning, tool calling, instruction following, and visual reasoning. Deploy across edge, single GPU, or data center environments with support for NeMo, TensorRT-LLM, vLLM, SGLang, and NIM microservices.

---

## Training Recipes

The Nemotron Training Cookbook provides complete, reproducible training pipelines that show the full journey from raw data to deployment-ready models. These implementations reflect how large language models are trained at leading AI labs—through rigorous, scientific processes with careful experimentation, validation gates, and systematic optimization.

### Why Complete Training Pipelines

Training a production model involves interconnected components where isolated examples miss critical interactions between stages. Complete pipelines show:

- **How data quality affects downstream performance** across pretraining, SFT, and RL stages
- **Which training techniques work together** in practice, not just theory
- **Where validation gates prevent failures** and ensure reproducibility
- **How to balance competing objectives** across training stages

Because these are complete systems, practitioners can extract specific techniques with confidence—each component has been proven to work in a production context.

### Available Recipes

| Model | Description | Stages | Guide |
|-------|-------------|--------|-------|
| **[Nemotron 3 Nano](docs/train/nano3/README.md)** | 3.6B active / 31.6B total MoE Hybrid Mamba-Transformer for agentic reasoning | Pretrain → SFT → RL | [Training Guide](docs/train/nano3/README.md) |

### Nemotron 3 Nano

A complete training recipe for the open, efficient Mixture-of-Experts hybrid Mamba-Transformer model optimized for agentic reasoning.

> **Open-Source Data Only**: These recipes train exclusively on the open-sourced subset of training data. Results will differ from the tech report benchmarks, which used additional proprietary data. Use these recipes as reference implementations to apply the methodology with your own data.

**Model Specifications**:
- 31.6B total parameters, 3.6B active per forward pass
- 25 trillion pretraining tokens with curriculum learning
- Up to 1M context length
- 3.3x higher inference throughput than similarly sized models

**What You Can Extract**:
- Curriculum-based pretraining with two-phase data mixture
- Long-context extension via CPT methodology
- Multi-domain SFT with 12+ data sources
- InfinityByte cross-domain code synthesis
- Tool-calling fine-tuning and budget-controlled reasoning
- Multi-environment RLVR with GRPO
- GenRM reward modeling with circular comparison
- DPO for tool hallucination reduction

**Resources**:
- [Training Guide](docs/train/nano3/README.md)
- [Tech Report](https://arxiv.org/abs/2506.XXXXX)
- [Model Weights (Base)](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16)
- [Model Weights (Instruct)](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- [Model Weights (FP8)](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8)

---

## Usage Cookbooks

Practical deployment and model usage guides for Nemotron models.

| Model | Best For | Key Features | Resources |
|-------|----------|--------------|-----------|
| [**Llama-3.3-Nemotron-Super-49B-v1.5**](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5) | Production deployments needing strong reasoning | 128K context, single H200 GPU, RAG & tool calling | [Cookbooks](./usage-cookbook/Llama-Nemotron-Super-49B-v1.5/) |
| [**NVIDIA-Nemotron-Nano-9B-v2**](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) | Resource-constrained environments | 9B params, hybrid Mamba-2, controllable reasoning | [Cookbooks](./usage-cookbook/Nemotron-Nano-9B-v2/) |
| [**NVIDIA-Nemotron-Nano-12B-v2-VL**](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL) | Document intelligence and video understanding | 12B VLM, video reasoning, Efficient Video Sampling | [Cookbooks](./usage-cookbook/Nemotron-Nano2-VL/) |
| [**Llama-3.1-Nemotron-Safety-Guard-8B-v3**](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3) | Multilingual content moderation | 9 languages, 23 safety categories | [Cookbooks](./usage-cookbook/Llama-3.1-Nemotron-Safety-Guard-V3/) |
| **Nemotron-Parse** | Document parsing for RAG and AI agents | Table extraction, semantic segmentation | [Cookbooks](./usage-cookbook/Nemotron-Parse-v1.1/) |

---

## Use Case Examples

End-to-end examples demonstrating practical applications in the [`use-case-examples/`](./use-case-examples/) directory:

- **Agentic Workflows** — Multi-step AI agents with planning, context management, and external tools
- **RAG Systems** — Pipelines combining retrieval with Nemotron models for grounded outputs
- **Tool Integration** — Structured tool calling, function execution, and data enrichment
- **Production Patterns** — Scalability, monitoring, and deployment architectures

### Each Recipe Includes
- 🎨 **Synthetic Data Generation** - Scripts to generate synthetic datasets using [NVIDIA-NeMo/DataDesigner](https://github.com/NVIDIA-NeMo/DataDesigner)
- 🗂️ **Data Curation** - Scripts to prepare training data using [NVIDIA NeMo Curator](https://github.com/NVIDIA/NeMo-Curator) for scalable data processing, filtering, and quality enhancement
- 🔁 **Training** - Complete training loops with hyperparameters using:
  - [NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main) for Megatron models
  - [NVIDIA-NeMo/Automodel](https://github.com/NVIDIA-NeMo/Automodel) for HuggingFace models
  - [NVIDIA-NeMo/NeMo-RL](https://github.com/NVIDIA-NeMo/RL/tree/main) when RL is needed
  - Includes GPU-accelerated last-mile data processing (tokenization + optional sequence packing) for optimal training efficiency
- 📊 **Evaluation** - Benchmark evaluation on standard suites using [NVIDIA NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator)
- 📖 **Documentation** - Detailed explanations of each stage

---

## Feature Requests

Have an idea for improving Nemotron models? Visit the **[Nemotron Ideas Portal](https://nemotron.ideas.nvidia.com/)** to vote on existing requests or submit your own.

---

## Documentation

- [Nemotron 3 Nano Training Guide](docs/train/nano3/README.md) — Complete training recipe
- [NeMo-Run Configuration](docs/train/nemo-run.md) — Execution profiles and job orchestration
- [Data Preparation](docs/train/data-prep.md) — Data preparation module documentation
- [Contributing Guidelines](CONTRIBUTING.md) — How to contribute
- [Changelog](CHANGELOG.md) — Version history

---

## Contributing

We welcome contributions—examples, recipes, or other tools. Please read the [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

---

## License

Apache 2.0 License — see [LICENSE](LICENSE) for details.

---

**NVIDIA Nemotron** — Open and efficient models for agentic AI.
