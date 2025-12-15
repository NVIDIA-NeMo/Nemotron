# NVIDIA Nemotron Developer Repository

**Open and efficient models for agentic AI.** Training recipes, deployment guides, and use-case examples for the Nemotron family.

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
| **Flexible Deployment** | Deploy anywhere: edge, single GPU, or data center with NIM microservices |

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

- **[Usage Cookbook](usage-cookbook/)** - Practical deployment and simple model usage guides for Nemotron models
- **[Use Case Examples](use-case-examples/)** - Practical use-case examples and apps *(more coming soon)*

---

## 💡 Feature Requests & Ideas

Have an idea for improving Nemotron models? Create a [Discussion](https://github.com/NVIDIA-NeMo/Nemotron/discussions) topic for it!

If you have a feature request, feel free to open an [Issue](https://github.com/NVIDIA-NeMo/Nemotron/issues) and tag it as `enhancement`.

Your feedback helps shape the future of Nemotron models!

---

## Documentation

Full, reproducible training pipelines will be included in the `nemotron` package at `src/nemotron/recipes/`.

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

## Model Specific Usage Cookbooks

Learn how to deploy and use the models through an API.

| Model | Best For | Key Features | Trade-offs | Resources |
|-------|----------|--------------|------------|-----------|
| [**Llama-3.3-Nemotron-Super-49B-v1.5**](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5) | Production deployments needing strong reasoning with efficiency | • 128K context<br>• Single H200 GPU<br>• RAG & tool calling<br>• Optimized via NAS | Balances accuracy & throughput | [📁 Cookbooks](./usage-cookbook/Llama-Nemotron-Super-49B-v1.5/) |
| [**NVIDIA-Nemotron-Nano-9B-v2**](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) | Resource-constrained environments needing flexible reasoning | • 9B params<br>• Hybrid Mamba-2 architecture<br>• Controllable reasoning traces<br>• Unified reasoning/non-reasoning | Smaller model with configurable reasoning | [📁 Cookbooks](./usage-cookbook/Nemotron-Nano-9B-v2/) |
| [**NVIDIA-Nemotron-Nano-12B-v2-VL**](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL) | Document intelligence and video understanding | • 12B VLM<br>• Video & multi-image reasoning<br>• Controllable reasoning (/think mode)<br>• Efficient Video Sampling (EVS) | Vision-language with configurable reasoning | [📁 Cookbooks](./usage-cookbook/Nemotron-Nano2-VL/) |
| [**Llama-3.1-Nemotron-Safety-Guard-8B-v3**](https://huggingface.co/nvidia/Llama-3.1-Nemotron-Safety-Guard-8B-v3) | Multilingual content moderation with cultural nuance | • 9 languages<br>• 23 safety categories<br>• Cultural sensitivity<br>• NeMo Guardrails integration | Focused on safety/moderation tasks | [📁 Cookbooks](./usage-cookbook/Llama-3.1-Nemotron-Safety-Guard-V3/) |
| **Nemotron-Parse** (link coming soon!) | Document parsing for RAG and AI agents | • VLM for document parsing<br>• Table extraction (LaTeX)<br>• Semantic segmentation<br>• Spatial grounding (bbox) | Specialized for document structure | [📁 Cookbooks](./usage-cookbook/Nemotron-Parse-v1.1/) |



## Nemotron Use Case Examples

Below is an outline of the end-to-end use case examples provided in the [`use-case-examples`](./use-case-examples/) directory. These scenarios demonstrate practical applications that go beyond basic model inference.

### What You'll Find

- **Agentic Workflows**  
  Orchestration of multi-step AI agents, integrating planning, context management, and external tools/APIs.

- **Retrieval-Augmented Generation (RAG) Systems**  
  Building pipelines that combine retrieval components (vector databases, search APIs) with Nemotron models for grounded, accurate outputs.

- **Integration with External Tools & APIs**  
  Examples of Nemotron models powering applications with structured tool calling, function execution, or data enrichment.

- **Production-Ready Application Patterns**  
  Architectures supporting scalability, monitoring, data pipelines, and real-world deployment considerations.

> See the [`use-case-examples/`](./use-case-examples/) subfolders for in-depth, runnable examples illustrating these concepts.

## Contributing

We welcome contributions: examples, recipes, or other tools. Please read the [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

---

## Security

To report any vulnerabilities, please reach out to [security@nvidia.com](mailto:security@nvidia.com)

---

## License

Apache 2.0 License — see [LICENSE](LICENSE) for details.

---

**NVIDIA Nemotron** — Open and efficient models for agentic AI.
