# NVIDIA Nemotron Developer Repository

Developer companion repo for working with NVIDIA's Nemotron models: inference, fine-tuning, agents, visual reasoning, deployment.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## 📂 Repo Layout

```
nemotron/
│
├── usage-cookbook/        Usage cookbooks (how to deploy, and simple model usage guides)
│
│
└── use-case-examples/     Examples of leveraging Nemotron Models in Agentic Workflows and more 
```

---

## What is Nemotron?

[NVIDIA Nemotron™](https://developer.nvidia.com/nemotron) is a family of open, high-efficiency models with fully transparent training data, weights, and recipes.

Nemotron models are designed for **agentic AI workflows** — they excel at coding, math, scientific reasoning, tool calling, instruction following, and visual reasoning (for the VL models).

They are optimized for deployment across a spectrum of compute tiers (edge, single GPU, data center) and support frameworks like NeMo and TensorRT-LLM, vLLM, and SGLang, with NIM microservice options for scalable serving.

---

### More Resources

- **[Usage Cookbook](usage-cookbook/)** - Practical deployment and simple model usage guides for Nemotron models
- **[Use Case Examples](use-case-examples/)** - Practical use-case examples and apps
- **[Nemotron Developer Page](https://developer.nvidia.com/nemotron)** - Developer resources for the Nemotron family of models
- **[Nemotron Research Hub](https://research.nvidia.com/labs/nemotron/)** - Research affiliated with the Nemotron effort
- **[Nemotron Datasets](https://huggingface.co/nvidia/datasets?search=nemotron)** - Datasets part of various Nemotron collections, from pre-training to post-training

---

## 💡 Feature Requests & Ideas

Have an idea for improving Nemotron models? Visit the **[Nemotron Ideas Portal](https://nemotron.ideas.nvidia.com/)** to:

- 🗳️ **Vote** on existing feature requests
- 💭 **Submit** your own ideas and suggestions
- 📊 **See** what the community is requesting

Your feedback helps shape the future of Nemotron models!

---

## Training Recipes (Coming Soon)

Full, reproducible training pipelines will be included in the `nemotron` package at `src/nemotron/recipes/`.

### Each Recipe Includes
- 🗂️ **Data Curation** - Scripts to prepare training data using [NVIDIA-NeMo/Curator](https://github.com/NVIDIA-NeMo/Curator)
- 🔁 **Training** - Complete training loops with hyperparameters using:
  - [NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main) for Megatron models
  - [NVIDIA-NeMo/Automodel](https://github.com/NVIDIA-NeMo/Automodel) for HuggingFace models
  - [NVIDIA-NeMo/NeMo-RL](https://github.com/NVIDIA-NeMo/RL/tree/main) when RL is needed
- 📊 **Evaluation** - Benchmark evaluation on standard suites using [NVIDIA-NeMo/Evaluator](https://github.com/NVIDIA-NeMo/Evaluator)
- 📖 **Documentation** - Detailed explanations of each stage

---

## Model Specific Usage Cookbooks

Learn how to deploy and use the models through an API.

| Model | Best For | Key Features | Trade-offs | Resources |
|-------|----------|--------------|------------|-----------|
| [**NVIDIA-Nemotron-3-Nano**](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B) | High-throughput agentic workflows, reasoning, tool-use, chat | • 31.6B total / 3.6B active (MoE)<br>• Hybrid Mamba-Transformer MoE<br>• 1M-token context window<br>• Reasoning ON/OFF + thinking budget | Sparse MoE trades total params for efficiency | [📁 Cookbooks](./usage-cookbook/Nemotron-3-Nano/) |
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

We welcome contributions! Whether it's examples, recipes, or other tools you'd find useful.

Please read our **[Contributing Guidelines](CONTRIBUTING.md)** before submitting pull requests.

## Documentation

- **[Contributing Guidelines](CONTRIBUTING.md)** - How to contribute to this project
- **[Changelog](CHANGELOG.md)** - Version history and changes

---

## License

Apache 2.0 License - see [LICENSE](LICENSE) file for details.

---

**NVIDIA Nemotron** - Open, transparent, and reproducible.
