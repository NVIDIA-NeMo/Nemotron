# Nemotron Use Case Examples

This directory contains end-to-end examples demonstrating practical applications of NVIDIA Nemotron models.

---

## 📋 Overview

These examples showcase **complete implementations** of agentic workflows, RAG systems, and other advanced use cases. Unlike the [usage-cookbook](../usage-cookbook/README.md) which focuses on deployment and basic model usage, these examples demonstrate how to build production-ready applications.

---

## Usage Examples Table of Contents

| Example | Description |
|---------|-------------|
| [RAG Agent with Nemotron RAG Models](RAG Agent with Nemotron RAG Models/README.md) | End-to-end example of a Retrieval-Augmented Generation (RAG) agent workflow using Nemotron RAG models through Hugging Face and Nemotron 9B hosted through build.nvidia.com models |
| [Data Science ML Agent](Data Science ML Agent/README.md) | End-to-end example of a natural language-driven data science and machine learning agent powered by NVIDIA GPUs. The agent allows users to perform data exploration, model training, and hyperparameter optimization interactively using RAPIDS cuDF and cuML for GPU acceleration.|
| [LangGraph CLI Agent with Synthetic Data and GRPO](Build_a_Bash_Agent_with_Synthetic_Data/README.md) | Generate deterministic CLI training labels with NeMo Data Designer, train Nemotron 3 Nano with Gym-backed GRPO rewards, and run validated human-confirmed LangGraph CLI tool calls. |
| [Long-Context Chat Synthetic Data Generation](long-context-chat-sdg/README.md) | Generate multi-turn, retrieval-grounded tool-call SFT conversations with an external HTTP retriever or an explicitly labeled simulation mode. |
| [French Public-Service Assistant](french-public-service-assistant/README.md) | Companion code for the blog post [Fine-Tuning NVIDIA Nemotron to Know Local Rules, Not Just Local Languages](https://developer.nvidia.com/blog/fine-tuning-nvidia-nemotron-to-know-local-rules-not-just-local-languages/) on post-training Nemotron for French-language applications. It adapts grounded synthetic data with Nemotron-Personas-France and NeMo Data Designer, LoRA fine-tuning with NeMo AutoModel, and a base-vs-adapted comparison. |
| [Build Your Own Benchmark](build-your-own-benchmark/README.md) | End-to-end BYOB MCQ benchmark example using the `src/nemotron/steps/byob` step, packaged step configs, and the `nemotron steps run byob/mcq` CLI. |
| [Nemotron 3.5 Super VL RxnCaption LoRA](nemotron-3.5-super-vl-rxncaption/lora-1-node/README.md) | Single-node LoRA fine-tuning, adapter merge, and evaluation for structured chemical-reaction parsing from numbered diagrams. |
| [Nemotron 3.5 Super VL RxnCaption Full SFT](nemotron-3.5-super-vl-rxncaption/sft-4-node/README.md) | Four-node, 16-GPU full-SFT workflow with shared data preparation, checkpoint export, and benchmark evaluation utilities. |
