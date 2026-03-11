# Nemotron-3-Super Notebooks

A collection of notebooks demonstrating deployment and fine-tuning cookbooks for **NVIDIA Nemotron-3-Super**.

## Overview

These notebooks provide end-to-end recipes for deploying and customizing Nemotron-3-Super.

## What's Inside

### Deployment

- **[vllm_cookbook.ipynb](vllm_cookbook.ipynb)** — Deploy Nemotron-3-Super with vLLM.
- **[sglang_cookbook.ipynb](sglang_cookbook.ipynb)** — Deploy Nemotron-3-Super with SGLang.
- **[trtllm_cookbook.ipynb](trtllm_cookbook.ipynb)** — Deploy Nemotron-3-Super with TensorRT-LLM.
- **{doc}`AdvancedDeploymentGuide <AdvancedDeploymentGuide/README>`** — Production deployment configurations for vLLM, SGLang, and TRT-LLM across GPU topologies (GB200, B200, DGX Spark), including MTP speculative decoding, expert parallelism, and tuning guidance.

### Fine-Tuning

- **{doc}`grpo-dapo <grpo-dapo/README>`** — Full-weight RL training with GRPO/DAPO algorithm, reproducing emergent math reasoning from a base model.
- **{doc}`lora-text2sql <lora-text2sql/README>`** — Supervised fine-tuning (LoRA) recipe for the Text2SQL use case, including dataset preparation and training with [NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) and [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) libraries.

### Agentic

- **{doc}`OpenScaffoldingResources <OpenScaffoldingResources/README>`** — Guides for using Nemotron-3-Super with agentic coding tools (OpenCode, OpenClaw, Kilo Code CLI, OpenHands) via OpenRouter and build.nvidia.com.