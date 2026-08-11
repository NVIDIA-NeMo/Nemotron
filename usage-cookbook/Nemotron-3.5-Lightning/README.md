# Nemotron-3.5 Lightning Notebooks

A collection of notebooks and guides for deploying, fine-tuning, and using
**NVIDIA Nemotron-3.5 Lightning**.

## Overview

Nemotron-3.5 Lightning is a 30B total / 3B active-parameter hybrid
Mamba-Transformer MoE model for fast reasoning, coding, and agentic workflows.
It is sized for single-node deployment, while supporting long-context use and
structured tool calling.

## What's Inside

### Deployment

- **[vllm_cookbook.ipynb](vllm_cookbook.ipynb)** - Deploy Nemotron-3.5 Lightning with vLLM.
- **[sglang_cookbook.ipynb](sglang_cookbook.ipynb)** - Deploy Nemotron-3.5 Lightning with SGLang.
- **[trtllm_cookbook.ipynb](trtllm_cookbook.ipynb)** - Deploy Nemotron-3.5 Lightning with TensorRT-LLM.

### Fine-Tuning

- **[RL](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Lightning/RL/README.md)** - DAPO/GRPO RL training with NeMo RL, including native math-environment and NeMo Gym variants.
- **[lora-text2sql/nemo-megatron-bridge](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Lightning/lora-text2sql/nemo-megatron-bridge/README.md)** - LoRA fine-tuning recipe for Text2SQL using NeMo Megatron-Bridge.

### Agentic

- **[OpenScaffoldingResources](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Lightning/OpenScaffoldingResources/README.md)** - Config-based guides for using Nemotron-3.5 Lightning with agentic coding tools through the NVIDIA-hosted API or a local vLLM deployment.

## Model Resources

- **build.nvidia.com:** [nvidia/nemotron-3.5-lightning-30b-a3b](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b)
- **Hugging Face (BF16):** [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
- **Hugging Face (NVFP4):** [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
