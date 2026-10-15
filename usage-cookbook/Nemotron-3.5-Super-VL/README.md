# Nemotron-3.5 Super VL Notebooks

A collection of notebooks and guides for deploying, fine-tuning, and using
**NVIDIA Nemotron-3.5 Super VL**.

## Overview

Nemotron-3.5 Super VL is a 120B total / 12B active-parameter hybrid
Mamba-Transformer MoE vision-language model with a
[C-RADIO v4-H](https://huggingface.co/nvidia/C-RADIOv4-H) vision encoder. It
accepts text, images, and video, supports up to 1M tokens of context, and is
built for document understanding, visual reasoning, and multimodal agentic
workflows. The usage cookbooks cover single-GPU deployment, LoRA fine-tuning,
multimodal RL post-training, and a visual investigation agent.

## What's Inside

### Deployment

- **[vllm_cookbook.ipynb](vllm_cookbook.ipynb)** - Deploy Nemotron-3.5 Super VL with vLLM.
- **[sglang_cookbook.ipynb](sglang_cookbook.ipynb)** - Deploy Nemotron-3.5 Super VL with SGLang.
- **[trtllm_cookbook.ipynb](trtllm_cookbook.ipynb)** - Deploy Nemotron-3.5 Super VL with TensorRT-LLM.
- **[Dynamo Recipes](https://docs.nvidia.com/dynamo/dev/recipes/nemotron-3-5-super-vl)** - Deploy Nemotron-3.5 Super VL with NVIDIA Dynamo.

### Fine-Tuning

You may use the [Nemotron Post-training Calculator](../../docs/train-models/sizing-calculator.md) for directional estimates of GPU count, runtime, and compute cost for SFT, LoRA, or GRPO.

- **[RL](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Super-VL/RL/README.md)** - Full-weight multimodal GRPO training on synthetic star-count images with NeMo RL and NeMo Gym.
- **[lora-finance/nemo-megatron-bridge](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Super-VL/lora-finance/nemo-megatron-bridge/README.md)** - LoRA fine-tuning recipe for financial question answering over document images and text using NeMo Megatron-Bridge.

### Agentic

- **[visual-investigation-agent](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Super-VL/visual-investigation-agent/README.md)** - Build an agent that reads dashboards and scanned documents, works out why a metric moved, and proposes a fix for a person to approve.

## Model Resources

- **build.nvidia.com:** [nvidia/nemotron-3.5-super-vl-120b-a12b](https://build.nvidia.com/nvidia/nemotron-3.5-super-vl-120b-a12b)
- **Hugging Face (BF16):** [nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16)
- **Hugging Face (NVFP4):** [nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4)
