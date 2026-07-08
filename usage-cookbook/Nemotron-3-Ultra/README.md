# Nemotron-3-Ultra Notebooks

A collection of notebooks and guides for deploying, fine-tuning, and using
**NVIDIA Nemotron-3-Ultra**.

## Overview

Nemotron-3-Ultra is a 550B total / 55B active-parameter hybrid
Mamba-Transformer MoE model for long-running agentic workflows across coding,
research, and enterprise tasks. The usage cookbooks cover hosted agent harness
configuration, multi-GPU deployment, LoRA fine-tuning, and RL post-training.

## What's Inside

### Deployment

- **[vllm_cookbook.ipynb](vllm_cookbook.ipynb)** - Deploy Nemotron-3-Ultra with vLLM.
- **[sglang_cookbook.ipynb](sglang_cookbook.ipynb)** - Deploy Nemotron-3-Ultra with SGLang.
- **[trtllm_cookbook.ipynb](trtllm_cookbook.ipynb)** - Deploy Nemotron-3-Ultra with TensorRT-LLM.
- **[SparkDeploymentGuide](SparkDeploymentGuide/README.md)** - Deploy Nemotron-3-Ultra across a 4x DGX Spark cluster with vLLM, then benchmark it with NVIDIA AIPerf.
- **[StationDeploymentGuide](StationDeploymentGuide/README.md)** - Deploy Nemotron-3-Ultra on a single GB300-based DGX Station with vLLM and selective expert offloading.

### Fine-Tuning

- **[RL](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Ultra/RL/README.md)** - Full-weight RL training with DAPO/GRPO, including direct NeMo RL and NeMo Gym variants.
- **[lora-text2sql/nemo-automodel](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Ultra/lora-text2sql/nemo-automodel/README.md)** - LoRA fine-tuning recipe for Text2SQL using NeMo AutoModel.
- **[lora-text2sql/nemo-megatron-bridge](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Ultra/lora-text2sql/nemo-megatron-bridge/README.md)** - LoRA fine-tuning recipe for Text2SQL using NeMo Megatron-Bridge.

### Agentic

- **[OpenScaffoldingResources](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Ultra/OpenScaffoldingResources/README.md)** - Config-based guides for using Nemotron-3-Ultra with agentic coding tools via OpenRouter and build.nvidia.com.

## Model Resources

- **build.nvidia.com:** [nvidia/nemotron-3-ultra-550b-a55b](https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b)
- **Hugging Face:** [nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16)
- **Technical report:** [NVIDIA Nemotron 3 Ultra Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf)
