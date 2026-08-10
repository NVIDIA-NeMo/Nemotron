# Nemotron-3.5 Lightning Notebooks

A collection of notebooks and guides for deploying, fine-tuning, and using
**NVIDIA Nemotron-3.5 Lightning**.

## Overview

Nemotron-3.5 Lightning is a 30B total / 3B active-parameter hybrid Mamba-Transformer MoE model.
It fits on a single node, so the cookbooks here run without a multi-node cluster.

## What's Inside

### Fine-Tuning

- **[RL](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Lightning/RL/README.md)** - DAPO/GRPO RL training with NeMo RL, with both a native math-environment recipe and a NeMo Gym variant.
- **[lora-text2sql/nemo-megatron-bridge](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Lightning/lora-text2sql/nemo-megatron-bridge/README.md)** - LoRA fine-tuning recipe for Text2SQL using NeMo Megatron-Bridge.

## Model Resources

- **Hugging Face:** [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
