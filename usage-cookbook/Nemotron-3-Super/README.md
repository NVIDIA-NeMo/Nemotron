# Nemotron-3-Super Notebooks

A collection of notebooks demonstrating deployment and fine-tuning cookbooks for **NVIDIA Nemotron-3-Super**.

## Overview

These notebooks provide end-to-end recipes for deploying and customizing Nemotron-3-Super.

## What's Inside

- **[grpo-dapo](grpo-dapo/)** — Full-weight RL training with GRPO/DAPO algorithm, reproducing emergent math reasoning from a base model.
- **[lora-text2sql](lora-text2sql/)** — Supervised fine-tuning (LoRA) recipe for the Text2SQL use case, including dataset preparation and training with [NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) and [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) libraries.