# Nemotron-3-Super Text2SQL Fine-tuning

This directory demonstrates customizing Nemotron-3-Super for the Text2SQL use case.

## Overview

This directory contains two distinct LoRA fine-tuning tutorials for Text2SQL:

- {doc}`nemo-megatron-bridge <nemo-megatron-bridge/README>` — Recipe using [NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
- {doc}`nemo-automodel <nemo-automodel/README>` — Recipe using [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel)

## Requirements

- 8x H100 80GB (or equivalent, e.g., 8x A100 80GB, 8x H200, or newer)
- 600GB disk space for checkpoints
