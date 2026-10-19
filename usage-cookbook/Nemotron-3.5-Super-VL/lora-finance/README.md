# Nemotron-3.5 Super VL Finance Fine-tuning

This directory demonstrates customizing Nemotron-3.5 Super VL for financial question answering over
document images and text.

## Overview

- [nemo-megatron-bridge](nemo-megatron-bridge/README.md) — LoRA recipe using [NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)

## Requirements

- One node with 8x H200, or 8x H100 80GB with a different layout. Training also runs on four
  H200s, more slowly — see the recipe's README for the measured memory/time table.
- ~800GB disk space for the downloaded and converted checkpoints, the merged export, and the data.
