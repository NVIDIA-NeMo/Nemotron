# Nemotron-3.5 Lightning Text2SQL Fine-tuning

This directory demonstrates customizing Nemotron-3.5 Lightning for the Text2SQL use case.

## Overview

- [nemo-megatron-bridge](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Lightning/lora-text2sql/nemo-megatron-bridge/README.md) — LoRA recipe using [NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)

## Requirements

- At least 1x H100 80GB (or equivalent). More GPUs train proportionally faster — see the recipe's
  README for the measured memory/time table.
- ~130GB disk space for the downloaded and converted checkpoints, plus room for the merged export.
