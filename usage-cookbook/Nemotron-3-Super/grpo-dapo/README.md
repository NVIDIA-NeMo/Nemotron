# Nemotron-3-Super GRPO/DAPO Training

This directory contains Reinforcement Learning (RL) training assets for the Nemotron-3-Super-120B-A20B model.

## Overview

- Full weight RL training with GRPO/DAPO algorithm using [NeMo RL](https://github.com/NVIDIA/NeMo-RL) from a base model: [grpo_training_cookbook.ipynb](grpo_training_cookbook.ipynb)

  This experiment reproduces the so-called Deepseek "aha" [moment](https://www.reddit.com/r/OpenAI/comments/1i6jsr2/deepseek_discovered_their_new_model_having_an_aha).
  The GRPO/DAPO training process can help a model fresh out of pretraining discover advanced math reasoning entirely by itself.

## Requirements

- 5x GB200 nodes on the same GB200 rack (4xGPUs each, i.e. 20x GB200 189GB GPUs in total), or
- 3x B200 nodes (8xGPUs each, i.e. 24x B200 183GB GPUs in total)
