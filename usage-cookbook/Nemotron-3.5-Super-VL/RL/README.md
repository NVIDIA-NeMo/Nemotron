# Nemotron 3.5 Super VL RL Training Cookbook

This directory documents multimodal RL post-training for Nemotron 3.5 Super
VL with NeMo RL's Megatron backend and NeMo Gym.

The included workflow trains on synthetic circle-count images and measures
exact-match accuracy on a deterministic, disjoint validation split. It uses a
four-node, 16-GPU GB200 topology with colocated vLLM generation.

- [`grpo-circle-count-nemo-gym/`](grpo-circle-count-nemo-gym/grpo_training_cookbook_nemo_gym.md):
  15-step GRPO run with validation at steps 0, 5, 10, and 15.

## Runtime requirements

Use NeMo RL branch `super-v3.5-posttraining` at revision `eb420d15034c` or a
compatible later revision. The branch contains the Super VL Megatron loader,
vLLM integration, and the NeMo Gym `circle_count` environment used here.

The tested model checkpoint is
`NVIDIA-Nemotron-3.5-Super-EA-09112026`. The recipe performs full-weight
updates with Megatron tensor parallelism 4 and expert parallelism 16.

## Tested result

The held-out split contains 256 examples. A prior run with the same data,
model, topology, and optimizer settings measured 59.77% before RL and 96.88%
after step 15. Re-run the recipe to verify these values for a new runtime or
checkpoint copy.

