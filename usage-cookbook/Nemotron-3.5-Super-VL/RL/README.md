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

The held-out split contains 256 examples. Slurm job `7555680` completed the
recipe with exit code 0 and produced this validation curve:

| Step | Correct | Accuracy |
| ---: | ---: | ---: |
| 0 | 152/256 | 59.38% |
| 5 | 148/256 | 57.81% |
| 10 | 168/256 | 65.62% |
| 15 | 209/256 | 81.64% |

The exact run is recorded in [W&B](https://wandb.ai/hwinf_dcm/nemotron-super-vl-35-circle-count/runs/awmpvucm).
An earlier run of the same 15-step training setup reached 96.88% at step 15,
so the short run has material rollout and optimization variance. Compare
step 15 with step 0 from the same run.
