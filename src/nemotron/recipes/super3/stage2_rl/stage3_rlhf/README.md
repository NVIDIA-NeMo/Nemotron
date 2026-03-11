# Stage 3: RLHF

Reinforcement Learning from Human Feedback (RLHF) is the final RL stage, run after RLVR and SWE-RL to improve model behavior and interaction quality.

## Approach

Uses a large **Generative Reward Model (GenRM)** to provide supervision during RL. The GenRM is a principle-following model trained as described in [RL-BFF](https://arxiv.org/abs/2505.18849), which allows guiding Nemotron Super 3's behavior on important domains like identity and safety-related topics.

### GenRM Model

- Initialized from **Qwen3-235B-A22B-Thinking-2507**
- Trained on [HelpSteer 3](https://huggingface.co/datasets/nvidia/HelpSteer3) preference data, commercially friendly subsets of lmarena-140k, and additional human preference data
- Used throughout multi-environment RLVR (Stage 1) AND as the sole reward signal in this RLHF stage
- Provides pairwise comparison judgments via the `genrm_compare` environment

### KL Penalty

Unlike the RLVR and SWE stages (which use KL=0), RLHF applies a **KL penalty of 1e-4** to prevent the model from drifting too far from the reference policy. This is critical for maintaining the capabilities learned in prior stages while improving interaction quality.

## Configuration

| Parameter | Value |
|-----------|-------|
| Nodes | 72 (576 GPUs) |
| Generation nodes | 32 (colocated=false) |
| NeMo Gym GPU nodes | 8 |
| Prompts/step | 128 |
| Generations/prompt | 16 |
| Batch size | 2,048 |
| Max sequence length | 49,152 |
| TP / CP | 4 / 4 |
| Learning rate | 1e-6 |
| KL penalty | 1e-4 |
| Overlong filtering | false |
| GenRM router DP size | 8 |

### Environments

The RLHF stage uses GenRM comparison as the primary reward signal, along with tool use evaluation:

- `genrm_compare` — Pairwise comparison using the GenRM model with principle-following prompts
- `single_step_tool_use_with_argument_comparison` — Tool use correctness

### Config files

- `config/default.yaml` — Full-scale 72-node config
- `config/small.yaml` — Reduced 24-node variant for testing

## Prerequisites

- **NeMo-RL repo**: Clone the `super-v3` branch of [NeMo-RL](https://github.com/NVIDIA-NeMo/RL)
- **Sandbox container**: Required for code execution environments

## Usage

> **Note**: RL sub-stages are not yet available as `nemotron` CLI commands because Megatron checkpoint consumption is not yet supported. Run directly inside the NeMo-RL repo using `super_launch.sh`.

```bash
EXP_NAME=stage3-rlhf \
CONFIG_PATH=examples/configs/super/stage3_rlhf.yaml \
MODEL_PATH=/path/to/swe2_checkpoint \
TRAIN_PATH=$DATA_DIR/rlhf/train-split.jsonl \
VAL_PATH=$DATA_DIR/rlhf/val-split.jsonl \
CONTAINER=nvcr.io/nvidia/nemo-rl:v0.5.0.nemotron_3_super \
SANDBOX_CONTAINER=$SANDBOX_CONTAINER \
PERSISTENT_CACHE=$PERSISTENT_CACHE \
EXTRA_MOUNTS=$EXTRA_MOUNTS \
SLURM_PARTITION=$SLURM_PARTITION \
SLURM_ACCOUNT=$SLURM_ACCOUNT \
bash super_launch.sh
```

See the [upstream training guide](https://github.com/NVIDIA-NeMo/RL/blob/super-v3/examples/nemotron_3_super/README.md) for full details on environment variables.

## References

- [Nemotron 3 Super Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)
- [RL-BFF: Reinforcement Learning with Best-of-F Feedback](https://arxiv.org/abs/2505.18849)
- [HelpSteer 3](https://huggingface.co/datasets/nvidia/HelpSteer3)
- [NeMo Gym](https://github.com/NVIDIA/NeMo-Gym)
- [NeMo RL](https://github.com/NVIDIA/NeMo-RL)
