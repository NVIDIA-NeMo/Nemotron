# RLHF (Stage 3)

Reinforcement Learning from Human Feedback is the final RL stage, run after RLVR and SWE-RL to improve model behavior and interaction quality.

---

## Approach

Uses a large **Generative Reward Model (GenRM)** to provide supervision during RL. The GenRM is a principle-following model trained as described in [RL-BFF](https://arxiv.org/abs/2505.18849), which allows guiding Nemotron Super 3's behavior on important domains like identity and safety-related topics.

### GenRM Model

| Parameter | Value |
|-----------|-------|
| **Initialization** | Qwen3-235B-A22B-Thinking-2507 |
| **Training Data** | [HelpSteer 3](https://huggingface.co/datasets/nvidia/HelpSteer3) + commercially friendly subsets of lmarena-140k + recently collected human preference data |
| **Approach** | Principle-following GenRM for guiding behavior on identity and safety domains |

The GenRM is used throughout both the multi-environment RL stage (RLVR) AND as the sole reward signal in this RLHF-only stage.

### KL Penalty

Unlike the RLVR and SWE stages (which use KL=0), RLHF applies a **KL penalty of 1e-4** to prevent the model from drifting too far from the reference policy. This is critical for maintaining the capabilities learned in prior stages while improving interaction quality.

---

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

### Config Files

- `stage3_rlhf/config/default.yaml` — Full-scale 72-node config
- `stage3_rlhf/config/small.yaml` — Reduced 24-node variant for testing

---

## Usage

### Using nemotron CLI

```bash
uv run nemotron super3 rl rlhf --run YOUR-CLUSTER
```

> **`--run YOUR-CLUSTER`** refers to a profile defined in your `env.toml` file.
> See the [env.toml setup guide](../README.md#envtoml-setup) for details.

### Using super_launch.sh

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

---

## References

- [RL-BFF: Reinforcement Learning with Best-of-F Feedback](https://arxiv.org/abs/2505.18849)
- [HelpSteer 3](https://huggingface.co/datasets/nvidia/HelpSteer3)

**Recipe Source**: `src/nemotron/recipes/super3/stage2_rl/stage3_rlhf/`
