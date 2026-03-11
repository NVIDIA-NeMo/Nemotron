# Stage 2.1: SWE-RL (SWE Pivot)

End-to-end RL for software engineering tasks using the SWE-pivot environment. This stage is run separately from multi-environment RLVR because SWE rollouts take substantially longer and require much longer context lengths, making them a throughput bottleneck when trained alongside other environments.

## Why a Separate Stage

SWE-RL has very different systems characteristics from the rest of the RL environments:

- Rollouts are substantially longer (agent loops with hundreds of turns)
- Context lengths are much larger (131K tokens vs 65K for RLVR)
- Each rollout requires isolated container execution for test evaluation
- Throughput bottleneck when mixed with shorter-horizon environments

Settings are tuned specifically for long-horizon, long-context trajectories.

## Algorithm

Uses the same **asynchronous GRPO** setup as Stage 1 RLVR with key differences:

- **Overlong filtering** enabled — critical for SWE where some trajectories exceed the sequence length
- Lower learning rate (1e-6 vs 3e-6)
- Fewer prompts per step (64 vs 256) due to longer rollout times
- Higher TP (8 vs 4) to handle 131K context

## Configuration

| Parameter | Value |
|-----------|-------|
| Nodes | 64 (512 GPUs) |
| Generation nodes | 32 (colocated=false) |
| Prompts/step | 64 |
| Generations/prompt | 16 |
| Batch size | 1,024 |
| Max sequence length | 131,072 |
| TP / CP | 8 / 8 |
| Learning rate | 1e-6 |
| KL penalty | 0 |
| Overlong filtering | true |
| Prefix caching | enabled |

### Config files

- `config/default.yaml` — Full-scale 64-node config
- `config/small.yaml` — Reduced 8-node variant for testing

## Environment

The SWE-pivot environment in NeMo Gym evaluates the model on software engineering tasks using a single-step tool use comparison approach. The model receives a code problem and must produce a solution that is evaluated against ground truth.

This is distinct from Stage 2.2 (SWE-bench) which uses full Apptainer-based agent loops.

## Prerequisites

- **NeMo-RL repo**: Clone the `super-v3` branch of [NeMo-RL](https://github.com/NVIDIA-NeMo/RL)
- **Sandbox container**: Required for code execution environments

## Usage

> **Note**: RL sub-stages are not yet available as `nemotron` CLI commands because Megatron checkpoint consumption is not yet supported. Run directly inside the NeMo-RL repo using `super_launch.sh`.

```bash
EXP_NAME=stage2.1-swe1 \
CONFIG_PATH=examples/configs/super/stage2_swe1.yaml \
MODEL_PATH=/path/to/rlvr3_checkpoint \
TRAIN_PATH=$DATA_DIR/swe1/train-split.jsonl \
VAL_PATH=$DATA_DIR/swe1/val-split.jsonl \
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
- [NeMo Gym](https://github.com/NVIDIA/NeMo-Gym)
- [NeMo RL](https://github.com/NVIDIA/NeMo-RL)
