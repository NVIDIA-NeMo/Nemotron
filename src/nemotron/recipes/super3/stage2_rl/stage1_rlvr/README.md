# Stage 1: Multi-Environment RLVR

Multi-environment Reinforcement Learning from Verifiable Rewards (RLVR) is the primary RL stage for Nemotron 3 Super. It trains on 21 environments and 37 datasets simultaneously, covering math, code, STEM, safety, chat, instruction following, long context, puzzles, and agentic tasks.

Training on all environments simultaneously yields stable gains — single-environment training leads to severe regressions on other benchmarks.

## Data

| Domain | Description |
|--------|-------------|
| **Math** | Competitive math problems, trained with and without python execution tool. Includes formal proof verification environment. |
| **Code** | Competition code data. |
| **STEM** | Scientific problems including newly curated difficult problems. |
| **Instruction Following** | Standard IF data plus a multi-challenge style dataset where the agent follows complex instructions scored against a rubric. |
| **Safety** | Two environments: (1) mitigating overrefusals on safety-related prompts, (2) robustness against jailbreak attacks using PAIR-generated adversarial prompts. |
| **Long Context** | Long context reasoning environment. |
| **Agentic Tool Use** | Conversational tool use and terminal use environments. |
| **Reasoning Gym** | Diverse reasoning tasks from [Reasoning Gym](https://github.com/reasoning-gym/reasoning-gym). |

Prompts where the SFT model consistently answers correctly are filtered out. Remaining samples are sorted via a difficulty-based curriculum.

### Low-Effort Reasoning

A subset of prompts (1-2%) are converted to low-effort mode. For each low-effort prompt, the reward accounts for both correctness and token count, encouraging efficient reasoning. The mix starts at 2% (math, STEM QA, competitive coding) and is later reduced to 1% (math and STEM QA only).

## Algorithm

Uses **asynchronous GRPO** where training and inference are decoupled across separate GPU devices:

- Inference workers continuously generate trajectories stored in a rollout buffer
- Batches are sent to the training engine once enough trajectories are collected
- Updated weights are pushed to inference workers as soon as a new model version is available
- **In-flight weight updates**: weights can be pushed mid-rollout without waiting for ongoing rollouts to finish
- KV cache is NOT recomputed after weight updates
- Policy lag is limited to at most one step behind the latest model version
- Importance sampling ratio masking stabilizes training under the training-inference mismatch

## Configuration

| Parameter | Value |
|-----------|-------|
| Nodes | 109 (872 GPUs) |
| Generation nodes | 72 (colocated=false) |
| NeMo Gym GPU nodes | 5 |
| Prompts/step | 256 |
| Generations/prompt | 16 |
| Batch size | 4,096 |
| Max sequence length | 65,536 |
| TP / CP | 4 / 8 |
| Learning rate | 3e-6 |
| KL penalty | 0 |
| Overlong filtering | false |

### Config files

- `config/default.yaml` — Full-scale 109-node config
- `config/small.yaml` — Reduced 21-node variant for testing

## Infrastructure

All RLVR experiments use an integrated NeMo RL + NeMo Gym infrastructure:

- **NeMo RL** acts as the RL training loop controller using Megatron-Core for model training
- **NeMo Gym** handles rollout environments using three server types: agents, models (vLLM), and resources (verifiers)
- **Ray** orchestrates resource management on SLURM — training workers, vLLM generation workers, Gym environments, and judge models all run on a single Ray cluster
- **Judge models**: Qwen3-235B-A22B for equivalence/instruction-following judging, Nemotron-Content-Safety-Reasoning-4B for safety, nvidia/Qwen3-Nemotron-235B-A22B-GenRM for GenRM comparison

### Resiliency at 1K GPU Scale

At 1K GPU scale, intermittent failures from hardware and software issues required:

- Parallelized initialization with prefetching of virtual environments and binaries
- Careful port management to avoid TOCTOU race conditions between Ray control plane, vLLM workers, TCP rendezvous, and NeMo Gym servers
- Caching in upstream repos (vLLM, flashinfer) to reduce startup time

## Prerequisites

- **NeMo-RL repo**: Clone the `super-v3` branch of [NeMo-RL](https://github.com/NVIDIA-NeMo/RL)
- **Sandbox container**: Required for code execution environments (NeMo-Skills tools, Lean4 proof verification). See [NeMo-Skills Dockerfile](https://github.com/NVIDIA-NeMo/Skills/blob/main/dockerfiles/Dockerfile.sandbox).

## Usage

> **Note**: RL sub-stages are not yet available as `nemotron` CLI commands because Megatron checkpoint consumption is not yet supported. Run directly inside the NeMo-RL repo using `super_launch.sh`.

The RLVR stage consists of 3 sub-stages with different data blends. All 3 use the same config (`stage1_rlvr.yaml`), only the data paths differ. Stage 1.1 starts from the SFT checkpoint; each subsequent stage takes the output of the previous one.

```bash
# Stage 1.1 — RLVR 1 (109 nodes)
EXP_NAME=stage1.1-rlvr1 \
CONFIG_PATH=examples/configs/super/stage1_rlvr.yaml \
MODEL_PATH=/path/to/sft_checkpoint \
TRAIN_PATH=$DATA_DIR/rlvr1/train-split.jsonl \
VAL_PATH=$DATA_DIR/rlvr1/val-split.jsonl \
CONTAINER=nvcr.io/nvidia/nemo-rl:v0.5.0.nemotron_3_super \
SANDBOX_CONTAINER=$SANDBOX_CONTAINER \
PERSISTENT_CACHE=$PERSISTENT_CACHE \
EXTRA_MOUNTS=$EXTRA_MOUNTS \
SLURM_PARTITION=$SLURM_PARTITION \
SLURM_ACCOUNT=$SLURM_ACCOUNT \
bash super_launch.sh

# Stage 1.2 — RLVR 2 (109 nodes)
EXP_NAME=stage1.2-rlvr2 \
CONFIG_PATH=examples/configs/super/stage1_rlvr.yaml \
MODEL_PATH=/path/to/rlvr1_checkpoint \
TRAIN_PATH=$DATA_DIR/rlvr2/train-split.jsonl \
VAL_PATH=$DATA_DIR/rlvr2/val-split.jsonl \
CONTAINER=nvcr.io/nvidia/nemo-rl:v0.5.0.nemotron_3_super \
SANDBOX_CONTAINER=$SANDBOX_CONTAINER \
PERSISTENT_CACHE=$PERSISTENT_CACHE \
EXTRA_MOUNTS=$EXTRA_MOUNTS \
SLURM_PARTITION=$SLURM_PARTITION \
SLURM_ACCOUNT=$SLURM_ACCOUNT \
bash super_launch.sh

# Stage 1.3 — RLVR 3 (109 nodes)
EXP_NAME=stage1.3-rlvr3 \
CONFIG_PATH=examples/configs/super/stage1_rlvr.yaml \
MODEL_PATH=/path/to/rlvr2_checkpoint \
TRAIN_PATH=$DATA_DIR/rlvr3/train-split.jsonl \
VAL_PATH=$DATA_DIR/rlvr3/val-split.jsonl \
CONTAINER=nvcr.io/nvidia/nemo-rl:v0.5.0.nemotron_3_super \
SANDBOX_CONTAINER=$SANDBOX_CONTAINER \
PERSISTENT_CACHE=$PERSISTENT_CACHE \
EXTRA_MOUNTS=$EXTRA_MOUNTS \
SLURM_PARTITION=$SLURM_PARTITION \
SLURM_ACCOUNT=$SLURM_ACCOUNT \
bash super_launch.sh
```

See the [upstream training guide](https://github.com/NVIDIA-NeMo/RL/blob/super-v3/examples/nemotron_3_super/README.md) for full details on environment variables (`DATA_DIR`, `SANDBOX_CONTAINER`, `PERSISTENT_CACHE`, `EXTRA_MOUNTS`).

## References

- [Nemotron 3 Super Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)
- [NeMo Gym](https://github.com/NVIDIA/NeMo-Gym)
- [NeMo RL](https://github.com/NVIDIA/NeMo-RL)
