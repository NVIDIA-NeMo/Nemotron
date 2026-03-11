# Multi-Environment RLVR (Stages 1.1–1.3)

Multi-environment Reinforcement Learning from Verifiable Rewards (RLVR) is the primary RL stage for Nemotron 3 Super. It trains on 21 environments and 37 datasets simultaneously, covering math, code, STEM, safety, chat, instruction following, long context, puzzles, and agentic tasks.

Training on all environments simultaneously yields stable gains — single-environment training leads to severe regressions on other benchmarks.

The RLVR stage consists of 3 sub-stages with different data blends. All 3 use the same config (`stage1_rlvr.yaml`), only the data paths differ.

---

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

**Data Curriculum:** Prompts where the SFT model consistently provides correct answers are filtered out. Remaining samples are sorted via a difficulty-based curriculum.

### Low-Effort Reasoning

A subset of prompts are converted to low-effort mode. For each low-effort prompt, the reward accounts for both correctness and token count, encouraging efficient reasoning.

| Phase | Scope | Proportion |
|-------|-------|------------|
| Early | Math, STEM QA, Competitive Coding | 2% of all RL prompts |
| Late | Math, STEM QA only | 1% of RL prompts |

---

## Algorithm

Uses **asynchronous GRPO** where training and inference are decoupled across separate GPU devices:

- Inference workers continuously generate trajectories stored in a rollout buffer
- Batches are sent to the training engine once enough trajectories are collected
- Updated weights are pushed to inference workers as soon as a new model version is available
- **In-flight weight updates**: weights can be pushed mid-rollout without waiting for ongoing rollouts to finish
- KV cache is NOT recomputed after weight updates
- Policy lag is limited to at most one step behind the latest model version
- Importance sampling ratio masking stabilizes training under the training-inference mismatch

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Nodes | 109 (872 GPUs) |
| Generation nodes | 72 (colocated=false) |
| NeMo Gym GPU nodes | 5 |
| Prompts/step | 256 |
| Generations/prompt | 16 |
| Batch size | 4,096 |
| Max sequence length | 49,152 → 65,536 (increased during training) |
| TP / CP | 4 / 8 |
| Learning rate | 3e-6 |
| KL penalty | 0 |
| Overlong filtering | false |

### Parallelism

| Parallelism | Value | Config Key |
|-------------|-------|------------|
| Tensor (TP) | 4 | `policy.megatron_cfg.tensor_model_parallel_size` |
| Pipeline (PP) | 1 | `policy.megatron_cfg.pipeline_model_parallel_size` |
| Context (CP) | 8 | `policy.megatron_cfg.context_parallel_size` |
| Expert (EP) | 8 | `policy.megatron_cfg.expert_model_parallel_size` |
| Sequence (SP) | Yes | `policy.megatron_cfg.sequence_parallel` |

### Config Files

- `stage1_rlvr/config/default.yaml` — Full-scale 109-node config
- `stage1_rlvr/config/small.yaml` — Reduced 21-node variant for testing

---

## Infrastructure

All RLVR experiments use an integrated NeMo RL + NeMo Gym infrastructure:

- **NeMo RL** acts as the RL training loop controller using Megatron-Core for model training
- **NeMo Gym** handles rollout environments using three server types: agents, models (vLLM), and resources (verifiers)
- **Ray** orchestrates resource management on SLURM — training workers, vLLM generation workers, Gym environments, and judge models all run on a single Ray cluster

### Judge Models

| Model | Purpose |
|-------|---------|
| Qwen3-235B-A22B | Equivalence/instruction-following judging |
| Nemotron-Content-Safety-Reasoning-4B | Safety evaluation |
| Qwen3-Nemotron-235B-A22B-GenRM | GenRM pairwise comparison |

### Resiliency at 1K GPU Scale

At 1K GPU scale, intermittent failures from hardware and software issues required:

- Parallelized initialization with prefetching of virtual environments and binaries
- Careful port management to avoid TOCTOU race conditions between Ray control plane, vLLM workers, TCP rendezvous, and NeMo Gym servers
- Caching in upstream repos (vLLM, flashinfer) to reduce startup time

---

## Usage

### Using nemotron CLI

```bash
# Stage 1.1–1.3: RLVR (uses base container)
uv run nemotron super3 rl rlvr -c rlvr1 --run YOUR-CLUSTER
uv run nemotron super3 rl rlvr -c rlvr2 --run YOUR-CLUSTER
uv run nemotron super3 rl rlvr -c rlvr3 --run YOUR-CLUSTER
```

> **`--run YOUR-CLUSTER`** refers to a profile defined in your `env.toml` file.
> See the [env.toml setup guide](../README.md#envtoml-setup) for details.

### Using super_launch.sh

#### Stage 1.1 — RLVR 1 (109 nodes)

```bash
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
```

#### Stage 1.2 — RLVR 2 (109 nodes)

```bash
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
```

#### Stage 1.3 — RLVR 3 (109 nodes)

```bash
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

See the [upstream training guide](https://github.com/NVIDIA-NeMo/RL/blob/super-v3/examples/nemotron_3_super/README.md) for full details on environment variables.

---

**Recipe Source**: `src/nemotron/recipes/super3/stage2_rl/stage1_rlvr/`
