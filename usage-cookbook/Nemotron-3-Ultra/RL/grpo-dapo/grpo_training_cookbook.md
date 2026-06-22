# Nemotron 3 Ultra DAPO Training Guide with NeMo RL

## Overview

This guide describes a practical DAPO/GRPO reinforcement-learning workflow for
[nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16)
on a Slurm/Ray cluster. The recipe trains on DAPO-Math-17K with verifiable math
rewards, using Megatron for policy training and non-colocated vLLM workers for
rollout generation.

The default profile is sized for a meaningful validation run on a frontier-scale
hybrid Mamba/MoE/Attention model:

- 16 GB200 nodes for Megatron policy training
- 2 GB200 nodes for vLLM generation
- 4 GPUs per node
- one 16-node training segment
- 4k total sequence length
- 2k-token rollout budget
- 64 prompts x 16 generations per step
- 100 training steps
- checkpoints every 10 steps

Use the interactive path first to validate the environment and data plumbing,
then use the batch path for longer runs and experiment tracking.

## Prerequisites

- **Compute**: 18 nodes with 4 GPUs per node. The guide uses 16 nodes for
  policy training and 2 nodes for vLLM generation.
- **Topology**: the 16 training nodes should be allocated as a single high-speed
  fabric segment where your cluster supports that control.
- **Storage**: a shared filesystem mounted into the container at `/shared`.
- **Container**: a NeMo RL container that includes the Ultra-compatible NeMo RL,
  Megatron, vLLM, and model runtime dependencies. Set it with `CONTAINER=...`
  when launching. The value can be either a path to a local squashfs (`.sqsh`)
  file or a Docker registry reference that the cluster can pull.
- **Credentials**: a Hugging Face token with access to the model repository.
  Optionally provide W&B credentials through `${SHARED_ROOT}/code/RL/.env`.

In the commands below, "login/head node" means outside the Slurm allocation.
"Slurm allocation/container" means the command runs after `sbatch` starts the
Ray job inside the training container.

## Step 1. Complete the Root Prerequisites

Before running this recipe, complete the common setup in the parent
[Nemotron 3 Ultra RL README](../README.md).

You should have:

- the model downloaded from
  [nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16)
  onto shared storage
- the compatibility overlay created with the bundled
  `checkpoint_compatibility` runtime files
- `SHARED_ROOT` set to the canonical shared-storage path from `realpath`
- the NeMo RL checkout available under `${SHARED_ROOT}/code/RL`
- container mounts that include both `/lustre:/lustre` and
  `${SHARED_ROOT}:/shared`

Inside the container, the prepared model path is:

```text
/shared/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-compat
```

The provided YAML uses this `/shared/...` path by default.

## Step 2. Install the Training Config

This directory contains one standalone YAML file:

- `dapo_ultra_starter.yaml`: 16 training + 2 generation node baseline

Copy it into your NeMo RL checkout:

Run from the login/head node:

```bash
export NEMO_RL="${SHARED_ROOT}/code/RL"
export COOKBOOK_DIR="${NEMO_RL}/Nemotron/usage-cookbook/Nemotron-3-Ultra/RL/grpo-dapo"

cp "${COOKBOOK_DIR}/dapo_ultra_starter.yaml" \
   "${NEMO_RL}/examples/configs/dapo_ultra_starter.yaml"
```

The recipe uses:

| Component | Setting | Notes |
|---|---:|---|
| Total nodes | 18 | 16 training + 2 generation |
| GPUs per node | 4 | adjust if your cluster shape differs |
| Megatron TP | 8 | tensor parallel group spans 2 four-GPU nodes |
| Megatron CP | 8 | context parallelism for Ultra |
| Megatron EP | 64 | one expert-parallel group across training GPUs |
| vLLM TP | 8 | one generation instance across 2 nodes |
| Max sequence length | 4k | policy and vLLM context budget |
| Max new tokens | 2048 | rollout generation budget |
| Rollout batch | 64 prompts x 16 generations | 1024 samples per step |
| Training steps | 100 | substantial validation run |
| Checkpoint period | 10 | save every 10 steps |

The YAML is self-contained and intentionally does not use `defaults:`
inheritance.

## Step 3. Launch

Set the site-specific values:

Run from the login/head node:

```bash
cd "${NEMO_RL}"

source .env 2>/dev/null || true

export CONTAINER=<PATH_TO_NEMO_RL_CONTAINER>
export SLURM_ACCOUNT=<SLURM_ACCOUNT>
export PARTITION=<SLURM_PARTITION>
export RUN_NAME=dapo-ultra-starter
export CHECKPOINT_DIR=/shared/checkpoints/${RUN_NAME}
```

`PATH_TO_NEMO_RL_CONTAINER` can be either a local enroot squashfs path, or an accessible Docker registry image.

If your cluster has an interactive or debug partition, use it for the validation
run, then switch `PARTITION` to your normal training partition for batch runs.

### Recommended: Interactive Validation Run

Use this path first for quick iteration and debugging. It starts the Ray cluster
and keeps the allocation alive without immediately launching training.

Run from the login/head node:

```bash
# Login/head node: move to the NeMo RL checkout and configure ray.sub.
cd "${NEMO_RL}"

export GPUS_PER_NODE=4
export MOUNTS="/lustre:/lustre,${SHARED_ROOT}:/shared,${PWD}/examples/configs:/opt/nemo-rl/examples/configs,${PWD}/nemo_rl:/opt/nemo-rl/nemo_rl"
unset COMMAND

sbatch \
  --nodes=18 \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --job-name="interactive-${RUN_NAME}" \
  --time=02:00:00 \
  --gres=gpu:4 \
  --exclusive \
  --mem=0 \
  ray.sub
```

When Slurm prints the job ID, wait for Ray to create the attach script, then
attach to the Ray head node:

Run from the login/head node:

```bash
bash ./<jobid>-attach.sh
```

Run a short validation job from the attached Slurm allocation/container:

```bash
cd /opt/nemo-rl

if [ -f /shared/code/RL/.env ]; then source /shared/code/RL/.env; fi
source /opt/nemo-rl/3rdparty/vllm/nemo-rl.env

export HF_HOME=/shared/HF_HOME
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NRL_VLLM_USE_V1=1
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800
export RUN_NAME=dapo-ultra-starter
export CHECKPOINT_DIR=/shared/checkpoints/${RUN_NAME}

uv run ./examples/run_grpo.py \
  --config examples/configs/dapo_ultra_starter.yaml \
  checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
  logger.log_dir=/shared/checkpoints/${RUN_NAME}/logs \
  logger.wandb_enabled=False \
  cluster.num_nodes=18 \
  cluster.segment_size=16 \
  cluster.gpus_per_node=4 \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.num_nodes=2 \
  policy.generation.colocated.resources.gpus_per_node=4 \
  grpo.max_num_steps=10
```

When you are finished debugging, release the allocation with `scancel <jobid>`
from the login/head node.

### Batch Submission

Once the interactive validation run is stable, submit the same training payload
as a batch job with `ray.sub`.

Run from the login/head node. The value assigned to `COMMAND` is the payload
that Ray runs inside the Slurm allocation and training container:

```bash
# Login/head node: configure ray.sub and the container payload.
export GPUS_PER_NODE=4
export MOUNTS="/lustre:/lustre,${SHARED_ROOT}:/shared,${PWD}/examples/configs:/opt/nemo-rl/examples/configs,${PWD}/nemo_rl:/opt/nemo-rl/nemo_rl"

# Slurm allocation/container: Ray executes this payload after allocation starts.
export COMMAND="cd /opt/nemo-rl && \
if [ -f /shared/code/RL/.env ]; then source /shared/code/RL/.env; fi && \
source /opt/nemo-rl/3rdparty/vllm/nemo-rl.env && \
HF_HOME=/shared/HF_HOME \
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
NRL_VLLM_USE_V1=1 \
NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800 \
uv run ./examples/run_grpo.py \
  --config examples/configs/dapo_ultra_starter.yaml \
  checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
  logger.log_dir=/shared/checkpoints/${RUN_NAME}/logs \
  logger.wandb_enabled=False \
  cluster.num_nodes=18 \
  cluster.segment_size=16 \
  cluster.gpus_per_node=4 \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.num_nodes=2 \
  policy.generation.colocated.resources.gpus_per_node=4 \
  grpo.max_num_steps=100"

# Login/head node: submit the allocation and Ray job.
sbatch \
  --nodes=18 \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --job-name="${RUN_NAME}" \
  --time=02:00:00 \
  --gres=gpu:4 \
  --exclusive \
  --mem=0 \
  ray.sub
```

If your cluster supports rack-aware Slurm segments, add the appropriate segment
flag to the `sbatch` commands, for example `--segment=18`. For exploratory
debugging, reduce `grpo.max_num_steps` to `10`; for production experiments,
increase steps and validation cadence according to your training plan.

## Step 4. Monitor Training

Run from the login/head node:

```bash
squeue -j <jobid> -o '%i %T %M %l %D %R'
tail -f slurm-<jobid>.out
```

Run artifacts are written under:

```text
/shared/checkpoints/<run-name>/
|____logs
|____step_10
|____step_20
```

If W&B is enabled through `.env` or command-line overrides, the run URL is
printed near startup.

## Step 5. Resume Behavior

Ultra checkpoints are large, and restoring the full distributed optimizer state
can exceed memory on resume. There are two useful resume modes.

For weights-only resume, load policy weights and initialize optimizer/RNG state
fresh by adding these overrides to the training command. Add them from the
login/head node when constructing the `COMMAND` payload; `run_grpo.py` consumes
them inside the Slurm allocation:

```bash
CHECKPOINT_DIR="/shared/checkpoints/<previous-run>" \
grpo.max_num_steps=<next_step> \
+policy.megatron_cfg.checkpoint.finetune=true
```

Despite the name, `finetune=true` is a Megatron checkpoint-loading mode. In this
RL workflow it means "load model weights only"; it does not change the outer
algorithm from DAPO/GRPO to supervised fine-tuning.

If your NeMo RL build includes `checkpointing.save_optimizer`, you can also
produce lighter checkpoints:

Add this override to the same training command from the login/head node:

```bash
+checkpointing.save_optimizer=false
```

Checkpoints saved without optimizer state should resume with fresh optimizer
state.

## Tuning Knobs

- **Run size**: `grpo.max_num_steps`, `grpo.num_prompts_per_step`,
  `grpo.num_generations_per_prompt`
- **Rollout length**: `policy.generation.max_new_tokens`,
  `grpo.reward_shaping.max_response_length`
- **Generation throughput**: `policy.generation_batch_size`,
  `policy.generation.vllm_cfg.gpu_memory_utilization`
- **Topology**: `cluster.segment_size`,
  `policy.generation.colocated.resources.num_nodes`
- **Resume strategy**: `policy.megatron_cfg.checkpoint.finetune`,
  `checkpointing.save_optimizer`

## Troubleshooting

- If a checkpoint fails during model initialization, rerun the compatibility
  overlay step from the parent
  [Nemotron 3 Ultra RL README](../README.md) and ensure
  `policy.model_name` and `policy.tokenizer.name` both point at
  `/shared/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-compat`.
- The vLLM/Megatron weight handoff may print warnings for Ultra MTP layer-name
  mappings. Warnings alone are not fatal; verify whether rollout generation and
  training continue.
- Final process teardown can emit Ray, TCPStore, NCCL, or SIGTERM messages after
  "Max number of steps has been reached". Verify success with
  `/shared/checkpoints/<run-name>/step_*/training_info.json` and training
  metrics.
- Release Slurm allocations promptly when you are done:

Run from the login/head node:

```bash
scancel <jobid>
```
