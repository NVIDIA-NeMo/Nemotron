# Nemotron 3 Ultra DAPO Training Guide with NeMo Gym

## Overview

This guide describes a practical DAPO/GRPO reinforcement-learning workflow for
[nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16)
using NeMo RL for policy training and NeMo Gym for rollout and reward
orchestration.

The default profile is sized for a meaningful validation run on a frontier-scale
hybrid Mamba/MoE/Attention model:

- 16 GB200 nodes for Megatron policy training
- 2 GB200 nodes for non-colocated vLLM generation
- 4 GPUs per node
- one 16-node training segment
- Megatron TP=8, CP=8, EP=64
- vLLM TP=8
- 4k total sequence length
- 2k-token rollout budget
- 64 prompts x 16 generations per step
- 100 training steps
- checkpoints every 10 steps

NeMo Gym talks to vLLM through the OpenAI-compatible HTTP server exposed by the
generation worker, then runs `math_with_judge` verification over generated
responses.

## Files

This directory contains:

- `dapo_ultra_starter_nemo_gym.yaml`: standalone baseline training config
- `prepare_hf_dapo_data_for_nemo_gym.py`: HF/local data to NeMo Gym JSONL
  converter

The YAML does not use `defaults:` inheritance, so it can be copied and
launched by itself.

## Prerequisites

- **Compute**: 18 nodes with 4 GPUs per node. The guide uses 16 nodes for
  policy training and 2 nodes for vLLM generation.
- **Topology**: the 16 training nodes should be allocated as a single
  high-speed fabric segment where your cluster supports that control.
- **Container**: a NeMo RL container with Ultra-compatible NeMo RL, Megatron,
  vLLM, and NeMo Gym dependencies. Set `CONTAINER` to either a path to a local
  squashfs (`.sqsh`) file or a Docker registry reference that the cluster can
  pull.
- **Storage**: shared storage mounted into the container at `/shared`.
- **Credentials**: Hugging Face access to
  [nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16).
- Optional `.env` at `/shared/code/RL/.env` with `HF_TOKEN`, `WANDB_API_KEY`,
  or other site-local settings.

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
- `NEMO_RL` set to `${SHARED_ROOT}/code/RL`
- container mounts that include both `/lustre:/lustre` and
  `${SHARED_ROOT}:/shared`

The recipe expects the prepared model inside the container at:

```text
/shared/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-compat
```

## Step 2. Prepare NeMo Gym Data

The recipe mirrors the direct NeMo RL DAPO path and uses the public
DAPO-Math-17K dataset:

```text
BytedTsinghua-SIA/DAPO-Math-17k
```

Prepare this data from the login/head node before launching Slurm. The
conversion uses the Hugging Face `datasets` package locally, writes the
converted JSONL into the NeMo RL checkout, and then the training container reads
those files through the `/shared` mount.

DAPO-Math-17K is hosted as a Hugging Face dataset with columns such as `prompt`
and `reward_model.ground_truth`. The converter below reads it through
`datasets.load_dataset` and writes the NeMo Gym JSONL rows expected by the
`math_with_judge_simple_agent`.

Install the head-node Python dependencies if you did not already install them
from the parent README:

Run from the login/head node:

```bash
python -m pip install --upgrade --user "huggingface_hub[cli]" datasets
```

If `huggingface-cli` is not on `PATH` after the install, add
`${HOME}/.local/bin` to `PATH`.

Set the shared Hugging Face cache paths and dataset name:

Run from the login/head node:

```bash
cd "${NEMO_RL}"

if [ -f "${NEMO_RL}/.env" ]; then source "${NEMO_RL}/.env"; fi

export HF_HOME="${HF_HOME:-${SHARED_ROOT}/HF_HOME}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_DATASET_ID=BytedTsinghua-SIA/DAPO-Math-17k
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" examples/data
```

Populate the HF Hub cache on shared storage. This step is optional if the
dataset is already cached, but it makes the later conversion and Slurm runs
less dependent on live network access:

Run from the login/head node:

```bash
huggingface-cli download "${HF_DATASET_ID}" \
  --repo-type dataset \
  --cache-dir "${HF_HUB_CACHE}"
```

Inspect your source dataset before converting it:

Run from the login/head node:

```bash
cd "${NEMO_RL}"

python Nemotron/usage-cookbook/Nemotron-3-Ultra/RL/grpo-dapo-nemo-gym/prepare_hf_dapo_data_for_nemo_gym.py \
  --dataset "${HF_DATASET_ID}" \
  --split train \
  --cache-dir "${HF_DATASETS_CACHE}" \
  --inspect
```

Convert train and validation data to the file names used by the YAML:

Run from the login/head node:

```bash
python Nemotron/usage-cookbook/Nemotron-3-Ultra/RL/grpo-dapo-nemo-gym/prepare_hf_dapo_data_for_nemo_gym.py \
  --dataset "${HF_DATASET_ID}" \
  --split train \
  --cache-dir "${HF_DATASETS_CACHE}" \
  --output examples/data/dapo17k_nemo_gym_train6400.jsonl \
  --limit 6400

python Nemotron/usage-cookbook/Nemotron-3-Ultra/RL/grpo-dapo-nemo-gym/prepare_hf_dapo_data_for_nemo_gym.py \
  --dataset "${HF_DATASET_ID}" \
  --split train \
  --cache-dir "${HF_DATASETS_CACHE}" \
  --skip 6400 \
  --output examples/data/dapo17k_nemo_gym_validation256.jsonl \
  --limit 256
```

Quickly verify the converted files:

Run from the login/head node:

```bash
wc -l examples/data/dapo17k_nemo_gym_train6400.jsonl \
      examples/data/dapo17k_nemo_gym_validation256.jsonl

python - <<'PY'
import json
from pathlib import Path

for path, expected in [
    ("examples/data/dapo17k_nemo_gym_train6400.jsonl", 6400),
    ("examples/data/dapo17k_nemo_gym_validation256.jsonl", 256),
]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    assert len(rows) == expected, (path, len(rows), expected)
    assert rows[0]["agent_ref"]["name"] == "math_with_judge_simple_agent"
    assert "expected_answer" in rows[0]
    assert "responses_create_params" in rows[0]
print("DAPO-Math-17K NeMo Gym data is ready.")
PY
```

The converter accepts an HF dataset name, a `datasets.load_from_disk` directory,
or a local `.jsonl`/`.json` file. It auto-detects common fields such as
`prompt`, `question`, `problem`, `input`, and `reward_model.ground_truth`, and
passes through rows that already contain `responses_create_params` and
`agent_ref`. For DAPO-Math-17K, it maps `reward_model.ground_truth` to
`expected_answer` and routes each row to the NeMo Gym `math_with_judge` server.

Each output row is a Gym example:

```json
{
  "responses_create_params": {
    "input": [{"role": "user", "content": "..."}]
  },
  "question": "...",
  "expected_answer": "34",
  "agent_ref": {
    "type": "responses_api_agents",
    "name": "math_with_judge_simple_agent"
  },
  "dataset": "BytedTsinghua-SIA/DAPO-Math-17k"
}
```

## Step 3. Install the Config

This directory contains one standalone YAML file:

- `dapo_ultra_starter_nemo_gym.yaml`: 16 training + 2 generation node baseline

Copy it into your NeMo RL checkout:

Run from the login/head node:

```bash
cd "${NEMO_RL}"

cp Nemotron/usage-cookbook/Nemotron-3-Ultra/RL/grpo-dapo-nemo-gym/dapo_ultra_starter_nemo_gym.yaml \
   examples/configs/dapo_ultra_starter_nemo_gym.yaml
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

## Step 4. Launch

Set the site-specific values:

Run from the login/head node:

```bash
export CONTAINER=<PATH_TO_NEMO_RL_CONTAINER>
export SLURM_ACCOUNT=<SLURM_ACCOUNT>
export PARTITION=<SLURM_PARTITION>
export RUN_NAME=dapo-ultra-starter-nemo-gym
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
export MOUNTS="/lustre:/lustre,${SHARED_ROOT}:/shared,${NEMO_RL}/examples:/opt/nemo-rl/examples,${NEMO_RL}/nemo_rl:/opt/nemo-rl/nemo_rl"
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
export RUN_NAME=dapo-ultra-starter-nemo-gym
export CHECKPOINT_DIR=/shared/checkpoints/${RUN_NAME}

uv run ./examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/configs/dapo_ultra_starter_nemo_gym.yaml \
  checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
  env.nemo_gym.nemo_gym_log_dir=/shared/checkpoints/${RUN_NAME}/nemo_gym_logs \
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
# Login/head node: move to the NeMo RL checkout and configure ray.sub.
cd "${NEMO_RL}"

export GPUS_PER_NODE=4
export MOUNTS="/lustre:/lustre,${SHARED_ROOT}:/shared,${NEMO_RL}/examples:/opt/nemo-rl/examples,${NEMO_RL}/nemo_rl:/opt/nemo-rl/nemo_rl"

# Slurm allocation/container: Ray executes this payload after allocation starts.
export COMMAND="cd /opt/nemo-rl && \
if [ -f /shared/code/RL/.env ]; then source /shared/code/RL/.env; fi && \
source /opt/nemo-rl/3rdparty/vllm/nemo-rl.env && \
HF_HOME=/shared/HF_HOME \
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
NRL_VLLM_USE_V1=1 \
NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800 \
uv run ./examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/configs/dapo_ultra_starter_nemo_gym.yaml \
  checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
  env.nemo_gym.nemo_gym_log_dir=/shared/checkpoints/${RUN_NAME}/nemo_gym_logs \
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
flag to the `sbatch` commands, for example `--segment=18`.

For exploratory debugging, reduce `grpo.max_num_steps` to `10`; for production
experiments, increase steps and validation cadence according to your training
plan.

## Step 5. Resume

For weights-only resume, point `CHECKPOINT_DIR` at the existing checkpoint
directory, set `grpo.max_num_steps` above the saved step, and add this override
to the training command. Add it from the login/head node when constructing the
`COMMAND` payload; `run_grpo_nemo_gym.py` consumes it inside the Slurm
allocation:

```bash
+policy.megatron_cfg.checkpoint.finetune=true
```

Despite the name, `finetune=true` is a Megatron checkpoint-loading mode. It
loads policy weights and initializes fresh optimizer/RNG state; the outer
algorithm remains DAPO/GRPO.

## Step 6. Monitor Training

Run from the login/head node:

```bash
squeue -j <jobid> -o '%i %T %M %l %D %R %j'
tail -f slurm-<jobid>.out
```

A successful validation run should show:

- NeMo Gym servers started
- vLLM serving `/shared/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-compat`
- Megatron policy workers initialized
- weight handoff/refit completed
- rollout, logprob, and policy training completed
- `step_10/training_info.json` written under `CHECKPOINT_DIR` for the
  interactive validation run

## Troubleshooting

- If model loading fails, rerun the compatibility overlay step from the parent
  [Nemotron 3 Ultra RL README](../README.md) and ensure
  `policy.model_name`, `policy.tokenizer.name`, and
  `policy.generation.vllm_cfg.http_server_serving_chat_kwargs.reasoning_parser_plugin`
  all point at `/shared/models/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-compat`.
- Keep `grpo.max_val_samples: null` for Gym recipes unless your local
  `run_grpo_nemo_gym.py` supports truncating Gym validation data.
- Final process teardown can emit Ray, TCPStore, NCCL, or SIGTERM messages after
  the max-step stop. Verify success with `training_info.json` and training
  metrics.
- Release Slurm allocations promptly when you are done:

Run from the login/head node:

```bash
scancel <jobid>
```
