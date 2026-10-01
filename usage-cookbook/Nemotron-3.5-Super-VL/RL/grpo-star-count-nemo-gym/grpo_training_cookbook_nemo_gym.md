# GRPO with Nemotron 3.5 Super VL and NeMo Gym Star Count

A page of colored stars looks like a simple visual puzzle. Solving it reliably,
however, requires a multimodal model to find the relevant objects, distinguish
their colors, count them, and follow a precise answer format. This guide turns
that compact task into an end-to-end reinforcement learning example with a
clear, automatically verifiable reward.

The workflow uses NeMo RL's Megatron backend for full-weight GRPO, colocated
vLLM for generation, and NeMo Gym for task execution and verification. Begin
with the repository, container, checkpoint, and shared-storage setup in
[`../README.md`](../README.md), then use
[`super_vl_3_5_star_count_megatron.yaml`](super_vl_3_5_star_count_megatron.yaml)
for training.

## The learning task

Each example presents a square image containing colored stars and asks the
policy to count the stars of one specified color. The model returns its final
answer in `\boxed{}` format. NeMo Gym extracts the boxed integer and compares
it with the count in the example metadata:

- a correct count receives reward `1.0`;
- an incorrect, missing, or malformed count receives reward `0.0`.

There is no partial credit and no judge model. This sparse binary reward makes
the task easy to interpret: an increase in validation accuracy directly means
that the policy solves more held-out images.

The environment is a single-turn `circle_count_simple_agent` interaction with
no tools. The generator retains the verifier's `circles` metadata key for
compatibility, while the rendered images and prompts contain stars throughout.

## Dataset design

The deterministic generator creates two disjoint splits:

| Split | Examples | Seeds |
| --- | ---: | --- |
| Training | 1,024 | 0–1,023 |
| Validation | 256 | 1,000,000–1,000,255 |

For every example, it samples a square canvas from 800 x 800 through
1,200 x 1,200 pixels, draws 1–30 non-overlapping stars, and assigns colors
from a fixed eight-color palette. Every selected color appears at least once,
so each question has a positive answer. The PNG is embedded in its JSONL row
as a base64 data URL, which keeps every example self-contained across workers.

## Configuration overview

The reference recipe uses the following settings:

| Component | Setting |
| --- | --- |
| Algorithm | Synchronous GRPO |
| Backend | Megatron, full-weight BF16 |
| Resources | 4 nodes x 4 GPUs |
| Training parallelism | TP=4, EP=16 |
| Generation | Colocated vLLM, TP=4 |
| Rollout batch | 16 prompts x 8 generations |
| Policy global batch | 128 responses |
| Validation | 256 held-out examples, greedy decoding |
| Evaluation cadence | Before RL, then every 2 steps through step 10 |
| Maximum response length | 256 tokens |

The 16 x 8 rollout batch gives GRPO eight candidate responses for each prompt.
NeMo RL converts their binary rewards into group-relative advantages using
reward normalization and a leave-one-out baseline. Reward shaping and reward
scaling remain disabled.

The policy trains both the language and vision components. During each
colocated weight refit, the recipe temporarily moves distributed optimizer
state out of GPU memory so the full tensor-parallel weight gather has enough
headroom.

## Generate the train and validation data

Run the generator inside the NeMo RL container or from an attached allocation.
The commands use the `/shared` layout established in the parent README:

```bash
export NEMO_RL=/shared/code/RL
export NEMOTRON_REPO=/shared/code/Nemotron
export DATA_DIR=/shared/runs/super35-star-count/data
export GENERATOR="${NEMOTRON_REPO}/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-star-count-nemo-gym/prepare_star_count_data.py"
mkdir -p "${DATA_DIR}"

python "${GENERATOR}" \
  --out "${DATA_DIR}/train.jsonl" \
  --num-samples 1024 \
  --seed-offset 0 \
  --canvas-size-min 800 \
  --canvas-size-max 1200 \
  --radius-min 24 \
  --radius-max 48 \
  --num-stars-min 1 \
  --num-stars-max 30

python "${GENERATOR}" \
  --out "${DATA_DIR}/validation.jsonl" \
  --num-samples 256 \
  --seed-offset 1000000 \
  --canvas-size-min 800 \
  --canvas-size-max 1200 \
  --radius-min 24 \
  --radius-max 48 \
  --num-stars-min 1 \
  --num-stars-max 30
```

Validate the row counts, image dimensions, star bounds, agent routing, and
split separation before launching training:

```bash
python - <<'PY'
import base64
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

root = Path('/shared/runs/super35-star-count/data')
fingerprints = {}

for split, expected in [('train', 1024), ('validation', 256)]:
    rows = [
        json.loads(line)
        for line in (root / f'{split}.jsonl').read_text().splitlines()
    ]
    assert len(rows) == expected
    assert all(
        row['agent_ref']['name'] == 'circle_count_simple_agent'
        for row in rows
    )
    assert all(1 <= len(row['circles']) <= 30 for row in rows)

    for row in rows:
        content = row['responses_create_params']['input'][1]['content']
        image_url = content[0]['image_url']
        image = Image.open(
            io.BytesIO(base64.b64decode(image_url.split(',', 1)[1]))
        )
        assert image.width == image.height
        assert 800 <= image.width <= 1200
        assert 'stars' in content[1]['text']

    fingerprints[split] = {
        hashlib.sha256(
            json.dumps(row['responses_create_params'], sort_keys=True).encode()
        ).hexdigest()
        for row in rows
    }

assert fingerprints['train'].isdisjoint(fingerprints['validation'])
print('Star-count train and validation splits are valid and disjoint.')
PY
```

## Interactive run

Use an interactive allocation when bringing up the recipe for the first time,
inspecting logs, or trying configuration overrides. From the login node, set
the scheduler and container values for your cluster:

```bash
export NUM_NODES=4
export GPUS_PER_NODE=4
export SLURM_ACCOUNT=<SLURM_ACCOUNT>
export PARTITION=<SLURM_PARTITION>
export CONTAINER=<NEMO_RL_CONTAINER_OR_SQUASHFS>
export MOUNTS="${SHARED_ROOT}:${SHARED_ROOT},${SHARED_ROOT}:/shared"
unset COMMAND

cd "${NEMO_RL}"
sbatch \
  --nodes="${NUM_NODES}" \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --job-name=interactive-super-vl-star-count \
  --time=04:00:00 \
  --gres=gpu:"${GPUS_PER_NODE}" \
  --exclusive \
  ray.sub
```

After the allocation starts, attach with the helper created by `ray.sub`:

```bash
cd "${NEMO_RL}"
bash ./<jobid>-attach.sh
```

Inside the attached container, configure persistent caches and launch the
training driver:

```bash
export NEMO_RL=/shared/code/RL
export NEMOTRON_REPO=/shared/code/Nemotron
export MODEL_DIR=/shared/models/NVIDIA-Nemotron-3.5-Super-VL-09212026
export RUN_DIR=/shared/runs/super35-star-count
export RECIPE="${NEMOTRON_REPO}/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-star-count-nemo-gym/super_vl_3_5_star_count_megatron.yaml"

mkdir -p \
  "${RUN_DIR}/logs" \
  "${RUN_DIR}/cache/hf_modules" \
  "${RUN_DIR}/cache/hf_config_locks" \
  "${RUN_DIR}/cache/megatron_ckpt" \
  "${RUN_DIR}/cache/vllm"

export HF_MODULES_CACHE="${RUN_DIR}/cache/hf_modules"
export MEGATRON_CONFIG_LOCK_DIR="${RUN_DIR}/cache/hf_config_locks"
export NRL_MEGATRON_CHECKPOINT_DIR="${RUN_DIR}/cache/megatron_ckpt"
export VLLM_CACHE_ROOT="${RUN_DIR}/cache/vllm"
export MEGATRON_BRIDGE="${NEMO_RL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
export MEGATRON_LM="${MEGATRON_BRIDGE}/3rdparty/Megatron-LM"
export PYTHONPATH="${HF_MODULES_CACHE}:${NEMO_RL}:${MEGATRON_BRIDGE}/src:${MEGATRON_LM}:${PYTHONPATH:-}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NRL_WG_USE_RAY_REF=1
export NRL_VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NEMO_GYM_VENV_DIR=/opt/gym_venvs

cd "${NEMO_RL}"
python -u examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "${RECIPE}" \
  policy.model_name="${MODEL_DIR}" \
  policy.tokenizer.name="${MODEL_DIR}" \
  logger.log_dir="${RUN_DIR}/logs"
```

For a pipeline check without external experiment tracking, append
`logger.wandb_enabled=false`. To use W&B, provide `WANDB_API_KEY` through the
job environment or a protected environment file.

## Batch run

For an unattended experiment, create a driver script on shared storage and
pass its container path to `ray.sub` through `COMMAND`. Run the following setup
from the login node:

```bash
export RUN_NAME=super35-star-count-$(date +%Y%m%d-%H%M%S)
export HOST_RUN_DIR="${SHARED_ROOT}/runs/${RUN_NAME}"
export RUN_SCRIPT="${HOST_RUN_DIR}/run.sh"
mkdir -p "${HOST_RUN_DIR}"

cat > "${RUN_SCRIPT}" <<'RUN'
#!/usr/bin/env bash
set -euo pipefail

: "${RUN_NAME:?RUN_NAME must be exported before submission}"

export NEMO_RL=/shared/code/RL
export NEMOTRON_REPO=/shared/code/Nemotron
export MODEL_DIR=/shared/models/NVIDIA-Nemotron-3.5-Super-VL-09212026
export RUN_DIR="/shared/runs/${RUN_NAME}"
export CACHE_DIR=/shared/runs/super35-star-count/cache
export RECIPE="${NEMOTRON_REPO}/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-star-count-nemo-gym/super_vl_3_5_star_count_megatron.yaml"

mkdir -p \
  "${RUN_DIR}/logs" \
  "${CACHE_DIR}/hf_modules" \
  "${CACHE_DIR}/hf_config_locks" \
  "${CACHE_DIR}/megatron_ckpt" \
  "${CACHE_DIR}/vllm"

export HF_MODULES_CACHE="${CACHE_DIR}/hf_modules"
export MEGATRON_CONFIG_LOCK_DIR="${CACHE_DIR}/hf_config_locks"
export NRL_MEGATRON_CHECKPOINT_DIR="${CACHE_DIR}/megatron_ckpt"
export VLLM_CACHE_ROOT="${CACHE_DIR}/vllm"
export MEGATRON_BRIDGE="${NEMO_RL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
export MEGATRON_LM="${MEGATRON_BRIDGE}/3rdparty/Megatron-LM"
export PYTHONPATH="${HF_MODULES_CACHE}:${NEMO_RL}:${MEGATRON_BRIDGE}/src:${MEGATRON_LM}:${PYTHONPATH:-}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NRL_WG_USE_RAY_REF=1
export NRL_VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export NEMO_GYM_VENV_DIR=/opt/gym_venvs

cd "${NEMO_RL}"
exec python -u examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "${RECIPE}" \
  policy.model_name="${MODEL_DIR}" \
  policy.tokenizer.name="${MODEL_DIR}" \
  logger.log_dir="${RUN_DIR}/logs" \
  logger.wandb.name="${RUN_NAME}"
RUN
chmod 700 "${RUN_SCRIPT}"

export NUM_NODES=4
export GPUS_PER_NODE=4
export SLURM_ACCOUNT=<SLURM_ACCOUNT>
export PARTITION=<SLURM_PARTITION>
export CONTAINER=<NEMO_RL_CONTAINER_OR_SQUASHFS>
export MOUNTS="${SHARED_ROOT}:${SHARED_ROOT},${SHARED_ROOT}:/shared"
export COMMAND="/shared/runs/${RUN_NAME}/run.sh"

cd "${NEMO_RL}"
sbatch \
  --nodes="${NUM_NODES}" \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --job-name="${RUN_NAME}" \
  --time=04:00:00 \
  --gres=gpu:"${GPUS_PER_NODE}" \
  --exclusive \
  ray.sub
```

Slurm exports `RUN_NAME` and the other submission variables by default. If
your cluster uses a restricted export policy, add `--export=ALL` to `sbatch`.
Provide `WANDB_API_KEY` through the submission environment when experiment
tracking is enabled. To run without W&B, add
`logger.wandb_enabled=false` to the Python command in `run.sh`.

## Monitor training

Monitor the allocation and driver log from the NeMo RL checkout:

```bash
squeue -j <jobid> -o '%i %T %M %l %D %R'
tail -f <jobid>-logs/ray-driver.log
```

The driver reports exact-match validation accuracy before training and after
steps 2, 4, 6, 8, and 10. It also records training reward, loss, response
length, throughput, timing, and GPU utilization through the configured logger.

A successful run reaches step 10, completes the final validation pass, shuts
down the NeMo Gym services, flushes the logger, and exits with status zero.

## Interpreting the result

One reference run produced the following held-out accuracy curve:

| Step | Validation accuracy |
| ---: | ---: |
| 0 | 11.33% |
| 2 | 13.67% |
| 4 | 13.67% |
| 6 | 55.47% |
| 8 | 71.09% |
| 10 | 70.31% |

The curve tells a useful story. The first updates produced little visible
change, improvement became clear by step 6, and the final two measurements
showed that individual GRPO updates can remain noisy even after substantial
learning. The run improved by 58.98 percentage points from its pre-RL
baseline, while its best measured accuracy occurred at step 8.

Treat this result as a pipeline reference rather than a benchmark claim.
Sampled rollouts, hardware, software revisions, and model checkpoints can all
affect the curve. For model-quality comparisons, repeat the experiment with
multiple seeds, preserve the same validation split, and report both the final
and best validation accuracy.
