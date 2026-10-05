# Train Nemotron 3.5 Super VL to Count Stars with GRPO

Counting colored stars is a compact way to exercise the complete multimodal RL
pipeline. The policy must inspect an image, identify the requested color, count
the matching objects, and return an answer that an automatic verifier can
score. This guide runs that task with full-weight GRPO, NeMo RL's Megatron
backend, colocated vLLM generation, and NeMo Gym.

## How the task works

Each example contains a generated PNG and a question such as, “How many cyan
stars are in the image?” The expected response is a plain non-negative integer
inside `\boxed{}`, for example `\boxed{3}`. The verifier gives reward `1.0` for
an exact match and `0.0` for an incorrect or malformed answer.

The task builds on NeMo Gym's circle-count environment. The generator replaces
circles with five-point stars while retaining the environment's established
data contract and reward mechanism:

| Mechanism | Star-count behavior |
| --- | --- |
| Agent | `circle_count_simple_agent`, with one user turn and no tools |
| Request | A system message followed by a user message containing a base64 PNG and question |
| Object metadata | Stars use the existing `circles` key with `x`, `y`, `radius`, and `color` fields |
| Target | `target_color` identifies the color to count |
| Answer | Strict `\boxed{<digits>}` format |
| Reward | Exact comparison between the boxed integer and the number of matching metadata entries |

The verifier reads object colors from the metadata rather than inspecting the
rendered shape. This lets the star task use the existing environment without a
custom NeMo Gym service.

The deterministic generator creates 1,024 training examples with seeds
0–1,023 and 256 held-out examples with seeds 1,000,000–1,000,255. Images range
from 800 x 800 to 1,200 x 1,200 pixels and contain 1–30 non-overlapping stars
drawn from 2–4 colors. Every requested color appears at least once.

<table>
  <tr>
    <td width="50%"><img src="assets/star_count_sample_1.png" alt="Sparse colored-star example" width="360"><br><strong>Question:</strong> How many cyan stars are in the image?<br><strong>Answer:</strong> <code>\boxed{1}</code></td>
    <td width="50%"><img src="assets/star_count_sample_2.png" alt="Colored-star counting example" width="360"><br><strong>Question:</strong> How many red stars are in the image?<br><strong>Answer:</strong> <code>\boxed{5}</code></td>
  </tr>
  <tr>
    <td width="50%"><img src="assets/star_count_sample_3.png" alt="Orange and purple star example" width="360"><br><strong>Question:</strong> How many orange stars are in the image?<br><strong>Answer:</strong> <code>\boxed{10}</code></td>
    <td width="50%"><img src="assets/star_count_sample_4.png" alt="Crowded colored-star example" width="360"><br><strong>Question:</strong> How many red stars are in the image?<br><strong>Answer:</strong> <code>\boxed{8}</code></td>
  </tr>
</table>

## Reference configuration

The included
[`super_vl_3_5_star_count_megatron.yaml`](super_vl_3_5_star_count_megatron.yaml)
uses the following settings:

| Component | Setting |
| --- | --- |
| Compute | 4 nodes x 4 GPUs |
| Training | Full-weight BF16, TP=4, EP=16 |
| Generation | Colocated vLLM, TP=4 |
| GRPO batch | 16 prompts x 8 responses |
| Schedule | 10 updates; validation before RL and every 2 updates |
| Sequence limit | 4,096 total tokens; 256 generated tokens |
| Checkpoints | Disabled |

The short schedule uses the first 160 rows because data shuffling is disabled.
It is intended as a reproducible pipeline example rather than a full training
run over all 1,024 examples. TP=4 shards dense and attention tensors within a
node. EP=16 distributes the 512 routed experts across all GPUs, with 32 experts
per expert-parallel rank.

## Prerequisites

Prepare shared storage that is visible to every compute node and contains:

```text
<SHARED_ROOT>/
|-- code/RL/                 # NeMo RL, branch super-v3.5-posttraining
|-- code/Nemotron/           # This repository
|-- models/NVIDIA-Nemotron-3.5-Super-VL-09212026/
`-- runs/
```

Use a NeMo RL container built from the `super-v3.5-posttraining` branch, or a
compatible prebuilt image newer than v0.7. The parent
[`README.md`](../README.md) provides the source build command and storage
estimates. The commands below assume that the shared root is also mounted at
`/shared` inside the container.

On the login or head node, define the site-specific values once:

```bash
export SHARED_ROOT=</YOUR/SHARED/STORAGE>
export NEMO_RL="${SHARED_ROOT}/code/RL"
export CONTAINER=<NEMO_RL_CONTAINER_OR_SQUASHFS>
export SLURM_ACCOUNT=<SLURM_ACCOUNT>
export PARTITION=<SLURM_PARTITION>
export GPUS_PER_NODE=4
export MOUNTS="${SHARED_ROOT}:${SHARED_ROOT},${SHARED_ROOT}:/shared"
```

## Interactive path

Use the interactive path when trying the recipe for the first time or watching
the training process directly.

### 1. Reserve four nodes — login or head node

Run from the NeMo RL repository root:

```bash
cd "${NEMO_RL}"
unset COMMAND
sbatch \
  --nodes=4 \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --job-name=super-vl-star-count \
  --time=04:00:00 \
  --gres=gpu:4 \
  --mem=0 \
  --exclusive \
  ray.sub
```

`--mem=0` requests all host memory on each node. The colocated full-weight
recipe temporarily moves optimizer state to host memory during weight refits.

### 2. Attach — login or head node

After the allocation starts, use the helper created by `ray.sub`:

```bash
cd "${NEMO_RL}"
bash ./<jobid>-attach.sh
```

### 3. Generate data and train — attached Ray-head container

Run this block in the attached container. It contains all runtime paths and
cache settings; the only separate recipe file is the checked-in YAML.

```bash
set -euo pipefail

export NEMO_RL=/shared/code/RL
export NEMOTRON_REPO=/shared/code/Nemotron
export MODEL_DIR=/shared/models/NVIDIA-Nemotron-3.5-Super-VL-09212026
export RUN_DIR=/shared/runs/super35-star-count
export DATA_DIR="${RUN_DIR}/data"
export CACHE_DIR="${RUN_DIR}/cache"
export EXAMPLE_DIR="${NEMOTRON_REPO}/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-star-count-nemo-gym"

mkdir -p "${DATA_DIR}" "${RUN_DIR}/logs" \
  "${CACHE_DIR}"/{hf_modules,hf_config_locks,megatron_ckpt,vllm}

export HF_MODULES_CACHE="${CACHE_DIR}/hf_modules"
export MEGATRON_CONFIG_LOCK_DIR="${CACHE_DIR}/hf_config_locks"
export NRL_MEGATRON_CHECKPOINT_DIR="${CACHE_DIR}/megatron_ckpt"
export VLLM_CACHE_ROOT="${CACHE_DIR}/vllm"
export MEGATRON_BRIDGE="${NEMO_RL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
export MEGATRON_LM="${MEGATRON_BRIDGE}/3rdparty/Megatron-LM"
export PYTHONPATH="${HF_MODULES_CACHE}:${NEMO_RL}:${MEGATRON_BRIDGE}/src:${MEGATRON_LM}:${PYTHONPATH:-}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NRL_WG_USE_RAY_REF=1
export NEMO_GYM_VENV_DIR=/opt/gym_venvs

cd "${NEMO_RL}"
uv run --no-sync python "${EXAMPLE_DIR}/prepare_star_count_data.py" \
  --out "${DATA_DIR}/train.jsonl" --num-samples 1024 --seed-offset 0
uv run --no-sync python "${EXAMPLE_DIR}/prepare_star_count_data.py" \
  --out "${DATA_DIR}/validation.jsonl" --num-samples 256 --seed-offset 1000000

uv run --no-sync python -u examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "${EXAMPLE_DIR}/super_vl_3_5_star_count_megatron.yaml" \
  policy.model_name="${MODEL_DIR}" \
  policy.tokenizer.name="${MODEL_DIR}" \
  logger.log_dir="${RUN_DIR}/logs"
```

Append `logger.wandb_enabled=false` to the final command when external
experiment tracking is not desired.

## Batch path

For an unattended run, submit the same work through `COMMAND`. `ray.sub`
materializes this value inside its job log directory, so no additional launch
script is needed.

Run the following block from the NeMo RL repository root on the login or head
node. It assumes the prerequisite variables above are still defined.

```bash
cd "${NEMO_RL}"

read -r -d '' COMMAND <<'RUN' || true
set -euo pipefail
export NEMO_RL=/shared/code/RL
export NEMOTRON_REPO=/shared/code/Nemotron
export MODEL_DIR=/shared/models/NVIDIA-Nemotron-3.5-Super-VL-09212026
export RUN_DIR=/shared/runs/super35-star-count
export DATA_DIR="${RUN_DIR}/data"
export CACHE_DIR="${RUN_DIR}/cache"
export EXAMPLE_DIR="${NEMOTRON_REPO}/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-star-count-nemo-gym"

mkdir -p "${DATA_DIR}" "${RUN_DIR}/logs" \
  "${CACHE_DIR}"/{hf_modules,hf_config_locks,megatron_ckpt,vllm}
export HF_MODULES_CACHE="${CACHE_DIR}/hf_modules"
export MEGATRON_CONFIG_LOCK_DIR="${CACHE_DIR}/hf_config_locks"
export NRL_MEGATRON_CHECKPOINT_DIR="${CACHE_DIR}/megatron_ckpt"
export VLLM_CACHE_ROOT="${CACHE_DIR}/vllm"
export MEGATRON_BRIDGE="${NEMO_RL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
export MEGATRON_LM="${MEGATRON_BRIDGE}/3rdparty/Megatron-LM"
export PYTHONPATH="${HF_MODULES_CACHE}:${NEMO_RL}:${MEGATRON_BRIDGE}/src:${MEGATRON_LM}:${PYTHONPATH:-}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export NRL_WG_USE_RAY_REF=1
export NEMO_GYM_VENV_DIR=/opt/gym_venvs

cd "${NEMO_RL}"
uv run --no-sync python "${EXAMPLE_DIR}/prepare_star_count_data.py" \
  --out "${DATA_DIR}/train.jsonl" --num-samples 1024 --seed-offset 0
uv run --no-sync python "${EXAMPLE_DIR}/prepare_star_count_data.py" \
  --out "${DATA_DIR}/validation.jsonl" --num-samples 256 --seed-offset 1000000
exec uv run --no-sync python -u examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "${EXAMPLE_DIR}/super_vl_3_5_star_count_megatron.yaml" \
  policy.model_name="${MODEL_DIR}" \
  policy.tokenizer.name="${MODEL_DIR}" \
  logger.log_dir="${RUN_DIR}/logs"
RUN
export COMMAND

sbatch \
  --nodes=4 \
  --account="${SLURM_ACCOUNT}" \
  --partition="${PARTITION}" \
  --job-name=super-vl-star-count \
  --time=04:00:00 \
  --gres=gpu:4 \
  --mem=0 \
  --exclusive \
  ray.sub
```

Monitor a batch job from the login or head node:

```bash
squeue -j <jobid> -o '%i %T %M %l %D %R'
tail -f <jobid>-logs/ray-driver.log
```

## Reading the result

The driver reports held-out exact-match accuracy before RL and after steps 2,
4, 6, 8, and 10. One reference run increased from 29/256 (11.33%) before RL to
180/256 (70.31%) after step 10.

This metric combines visual counting, response completion, and strict answer
formatting. In the reference run, parseable boxed answers rose from 30/256 to
255/256 while mean response length fell from 249 to 155 tokens. Among
parseable answers, the initial policy was already correct on 29/30 examples.
The result therefore demonstrates that the RL pipeline teaches the policy to
produce concise, verifiable responses; it should not be read as a pure measure
of improved visual perception.
