# GRPO with Nemotron 3.5 Super VL and NeMo Gym Star Count

A page of colored stars looks like a simple visual puzzle. Solving it reliably,
however, requires a multimodal model to find the relevant objects, distinguish
their colors, count them, and follow a precise answer format. This guide turns
that compact task into an end-to-end reinforcement learning example with a
clear, automatically verifiable reward.

This is the cookbook's single training workflow. It uses NeMo RL's Megatron
backend for full-weight GRPO, colocated vLLM for generation, and NeMo Gym for
task execution and verification. Begin with the repository, container,
checkpoint, and shared-storage setup in [`../README.md`](../README.md), then use
[`super_vl_3_5_star_count_megatron.yaml`](super_vl_3_5_star_count_megatron.yaml)
for training.

## The learning task

Each example presents a square image containing colored stars and asks the
policy to count the stars of one specified color. The model returns its final
answer in `\boxed{}` format. NeMo Gym extracts the boxed integer and compares
it with the count in the example metadata:

- a correct count receives reward `1.0`;
- an incorrect, missing, or malformed count receives reward `0.0`.

There is no partial credit and no judge model. Exact-match accuracy therefore
combines several behaviors: perceiving and counting the target stars, emitting
a parseable integer, and following the required boxed-answer format. Report
format coverage alongside accuracy before attributing a gain to visual
counting alone.

### Built on the NeMo Gym circle-count environment

The star-count task is an adaptation of NeMo Gym's circle-count environment.
It replaces the rendered circles with five-point stars and changes the prompt
to ask about stars. It deliberately retains the environment's data contract,
interaction pattern, answer format, and reward function:

| Mechanism | Retained behavior |
| --- | --- |
| Agent routing | Each row targets `circle_count_simple_agent` in a single-turn interaction with no tools. |
| Request format | `responses_create_params.input` contains a system message followed by one user message with a base64 PNG and text question. |
| Dataset keys | Each generated star is stored under the existing `circles` key with `x`, `y`, `radius`, and `color`; `target_color` identifies the requested class. |
| Answer format | The final answer must contain a plain non-negative integer in strict `\boxed{<digits>}` form. |
| Reward | The verifier counts entries in `circles` whose `color` equals `target_color`, extracts the first boxed integer, and returns `1.0` for an exact match or `0.0` otherwise. |

The verifier is therefore shape-agnostic: it reads the generated metadata and
does not inspect whether the rendered objects are circles or stars. The data
generator keeps the star image, prompt, and retained metadata aligned, which
allows the existing environment to score the new visual object without a
custom NeMo Gym service. This compatibility path was validated on the NeMo RL
`super-v3.5-posttraining` branch. Recheck the verifier after updating the NeMo
Gym submodule because the recipe depends on it continuing to read only each
item's `color` field.

## Dataset design

The deterministic generator creates two disjoint splits:

| Split | Examples | Seeds |
| --- | ---: | --- |
| Training | 1,024 | 0–1,023 |
| Validation | 256 | 1,000,000–1,000,255 |

For every example, it samples a square canvas from 800 x 800 through
1,200 x 1,200 pixels, draws 1–30 non-overlapping stars, and selects 2–4 colors
from a fixed eight-color palette. Every selected color appears at least once,
so each question has a positive answer. The PNG is embedded in its JSONL row
as a base64 data URL, which keeps every example self-contained across workers.

The palette follows the original synthetic task. Its red, orange, and pink
tones are closer than its other colors, and yellow appears as a dark gold.
This can make color identification part of the task rather than a perfectly
controlled counting variable. Keep the palette fixed when comparing runs, or
replace it with a perceptually validated palette and regenerate both splits.

### Sample examples

The following examples come directly from the deterministic training split.
Together they show how the same task ranges from a sparse scene to a more
crowded visual counting problem.

<table>
  <tr>
    <td width="50%">
      <img src="assets/star_count_sample_1.png" alt="Three colored stars on a white canvas" width="360"><br>
      <strong>Prompt:</strong> How many cyan stars are in the image?<br>
      <strong>Expected response:</strong> <code>\boxed{1}</code>
    </td>
    <td width="50%">
      <img src="assets/star_count_sample_2.png" alt="Thirteen colored stars on a white canvas" width="360"><br>
      <strong>Prompt:</strong> How many red stars are in the image?<br>
      <strong>Expected response:</strong> <code>\boxed{5}</code>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="assets/star_count_sample_3.png" alt="Eighteen orange and purple stars on a white canvas" width="360"><br>
      <strong>Prompt:</strong> How many orange stars are in the image?<br>
      <strong>Expected response:</strong> <code>\boxed{10}</code>
    </td>
    <td width="50%">
      <img src="assets/star_count_sample_4.png" alt="Twenty-nine red, purple, and yellow stars on a white canvas" width="360"><br>
      <strong>Prompt:</strong> How many red stars are in the image?<br>
      <strong>Expected response:</strong> <code>\boxed{8}</code>
    </td>
  </tr>
</table>

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
| Training coverage | 160 prompts: the first 160 rows of the 1,024-row split |
| Checkpointing | Disabled; the example does not save trained weights |

The 16 x 8 rollout batch gives GRPO eight candidate responses for each prompt.
NeMo RL converts their binary rewards into group-relative advantages using
reward normalization and a leave-one-out baseline. Reward shaping and reward
scaling remain disabled.

The policy trains both the language and vision components. During each
colocated weight refit, the recipe temporarily moves distributed optimizer
state out of GPU memory so the full tensor-parallel weight gather has enough
headroom.

The ten-step schedule is a smoke test and short convergence demonstration
rather than a full dataset epoch: 10 steps x 16 prompts consume 160 rows. With
`data.shuffle: false`, these are rows 0–159 in seed order. Increase the step
count or enable shuffling for broader training coverage. `max_num_epochs` is
set explicitly to one because the step limit ends this example before the
first epoch completes.

TP=4 shards dense and attention tensors across four ranks. EP=16 independently
distributes the model's 512 routed experts across all 16 GPUs, leaving 32
routed experts per expert-parallel rank. The TP and EP values describe
different parallel dimensions and do not imply a 64-GPU allocation.

The launch commands below make the vLLM V1 and `FLASH_ATTN` runtime choices
explicit and place model-conversion caches on shared storage.

## Generate the train and validation data

Run the generator inside the NeMo RL container or from an attached allocation.
The commands use the `/shared` layout established in the parent README:

```bash
export NEMO_RL=/shared/code/RL
export NEMOTRON_REPO=/shared/code/Nemotron
export MODEL_DIR=/shared/models/NVIDIA-Nemotron-3.5-Super-VL-09212026
export DATA_DIR=/shared/runs/super35-star-count/data
export GENERATOR="${NEMOTRON_REPO}/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-star-count-nemo-gym/prepare_star_count_data.py"
mkdir -p "${DATA_DIR}"

python "${GENERATOR}" \
  --out "${DATA_DIR}/train.jsonl" \
  --num-samples 1024 \
  --seed-offset 0

python "${GENERATOR}" \
  --out "${DATA_DIR}/validation.jsonl" \
  --num-samples 256 \
  --seed-offset 1000000
```

The generator writes self-contained JSONL rows with embedded PNG images. The
training and validation seeds are disjoint.

## Interactive run

Use an interactive allocation when bringing up the recipe for the first time,
inspecting logs, or trying configuration overrides. From the login node, set
the scheduler and container values for your cluster. The parent README defines
`SHARED_ROOT`, `NEMO_RL`, `CONTAINER`, and the `/shared` mount convention used
below:

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
export CACHE_DIR="${RUN_DIR}/cache"
export RECIPE="${NEMOTRON_REPO}/usage-cookbook/Nemotron-3.5-Super-VL/RL/grpo-star-count-nemo-gym/super_vl_3_5_star_count_megatron.yaml"

mkdir -p "${RUN_DIR}/logs" "${CACHE_DIR}"/{hf_modules,hf_config_locks,megatron_ckpt,vllm}
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
export NRL_VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

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

mkdir -p "${RUN_DIR}/logs" "${CACHE_DIR}"/{hf_modules,hf_config_locks,megatron_ckpt,vllm}
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
export NRL_VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

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

## Interpreting the reference result

One reference run produced the following held-out exact-match counts:

| Step | Correct | Validation accuracy |
| ---: | ---: | ---: |
| 0 | 29/256 | 11.33% |
| 2 | 35/256 | 13.67% |
| 4 | 35/256 | 13.67% |
| 6 | 142/256 | 55.47% |
| 8 | 182/256 | 71.09% |
| 10 | 180/256 | 70.31% |

Format diagnostics change how this curve should be read:

| Diagnostic | Step 0 | Step 10 |
| --- | ---: | ---: |
| Parseable boxed integer | 30/256 (11.72%) | 255/256 (99.61%) |
| Correct boxed integer | 29/256 (11.33%) | 180/256 (70.31%) |
| Correct among parseable answers | 29/30 (96.67%) | 180/255 (70.59%) |
| Mean response length | 249.2 tokens | 155.2 tokens |
| Responses in the 254–256 token histogram bin | 227/256 | 1/256 |

The policy learned to emit shorter, parseable boxed answers. Since 29 of the
30 parseable step-0 responses were already correct, much of the measured gain
comes from output completion and formatting and cannot be attributed solely
to better visual counting. The step-8 and step-10 results differ by only two
examples and are statistically indistinguishable at this sample size. Treat
this short run as a pipeline reference rather than a benchmark result.
