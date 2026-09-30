# GRPO with Nemotron 3.5 Super VL and NeMo Gym Circle Count

This guide runs multimodal GRPO through NeMo RL's Megatron backend. The
four-node reference performs full-weight updates, and the two-node variant
uses LoRA. NeMo Gym presents synthetic circle images to the policy through
the Responses API, routes each response through `circle_count_simple_agent`,
and returns a binary exact-match reward.

Use [`super_vl_3_5_circle_count_megatron.yaml`](super_vl_3_5_circle_count_megatron.yaml)
as the training configuration. Complete the repository, container, checkpoint,
and shared-storage setup in [`../README.md`](../README.md) first.

## Training goal and data

The task asks the policy to count circles of a specified color and return the
answer in `\boxed{}` format. The data generator embeds each image directly in
the JSONL record as a data URL. This makes every row self-contained and avoids
image path differences across distributed workers.

The reference split contains 1,024 training examples and 256 validation
examples. Training uses seeds 0–1,023; validation uses seeds
1,000,000–1,000,255. Both splits use 256 x 256 images with 1–8 circles and
circle radii from 12–24 pixels.

## Configuration overview

The recipe uses the following reference settings:

| Component | Setting |
| --- | --- |
| Backend | Megatron, full-weight BF16 |
| Training parallelism | TP=4, EP=16 |
| Generation | colocated vLLM, TP=4 |
| Rollout batch | 8 prompts x 2 generations |
| Policy global batch | 16 |
| Validation | 256 examples, greedy decoding |
| Evaluation cadence | before RL, then steps 5, 10, and 15 |

The short schedule is intended to verify the complete multimodal RL pipeline
and expose a before/after learning signal. Treat it as a starting point for
task-specific experiments rather than a general visual-reasoning benchmark.

### Two-node pipeline check

[`super_vl_3_5_circle_count_megatron_2n.yaml`](super_vl_3_5_circle_count_megatron_2n.yaml)
is an overlay for two 4-GPU GB200 nodes. It inherits the data, rollout, and
evaluation settings above, then changes training to TP=8 and EP=8 with
rank-16 LoRA. It keeps one node-local TP=4 vLLM group per node, enables
expandable CUDA allocation, and holds the vision encoder and projection fixed.

To use it in the commands below, select two nodes and the overlay filename:

```bash
export NUM_NODES=2
export RECIPE_NAME=super_vl_3_5_circle_count_megatron_2n.yaml
```

The topology was validated end to end with a one-step smoke run on 16 held-out
examples. Exact-match accuracy was 0.75 both before and after that update. The
unchanged score is expected to be inconclusive at this scale: the smoke run
checks distributed initialization, rollout, optimization, adapter refit, and
post-update inference. Run the full schedule and validation split for an
accuracy comparison.

## Generate the train and validation data

Run the generator inside the NeMo RL image or an attached allocation. The
commands below use the container-side `/shared` paths from the parent README:

```bash
export NEMO_RL=/shared/code/RL
export DATA_DIR=/shared/runs/super35-circle-count/data
mkdir -p "${DATA_DIR}"

cd "${NEMO_RL}"
python examples/nemo_gym/nemotron-3-super-omni/prepare_circle_count_mopd_data.py \
  --out "${DATA_DIR}/train.jsonl" \
  --num-samples 1024 \
  --seed-offset 0 \
  --image-size 256 \
  --radius-min 12 \
  --radius-max 24 \
  --num-circles-min 1 \
  --num-circles-max 8

python examples/nemo_gym/nemotron-3-super-omni/prepare_circle_count_mopd_data.py \
  --out "${DATA_DIR}/validation.jsonl" \
  --num-samples 256 \
  --seed-offset 1000000 \
  --image-size 256 \
  --radius-min 12 \
  --radius-max 24 \
  --num-circles-min 1 \
  --num-circles-max 8
```

Validate the row counts, agent routing, and split separation:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('/shared/runs/super35-circle-count/data')
fingerprints = {}
for split, expected in [('train', 1024), ('validation', 256)]:
    rows = [json.loads(line) for line in (root / f'{split}.jsonl').read_text().splitlines()]
    assert len(rows) == expected
    assert all(row['agent_ref']['name'] == 'circle_count_simple_agent' for row in rows)
    fingerprints[split] = {
        hashlib.sha256(
            json.dumps(row['responses_create_params'], sort_keys=True).encode()
        ).hexdigest()
        for row in rows
    }

assert fingerprints['train'].isdisjoint(fingerprints['validation'])
print('Circle-count train and validation splits are valid and disjoint.')
PY
```

## Interactive reference run

Start a persistent allocation from the login node. Replace the Slurm account,
partition, and container values with settings for your cluster:

```bash
export NUM_NODES=${NUM_NODES:-4}
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
  --job-name=interactive-super-vl-circle-count \
  --time=02:00:00 \
  --gres=gpu:"${GPUS_PER_NODE}" \
  --exclusive \
  ray.sub
```

After the allocation starts, attach with the helper created by `ray.sub`:

```bash
cd "${NEMO_RL}"
bash ./<jobid>-attach.sh
```

Inside the attached container, prepare shared caches and launch the recipe:

```bash
export NEMO_RL=/shared/code/RL
export NEMOTRON_REPO=/shared/code/Nemotron
export MODEL_DIR=/shared/models/NVIDIA-Nemotron-3.5-Super-VL-09212026
export RUN_DIR=/shared/runs/super35-circle-count
export COOKBOOK_DIR="${NEMOTRON_REPO}/usage-cookbook/Nemotron-3.5-Super-VL/RL"
export RECIPE="${COOKBOOK_DIR}/grpo-circle-count-nemo-gym/${RECIPE_NAME:-super_vl_3_5_circle_count_megatron.yaml}"

mkdir -p \
  "${RUN_DIR}/hf_modules" \
  "${RUN_DIR}/hf_config_locks" \
  "${RUN_DIR}/megatron_ckpt_cache" \
  "${RUN_DIR}/vllm_compile_cache" \
  "${RUN_DIR}/logs"

export HF_MODULES_CACHE="${RUN_DIR}/hf_modules"
export MEGATRON_CONFIG_LOCK_DIR="${RUN_DIR}/hf_config_locks"
export NRL_MEGATRON_CHECKPOINT_DIR="${RUN_DIR}/megatron_ckpt_cache"
export VLLM_CACHE_ROOT="${RUN_DIR}/vllm_compile_cache"
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

For a pipeline check without external logging, add
`logger.wandb_enabled=false`. For experiment tracking, leave W&B enabled and
provide `WANDB_API_KEY` through the job environment.

## Batch run

For an unattended run, create a small driver script on shared storage and pass
its container path to `ray.sub` as `COMMAND`. Run this setup on the login
node:

```bash
export RUN_NAME=super35-circle-count-$(date +%Y%m%d-%H%M%S)
export HOST_RUN_DIR="${SHARED_ROOT}/runs/${RUN_NAME}"
export RUN_SCRIPT="${HOST_RUN_DIR}/run.sh"
mkdir -p "${HOST_RUN_DIR}"

cat > "${RUN_SCRIPT}" <<'RUN'
#!/usr/bin/env bash
set -euo pipefail

export NEMO_RL=/shared/code/RL
export NEMOTRON_REPO=/shared/code/Nemotron
export MODEL_DIR=/shared/models/NVIDIA-Nemotron-3.5-Super-VL-09212026
export RUN_DIR=/shared/runs/${RUN_NAME}
export CACHE_DIR=/shared/runs/super35-circle-count/cache
export COOKBOOK_DIR="${NEMOTRON_REPO}/usage-cookbook/Nemotron-3.5-Super-VL/RL"
export RECIPE="${COOKBOOK_DIR}/grpo-circle-count-nemo-gym/${RECIPE_NAME:-super_vl_3_5_circle_count_megatron.yaml}"

mkdir -p "${RUN_DIR}/logs" "${CACHE_DIR}"/{hf_modules,hf_config_locks,megatron_ckpt_cache,vllm_compile_cache}
export HF_MODULES_CACHE="${CACHE_DIR}/hf_modules"
export MEGATRON_CONFIG_LOCK_DIR="${CACHE_DIR}/hf_config_locks"
export NRL_MEGATRON_CHECKPOINT_DIR="${CACHE_DIR}/megatron_ckpt_cache"
export VLLM_CACHE_ROOT="${CACHE_DIR}/vllm_compile_cache"
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

export NUM_NODES=${NUM_NODES:-4}
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
  --time=02:00:00 \
  --gres=gpu:"${GPUS_PER_NODE}" \
  --exclusive \
  ray.sub
```

Export `RUN_NAME` with the submitted environment, as shown above. Provide
`WANDB_API_KEY` through the submission environment or a protected environment
file before calling `sbatch`.

Monitor the allocation and driver log from the NeMo RL checkout:

```bash
squeue -j <jobid> -o '%i %T %M %l %D %R'
tail -f <jobid>-logs/ray-driver.log
```

## Expected output

The driver logs `val:accuracy` before the first optimizer update and after
steps 5, 10, and 15. Compare step 15 with step 0 from the same run. A normal
run reaches `Max number of steps has been reached`, shuts down NeMo Gym,
syncs the logger, and exits successfully.

For the two-node configuration, successful completion also confirms that the
LoRA optimizer update was refit into the colocated vLLM workers before the
next validation pass.

The short synthetic task can vary across runs because training rollouts are
sampled. Use repeated runs, a longer schedule, and a task-specific validation
set when measuring model quality.
