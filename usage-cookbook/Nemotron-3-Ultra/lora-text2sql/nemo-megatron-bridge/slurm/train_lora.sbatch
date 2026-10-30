#!/bin/bash
# Nemotron-3 Ultra LoRA fine-tune on local text2sql data, PACKED sequences.
# Runs train_lora.py (builds a packed FinetuningDatasetConfig on our local data).
#SBATCH --job-name=nultra-text2sql-lora
#SBATCH --exclusive

set -euo pipefail
source "${MBRIDGE_CONFIG:?set by the notebook launch cell}"
mkdir -p "$LOG_DIR" "$TRAIN_OUTPUT_DIR"
[ -e "$MEGATRON_MODEL_PATH/latest_checkpointed_iteration.txt" ] || { echo "ERROR: run convert first ($MEGATRON_MODEL_PATH)"; exit 1; }

export SAVE_DIR="${TRAIN_OUTPUT_DIR}/${EXPERIMENT_NAME}"
export DATASET_DIR MEGATRON_MODEL_PATH HF_MODEL_PATH MAX_SEQ_LEN GLOBAL_BS MICRO_BS EPOCHS
export TRAIN_TP TRAIN_PP TRAIN_EP TRAIN_ETP TRAIN_CP LR MIN_LR GPUS_PER_NODE WORKDIR COOKBOOK_DIR
export WANDB_MODE=${WANDB_MODE:-disabled}
export TORCH_NCCL_AVOID_RECORD_STREAMS=1 NCCL_NVLS_ENABLE=0 NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
MASTER_PORT=$((18000 + SLURM_JOB_ID % 40000))
export MASTER_ADDR MASTER_PORT

echo "packed lora: base=${MEGATRON_MODEL_PATH} data=${DATASET_DIR} save=${SAVE_DIR} nodes=${SLURM_JOB_NUM_NODES}"

SRUN_CMD=(srun --mpi=pmix --no-kill --container-image="${CONTAINER_IMAGE}" --no-container-mount-home)
MOUNTS="${COOKBOOK_DIR}:${COOKBOOK_DIR}"
[ -n "${CONTAINER_MOUNTS:-}" ] && MOUNTS="${MOUNTS},${CONTAINER_MOUNTS}"
SRUN_CMD+=(--container-mounts="${MOUNTS}")

"${SRUN_CMD[@]}" bash -c '
set -euo pipefail
cd "$WORKDIR"
export PYTHONPATH="$COOKBOOK_DIR:$WORKDIR/src:$WORKDIR/3rdparty/Megatron-LM:${PYTHONPATH:-}"
export RANK="${SLURM_PROCID}" WORLD_SIZE="${SLURM_NTASKS}" LOCAL_RANK="${SLURM_LOCALID}" LOCAL_WORLD_SIZE="${GPUS_PER_NODE}"
uv run --no-sync python "$COOKBOOK_DIR/train_lora.py"
'
echo LORA_TRAIN_DONE
