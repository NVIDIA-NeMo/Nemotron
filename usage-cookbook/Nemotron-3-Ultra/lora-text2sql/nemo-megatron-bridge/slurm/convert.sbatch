#!/bin/bash
# Distributed HF -> Megatron-Bridge import of the 550B base (CPU import infeasible at this size).
# World size (CONV_TP*CONV_PP*CONV_EP) must equal nodes * GPUS_PER_NODE.
#SBATCH --job-name=nultra-text2sql-convert
#SBATCH --exclusive

set -euo pipefail
source "${MBRIDGE_CONFIG:?set by the notebook launch cell}"
mkdir -p "$LOG_DIR" "$(dirname "$MEGATRON_MODEL_PATH")"

TP=${CONV_TP}; PP=${CONV_PP}; EP=${CONV_EP}; ETP=${CONV_ETP:-1}

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

if [ "$((TP * PP * EP))" -ne "$((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))" ]; then
    echo "ERROR: CONV_TP*CONV_PP*CONV_EP ($((TP*PP*EP))) must equal nodes*GPUS_PER_NODE ($((SLURM_JOB_NUM_NODES*GPUS_PER_NODE)))."
    exit 2
fi
if [ -e "${MEGATRON_MODEL_PATH}/latest_checkpointed_iteration.txt" ]; then
    echo "Megatron checkpoint already present at ${MEGATRON_MODEL_PATH}; nothing to do."
    echo IMPORT_DONE
    exit 0
fi

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
MASTER_PORT=$((18000 + SLURM_JOB_ID % 40000))
export MASTER_ADDR MASTER_PORT HF_MODEL_PATH MEGATRON_MODEL_PATH TP PP EP ETP WORKDIR GPUS_PER_NODE

echo "import: ${HF_MODEL_PATH} -> ${MEGATRON_MODEL_PATH}  (TP${TP} PP${PP} EP${EP}, ${SLURM_JOB_NUM_NODES} nodes)"

SRUN_CMD=(srun --mpi=pmix --no-kill --container-image="${CONTAINER_IMAGE}" --no-container-mount-home)
[ -n "${CONTAINER_MOUNTS:-}" ] && SRUN_CMD+=(--container-mounts="${CONTAINER_MOUNTS}")

"${SRUN_CMD[@]}" bash -c '
set -euo pipefail
cd "$WORKDIR"
export PYTHONPATH="$WORKDIR/src:$WORKDIR/3rdparty/Megatron-LM:${PYTHONPATH:-}"
export RANK="${SLURM_PROCID}" WORLD_SIZE="${SLURM_NTASKS}" LOCAL_RANK="${SLURM_LOCALID}" LOCAL_WORLD_SIZE="${GPUS_PER_NODE}"
uv run --no-sync python examples/conversion/convert_checkpoints_multi_gpu.py import \
    --hf-model "$HF_MODEL_PATH" \
    --megatron-path "$MEGATRON_MODEL_PATH" \
    --tp "$TP" --pp "$PP" --ep "$EP" --etp "$ETP" \
    --torch-dtype bfloat16 \
    --distributed-timeout-minutes 90
'
echo IMPORT_DONE
