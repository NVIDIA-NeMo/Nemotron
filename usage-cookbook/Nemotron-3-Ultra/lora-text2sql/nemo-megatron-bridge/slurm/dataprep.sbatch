#!/bin/bash
# Build the (small) BIRD text2sql training.jsonl inside the container (CPU-only).
#SBATCH --job-name=nultra-text2sql-dataprep
#SBATCH --ntasks-per-node=1

set -euo pipefail
source "${MBRIDGE_CONFIG:?set by the notebook launch cell}"
mkdir -p "$LOG_DIR" "$DATAPREP_OUTPUT_DIR"

# MODEL_ID provides the tokenizer/chat template (local snapshot -> no model download).
export MODEL_ID="${HF_MODEL_PATH}"
export MAX_SEQ_LEN DATAPREP_OUTPUT_DIR MAX_TRAIN_SAMPLES HF_HOME

SRUN_CMD=(srun --mpi=pmix --container-image="${CONTAINER_IMAGE}" --no-container-mount-home)
MOUNTS="${COOKBOOK_DIR}:${COOKBOOK_DIR}"
[ -n "${CONTAINER_MOUNTS:-}" ] && MOUNTS="${MOUNTS},${CONTAINER_MOUNTS}"
SRUN_CMD+=(--container-mounts="${MOUNTS}")

echo "dataprep: MODEL_ID=${MODEL_ID} max_seq_len=${MAX_SEQ_LEN} max_samples=${MAX_TRAIN_SAMPLES}"
"${SRUN_CMD[@]}" bash -c "
set -euo pipefail
export PYTHONPATH='${COOKBOOK_DIR}':\${PYTHONPATH:-}
cd '${WORKDIR}'
uv run --no-sync python '${COOKBOOK_DIR}/dataprep.py'
"
echo DATAPREP_DONE
