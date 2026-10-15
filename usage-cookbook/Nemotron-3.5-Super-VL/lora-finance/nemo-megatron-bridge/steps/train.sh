#!/bin/bash
# Step 4: LoRA fine-tuning; the adapter lands in $WORKSPACE/trained/$EXPERIMENT_NAME.
# On one machine this starts one process per GPU. Under Slurm it runs once per node and the
# SLURM_* variables join the nodes into one job.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source config.env; set +a

torchrun --nproc-per-node="$GPUS_PER_NODE" \
  --nnodes="${SLURM_NNODES:-1}" --node-rank="${SLURM_NODEID:-0}" \
  --master-addr="${MASTER_ADDR:-localhost}" --master-port=29500 \
  train.py
