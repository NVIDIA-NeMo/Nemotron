#!/bin/bash
# Step 3: convert the checkpoint to Megatron-Bridge format in $WORKSPACE/base. 4 GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source config.env; set +a

if [ -e "$WORKSPACE/base/latest_checkpointed_iteration.txt" ]; then
  echo "Already converted: $WORKSPACE/base"
  exit 0
fi
torchrun --nproc-per-node=4 /opt/Megatron-Bridge/scripts/conversion/run_conversion.py import \
  --device gpu --hf-model "$MODEL_DIR" --trust-remote-code --torch-dtype bfloat16 \
  --tp 1 --pp 2 --ep 2 --etp 1 --low-memory-save --distributed-timeout-minutes 120 \
  --megatron-path "$WORKSPACE/base"
