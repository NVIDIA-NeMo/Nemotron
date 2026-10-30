#!/bin/bash
# Step 5: merge the adapter into the base weights and export a Hugging Face checkpoint to
# $WORKSPACE/trained/$EXPERIMENT_NAME/merged_hf. 4 GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source config.env; set +a

RUN_DIR=$WORKSPACE/trained/$EXPERIMENT_NAME
CKPT=$RUN_DIR/iter_$(printf %07d "$(cat "$RUN_DIR/latest_checkpointed_iteration.txt")")
torchrun --nproc-per-node=4 /opt/Megatron-Bridge/examples/peft/merge_lora.py \
  --lora-checkpoint "$CKPT" --hf-model-path "$MODEL_DIR" --pretrained "$WORKSPACE/base" \
  --output "$RUN_DIR/merged_hf" --tp 1 --pp 2 --ep 2
