#!/bin/bash
# Step 6: answer held-out questions with vLLM.
#   bash steps/try.sh        the merged, fine-tuned model
#   bash steps/try.sh base   the base model, for comparison
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source config.env; set +a

MODEL=$WORKSPACE/trained/$EXPERIMENT_NAME/merged_hf
[ "${1:-}" = base ] && MODEL=$MODEL_DIR
python try_model.py "$MODEL"
