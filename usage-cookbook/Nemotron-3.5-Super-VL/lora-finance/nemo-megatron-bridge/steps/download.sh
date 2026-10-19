#!/bin/bash
# Step 1: download the checkpoint to $MODEL_DIR. CPU only.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source config.env; set +a

python -c "from huggingface_hub import snapshot_download
snapshot_download('$HF_MODEL', local_dir='$MODEL_DIR')"

# The checkpoint ships its own model code. Load it once here, so the multi-GPU steps reuse the
# cached copy instead of every process writing it at the same time.
python -c "from transformers import AutoConfig, AutoProcessor
AutoConfig.from_pretrained('$MODEL_DIR', trust_remote_code=True)
AutoProcessor.from_pretrained('$MODEL_DIR', trust_remote_code=True)"
echo "Download complete: $MODEL_DIR"
