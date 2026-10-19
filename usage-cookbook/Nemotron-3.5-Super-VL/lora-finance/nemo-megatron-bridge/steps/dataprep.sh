#!/bin/bash
# Step 2: build the training data in $WORKSPACE/dataprep. CPU only.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source config.env; set +a

pip install -q pymupdf  # renders the TAT-DQA PDF pages; not in the container
python dataprep.py
