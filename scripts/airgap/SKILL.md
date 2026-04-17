# SKILL: Airgap Deployment for Nemotron Customization Recipes

> Pre-download, transfer, and deploy the full Nemotron customization pipeline in environments with no internet access.

---

## Overview

The Nemotron customization pipeline (CPT, SFT, RL, BYOB, Eval, Quantization) normally requires internet access for downloading HuggingFace models, datasets, NLP assets (NLTK, spaCy, FastText), and calling cloud APIs. Airgap deployment eliminates all network dependencies by pre-downloading everything into a portable bundle.

### Components

| Script | Purpose | Where to run |
|--------|---------|--------------|
| `scripts/airgap/download_assets.sh` | Pre-download all required assets | Internet-connected machine |
| `scripts/airgap/deploy_airgap.sh` | Deploy the bundle in the airgap environment | Airgap target machine |
| `deploy/nemotron/customization_recipes/docker-compose.airgap.yaml` | Docker Compose override for offline mode | Airgap target machine |

---

## Step 1: Pre-Download (Internet-Connected Machine)

### Prerequisites

- Python 3.8+ with `huggingface_hub`, `datasets`, `nltk`, `spacy` installed
- `huggingface-cli` (comes with `huggingface_hub`)
- HuggingFace API token (for gated models)
- Docker (if saving container images)
- 100-300 GB free disk space depending on options

### Basic Download

```bash
# Download core assets for Nemotron Nano customization
./scripts/airgap/download_assets.sh \
  --output-dir /data/airgap-bundle \
  --model-family nemotron-nano \
  --hf-token $HF_TOKEN
```

### Full Download (All Options)

```bash
# Download everything: both model families, NIM, benchmarks, Docker images
./scripts/airgap/download_assets.sh \
  --output-dir /data/airgap-bundle \
  --model-family all \
  --include-nim \
  --include-benchmarks \
  --include-docker \
  --include-chat-model \
  --hf-token $HF_TOKEN
```

### Dry Run (Preview)

```bash
# See what would be downloaded without downloading anything
./scripts/airgap/download_assets.sh \
  --output-dir /data/airgap-bundle \
  --model-family all \
  --include-nim \
  --include-benchmarks \
  --dry-run
```

### What Gets Downloaded

| Category | Assets | Approximate Size |
|----------|--------|-----------------|
| HF Models (Nano) | Nemotron-3-Nano-30B-A3B-Base-BF16, Nemotron-3-Nano-30B-A3B-BF16 | ~60 GB |
| HF Models (Super) | Nemotron-3-Super-49B-v1, Nemotron-3-Super-49B-Instruct-v1 | ~100 GB |
| Shared Models | multilingual-domain-classifier, all-MiniLM-L6-v2 | ~1 GB |
| Chat Template Model | Mistral-Small-24B-Instruct-2501 (optional) | ~48 GB |
| Datasets | cais/mmlu, ultrachat_200k, Nemotron-Pretraining-Dataset-sample, cnn_dailymail | ~10 GB |
| FastText | lid.176.bin (language identification) | ~125 MB |
| NLTK | punkt, punkt_tab, stopwords, averaged_perceptron_tagger_eng | ~50 MB |
| spaCy | en_core_web_sm, xx_sent_ud_sm | ~50 MB |
| Docker Images | nemo trainer, nemo-curator, NIM (optional) | ~20-40 GB each |
| Benchmarks | NeMo-Skills, Gorilla, MMLU-Pro (optional) | ~5 GB |

### Bundle Structure

```
airgap-bundle/
  models/
    huggingface/
      nvidia_NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16/
      nvidia_NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/
      nvidia_multilingual-domain-classifier/
      sentence-transformers_all-MiniLM-L6-v2/
    fasttext/
      lid.176.bin
    spacy/
      en_core_web_sm/
      xx_sent_ud_sm/
  datasets/
    cais_mmlu/
    HuggingFaceH4_ultrachat_200k/
    nvidia_Nemotron-Pretraining-Dataset-sample/
    cnn_dailymail/
  nltk_data/
    tokenizers/punkt/
    tokenizers/punkt_tab/
    corpora/stopwords/
    taggers/averaged_perceptron_tagger_eng/
  docker_images/         (if --include-docker)
    nvcr.io_nvidia_nemo_25.11.nemotron_3_nano.tar
    nvcr.io_nvidia_nemo-curator_26.02.tar
  benchmarks/            (if --include-benchmarks)
    NeMo-Skills/
    gorilla/
    mmlu_pro/
  configs/
    airgap-env.toml
    airgap-overrides.yaml
  manifest.json
```

---

## Step 2: Transfer the Bundle

Transfer the airgap bundle to the target environment using your preferred method:

```bash
# Option A: rsync over network (if temporary connectivity available)
rsync -avP --progress /data/airgap-bundle/ airgap-host:/data/airgap-bundle/

# Option B: tar + physical media
tar -cf airgap-bundle.tar -C /data airgap-bundle/
# Copy to USB drive, ship to airgap site

# Option C: Split tar for size limits
tar -cf - -C /data airgap-bundle/ | split -b 50G - airgap-bundle.tar.part.
# On target: cat airgap-bundle.tar.part.* | tar -xf - -C /data
```

---

## Step 3: Deploy in Airgap Environment

### Verify the Bundle

```bash
# Check all expected assets are present and checksums match
./scripts/airgap/deploy_airgap.sh \
  --bundle-dir /data/airgap-bundle \
  --verify-only
```

### Deploy Assets

```bash
# Deploy assets and load Docker images
./scripts/airgap/deploy_airgap.sh \
  --bundle-dir /data/airgap-bundle \
  --workspace /workspace \
  --load-docker
```

### Deploy with Private Registry

```bash
# Load images, re-tag for private registry, and push
./scripts/airgap/deploy_airgap.sh \
  --bundle-dir /data/airgap-bundle \
  --workspace /workspace \
  --registry harbor.internal:5000/nvidia \
  --load-docker
```

---

## Step 4: Run Customization in Airgap

### Start the Container

```bash
cd deploy/nemotron/customization_recipes

# Using environment file (recommended)
docker compose --env-file /workspace/.env.airgap \
  -f docker-compose.yaml \
  -f docker-compose.airgap.yaml up -d

# Or with explicit paths
AIRGAP_MODELS_DIR=/data/airgap-bundle/models \
AIRGAP_DATASETS_DIR=/data/airgap-bundle/datasets \
AIRGAP_NLTK_DIR=/data/airgap-bundle/nltk_data \
AIRGAP_CONFIGS_DIR=/data/airgap-bundle/configs \
AIRGAP_BENCHMARKS_DIR=/data/airgap-bundle/benchmarks \
  docker compose -f docker-compose.yaml -f docker-compose.airgap.yaml up -d
```

### Enter the Container

```bash
docker compose exec nemotron-orchestrator bash
```

### Run Customization Stages

Inside the container, all model paths point to local pre-downloaded assets:

```bash
# CPT (Continued Pretraining) with local model
nemotron customize cpt \
  model.pretrained_model_name_or_path=/workspace/models/huggingface/nvidia_NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16

# SFT (Supervised Fine-Tuning) with local model and dataset
nemotron customize sft \
  model.pretrained_model_name_or_path=/workspace/models/huggingface/nvidia_NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  dataset.dataset_name=/workspace/datasets/HuggingFaceH4_ultrachat_200k

# Evaluation with local model
nemotron customize eval --step model \
  model_eval.model_name_or_path=/workspace/models/huggingface/nvidia_NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

# Quantization with local calibration data
nemotron customize quantize \
  model.name_or_path=/workspace/models/huggingface/nvidia_NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  quantization.calibration.dataset=/workspace/datasets/cnn_dailymail
```

---

## Step 5: Verification

### Verify Offline Mode

```bash
# Inside the container, verify environment variables
env | grep -E "(OFFLINE|WANDB|HF_HOME|NLTK)"

# Expected output:
# TRANSFORMERS_OFFLINE=1
# HF_DATASETS_OFFLINE=1
# HF_HUB_OFFLINE=1
# HF_HOME=/workspace/models
# NLTK_DATA=/workspace/nltk_data
# WANDB_MODE=offline
```

### Verify Model Access

```bash
# Test that the model loads without network
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(
    '/workspace/models/huggingface/nvidia_NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
    trust_remote_code=True
)
print(f'Tokenizer loaded: vocab_size={tok.vocab_size}')
"
```

### Verify NLTK

```bash
python3 -c "
import nltk
nltk.data.path.insert(0, '/workspace/nltk_data')
from nltk.tokenize import sent_tokenize
print(sent_tokenize('Hello world. This is a test.'))
"
```

---

## Troubleshooting

### Problem: "Connection error" during model loading

**Cause:** `TRANSFORMERS_OFFLINE=1` is not set, or the model path is wrong.

**Fix:**
```bash
# Verify the env var
echo $TRANSFORMERS_OFFLINE  # Should be "1"

# Check the model directory exists and has config.json
ls /workspace/models/huggingface/nvidia_NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/config.json
```

### Problem: NLTK LookupError

**Cause:** `NLTK_DATA` not pointing to the pre-downloaded data.

**Fix:**
```bash
export NLTK_DATA=/workspace/nltk_data
# Or in Python:
import nltk
nltk.data.path.insert(0, '/workspace/nltk_data')
```

### Problem: spaCy model not found

**Cause:** spaCy models need to be loaded by explicit path in airgap mode.

**Fix:**
```python
import spacy
# Instead of: nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("/workspace/models/spacy/en_core_web_sm")
```

### Problem: Docker images fail to load

**Cause:** Corrupt tar file or insufficient disk space.

**Fix:**
```bash
# Verify tar integrity
tar -tf /data/airgap-bundle/docker_images/nvcr.io_nvidia_nemo_25.11.nemotron_3_nano.tar > /dev/null

# Check disk space
df -h /var/lib/docker
```

### Problem: "Dataset not found" errors

**Cause:** HuggingFace datasets library tries to reach the Hub even with `HF_DATASETS_OFFLINE=1` if the dataset was not saved in the expected cache format.

**Fix:**
```python
from datasets import load_from_disk
# Instead of: ds = load_dataset("cais/mmlu")
ds = load_from_disk("/workspace/datasets/cais_mmlu")
```

### Problem: FastText language ID model not found

**Cause:** `lid_model_path` not set in the config.

**Fix:**
```bash
# Set in config override or environment
export FASTTEXT_LID_MODEL=/workspace/models/fasttext/lid.176.bin

# Or pass as config override:
nemotron customize data-prep \
  lid_model_path=/workspace/models/fasttext/lid.176.bin
```

### Problem: BYOB/SDG stages fail (no API access)

**Cause:** BYOB and SDG stages require LLM inference. In airgap mode, cloud APIs are unavailable.

**Fix:** Deploy a local NIM instance for inference:
```bash
# Load the NIM image
docker load -i /data/airgap-bundle/docker_images/nim_mistral.tar

# Run NIM locally
docker run --gpus all -p 8000:8000 \
  -v /workspace/models:/models \
  nvcr.io/nim/mistralai/mistral-7b-instruct-v0.3:1.12.0

# Point BYOB/SDG to local NIM
nemotron customize byob \
  generation_model_config.provider=local \
  generation_model_config.model=http://localhost:8000/v1
```

### Problem: W&B sync fails

**Cause:** `WANDB_MODE=offline` stores runs locally. They need to be synced when connectivity is restored.

**Fix:**
```bash
# When connectivity is available:
wandb sync /workspace/results/wandb/offline-*
```

---

## Airgap Limitations

| Feature | Airgap Status | Workaround |
|---------|--------------|------------|
| CPT (Continued Pretraining) | Fully supported | Local model + data |
| SFT (Supervised Fine-Tuning) | Fully supported | Local model + data |
| RL (DPO/GRPO) | Fully supported | Local model + data |
| Data Prep (acquire/filter) | Fully supported | Pre-downloaded corpora |
| Language ID | Fully supported | Local FastText model |
| Domain Classification | Fully supported | Local classifier model |
| SDG (Synthetic Data Gen) | Requires local NIM | Deploy NIM in airgap |
| BYOB (Benchmark Gen) | Requires local NIM | Deploy NIM in airgap |
| Evaluation (model) | Fully supported | Local model + benchmarks |
| Evaluation (data quality) | Requires local NIM for LLM-based eval | Deploy NIM or use rule-based only |
| Translation (Google/AWS) | Not available | Use local NMT model (NLLB) |
| Quantization | Fully supported | Local model + calibration data |
| W&B Tracking | Offline mode | Sync when connectivity restored |

---

## Security Considerations

- All assets are downloaded over HTTPS from official sources (HuggingFace, Meta, NVIDIA NGC)
- The `manifest.json` contains SHA-256 checksums for integrity verification
- Docker images are loaded from signed tars; verify with `docker trust inspect` if needed
- No credentials are stored in the bundle; HF tokens are only used during download
- The docker-compose.airgap.yaml mounts data volumes as read-only (`:ro`) where possible
