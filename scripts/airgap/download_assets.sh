#!/usr/bin/env bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
# Airgap Pre-Download Script for Nemotron Customization Recipes
#
# Downloads all models, datasets, NLP assets, and optionally Docker images
# required to run the Nemotron customization pipeline without internet access.
#
# Usage:
#   ./download_assets.sh \
#     --output-dir /path/to/airgap-bundle \
#     --model-family nemotron-nano \
#     --include-nim \
#     --include-benchmarks \
#     --hf-token $HF_TOKEN
#
# The output bundle can then be transferred to the airgap environment
# and deployed using deploy_airgap.sh.
# =============================================================================

set -euo pipefail

# ---- Constants ---------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

# HuggingFace models per family
declare -A HF_MODELS_NANO=(
    ["base"]="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
    ["instruct"]="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
)

declare -A HF_MODELS_SUPER=(
    ["base"]="nvidia/NVIDIA-Nemotron-3-Super-49B-v1"
    ["instruct"]="nvidia/NVIDIA-Nemotron-3-Super-49B-Instruct-v1"
)

# Shared models (needed regardless of family)
SHARED_MODELS=(
    "nvidia/multilingual-domain-classifier"
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Chat template model (for data acquisition)
CHAT_TEMPLATE_MODEL="mistralai/Mistral-Small-24B-Instruct-2501"

# HuggingFace datasets
HF_DATASETS=(
    "cais/mmlu"
    "HuggingFaceH4/ultrachat_200k"
    "nvidia/Nemotron-Pretraining-Dataset-sample"
)

# Calibration dataset for quantization
CALIBRATION_DATASETS=(
    "cnn_dailymail"
)

# NLTK data packages
NLTK_PACKAGES=(
    "punkt"
    "punkt_tab"
    "stopwords"
    "averaged_perceptron_tagger_eng"
)

# spaCy models
SPACY_MODELS=(
    "en_core_web_sm"
    "xx_sent_ud_sm"
)

# FastText language ID model
FASTTEXT_LID_URL="https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

# Docker images
DOCKER_IMAGE_CURATOR="nvcr.io/nvidia/nemo-curator:26.02"
DOCKER_IMAGE_TRAINER="nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
DOCKER_IMAGE_NIM="nvcr.io/nim/mistralai/mistral-7b-instruct-v0.3:1.12.0"
DOCKER_IMAGE_CUSTOMIZE="nemotron-customize:latest"

# Eval benchmarks
NEMO_SKILLS_REPO="https://github.com/NVIDIA/NeMo-Skills.git"
NEMO_SKILLS_COMMIT="63cf71f4706b9c4ad959be7563ee9b88864da1eb"
GORILLA_REPO="https://github.com/ShishirPatil/gorilla.git"
GORILLA_COMMIT="d2177992bbba9aa228b53c0645bf8f5613a5a7c6"

# ---- Globals -----------------------------------------------------------------

OUTPUT_DIR=""
MODEL_FAMILY="nemotron-nano"
INCLUDE_NIM=false
INCLUDE_BENCHMARKS=false
INCLUDE_DOCKER=false
INCLUDE_CHAT_TEMPLATE_MODEL=false
HF_TOKEN=""
DRY_RUN=false
SKIP_EXISTING=true

# ---- Logging -----------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC}  $(date +%H:%M:%S) $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date +%H:%M:%S) $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date +%H:%M:%S) $*" >&2; }
log_step()  { echo -e "${BLUE}[STEP]${NC}  $(date +%H:%M:%S) ====== $* ======"; }

# ---- Usage -------------------------------------------------------------------

usage() {
    cat <<'USAGE'
Usage: download_assets.sh [OPTIONS]

Pre-download all assets required for airgap deployment of Nemotron
customization recipes.

Options:
  --output-dir DIR          Output directory for the airgap bundle (required)
  --model-family FAMILY     Model family: nemotron-nano, nemotron-super, all
                            (default: nemotron-nano)
  --include-nim             Download NIM model image for local inference
  --include-benchmarks      Download evaluation benchmark repos
  --include-docker          Save Docker images as tar files
  --include-chat-model      Download chat template model (Mistral-Small-24B)
  --hf-token TOKEN          HuggingFace API token (or set HF_TOKEN env var)
  --skip-existing           Skip assets that already exist (default: true)
  --no-skip-existing        Re-download all assets even if they exist
  --dry-run                 Show what would be downloaded without downloading
  -h, --help                Show this help message

Examples:
  # Minimal download (Nano models + core assets)
  ./download_assets.sh --output-dir ./airgap-bundle --hf-token $HF_TOKEN

  # Full download with NIM and Docker images
  ./download_assets.sh \
    --output-dir ./airgap-bundle \
    --model-family all \
    --include-nim \
    --include-benchmarks \
    --include-docker \
    --include-chat-model \
    --hf-token $HF_TOKEN

  # Dry run to see what would be downloaded
  ./download_assets.sh --output-dir ./airgap-bundle --dry-run
USAGE
    exit 0
}

# ---- Argument Parsing --------------------------------------------------------

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --output-dir)
                OUTPUT_DIR="$2"; shift 2 ;;
            --model-family)
                MODEL_FAMILY="$2"; shift 2 ;;
            --include-nim)
                INCLUDE_NIM=true; shift ;;
            --include-benchmarks)
                INCLUDE_BENCHMARKS=true; shift ;;
            --include-docker)
                INCLUDE_DOCKER=true; shift ;;
            --include-chat-model)
                INCLUDE_CHAT_TEMPLATE_MODEL=true; shift ;;
            --hf-token)
                HF_TOKEN="$2"; shift 2 ;;
            --skip-existing)
                SKIP_EXISTING=true; shift ;;
            --no-skip-existing)
                SKIP_EXISTING=false; shift ;;
            --dry-run)
                DRY_RUN=true; shift ;;
            -h|--help)
                usage ;;
            *)
                log_error "Unknown option: $1"
                usage ;;
        esac
    done

    # Validate required args
    if [[ -z "$OUTPUT_DIR" ]]; then
        log_error "--output-dir is required"
        usage
    fi

    # Accept HF_TOKEN from environment if not passed as argument
    if [[ -z "$HF_TOKEN" ]]; then
        HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
    fi

    # Validate model family
    case "$MODEL_FAMILY" in
        nemotron-nano|nemotron-super|all) ;;
        *)
            log_error "Invalid --model-family: $MODEL_FAMILY (expected: nemotron-nano, nemotron-super, all)"
            exit 1 ;;
    esac
}

# ---- Prerequisite Checks ----------------------------------------------------

check_prerequisites() {
    log_step "Checking prerequisites"

    local missing=()

    if ! command -v huggingface-cli &>/dev/null; then
        missing+=("huggingface-cli (pip install huggingface_hub)")
    fi

    if ! command -v python3 &>/dev/null; then
        missing+=("python3")
    fi

    if ! command -v wget &>/dev/null && ! command -v curl &>/dev/null; then
        missing+=("wget or curl")
    fi

    if [[ "$INCLUDE_DOCKER" == true ]] && ! command -v docker &>/dev/null; then
        missing+=("docker (needed for --include-docker)")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing prerequisites:"
        for m in "${missing[@]}"; do
            log_error "  - $m"
        done
        exit 1
    fi

    # Log into HuggingFace if token is provided
    if [[ -n "$HF_TOKEN" ]]; then
        log_info "HuggingFace token provided; setting for downloads"
        export HF_TOKEN
        export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    else
        log_warn "No HuggingFace token provided. Some models may fail to download."
        log_warn "Set --hf-token or HF_TOKEN environment variable."
    fi

    log_info "All prerequisites satisfied"
}

# ---- Directory Setup ---------------------------------------------------------

setup_directories() {
    log_step "Setting up output directories"

    local dirs=(
        "$OUTPUT_DIR"
        "$OUTPUT_DIR/models/huggingface"
        "$OUTPUT_DIR/models/fasttext"
        "$OUTPUT_DIR/models/spacy"
        "$OUTPUT_DIR/models/sentence-transformers"
        "$OUTPUT_DIR/datasets"
        "$OUTPUT_DIR/nltk_data"
        "$OUTPUT_DIR/docker_images"
        "$OUTPUT_DIR/benchmarks"
        "$OUTPUT_DIR/configs"
    )

    for dir in "${dirs[@]}"; do
        if [[ "$DRY_RUN" == true ]]; then
            log_info "[DRY RUN] Would create: $dir"
        else
            mkdir -p "$dir"
        fi
    done
}

# ---- Download Functions ------------------------------------------------------

download_hf_model() {
    local model_id="$1"
    local target_dir="$OUTPUT_DIR/models/huggingface/$(echo "$model_id" | tr '/' '_')"

    if [[ "$SKIP_EXISTING" == true ]] && [[ -d "$target_dir" ]] && [[ -f "$target_dir/config.json" || -f "$target_dir/tokenizer.json" ]]; then
        log_info "Skipping (already exists): $model_id"
        return 0
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would download HF model: $model_id -> $target_dir"
        return 0
    fi

    log_info "Downloading HuggingFace model: $model_id"
    mkdir -p "$target_dir"

    local hf_args=("download" "$model_id" "--local-dir" "$target_dir")
    if [[ -n "$HF_TOKEN" ]]; then
        hf_args+=("--token" "$HF_TOKEN")
    fi

    if huggingface-cli "${hf_args[@]}"; then
        log_info "Downloaded: $model_id -> $target_dir"
    else
        log_error "Failed to download: $model_id"
        return 1
    fi
}

download_hf_dataset() {
    local dataset_id="$1"
    local target_dir="$OUTPUT_DIR/datasets/$(echo "$dataset_id" | tr '/' '_')"

    if [[ "$SKIP_EXISTING" == true ]] && [[ -d "$target_dir" ]] && [[ "$(ls -A "$target_dir" 2>/dev/null)" ]]; then
        log_info "Skipping (already exists): $dataset_id"
        return 0
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would download HF dataset: $dataset_id -> $target_dir"
        return 0
    fi

    log_info "Downloading HuggingFace dataset: $dataset_id"
    mkdir -p "$target_dir"

    # Use Python datasets library for reliable dataset download
    python3 -c "
from datasets import load_dataset
import os
ds = load_dataset('${dataset_id}', trust_remote_code=True)
ds.save_to_disk('${target_dir}')
print(f'Saved {dataset_id} to ${target_dir}')
" 2>&1 || {
        # Fallback: use huggingface-cli
        log_warn "Python datasets download failed; trying huggingface-cli"
        local hf_args=("download" "$dataset_id" "--repo-type" "dataset" "--local-dir" "$target_dir")
        if [[ -n "$HF_TOKEN" ]]; then
            hf_args+=("--token" "$HF_TOKEN")
        fi
        huggingface-cli "${hf_args[@]}" || {
            log_error "Failed to download dataset: $dataset_id"
            return 1
        }
    }

    log_info "Downloaded dataset: $dataset_id -> $target_dir"
}

download_fasttext_lid() {
    local target_file="$OUTPUT_DIR/models/fasttext/lid.176.bin"

    if [[ "$SKIP_EXISTING" == true ]] && [[ -f "$target_file" ]]; then
        log_info "Skipping (already exists): FastText lid.176.bin"
        return 0
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would download FastText LID model -> $target_file"
        return 0
    fi

    log_info "Downloading FastText language identification model (lid.176.bin)"
    mkdir -p "$(dirname "$target_file")"

    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$target_file" "$FASTTEXT_LID_URL"
    else
        curl -L --progress-bar -o "$target_file" "$FASTTEXT_LID_URL"
    fi

    log_info "Downloaded: lid.176.bin -> $target_file"
}

download_nltk_data() {
    local target_dir="$OUTPUT_DIR/nltk_data"

    if [[ "$DRY_RUN" == true ]]; then
        for pkg in "${NLTK_PACKAGES[@]}"; do
            log_info "[DRY RUN] Would download NLTK package: $pkg"
        done
        return 0
    fi

    log_info "Downloading NLTK data packages"
    mkdir -p "$target_dir"

    python3 -c "
import nltk
import os
target = '${target_dir}'
os.makedirs(target, exist_ok=True)
packages = $(printf "'%s'," "${NLTK_PACKAGES[@]}" | sed 's/,$//')
for pkg in [${packages}]:
    print(f'Downloading NLTK: {pkg}')
    nltk.download(pkg, download_dir=target)
print('NLTK downloads complete')
"
    log_info "NLTK data downloaded to: $target_dir"
}

download_spacy_models() {
    local target_dir="$OUTPUT_DIR/models/spacy"

    if [[ "$DRY_RUN" == true ]]; then
        for model in "${SPACY_MODELS[@]}"; do
            log_info "[DRY RUN] Would download spaCy model: $model"
        done
        return 0
    fi

    log_info "Downloading spaCy models"
    mkdir -p "$target_dir"

    for model in "${SPACY_MODELS[@]}"; do
        local model_dir="$target_dir/$model"
        if [[ "$SKIP_EXISTING" == true ]] && [[ -d "$model_dir" ]] && [[ "$(ls -A "$model_dir" 2>/dev/null)" ]]; then
            log_info "Skipping (already exists): spaCy $model"
            continue
        fi

        log_info "Downloading spaCy model: $model"
        # Download and then copy the installed model to our target dir
        python3 -m spacy download "$model"
        python3 -c "
import spacy
import shutil
nlp = spacy.load('${model}')
model_path = nlp.path
target = '${model_dir}'
print(f'Copying {model_path} -> {target}')
shutil.copytree(str(model_path), target, dirs_exist_ok=True)
"
        log_info "Downloaded spaCy model: $model -> $model_dir"
    done
}

save_docker_images() {
    local images=("$DOCKER_IMAGE_TRAINER")

    if [[ "$INCLUDE_NIM" == true ]]; then
        images+=("$DOCKER_IMAGE_NIM")
    fi

    # Always include curator
    images+=("$DOCKER_IMAGE_CURATOR")

    if [[ "$DRY_RUN" == true ]]; then
        for img in "${images[@]}"; do
            local basename
            basename="$(echo "$img" | tr '/:' '_').tar"
            log_info "[DRY RUN] Would save Docker image: $img -> $OUTPUT_DIR/docker_images/$basename"
        done
        return 0
    fi

    log_info "Saving Docker images as tar files"

    for img in "${images[@]}"; do
        local basename
        basename="$(echo "$img" | tr '/:' '_').tar"
        local target="$OUTPUT_DIR/docker_images/$basename"

        if [[ "$SKIP_EXISTING" == true ]] && [[ -f "$target" ]]; then
            log_info "Skipping (already exists): $img"
            continue
        fi

        log_info "Pulling Docker image: $img"
        docker pull "$img" || {
            log_error "Failed to pull: $img"
            continue
        }

        log_info "Saving Docker image: $img -> $target"
        docker save "$img" -o "$target" || {
            log_error "Failed to save: $img"
            continue
        }

        log_info "Saved: $img -> $target"
    done
}

download_benchmarks() {
    local benchmarks_dir="$OUTPUT_DIR/benchmarks"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would clone NeMo-Skills repo"
        log_info "[DRY RUN] Would clone Gorilla repo"
        return 0
    fi

    log_info "Downloading evaluation benchmark repositories"

    # NeMo-Skills
    local nemo_skills_dir="$benchmarks_dir/NeMo-Skills"
    if [[ "$SKIP_EXISTING" == true ]] && [[ -d "$nemo_skills_dir/.git" ]]; then
        log_info "Skipping (already exists): NeMo-Skills"
    else
        log_info "Cloning NeMo-Skills"
        rm -rf "$nemo_skills_dir"
        git clone "$NEMO_SKILLS_REPO" "$nemo_skills_dir"
        (cd "$nemo_skills_dir" && git checkout "$NEMO_SKILLS_COMMIT")
        log_info "Cloned NeMo-Skills at commit $NEMO_SKILLS_COMMIT"
    fi

    # Gorilla (Berkeley Function Call Leaderboard)
    local gorilla_dir="$benchmarks_dir/gorilla"
    if [[ "$SKIP_EXISTING" == true ]] && [[ -d "$gorilla_dir/.git" ]]; then
        log_info "Skipping (already exists): Gorilla"
    else
        log_info "Cloning Gorilla"
        rm -rf "$gorilla_dir"
        git clone "$GORILLA_REPO" "$gorilla_dir"
        (cd "$gorilla_dir" && git checkout "$GORILLA_COMMIT")
        log_info "Cloned Gorilla at commit $GORILLA_COMMIT"
    fi

    # MMLU Pro benchmark data (used by eval stage)
    local mmlu_pro_dir="$benchmarks_dir/mmlu_pro"
    if [[ "$SKIP_EXISTING" == true ]] && [[ -d "$mmlu_pro_dir" ]] && [[ "$(ls -A "$mmlu_pro_dir" 2>/dev/null)" ]]; then
        log_info "Skipping (already exists): MMLU-Pro benchmark data"
    else
        log_info "Downloading MMLU-Pro benchmark data"
        mkdir -p "$mmlu_pro_dir"
        python3 -c "
from datasets import load_dataset
ds = load_dataset('TIGER-Lab/MMLU-Pro', trust_remote_code=True)
ds.save_to_disk('${mmlu_pro_dir}')
print('MMLU-Pro downloaded')
" || log_warn "Could not download MMLU-Pro; eval stage may need manual setup"
    fi
}

# ---- Config Generation -------------------------------------------------------

generate_airgap_env_toml() {
    local target="$OUTPUT_DIR/configs/airgap-env.toml"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would generate: $target"
        return 0
    fi

    log_info "Generating airgap environment config: $target"

    cat > "$target" <<'TOML'
# =============================================================================
# Airgap Environment Configuration
#
# Source this into your shell or reference from docker-compose.
# All paths assume the airgap bundle is mounted at /workspace/airgap-bundle.
# =============================================================================

[env]
TRANSFORMERS_OFFLINE = "1"
HF_DATASETS_OFFLINE = "1"
HF_HUB_OFFLINE = "1"
HF_HOME = "/workspace/models"
HF_DATASETS_CACHE = "/workspace/datasets"
NLTK_DATA = "/workspace/nltk_data"
WANDB_MODE = "offline"
WANDB_DISABLED = "true"
TOKENIZERS_PARALLELISM = "false"

# FastText language ID model path
FASTTEXT_LID_MODEL = "/workspace/models/fasttext/lid.176.bin"

# spaCy model paths (set via spacy.load with explicit path)
SPACY_EN_CORE_WEB_SM = "/workspace/models/spacy/en_core_web_sm"
SPACY_XX_SENT_UD_SM = "/workspace/models/spacy/xx_sent_ud_sm"

# Disable telemetry
DO_NOT_TRACK = "1"
ANONYMIZED_TELEMETRY = "false"

[executor]
# Default executor for airgap: local (no remote calls)
type = "local"
TOML

    log_info "Generated: $target"
}

generate_airgap_overrides_yaml() {
    local target="$OUTPUT_DIR/configs/airgap-overrides.yaml"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would generate: $target"
        return 0
    fi

    log_info "Generating airgap config overrides: $target"

    # Determine model paths based on family
    local base_model_path=""
    local instruct_model_path=""

    case "$MODEL_FAMILY" in
        nemotron-nano|all)
            base_model_path="/workspace/models/huggingface/nvidia_NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
            instruct_model_path="/workspace/models/huggingface/nvidia_NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
            ;;
        nemotron-super)
            base_model_path="/workspace/models/huggingface/nvidia_NVIDIA-Nemotron-3-Super-49B-v1"
            instruct_model_path="/workspace/models/huggingface/nvidia_NVIDIA-Nemotron-3-Super-49B-Instruct-v1"
            ;;
    esac

    cat > "$target" <<YAML
# =============================================================================
# Airgap Config Overrides for Nemotron Customization Recipes
#
# Apply these overrides to point all model/data references to local paths.
#
# Usage with OmegaConf:
#   nemotron customize cpt --config-overrides /workspace/configs/airgap-overrides.yaml
#
# Or merge manually:
#   python -c "from omegaconf import OmegaConf; ..."
# =============================================================================

# --- Model paths (local filesystem instead of HuggingFace Hub) ---
model:
  pretrained_model_name_or_path: ${base_model_path}

dataset:
  tokenizer:
    pretrained_model_name_or_path: ${base_model_path}

# --- Evaluator model path ---
model_eval:
  model_name_or_path: ${instruct_model_path}

# --- Domain classifier (local path) ---
domain_classifier:
  model: /workspace/models/huggingface/nvidia_multilingual-domain-classifier

# --- Sentence transformer (for BYOB semantic dedup/coverage/outlier) ---
semantic_deduplication_config:
  model_identifier: /workspace/models/huggingface/sentence-transformers_all-MiniLM-L6-v2

coverage_check_config:
  model_identifier: /workspace/models/huggingface/sentence-transformers_all-MiniLM-L6-v2

semantic_outlier_detection_config:
  model_identifier: /workspace/models/huggingface/sentence-transformers_all-MiniLM-L6-v2

# --- FastText language ID ---
lid_model_path: /workspace/models/fasttext/lid.176.bin

# --- Tokenizer for SFT data prep ---
tokenizer_model: ${base_model_path}

# --- Calibration dataset for quantization ---
quantization:
  calibration:
    dataset: /workspace/datasets/cnn_dailymail

# --- W&B disabled ---
wandb:
  entity: null
  project: nemotron-customization-airgap
YAML

    log_info "Generated: $target"
}

# ---- Manifest Generation -----------------------------------------------------

generate_manifest() {
    local manifest_file="$OUTPUT_DIR/manifest.json"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would generate manifest: $manifest_file"
        return 0
    fi

    log_step "Generating asset manifest"

    python3 <<PYEOF
import json
import os
import hashlib
from pathlib import Path
from datetime import datetime, timezone

bundle_dir = "${OUTPUT_DIR}"
manifest = {
    "version": "1.0",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "model_family": "${MODEL_FAMILY}",
    "include_nim": ${INCLUDE_NIM},
    "include_benchmarks": ${INCLUDE_BENCHMARKS},
    "include_docker": ${INCLUDE_DOCKER},
    "assets": []
}

def sha256_file(path, chunk_size=8192):
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except (OSError, PermissionError):
        return "error"

def dir_size(path):
    """Compute total size of a directory."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total

def human_size(num_bytes):
    """Convert bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"

# Walk the bundle and catalog assets
for category_dir in sorted(Path(bundle_dir).iterdir()):
    if not category_dir.is_dir():
        continue
    category = category_dir.name

    for asset_path in sorted(category_dir.rglob("*")):
        if asset_path.is_file():
            rel_path = str(asset_path.relative_to(bundle_dir))
            size = asset_path.stat().st_size

            # Only compute checksums for key files (configs, small models)
            # Skip checksums for very large files (> 1GB) to save time
            checksum = ""
            if size < 1_073_741_824:  # 1 GB
                checksum = sha256_file(str(asset_path))
            else:
                checksum = "skipped-large-file"

            manifest["assets"].append({
                "path": rel_path,
                "category": category,
                "size_bytes": size,
                "size_human": human_size(size),
                "sha256": checksum,
            })

# Summary stats
total_size = sum(a["size_bytes"] for a in manifest["assets"])
manifest["summary"] = {
    "total_assets": len(manifest["assets"]),
    "total_size_bytes": total_size,
    "total_size_human": human_size(total_size),
    "categories": {}
}

for asset in manifest["assets"]:
    cat = asset["category"]
    if cat not in manifest["summary"]["categories"]:
        manifest["summary"]["categories"][cat] = {"count": 0, "size_bytes": 0}
    manifest["summary"]["categories"][cat]["count"] += 1
    manifest["summary"]["categories"][cat]["size_bytes"] += asset["size_bytes"]

for cat in manifest["summary"]["categories"]:
    s = manifest["summary"]["categories"][cat]["size_bytes"]
    manifest["summary"]["categories"][cat]["size_human"] = human_size(s)

manifest_path = os.path.join(bundle_dir, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Manifest written to {manifest_path}")
print(f"Total assets: {manifest['summary']['total_assets']}")
print(f"Total size: {manifest['summary']['total_size_human']}")
PYEOF

    log_info "Manifest generated: $manifest_file"
}

# ---- Main Pipeline -----------------------------------------------------------

download_models() {
    log_step "Downloading HuggingFace models"

    # Determine which model family to download
    case "$MODEL_FAMILY" in
        nemotron-nano)
            for key in "${!HF_MODELS_NANO[@]}"; do
                download_hf_model "${HF_MODELS_NANO[$key]}"
            done
            ;;
        nemotron-super)
            for key in "${!HF_MODELS_SUPER[@]}"; do
                download_hf_model "${HF_MODELS_SUPER[$key]}"
            done
            ;;
        all)
            for key in "${!HF_MODELS_NANO[@]}"; do
                download_hf_model "${HF_MODELS_NANO[$key]}"
            done
            for key in "${!HF_MODELS_SUPER[@]}"; do
                download_hf_model "${HF_MODELS_SUPER[$key]}"
            done
            ;;
    esac

    # Always download shared models
    for model in "${SHARED_MODELS[@]}"; do
        download_hf_model "$model"
    done

    # Optionally download the chat template model
    if [[ "$INCLUDE_CHAT_TEMPLATE_MODEL" == true ]]; then
        download_hf_model "$CHAT_TEMPLATE_MODEL"
    fi
}

download_datasets() {
    log_step "Downloading HuggingFace datasets"

    for ds in "${HF_DATASETS[@]}"; do
        download_hf_dataset "$ds"
    done

    for ds in "${CALIBRATION_DATASETS[@]}"; do
        download_hf_dataset "$ds"
    done
}

download_nlp_assets() {
    log_step "Downloading NLP assets (FastText, NLTK, spaCy)"

    download_fasttext_lid
    download_nltk_data
    download_spacy_models
}

# ---- Summary -----------------------------------------------------------------

print_summary() {
    log_step "Download Summary"

    echo ""
    echo "============================================================"
    echo "  Airgap Bundle: $OUTPUT_DIR"
    echo "  Model Family:  $MODEL_FAMILY"
    echo "  Include NIM:   $INCLUDE_NIM"
    echo "  Include Bench: $INCLUDE_BENCHMARKS"
    echo "  Include Docker: $INCLUDE_DOCKER"
    echo "============================================================"
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        echo "  ** DRY RUN -- no files were downloaded **"
        echo ""
        return 0
    fi

    # Print directory sizes
    echo "Directory sizes:"
    if command -v du &>/dev/null; then
        du -sh "$OUTPUT_DIR"/*/  2>/dev/null || true
    fi
    echo ""

    echo "Next steps:"
    echo "  1. Transfer the bundle to your airgap environment:"
    echo "       rsync -avP $OUTPUT_DIR/ airgap-host:/path/to/airgap-bundle/"
    echo ""
    echo "  2. On the airgap host, run the deployment script:"
    echo "       ./deploy_airgap.sh --bundle-dir /path/to/airgap-bundle"
    echo ""
    echo "  3. Start the customization container:"
    echo "       cd deploy/nemotron/customization_recipes"
    echo "       docker compose -f docker-compose.yaml -f docker-compose.airgap.yaml up -d"
    echo ""
}

# ---- Entry Point -------------------------------------------------------------

main() {
    parse_args "$@"

    echo ""
    echo "============================================================"
    echo "  Nemotron Airgap Asset Download"
    echo "  Timestamp: $TIMESTAMP"
    echo "============================================================"
    echo ""

    check_prerequisites
    setup_directories

    # Core downloads (always run)
    download_models
    download_datasets
    download_nlp_assets

    # Optional: Docker images
    if [[ "$INCLUDE_DOCKER" == true ]]; then
        log_step "Saving Docker images"
        save_docker_images
    fi

    # Optional: Evaluation benchmarks
    if [[ "$INCLUDE_BENCHMARKS" == true ]]; then
        log_step "Downloading evaluation benchmarks"
        download_benchmarks
    fi

    # Generate configs
    log_step "Generating airgap configuration files"
    generate_airgap_env_toml
    generate_airgap_overrides_yaml

    # Generate manifest
    generate_manifest

    # Summary
    print_summary

    log_info "Airgap asset download complete!"
}

main "$@"
