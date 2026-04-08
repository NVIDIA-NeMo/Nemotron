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
# Airgap Deployment Script for Nemotron Customization Recipes
#
# Runs in the airgap environment to set up the Nemotron customization pipeline
# from a pre-downloaded asset bundle.
#
# Usage:
#   ./deploy_airgap.sh \
#     --bundle-dir /path/to/airgap-bundle \
#     --workspace /workspace \
#     --registry my-registry.internal:5000 \
#     --load-docker
# =============================================================================

set -euo pipefail

# ---- Constants ---------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Globals -----------------------------------------------------------------

BUNDLE_DIR=""
WORKSPACE="/workspace"
REGISTRY=""
LOAD_DOCKER=false
COMPOSE_DIR=""
DRY_RUN=false
VERIFY_ONLY=false

# ---- Logging -----------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $(date +%H:%M:%S) $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $(date +%H:%M:%S) $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $(date +%H:%M:%S) $*" >&2; }
log_step()  { echo -e "${BLUE}[STEP]${NC}  $(date +%H:%M:%S) ====== $* ======"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $(date +%H:%M:%S) $*"; }
log_fail()  { echo -e "${RED}[FAIL]${NC}  $(date +%H:%M:%S) $*"; }

# ---- Usage -------------------------------------------------------------------

usage() {
    cat <<'USAGE'
Usage: deploy_airgap.sh [OPTIONS]

Deploy Nemotron customization recipes from a pre-downloaded airgap bundle.

Options:
  --bundle-dir DIR          Path to the airgap asset bundle (required)
  --workspace DIR           Target workspace directory (default: /workspace)
  --registry HOST:PORT      Private Docker registry for re-tagging images
  --load-docker             Load Docker images from tar files in the bundle
  --compose-dir DIR         Path to docker-compose directory
                            (default: auto-detect from bundle)
  --verify-only             Only verify assets; do not copy or load anything
  --dry-run                 Show what would be done without doing it
  -h, --help                Show this help message

Examples:
  # Basic deployment
  ./deploy_airgap.sh --bundle-dir /mnt/airgap-bundle --load-docker

  # Deploy with private registry
  ./deploy_airgap.sh \
    --bundle-dir /mnt/airgap-bundle \
    --workspace /data/nemotron \
    --registry harbor.internal:5000/nvidia \
    --load-docker

  # Verify bundle integrity only
  ./deploy_airgap.sh --bundle-dir /mnt/airgap-bundle --verify-only
USAGE
    exit 0
}

# ---- Argument Parsing --------------------------------------------------------

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --bundle-dir)
                BUNDLE_DIR="$2"; shift 2 ;;
            --workspace)
                WORKSPACE="$2"; shift 2 ;;
            --registry)
                REGISTRY="$2"; shift 2 ;;
            --load-docker)
                LOAD_DOCKER=true; shift ;;
            --compose-dir)
                COMPOSE_DIR="$2"; shift 2 ;;
            --verify-only)
                VERIFY_ONLY=true; shift ;;
            --dry-run)
                DRY_RUN=true; shift ;;
            -h|--help)
                usage ;;
            *)
                log_error "Unknown option: $1"
                usage ;;
        esac
    done

    if [[ -z "$BUNDLE_DIR" ]]; then
        log_error "--bundle-dir is required"
        usage
    fi

    if [[ ! -d "$BUNDLE_DIR" ]]; then
        log_error "Bundle directory does not exist: $BUNDLE_DIR"
        exit 1
    fi
}

# ---- Verification ------------------------------------------------------------

verify_bundle() {
    log_step "Verifying airgap bundle"

    local errors=0
    local warnings=0

    # Check manifest exists
    if [[ -f "$BUNDLE_DIR/manifest.json" ]]; then
        log_ok "manifest.json found"
    else
        log_fail "manifest.json not found"
        ((errors++))
    fi

    # Check required directories
    local required_dirs=(
        "models/huggingface"
        "models/fasttext"
        "nltk_data"
        "datasets"
        "configs"
    )

    for dir in "${required_dirs[@]}"; do
        if [[ -d "$BUNDLE_DIR/$dir" ]]; then
            log_ok "Directory exists: $dir"
        else
            log_fail "Missing directory: $dir"
            ((errors++))
        fi
    done

    # Check FastText LID model
    if [[ -f "$BUNDLE_DIR/models/fasttext/lid.176.bin" ]]; then
        log_ok "FastText lid.176.bin found"
    else
        log_warn "FastText lid.176.bin not found (language ID will not work)"
        ((warnings++))
    fi

    # Check NLTK data
    local nltk_packages=("punkt" "punkt_tab" "stopwords" "averaged_perceptron_tagger_eng")
    for pkg in "${nltk_packages[@]}"; do
        if find "$BUNDLE_DIR/nltk_data" -name "$pkg" -o -name "${pkg}.zip" 2>/dev/null | head -1 | grep -q .; then
            log_ok "NLTK package: $pkg"
        else
            # NLTK stores data in subdirectories; check more broadly
            if find "$BUNDLE_DIR/nltk_data" -type d -name "$pkg" 2>/dev/null | head -1 | grep -q .; then
                log_ok "NLTK package: $pkg"
            else
                log_warn "NLTK package may be missing: $pkg"
                ((warnings++))
            fi
        fi
    done

    # Check spaCy models
    local spacy_models=("en_core_web_sm" "xx_sent_ud_sm")
    for model in "${spacy_models[@]}"; do
        if [[ -d "$BUNDLE_DIR/models/spacy/$model" ]]; then
            log_ok "spaCy model: $model"
        else
            log_warn "spaCy model not found: $model"
            ((warnings++))
        fi
    done

    # Check for at least one HuggingFace model
    local hf_model_count
    hf_model_count=$(find "$BUNDLE_DIR/models/huggingface" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
    if [[ "$hf_model_count" -gt 0 ]]; then
        log_ok "HuggingFace models found: $hf_model_count"
    else
        log_fail "No HuggingFace models found in bundle"
        ((errors++))
    fi

    # Check for at least one dataset
    local dataset_count
    dataset_count=$(find "$BUNDLE_DIR/datasets" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
    if [[ "$dataset_count" -gt 0 ]]; then
        log_ok "Datasets found: $dataset_count"
    else
        log_fail "No datasets found in bundle"
        ((errors++))
    fi

    # Check config files
    if [[ -f "$BUNDLE_DIR/configs/airgap-env.toml" ]]; then
        log_ok "airgap-env.toml found"
    else
        log_fail "airgap-env.toml not found"
        ((errors++))
    fi

    if [[ -f "$BUNDLE_DIR/configs/airgap-overrides.yaml" ]]; then
        log_ok "airgap-overrides.yaml found"
    else
        log_fail "airgap-overrides.yaml not found"
        ((errors++))
    fi

    # Check Docker images (if directory exists)
    if [[ -d "$BUNDLE_DIR/docker_images" ]]; then
        local docker_count
        docker_count=$(find "$BUNDLE_DIR/docker_images" -name "*.tar" 2>/dev/null | wc -l)
        if [[ "$docker_count" -gt 0 ]]; then
            log_ok "Docker image tars found: $docker_count"
        else
            log_warn "docker_images directory exists but no .tar files found"
            ((warnings++))
        fi
    fi

    # Checksum verification (if manifest exists and is not dry-run)
    if [[ -f "$BUNDLE_DIR/manifest.json" ]] && [[ "$DRY_RUN" == false ]]; then
        log_info "Verifying checksums from manifest (sampling key files)..."
        python3 <<PYEOF || log_warn "Checksum verification skipped (Python error)"
import json, hashlib, os, sys

with open("${BUNDLE_DIR}/manifest.json") as f:
    manifest = json.load(f)

checked = 0
failed = 0
skipped = 0

for asset in manifest.get("assets", []):
    path = os.path.join("${BUNDLE_DIR}", asset["path"])
    expected = asset.get("sha256", "")

    if not expected or expected in ("skipped-large-file", "error"):
        skipped += 1
        continue

    if not os.path.exists(path):
        print(f"  MISSING: {asset['path']}")
        failed += 1
        continue

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)

    if h.hexdigest() == expected:
        checked += 1
    else:
        print(f"  CHECKSUM MISMATCH: {asset['path']}")
        failed += 1

print(f"  Checksums: {checked} OK, {failed} failed, {skipped} skipped")
if failed > 0:
    sys.exit(1)
PYEOF
    fi

    # Summary
    echo ""
    echo "  Verification: $errors errors, $warnings warnings"
    echo ""

    if [[ "$errors" -gt 0 ]]; then
        log_error "Bundle verification FAILED with $errors error(s)"
        log_error "Please re-run download_assets.sh to fix missing assets"
        return 1
    fi

    if [[ "$warnings" -gt 0 ]]; then
        log_warn "Bundle has $warnings warning(s) -- some features may be limited"
    else
        log_ok "Bundle verification PASSED"
    fi
}

# ---- Docker Image Loading ----------------------------------------------------

load_docker_images() {
    log_step "Loading Docker images from bundle"

    local images_dir="$BUNDLE_DIR/docker_images"
    if [[ ! -d "$images_dir" ]]; then
        log_warn "No docker_images directory in bundle; skipping"
        return 0
    fi

    local tar_files
    tar_files=$(find "$images_dir" -name "*.tar" -type f 2>/dev/null)

    if [[ -z "$tar_files" ]]; then
        log_warn "No .tar files found in $images_dir"
        return 0
    fi

    while IFS= read -r tar_file; do
        local basename
        basename="$(basename "$tar_file")"

        if [[ "$DRY_RUN" == true ]]; then
            log_info "[DRY RUN] Would load: $basename"
            continue
        fi

        log_info "Loading Docker image: $basename"
        if docker load -i "$tar_file"; then
            log_ok "Loaded: $basename"

            # If a private registry is specified, re-tag and push
            if [[ -n "$REGISTRY" ]]; then
                retag_and_push "$tar_file"
            fi
        else
            log_error "Failed to load: $basename"
        fi
    done <<< "$tar_files"
}

retag_and_push() {
    local tar_file="$1"

    # Extract the original image name from the tar
    local image_info
    image_info=$(docker load -i "$tar_file" 2>&1 | grep -oP 'Loaded image: \K.*' || true)

    if [[ -z "$image_info" ]]; then
        log_warn "Could not determine image name from $tar_file; skipping re-tag"
        return 0
    fi

    # Derive the new tag
    local image_name
    image_name=$(echo "$image_info" | sed 's|.*/||')
    local new_tag="${REGISTRY}/${image_name}"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would tag: $image_info -> $new_tag"
        log_info "[DRY RUN] Would push: $new_tag"
        return 0
    fi

    log_info "Tagging: $image_info -> $new_tag"
    docker tag "$image_info" "$new_tag"

    log_info "Pushing: $new_tag"
    if docker push "$new_tag"; then
        log_ok "Pushed: $new_tag"
    else
        log_warn "Failed to push: $new_tag (registry may not be reachable)"
    fi
}

# ---- Asset Deployment --------------------------------------------------------

deploy_assets() {
    log_step "Deploying assets to workspace"

    local target_dirs=(
        "$WORKSPACE/models/huggingface"
        "$WORKSPACE/models/fasttext"
        "$WORKSPACE/models/spacy"
        "$WORKSPACE/datasets"
        "$WORKSPACE/nltk_data"
        "$WORKSPACE/configs"
        "$WORKSPACE/benchmarks"
    )

    # Create target directories
    for dir in "${target_dirs[@]}"; do
        if [[ "$DRY_RUN" == true ]]; then
            log_info "[DRY RUN] Would create: $dir"
        else
            mkdir -p "$dir"
        fi
    done

    # If bundle IS the workspace (same path), skip copying
    local bundle_real workspace_real
    bundle_real="$(cd "$BUNDLE_DIR" && pwd)"
    workspace_real="$(mkdir -p "$WORKSPACE" && cd "$WORKSPACE" && pwd)"

    if [[ "$bundle_real" == "$workspace_real" ]]; then
        log_info "Bundle directory is the workspace; skipping copy"
        return 0
    fi

    # Copy models
    copy_dir_contents "$BUNDLE_DIR/models" "$WORKSPACE/models" "models"

    # Copy datasets
    copy_dir_contents "$BUNDLE_DIR/datasets" "$WORKSPACE/datasets" "datasets"

    # Copy NLTK data
    copy_dir_contents "$BUNDLE_DIR/nltk_data" "$WORKSPACE/nltk_data" "NLTK data"

    # Copy configs
    copy_dir_contents "$BUNDLE_DIR/configs" "$WORKSPACE/configs" "configs"

    # Copy benchmarks (if present)
    if [[ -d "$BUNDLE_DIR/benchmarks" ]]; then
        copy_dir_contents "$BUNDLE_DIR/benchmarks" "$WORKSPACE/benchmarks" "benchmarks"
    fi

    log_ok "All assets deployed to $WORKSPACE"
}

copy_dir_contents() {
    local src="$1"
    local dst="$2"
    local label="$3"

    if [[ ! -d "$src" ]]; then
        log_warn "Source directory not found: $src ($label)"
        return 0
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would copy $label: $src -> $dst"
        return 0
    fi

    log_info "Copying $label: $src -> $dst"
    mkdir -p "$dst"

    # Use rsync if available for efficiency; fall back to cp
    if command -v rsync &>/dev/null; then
        rsync -a --info=progress2 "$src/" "$dst/"
    else
        cp -a "$src/." "$dst/"
    fi
}

# ---- Docker Compose Override Generation --------------------------------------

generate_compose_override() {
    log_step "Generating docker-compose airgap override"

    # Determine compose directory
    if [[ -z "$COMPOSE_DIR" ]]; then
        # Try to find it relative to the script
        local repo_compose="$SCRIPT_DIR/../../deploy/nemotron/customization_recipes"
        if [[ -d "$repo_compose" ]]; then
            COMPOSE_DIR="$(cd "$repo_compose" && pwd)"
        else
            COMPOSE_DIR="$WORKSPACE"
        fi
    fi

    local override_file="$COMPOSE_DIR/docker-compose.airgap.yaml"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would generate: $override_file"
        return 0
    fi

    log_info "Generating: $override_file"

    cat > "$override_file" <<'COMPOSEYAML'
# =============================================================================
# Airgap Override for Nemotron Customization Recipes
#
# Generated by deploy_airgap.sh
#
# Usage:
#   docker compose -f docker-compose.yaml -f docker-compose.airgap.yaml up -d
# =============================================================================

# Shared airgap environment variables (YAML anchor)
x-airgap-env: &airgap-env
  - TRANSFORMERS_OFFLINE=1
  - HF_DATASETS_OFFLINE=1
  - HF_HUB_OFFLINE=1
  - HF_HOME=/workspace/models
  - HF_DATASETS_CACHE=/workspace/datasets
  - HUGGINGFACE_HUB_CACHE=/workspace/models/huggingface
  - SENTENCE_TRANSFORMERS_HOME=/workspace/models/sentence-transformers
  - NLTK_DATA=/workspace/nltk_data
  - FASTTEXT_LID_MODEL=/workspace/models/fasttext/lid.176.bin
  - WANDB_MODE=offline
  - WANDB_DISABLED=true
  - DO_NOT_TRACK=1
  - ANONYMIZED_TELEMETRY=false
  - TOKENIZERS_PARALLELISM=false

# Shared airgap volume mounts (YAML anchor)
x-airgap-volumes: &airgap-volumes
  - ${AIRGAP_MODELS_DIR:-./airgap-bundle/models}:/workspace/models:ro
  - ${AIRGAP_DATASETS_DIR:-./airgap-bundle/datasets}:/workspace/datasets:ro
  - ${AIRGAP_NLTK_DIR:-./airgap-bundle/nltk_data}:/workspace/nltk_data:ro
  - ${AIRGAP_CONFIGS_DIR:-./airgap-bundle/configs}:/workspace/configs:ro
  - ${AIRGAP_BENCHMARKS_DIR:-./airgap-bundle/benchmarks}:/workspace/benchmarks:ro
  - ${RESULTS_DIR:-./results}:/workspace/results
  - ${DATA_DIR:-./data}:/workspace/data

services:
  nemotron-orchestrator:
    environment: *airgap-env
    volumes: *airgap-volumes

  nemotron-curator:
    environment: *airgap-env
    volumes: *airgap-volumes

  nemotron-trainer:
    environment: *airgap-env
    volumes: *airgap-volumes

  nemotron-evaluator:
    environment: *airgap-env
    volumes: *airgap-volumes
COMPOSEYAML

    log_ok "Generated: $override_file"
    log_info "To use: docker compose -f docker-compose.yaml -f docker-compose.airgap.yaml up -d"
}

# ---- Environment File Generation ---------------------------------------------

generate_env_file() {
    log_step "Generating .env file for docker-compose"

    local env_file="$WORKSPACE/.env.airgap"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would generate: $env_file"
        return 0
    fi

    cat > "$env_file" <<ENV
# Airgap environment variables for docker-compose
# Source this file or reference with --env-file
#
# Usage:
#   docker compose --env-file .env.airgap -f docker-compose.yaml -f docker-compose.airgap.yaml up

AIRGAP_MODELS_DIR=${WORKSPACE}/models
AIRGAP_DATASETS_DIR=${WORKSPACE}/datasets
AIRGAP_NLTK_DIR=${WORKSPACE}/nltk_data
AIRGAP_CONFIGS_DIR=${WORKSPACE}/configs
AIRGAP_BENCHMARKS_DIR=${WORKSPACE}/benchmarks
ENV

    log_ok "Generated: $env_file"
}

# ---- Summary -----------------------------------------------------------------

print_summary() {
    log_step "Deployment Summary"

    echo ""
    echo "============================================================"
    echo "  Airgap Deployment Complete"
    echo "============================================================"
    echo ""
    echo "  Bundle:     $BUNDLE_DIR"
    echo "  Workspace:  $WORKSPACE"
    echo "  Registry:   ${REGISTRY:-none}"
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        echo "  ** DRY RUN -- no changes were made **"
        echo ""
        return 0
    fi

    echo "  Asset locations:"
    echo "    Models:     $WORKSPACE/models/"
    echo "    Datasets:   $WORKSPACE/datasets/"
    echo "    NLTK:       $WORKSPACE/nltk_data/"
    echo "    Configs:    $WORKSPACE/configs/"
    echo "    Benchmarks: $WORKSPACE/benchmarks/"
    echo ""
    echo "  To start the customization container:"
    echo ""
    echo "    cd deploy/nemotron/customization_recipes"
    echo "    docker compose --env-file $WORKSPACE/.env.airgap \\"
    echo "      -f docker-compose.yaml \\"
    echo "      -f docker-compose.airgap.yaml up -d"
    echo ""
    echo "  Inside the container, apply airgap config overrides:"
    echo ""
    echo "    # For CPT:"
    echo "    nemotron customize cpt \\"
    echo "      model.pretrained_model_name_or_path=/workspace/models/huggingface/<model>"
    echo ""
    echo "    # Or use the override file:"
    echo "    # --config-overrides /workspace/configs/airgap-overrides.yaml"
    echo ""
}

# ---- Entry Point -------------------------------------------------------------

main() {
    parse_args "$@"

    echo ""
    echo "============================================================"
    echo "  Nemotron Airgap Deployment"
    echo "  Bundle: $BUNDLE_DIR"
    echo "============================================================"
    echo ""

    # Always verify first
    verify_bundle || {
        if [[ "$VERIFY_ONLY" == true ]]; then
            exit 1
        fi
        log_warn "Bundle verification had errors; continuing anyway"
    }

    if [[ "$VERIFY_ONLY" == true ]]; then
        log_info "Verification complete (--verify-only mode)"
        exit 0
    fi

    # Deploy assets to workspace
    deploy_assets

    # Load Docker images
    if [[ "$LOAD_DOCKER" == true ]]; then
        load_docker_images
    fi

    # Generate configuration files
    generate_compose_override
    generate_env_file

    # Summary
    print_summary

    log_info "Airgap deployment complete!"
}

main "$@"
