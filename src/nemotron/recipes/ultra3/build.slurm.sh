#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

# ==============================================================================
# Nemotron 3 Ultra — cluster container build, transparent (no-CLI) path.
#
# The env.toml / nemo_runspec driven path is `nemotron kit slurm build`. This
# script is the dependency-free fallback: run it with bare `sbatch` on a
# cluster-visible checkout. It only resolves the ultra3 stage -> Dockerfile/tag/
# sqsh, then launches the SHARED inner script
# (src/nemotron/cli/kit/slurm/build_container.sh) inside a pyxis podman
# container — the exact same build logic `kit slurm build` runs.
#
# USAGE
#   sbatch --partition=<cpu-part> --account=<acct> \
#       src/nemotron/recipes/ultra3/build.slurm.sh pretrain
#
#   NGC_API_KEY=<key> \
#   BUILD_ARGS="--build-arg MEGATRON_BRIDGE_BRANCH=<branch> --build-arg MEGATRON_CORE_BRANCH=<branch>" \
#   sbatch --partition=<cpu-part> --account=<acct> \
#       src/nemotron/recipes/ultra3/build.slurm.sh pretrain
#
#   DRY_RUN=1 bash src/nemotron/recipes/ultra3/build.slurm.sh pretrain   # preview
#
# Stages: pretrain (stage0_pretrain), sft (stage1_sft).
# ==============================================================================

#SBATCH --job-name=ultra3-container-build
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/ultra3_build_%j.log
# NOTE: set --partition and --account on the `sbatch` command line. This is a
# CPU-only build; a GPU-only training partition will be rejected by sbatch.

set -euo pipefail

STAGE="${1:-${STAGE:-}}"
if [ -z "${STAGE}" ]; then
    echo "ERROR: stage required. Usage: sbatch build.slurm.sh <pretrain|sft>" >&2
    exit 1
fi

# Repo root: default to this script's location (recipe root is four levels up).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
INNER="${REPO_ROOT}/src/nemotron/cli/kit/slurm/build_container.sh"

# Ultra3 stage registry (matches the RECIPES table in kit/slurm/build.py).
case "${STAGE}" in
    pretrain|stage0_pretrain) STAGE_DIR="stage0_pretrain"; IMAGE_TAG="nemotron/ultra3-pretrain:latest"; SQSH_NAME="ultra3-pretrain.sqsh" ;;
    sft|stage1_sft)           STAGE_DIR="stage1_sft";      IMAGE_TAG="nemotron/ultra3-sft:latest";      SQSH_NAME="ultra3-sft.sqsh" ;;
    *) echo "ERROR: unknown stage '${STAGE}'. Expected: pretrain, sft." >&2; exit 1 ;;
esac

# Explicit inputs for the shared inner script.
export DOCKERFILE="${REPO_ROOT}/src/nemotron/recipes/ultra3/${STAGE_DIR}/Dockerfile"
export CONTEXT="${REPO_ROOT}/src/nemotron/recipes/ultra3/${STAGE_DIR}"
export IMAGE_TAG
export BUILD_CACHE_DIR="${BUILD_CACHE_DIR:-${HOME}/.cache/nemotron}"
export SQSH="${BUILD_CACHE_DIR}/containers/${SQSH_NAME}"
export MANIFEST="${BUILD_CACHE_DIR}/containers/manifest.yaml"
export MANIFEST_KEY="ultra3-${STAGE_DIR}"
export ENROOT_VERSION="${ENROOT_VERSION:-3.5.0}"
export BUILD_ARGS="${BUILD_ARGS:-}"

PODMAN_IMAGE="${PODMAN_IMAGE:-docker://quay.io#podman/stable:v5.3}"
EXTRA_MOUNTS="${EXTRA_MOUNTS:-}"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -f "${INNER}" ]; then
    echo "ERROR: shared inner build script not found: ${INNER}" >&2
    exit 1
fi
if [ ! -f "${DOCKERFILE}" ]; then
    echo "ERROR: stage '${STAGE_DIR}' is missing ${DOCKERFILE}." >&2
    exit 1
fi
mkdir -p logs

MOUNTS="${REPO_ROOT}:${REPO_ROOT},${BUILD_CACHE_DIR}:${BUILD_CACHE_DIR}"
[ -n "${EXTRA_MOUNTS}" ] && MOUNTS="${MOUNTS},${EXTRA_MOUNTS}"

SRUN_CMD=(srun --mpi=none --ntasks=1
    --container-image="${PODMAN_IMAGE}"
    --container-mounts="${MOUNTS}"
    --no-container-mount-home
    --export=ALL
    bash "${INNER}")

echo "======================================"
echo "Nemotron 3 Ultra container build (cluster / transparent)"
echo "======================================"
echo "Stage:          ${STAGE} -> ${STAGE_DIR}"
echo "Repo root:      ${REPO_ROOT}"
echo "Shared inner:   ${INNER}"
echo "Dockerfile:     ${DOCKERFILE}"
echo "Build image:    ${PODMAN_IMAGE}"
echo "Output sqsh:    ${SQSH}"
echo "Manifest:       ${MANIFEST} (key: ${MANIFEST_KEY})"
echo "Build args:     ${BUILD_ARGS:-<none>}"
echo "Mounts:         ${MOUNTS}"
echo "======================================"

if [ "${DRY_RUN}" = "1" ]; then
    echo "[DRY_RUN] would run:"
    printf '  %q' "${SRUN_CMD[@]}"; echo
    exit 0
fi

exec "${SRUN_CMD[@]}"
