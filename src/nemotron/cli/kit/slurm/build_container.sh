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
# Generic container-build steps — single source of truth.
#
# Runs INSIDE the podman/stable (pyxis) container. Recipe-agnostic: every input
# is explicit, so the same script serves `nemotron kit slurm build` (env.toml /
# nemo_runspec driven) and the transparent per-recipe build.slurm.sh wrappers.
# It installs enroot at runtime, builds a Dockerfile with podman, imports the
# image to an enroot .sqsh, and upserts the shared container manifest that `run`
# consumes via `container_image=<path>`.
#
# Required inputs (environment):
#   DOCKERFILE       abs path to the recipe-owned Dockerfile
#   CONTEXT          abs path to the build context directory
#   IMAGE_TAG        build-time podman image tag (transient)
#   SQSH             abs output .sqsh path
#   BUILD_CACHE_DIR  cluster-visible scratch dir for enroot (Lustre)
# Optional:
#   MANIFEST_KEY     manifest key (default: basename of SQSH without .sqsh)
#   MANIFEST         manifest path (default: <dir of SQSH>/manifest.yaml)
#   ENROOT_VERSION   enroot to install in the build container (default 3.5.0)
#   BUILD_ARGS       passthrough to `podman build` (e.g. --build-arg FOO=bar)
#   NGC_API_KEY      optional; `podman login nvcr.io` for the FROM base image
# ==============================================================================

set -euo pipefail
export TERM=dumb NO_COLOR=1

DOCKERFILE="${DOCKERFILE:?DOCKERFILE required}"
CONTEXT="${CONTEXT:?CONTEXT required}"
IMAGE_TAG="${IMAGE_TAG:?IMAGE_TAG required}"
SQSH="${SQSH:?SQSH required (abs output .sqsh path)}"
BUILD_CACHE_DIR="${BUILD_CACHE_DIR:?BUILD_CACHE_DIR required (cluster-visible scratch dir)}"
ENROOT_VERSION="${ENROOT_VERSION:-3.5.0}"
BUILD_ARGS="${BUILD_ARGS:-}"
CONTAINERS_DIR="$(dirname "${SQSH}")"
MANIFEST="${MANIFEST:-${CONTAINERS_DIR}/manifest.yaml}"
MANIFEST_KEY="${MANIFEST_KEY:-$(basename "${SQSH}" .sqsh)}"

if [ ! -f "${DOCKERFILE}" ]; then
    echo "ERROR: Dockerfile not found: ${DOCKERFILE}" >&2
    exit 1
fi
mkdir -p "${CONTAINERS_DIR}" "${BUILD_CACHE_DIR}"

# ------------------------------------------------------------------------------
# 1. Install enroot at runtime. dnf5 on Fedora 41 crashes the *final* transaction
# step (std::length_error in basic_string::_M_replace_aux) regardless of packages
# — files land, then bookkeeping segfaults. Tolerate the non-zero exit and verify
# binaries. Excluding ncurses keeps the dep set minimal (avoids
# parallel -> perl-Term-Cap -> ncurses, which the import-from-podman path skips).
# ------------------------------------------------------------------------------
echo "[kit-build] installing enroot v${ENROOT_VERSION} runtime deps ..."
set +e
dnf install -y --quiet --exclude=ncurses jq squashfs-tools
dnf_rc=$?
set -e
if ! command -v jq >/dev/null 2>&1 || ! command -v mksquashfs >/dev/null 2>&1; then
    echo "[kit-build] dnf install actually failed (rc=${dnf_rc}); deps missing." >&2
    exit 1
fi
echo "[kit-build] dnf install rc=${dnf_rc} (non-zero is the dnf5 finish bug; files installed)."

echo "[kit-build] fetching + rpm --nodeps installing enroot ..."
RPM_URL_BASE="https://github.com/NVIDIA/enroot/releases/download/v${ENROOT_VERSION}"
mkdir -p /tmp/enroot-rpms
curl -fsSL -o /tmp/enroot-rpms/enroot.rpm \
    "${RPM_URL_BASE}/enroot-${ENROOT_VERSION}-1.el8.x86_64.rpm"
curl -fsSL -o /tmp/enroot-rpms/enroot-caps.rpm \
    "${RPM_URL_BASE}/enroot+caps-${ENROOT_VERSION}-1.el8.x86_64.rpm"
rpm -i --nodeps /tmp/enroot-rpms/enroot.rpm /tmp/enroot-rpms/enroot-caps.rpm

# ------------------------------------------------------------------------------
# 2. Redirect enroot scratch to the mounted cache. enroot defaults to $HOME (a
# small tmpfs inside the container) which fills during import of a multi-GB image.
# ------------------------------------------------------------------------------
export ENROOT_CACHE_PATH="${BUILD_CACHE_DIR}/enroot/cache"
export ENROOT_DATA_PATH="${BUILD_CACHE_DIR}/enroot/data"
export ENROOT_RUNTIME_PATH="${BUILD_CACHE_DIR}/enroot/runtime"
export ENROOT_TEMP_PATH="${BUILD_CACHE_DIR}/enroot/tmp"
mkdir -p "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}" "${ENROOT_TEMP_PATH}"
enroot version

# ------------------------------------------------------------------------------
# 3. Registry auth for the base image (FROM nvcr.io/...). Provide NGC_API_KEY, or
# rely on the cluster's podman credentials / a cached base image.
# ------------------------------------------------------------------------------
if [ -n "${NGC_API_KEY:-}" ]; then
    echo "[kit-build] podman login nvcr.io ..."
    echo "${NGC_API_KEY}" | podman login nvcr.io --username '$oauthtoken' --password-stdin
fi

# ------------------------------------------------------------------------------
# 4. Build + 5. import.
# ------------------------------------------------------------------------------
echo "[kit-build] podman build -t ${IMAGE_TAG} ..."
# shellcheck disable=SC2086  # BUILD_ARGS is an intentional word-split passthrough.
podman build ${BUILD_ARGS} -f "${DOCKERFILE}" -t "${IMAGE_TAG}" "${CONTEXT}"

echo "[kit-build] enroot import podman://${IMAGE_TAG} -> ${SQSH}"
rm -f "${SQSH}"
enroot import --output "${SQSH}" "podman://${IMAGE_TAG}"
ls -la "${SQSH}"

# ------------------------------------------------------------------------------
# 6. Upsert the shared container manifest (idempotent: drop prior block, append).
# `run` reads `container_image` from here; the local and air-gap backends emit
# the same schema.
# ------------------------------------------------------------------------------
if [ ! -f "${SQSH}" ]; then
    echo "ERROR: build reported success but ${SQSH} is missing." >&2
    exit 1
fi
SHA256="$(sha256sum "${SQSH}" | awk '{print $1}')"
BUILT_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
touch "${MANIFEST}"
if grep -q "^${MANIFEST_KEY}:" "${MANIFEST}" 2>/dev/null; then
    sed -i "/^${MANIFEST_KEY}:/,+3d" "${MANIFEST}"
fi
cat >> "${MANIFEST}" <<MANIFEST_EOF
${MANIFEST_KEY}:
  ref: ${SQSH}
  sha256: ${SHA256}
  built: ${BUILT_AT}
MANIFEST_EOF

echo "[kit-build] manifest updated: ${MANIFEST} (key: ${MANIFEST_KEY})"
echo "[kit-build] run with:  ... run.env.container_image=${SQSH}"
echo KIT_BUILD_DONE
