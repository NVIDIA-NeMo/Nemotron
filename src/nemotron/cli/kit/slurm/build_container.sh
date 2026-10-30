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
# Container build + push (single source of truth) — runs INSIDE the pyxis
# podman/stable container.
#
# It ONLY builds the recipe Dockerfile and pushes the image to a registry. The
# enroot import (image -> .sqsh) is deliberately NOT here: a nested unprivileged
# `enroot import` inside a pyxis container fails (it can't create the user
# namespace it needs to unpack — Permission denied). Instead the image is pushed
# to a registry and imported on the HOST via `enroot import docker://...` (the
# proven `kit squash` path). See the kit slurm build command for the host step.
#
# Required inputs (environment):
#   DOCKERFILE   abs path to the recipe-owned Dockerfile
#   CONTEXT      abs path to the build context directory
#   IMAGE_REF    full registry ref to push, e.g.
#                gitlab-master.nvidia.com:5005/<repo>/ultra3-pretrain:latest
# Optional:
#   BUILD_ARGS   passthrough to `podman build` (e.g. --build-arg FOO=bar)
#   REGISTRY_HOST + REGISTRY_TOKEN  podman login before push (otherwise a
#                mounted /root/.config/containers/auth.json is used).
# ==============================================================================

set -euo pipefail
export TERM=dumb NO_COLOR=1

DOCKERFILE="${DOCKERFILE:?DOCKERFILE required}"
CONTEXT="${CONTEXT:?CONTEXT required}"
IMAGE_REF="${IMAGE_REF:?IMAGE_REF required (registry ref to push)}"
BUILD_ARGS="${BUILD_ARGS:-}"

if [ ! -f "${DOCKERFILE}" ]; then
    echo "ERROR: Dockerfile not found: ${DOCKERFILE}" >&2
    exit 1
fi

# Optional explicit login (otherwise rely on a mounted auth.json bridged from the
# host enroot credentials).
if [ -n "${REGISTRY_TOKEN:-}" ] && [ -n "${REGISTRY_HOST:-}" ]; then
    echo "[kit-build] podman login ${REGISTRY_HOST} ..."
    echo "${REGISTRY_TOKEN}" | podman login "${REGISTRY_HOST}" --username "${REGISTRY_USER:-oauth2}" --password-stdin
fi

echo "[kit-build] podman build -t ${IMAGE_REF} ..."
# shellcheck disable=SC2086  # BUILD_ARGS is an intentional word-split passthrough.
podman build ${BUILD_ARGS} -f "${DOCKERFILE}" -t "${IMAGE_REF}" "${CONTEXT}"

echo "[kit-build] podman push ${IMAGE_REF} ..."
podman push "${IMAGE_REF}"

echo "[kit-build] pushed ${IMAGE_REF}"
echo KIT_BUILD_PUSH_DONE
