# syntax=docker/dockerfile:1.7
#
# Omni3 RL container.
#
# We mirror NeMo-RL's official ``docker/Dockerfile`` body here rather
# than pulling a pre-built image, because:
#   - The omni branch (``nano-v3-omni``) carries a vllm submodule
#     pointing at the omni-specific fork
#     (``https://github.com/aroshanghias-nvd/vllm.git``,
#     branch ``nano-v3-vl``); we want our build to use that submodule
#     verbatim, which means ``git clone --recurse-submodules`` + the
#     ``BUILD_CUSTOM_VLLM=1`` build flow.
#   - Keeping the omni3 layer thin (mostly env vars + the NeMo-RL
#     build steps) means upstream changes to NeMo-RL's Dockerfile
#     can be re-pulled in by bumping ``NEMO_RL_REF``.
#
# Bump ``NEMO_RL_REF`` to the post-release branch (e.g. ``main`` or
# a versioned tag) once the model ships and the omni branch is merged.

ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:25.05-cuda12.9-devel-ubuntu24.04
ARG NEMO_RL_REF=nano-v3-omni
ARG UV_VERSION=0.9.7
ARG PYTHON_VERSION=3.12
ARG MAX_JOBS=4
ARG NVIDIA_BUILD_ID
ARG NVIDIA_BUILD_REF


FROM ${BASE_IMAGE} AS base

ARG NEMO_RL_REF
ARG UV_VERSION
ARG PYTHON_VERSION

ENV NRL_CONTAINER=1
USER root

RUN <<EOF bash -exu -o pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    jq curl git rsync wget less vim ca-certificates gnupg
apt-get clean
rm -rf /var/lib/apt/lists/*
EOF

# Install uv + python (mirrors NeMo-RL's Dockerfile).
ENV PATH="/root/.local/bin:$PATH"
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh && \
    uv python install ${PYTHON_VERSION}

ENV RAY_USAGE_STATS_ENABLED=0
ENV RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
ENV NEMO_RL_VENV_DIR=/opt/ray_venvs
ENV UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
ENV UV_LINK_MODE=copy
# Hopper (H100) + Blackwell (B200/GB200). DeepEP and the mcore inference
# unified-memory API both require this to be set.
ENV TORCH_CUDA_ARCH_LIST="9.0 10.0"


FROM base AS hermetic

WORKDIR /opt/nemo-rl

# Recursive clone brings in NeMo-RL's submodules — most importantly the
# vllm fork at ``3rdparty/vllm`` (configured in the upstream
# ``.gitmodules`` to point at ``aroshanghias-nvd/vllm`` for the omni
# release).
RUN git clone --recurse-submodules --depth 1 --shallow-submodules \
        --branch ${NEMO_RL_REF} \
        https://github.com/NVIDIA/NeMo-RL.git /opt/nemo-rl

ARG MAX_JOBS
ENV MAX_JOBS=${MAX_JOBS}
# ``BUILD_CUSTOM_VLLM=1`` makes NeMo-RL's ``tools/build-custom-vllm.sh``
# build vllm from the in-tree ``3rdparty/vllm`` submodule instead of
# downloading a precompiled wheel. This is what wires our omni-specific
# vllm fork into the container.
ENV BUILD_CUSTOM_VLLM=1

RUN <<"EOF" bash -exu -o pipefail
uv venv --seed

# vllm builds without isolation, so seed setuptools first.
VIRTUAL_ENV=/opt/nemo_rl_venv uv pip install setuptools setuptools_scm
VIRTUAL_ENV=/opt/nemo_rl_venv uv pip install torch==2.9.1 --index https://download.pytorch.org/whl/cu129

# Build the omni vllm fork from the 3rdparty submodule. The script
# writes ``3rdparty/vllm/nemo-rl.env`` containing
# ``VLLM_PRECOMPILED_WHEEL_LOCATION`` + ``SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM``
# pointing at the freshly built wheel.
bash tools/build-custom-vllm.sh
source 3rdparty/vllm/nemo-rl.env

# Sync the training/inference backends one extra at a time to warm uv's
# cache and surface conflicts early; the final ``--all-groups`` sync
# resolves the full dep set.
uv sync --link-mode symlink --locked --no-install-project
uv sync --link-mode symlink --locked --extra vllm --no-install-project
uv sync --link-mode symlink --locked --extra mcore --no-install-project
uv sync --link-mode symlink --locked --extra automodel --no-install-project
uv sync --link-mode symlink --locked --all-groups --no-install-project
EOF

ENV PATH="/opt/nemo_rl_venv/bin:$PATH"


FROM hermetic AS release

ARG NVIDIA_BUILD_ID
ARG NVIDIA_BUILD_REF
ENV NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID:-<unknown>}
ENV NVIDIA_BUILD_REF=${NVIDIA_BUILD_REF:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"

# Prefetch venvs that NeMo-RL spawns at runtime (Ray actor environments).
WORKDIR /opt/nemo-rl
RUN UV_LINK_MODE=symlink uv run nemo_rl/utils/prefetch_venvs.py

# Container fingerprint for frozen-env support; lives outside
# /opt/nemo-rl so user mounts can't clobber it.
RUN python tools/generate_fingerprint.py > /opt/nemo_rl_container_fingerprint

WORKDIR /opt/nemo-rl
