# syntax=docker/dockerfile:1.7
#
# Super35 (Nemotron 3.5 Super VL) RL container.
#
# Every RL sub-stage (RLVR, specialist teachers, MOPD) runs NeMo-RL's
# ``super-v3.5-posttraining`` branch with its NeMo Gym and Megatron-Bridge
# submodules. NeMo-RL's own ``docker/Dockerfile`` expects the source tree as a
# named build context; ``nemotron kit slurm build`` only ships this
# Dockerfile, so we clone the branch here and mirror the relevant build steps
# (as omni3/stage1_rl/Dockerfile does for the Omni branch).
#
# Build:
#   nemotron kit slurm build <profile> --recipe super35 --stage rl
#
# ``NEMO_RL_REF`` pins the NeMo-RL ``super-v3.5-posttraining`` commit the
# Super 3.5 RL recipes were validated with; its Gym submodule is e24db4e6.
# Bump it to a release tag once these changes land on NeMo-RL main.

ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:26.05-cuda13.2-devel-ubuntu24.04
ARG NEMO_RL_REF=f299847f17c95f61767aef005fa07db704c7678d
ARG UV_VERSION=0.11.28
ARG PYTHON_VERSION=3.13.14
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
    jq curl git ninja-build rsync wget less vim ca-certificates gnupg ccache
apt-get clean
rm -rf /var/lib/apt/lists/*
EOF

# Install uv + python (mirrors NeMo-RL's Dockerfile on the same branch).
ENV PATH="/root/.local/bin:$PATH"
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh && \
    uv python install ${PYTHON_VERSION}

ENV RAY_USAGE_STATS_ENABLED=0
ENV RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
ENV NEMO_RL_VENV_DIR=/opt/ray_venvs
# NeMo Gym server venvs live in the image; the Super 3.5 RL configs reuse
# them via ``env.nemo_gym.uv_venv_dir: ${oc.env:NEMO_GYM_VENV_DIR,/opt/gym_venvs}``
# with ``skip_venv_if_present: true``.
ENV NEMO_GYM_VENV_DIR=/opt/gym_venvs
ENV UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
ENV UV_LINK_MODE=copy
# Hopper (H100) + Blackwell (B200/GB200).
ENV TORCH_CUDA_ARCH_LIST="9.0 10.0"
ENV CUDA_HOME=/usr/local/cuda


FROM base AS hermetic

WORKDIR /opt/nemo-rl

# Shallow fetch of the pinned commit with only the submodules the
# Super 3.5 RL stages need: NeMo Gym (environments / resource servers) and
# Megatron-Bridge (Megatron policy backend for the Super-VL model).
RUN <<EOF bash -exu -o pipefail
git init /opt/nemo-rl
cd /opt/nemo-rl
git remote add origin https://github.com/NVIDIA-NeMo/RL.git
git fetch --depth 1 origin ${NEMO_RL_REF}
git checkout FETCH_HEAD
git submodule update --init --depth 1 \
    3rdparty/Gym-workspace/Gym \
    3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
EOF

ARG MAX_JOBS
ENV MAX_JOBS=${MAX_JOBS}

RUN <<"EOF" bash -exu -o pipefail
uv venv --seed

# Sync the training/inference backends one extra at a time to warm uv's
# cache and surface conflicts early; the final ``--all-groups`` sync
# resolves the full dep set. SGLang / TRT-LLM backends are not used by the
# Super 3.5 RL recipes and are skipped.
UV_LINK_MODE=hardlink uv sync --frozen --no-install-project
UV_LINK_MODE=hardlink uv sync --frozen --extra vllm --no-install-project
uv sync --link-mode symlink --frozen --extra mcore --no-install-project
uv sync --link-mode symlink --frozen --extra automodel --no-install-project
uv sync --link-mode symlink --frozen --all-groups --no-install-project
EOF

ENV PATH="/opt/nemo_rl_venv/bin:$PATH"


FROM hermetic AS release

ARG NVIDIA_BUILD_ID
ARG NVIDIA_BUILD_REF
ENV NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID:-<unknown>}
ENV NVIDIA_BUILD_REF=${NVIDIA_BUILD_REF:-<unknown>}
LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"

WORKDIR /opt/nemo-rl

# Prefetch the Ray actor venvs NeMo-RL spawns at runtime (policy workers,
# vLLM generation, NeMo Gym actor).
RUN UV_LINK_MODE=symlink uv run nemo_rl/utils/prefetch_venvs.py --negative-filters sglang trtllm

# Prefetch NeMo Gym server venvs (math_with_judge, code_gen, ns_tools, ...)
# into ${NEMO_GYM_VENV_DIR}. The branch ships a config that enumerates every
# Gym environment used by the Super 3.5 RL stages; pass a different
# space-separated list of NeMo-RL configs to prefetch other environments.
# Gym pins its own vLLM; override it to the prebuilt CUDA wheel so Gym's venvs
# share NeMo-RL's torch/NVIDIA stack (same as NeMo-RL's Dockerfile).
ARG NEMO_GYM_PREFETCH_CONFIGS=examples/nemo_gym/prefetch_super35_all_envs.yaml
ARG NEMO_GYM_CUDA=cu130
ARG NEMO_GYM_VLLM_VERSION=0.24.0
ARG TARGETARCH
RUN <<"EOF" bash -exu
if [[ -n "${NEMO_GYM_PREFETCH_CONFIGS:-}" ]]; then
    case "${TARGETARCH:-amd64}" in arm64) WHEEL_ARCH=aarch64 ;; *) WHEEL_ARCH=x86_64 ;; esac
    GYM_VLLM_OVERRIDE=/opt/nemo-rl/.gym-vllm-override.txt
    printf 'vllm @ https://github.com/vllm-project/vllm/releases/download/v%s/vllm-%s-cp38-abi3-manylinux_2_28_%s.whl\n' \
        "${NEMO_GYM_VLLM_VERSION}" "${NEMO_GYM_VLLM_VERSION}" "${WHEEL_ARCH}" > "${GYM_VLLM_OVERRIDE}"
    UV_TORCH_BACKEND="${NEMO_GYM_CUDA}" \
    UV_OVERRIDE="${GYM_VLLM_OVERRIDE}" \
    UV_LINK_MODE=symlink uv run python examples/nemo_gym/prefetch_venvs.py ${NEMO_GYM_PREFETCH_CONFIGS}
    rm -f "${GYM_VLLM_OVERRIDE}"
fi
EOF

# Container fingerprint for frozen-env support; lives outside
# /opt/nemo-rl so user mounts can't clobber it.
RUN python tools/generate_fingerprint.py > /opt/nemo_rl_container_fingerprint

WORKDIR /opt/nemo-rl
