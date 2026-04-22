# syntax=docker/dockerfile:1.7
ARG MAX_JOBS=4
ARG UV_VERSION=0.9.30
ARG VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/27ca95b3c9e6e32ac02508e729c01865800fd036/vllm-0.14.0rc2.dev196%2Bg27ca95b3c-cp38-abi3-manylinux_2_31_x86_64.whl
ARG SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.14.0rc2.dev196+g27ca95b3c
ARG NEMO_RL_COMMIT
ARG NVIDIA_BUILD_ID
ARG NVIDIA_BUILD_REF
ARG RC_DATE=00.00
ARG TARGETARCH

FROM nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04 AS release

ARG MAX_JOBS
ARG UV_VERSION
ARG VLLM_PRECOMPILED_WHEEL_LOCATION
ARG SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM
ARG NEMO_RL_COMMIT
ARG NVIDIA_BUILD_ID
ARG NVIDIA_BUILD_REF
ARG RC_DATE
ARG TARGETARCH

ADD https://github.com/NVIDIA/NeMo-RL.git#nano-v3-omni-recipes /workspace/nemo-rl

ENV MAX_JOBS=${MAX_JOBS}
ENV UV_VERSION=${UV_VERSION}
ENV VLLM_PRECOMPILED_WHEEL_LOCATION=${VLLM_PRECOMPILED_WHEEL_LOCATION}
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM}
ENV NEMO_RL_COMMIT=${NEMO_RL_COMMIT:-}
ENV NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID:-}
ENV NVIDIA_BUILD_REF=${NVIDIA_BUILD_REF:-}

LABEL com.nvidia.build.id="${NVIDIA_BUILD_ID}"
LABEL com.nvidia.build.ref="${NVIDIA_BUILD_REF}"

WORKDIR /opt/nemo-rl-omni

# TODO(release): replace body with forked content from nemo-rl-omni/docker/Dockerfile release target.
# The exact omni fork Dockerfile was not accessible at authoring time, so keep the
# pinned ADD above plus the required build args and labels, then transcribe the
# upstream release body here at release (vLLM wheel install, uv sync, fingerprinting, etc.).
RUN mkdir -p /opt/nemo-rl-omni && \
    if [ -d /workspace/nemo-rl ]; then cp -a /workspace/nemo-rl/. /opt/nemo-rl-omni/; fi
