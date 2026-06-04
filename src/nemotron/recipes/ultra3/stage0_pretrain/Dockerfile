# syntax=docker/dockerfile:1.7
FROM nvcr.io/nvidia/nemo:26.04.01

# Ultra does not currently ship a released training container. This recipe
# builds one on top of the published NeMo base image by cloning a pinned
# Megatron-Bridge release branch. That branch pins its Megatron-LM submodule, so
# no separate Megatron-Core branch is needed. Build via:
#   nemotron kit slurm build <profile> --recipe ultra3 --stage pretrain \
#       --build-arg MEGATRON_BRIDGE_BRANCH=dev-0603
# Optionally override the submodule with --build-arg MEGATRON_CORE_BRANCH=<branch>.
ARG MEGATRON_BRIDGE_REPO=https://github.com/NVIDIA-NeMo/Megatron-Bridge.git
ARG MEGATRON_BRIDGE_BRANCH=<ultra-release-branch>
ARG MEGATRON_CORE_REPO=https://github.com/NVIDIA/Megatron-LM.git
# Empty by default: trust the Megatron-LM commit pinned by the Megatron-Bridge
# branch's submodule. Set only to override that pin.
ARG MEGATRON_CORE_BRANCH=

RUN test "${MEGATRON_BRIDGE_BRANCH}" != "<ultra-release-branch>" || \
        (echo "Set MEGATRON_BRIDGE_BRANCH to the Ultra release branch with --build-arg." >&2; exit 1)

# Use git clone rather than ADD <git-url>: BuildKit's ADD strips the .git
# directory, which breaks submodule update/fetch operations. --recurse-submodules
# checks out the Megatron-LM commit pinned by the branch.
RUN git clone --recurse-submodules --branch "${MEGATRON_BRIDGE_BRANCH}" \
        "${MEGATRON_BRIDGE_REPO}" /workspace/Megatron-Bridge

WORKDIR /workspace/Megatron-Bridge

# Materialize the pinned submodule. Only re-point/fetch a different branch when
# MEGATRON_CORE_BRANCH is explicitly set; otherwise trust the pin from the clone.
RUN git submodule update --init --recursive --depth 1 3rdparty/Megatron-LM && \
    if [ -n "${MEGATRON_CORE_BRANCH}" ]; then \
        git -C 3rdparty/Megatron-LM remote set-url origin "${MEGATRON_CORE_REPO}" && \
        git -C 3rdparty/Megatron-LM fetch --depth 1 origin "${MEGATRON_CORE_BRANCH}" && \
        git -C 3rdparty/Megatron-LM checkout -B "${MEGATRON_CORE_BRANCH}" FETCH_HEAD ; \
    fi

# The base image sets VIRTUAL_ENV=/opt/venv. Build a fresh uv-managed venv in
# the workspace, but inherit the base image's CUDA-built torch and other heavy
# GPU packages via --system-site-packages.
ENV UV_PROJECT_ENVIRONMENT=/workspace/Megatron-Bridge/.venv

RUN uv venv --system-site-packages /workspace/Megatron-Bridge/.venv \
 && uv lock && uv sync --locked

# Make the uv-synced venv the default Python environment for nemo-run/torchrun.
ENV PATH=/workspace/Megatron-Bridge/.venv/bin:$PATH
ENV PYTHONPATH=/workspace/Megatron-Bridge/src:/workspace/Megatron-Bridge/3rdparty/Megatron-LM

# Sanity: fail the build if Megatron-Core, Megatron-Bridge, or the Ultra
# pretrain recipe config cannot be imported from this checkout.
RUN python -c "import megatron.core, megatron.bridge" && \
    python -c "from megatron.bridge.recipes.nemotronh.nemotron_3_ultra import nemotron_3_ultra_pretrain_config"
