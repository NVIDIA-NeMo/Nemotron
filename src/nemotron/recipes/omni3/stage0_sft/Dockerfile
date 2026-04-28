# syntax=docker/dockerfile:1.7
# Base image: latest RC of the 26.04 series. The bare ``26.04`` tag is not
# published; nvcr.io publishes ``26.04.rcN`` release candidates instead.
# Bump as new RCs land and this image is re-tested.
FROM nvcr.io/nvidian/nemo:26.04.rc7

# Use ``git clone`` rather than ``ADD <git-url>``: BuildKit's ADD strips
# the .git directory, which breaks ``git submodule update`` and the
# downstream ``git fetch`` in 3rdparty/Megatron-LM.
#
# The ``nemotron_3_omni`` branch is the current release branch for
# the Nemotron 3 Omni model. Bump to a versioned tag (or ``main``)
# when these changes merge upstream.
RUN git clone --recurse-submodules --branch nemotron_3_omni \
        https://github.com/NVIDIA-NeMo/Megatron-Bridge.git /workspace/Megatron-Bridge

WORKDIR /workspace/Megatron-Bridge

RUN cd 3rdparty/Megatron-LM && \
    (git remote add adlr https://github.com/NVIDIA/Megatron-LM.git || true) && \
    git fetch adlr nemotron_3_omni && \
    git checkout adlr/nemotron_3_omni

# The base image sets ``VIRTUAL_ENV=/opt/venv``, which would make uv
# install into the pre-existing base venv. That venv has many packages
# installed outside of uv's tracking, producing
# ``File exists (os error 17)`` collisions on dist-info dirs. Pin uv to
# a fresh ``.venv`` in the workspace so the build is hermetic.
#
# Pre-create the venv with ``--system-site-packages`` so it inherits the
# CUDA-built torch (and other heavy GPU packages) from the base image's
# /opt/venv. ``uv sync`` then layers Megatron-Bridge's deps on top
# without re-installing torch — Megatron-Bridge's lockfile expects
# torch to be provided by the base image.
ENV UV_PROJECT_ENVIRONMENT=/workspace/Megatron-Bridge/.venv

RUN uv venv --system-site-packages /workspace/Megatron-Bridge/.venv \
 && uv lock && uv sync --locked

# Make the uv-synced venv the default Python environment for this image so
# downstream tooling (nemo-run, torchrun, the sft.py CLI) doesn't need to
# know that `uv` is how it was built. Putting `.venv/bin` first means
# `python`, `torchrun`, etc. resolve to the Megatron-Bridge venv without a
# `uv run` prefix. The base image's /opt/venv stays available as a fallback.
ENV PATH=/workspace/Megatron-Bridge/.venv/bin:$PATH
ENV PYTHONPATH=/workspace/Megatron-Bridge/src:/workspace/Megatron-Bridge/3rdparty/Megatron-LM

# Sanity: import Megatron + the Omni recipe configs the train script
# resolves at runtime. ``nemotron_omni_cord_v2_sft_config`` is the
# default recipe (see DEFAULT_RECIPE_NAME in
# stage0_sft/train.py); the valor32k variants are exercised by
# ``train.py -c valor32k|peft_valor32k``. If new recipe configs are
# added on the nemotron_3_omni branch, extend the import line so the build
# fails fast rather than at training time.
RUN python -c "import megatron.core, megatron.bridge" && \
    python -c "from megatron.bridge.recipes.nemotron_omni import \
        nemotron_omni_cord_v2_sft_config, \
        nemotron_omni_cord_v2_peft_config, \
        nemotron_omni_valor32k_sft_config, \
        nemotron_omni_valor32k_peft_config"
