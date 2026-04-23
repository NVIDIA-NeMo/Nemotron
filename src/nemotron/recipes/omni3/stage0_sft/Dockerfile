# syntax=docker/dockerfile:1.7
FROM nvcr.io/nvidian/nemo:26.04

ADD https://github.com/NVIDIA/Megatron-Bridge.git#dev/nomni /workspace/Megatron-Bridge

WORKDIR /workspace/Megatron-Bridge

RUN git submodule update --init --recursive || true

RUN cd 3rdparty/Megatron-LM && \
    (git remote add adlr https://github.com/NVIDIA/Megatron-LM.git || true) && \
    git fetch adlr chcui/nemotron-omni-mcore && \
    git checkout adlr/chcui/nemotron-omni-mcore

RUN rm -f /opt/venv/lib/python3.12/site-packages/__editable__*megatron*.pth \
          /opt/venv/lib/python3.12/site-packages/__editable__*megatron*.py

RUN uv lock && uv sync --locked

# Make the uv-synced venv the default Python environment for this image so
# downstream tooling (nemo-run, torchrun, the sft.py CLI) doesn't need to
# know that `uv` is how it was built. Putting `.venv/bin` first means
# `python`, `torchrun`, etc. resolve to the Megatron-Bridge venv without a
# `uv run` prefix. The base image's /opt/venv stays available as a fallback.
ENV PATH=/workspace/Megatron-Bridge/.venv/bin:$PATH
ENV PYTHONPATH=/workspace/Megatron-Bridge/src:/workspace/Megatron-Bridge/3rdparty/Megatron-LM

# Sanity: import Megatron + the Omni recipe module without the uv-run prefix.
RUN python -c "import megatron.core, megatron.bridge" && \
    python -c "from megatron.core.models.huggingface.fastconformer import configuration_fastconformer" && \
    python -c "from megatron.bridge.recipes.nemotron_omni import nemotron_omni_valor32k_energon_sft_config"
