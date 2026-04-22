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

ENV PYTHONPATH=/workspace/Megatron-Bridge/src:/workspace/Megatron-Bridge/3rdparty/Megatron-LM

RUN uv run python -c "import megatron.core, megatron.bridge" && \
    uv run python -c "from megatron.core.models.huggingface.fastconformer import configuration_fastconformer" && \
    uv run python -c "from megatron.bridge.recipes.nemotron_omni import nemotron_omni_valor32k_energon_sft_config"
