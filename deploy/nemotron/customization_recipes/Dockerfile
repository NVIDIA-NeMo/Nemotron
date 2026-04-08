# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# =============================================================================
# Nemotron Orchestrator Container (nemotron-orchestrator)
#
# Lightweight CLI + orchestration container. Routes work to the curator,
# trainer, evaluator, and NIM service containers. Does NOT include heavy
# ML frameworks (NeMo, Megatron, PyTorch) -- those live in dedicated
# service containers.
#
# This is part of the multi-container customization deployment:
#   - nemotron-orchestrator  (this image)  — CLI, orchestration, Docker client
#   - nemotron-curator       — NeMo Curator, data prep, SDG, BYOB
#   - nemotron-trainer       — NeMo + Megatron, CPT/SFT/RL training
#   - nemotron-evaluator     — Model evaluation, benchmarks
#   - nemotron-nim           — NIM for local LLM inference
#
# Build:
#   docker compose build nemotron-orchestrator
#
# Run:
#   docker compose run --rm nemotron-orchestrator nemotron customize --help
# =============================================================================

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG REMOTE_USER=nemotron
ARG REMOTE_UID=1000
ARG REMOTE_GID=1000

# Create the user/group (ignore if they already exist)
RUN groupadd --gid $REMOTE_GID $REMOTE_USER -f && \
    if [ -z "$(id -u $REMOTE_UID 2>/dev/null)" ]; then \
        useradd --uid $REMOTE_UID --gid $REMOTE_GID -m $REMOTE_USER; \
    fi

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo \
        ca-certificates \
        curl \
        git \
        git-lfs \
        wget \
        unzip \
        python3 \
        python3-pip \
        python3-dev \
    && update-ca-certificates \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Add user to sudoers
RUN REAL_USER=$(id -u -n ${REMOTE_UID} 2>/dev/null || echo $REMOTE_USER) && \
    echo "$REAL_USER ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$REAL_USER && \
    chmod 0440 /etc/sudoers.d/$REAL_USER

# Install Docker CLI (for orchestrating other containers)
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu jammy stable" \
        > /etc/apt/sources.list.d/docker.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin && \
    rm -rf /var/lib/apt/lists/*

# Install NGC CLI (required for data-designer persona downloads and model access)
RUN cd /tmp && \
    wget -q -O ngccli_linux.zip https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.41.4/files/ngccli_linux.zip && \
    unzip -q ngccli_linux.zip && \
    mv ngc-cli /opt/ngc-cli && \
    rm ngccli_linux.zip
ENV PATH="/opt/ngc-cli:${PATH}"

# Copy the Nemotron repo into the container
COPY --chown=$REMOTE_UID:$REMOTE_GID . /workspace/nemotron

WORKDIR /workspace/nemotron

# Install Nemotron CLI (lightweight — no heavy ML deps)
# The [customize] extras pull in orchestration + config deps only;
# heavy training/inference deps are in the trainer/curator containers.
RUN pip install --no-cache-dir -e ".[customize]"

# Mark this container as the orchestrator so the dispatcher knows to route
# commands to sibling containers via docker exec instead of running locally.
ENV NEMOTRON_ORCHESTRATOR=1
ENV NEMOTRON_CONTAINER=orchestrator

# Switch to the user
USER $REMOTE_UID

CMD ["tail", "-f", "/dev/null"]
