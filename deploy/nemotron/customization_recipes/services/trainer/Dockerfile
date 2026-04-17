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
# Nemotron Trainer Container (nemotron-trainer)
#
# Model training container for the customization pipeline. Handles:
#   - Continued Pre-Training (CPT)
#   - Supervised Fine-Tuning (SFT)
#   - Reinforcement Learning (DPO, GRPO)
#
# Base image: NeMo Nemotron (has NeMo Framework, Megatron-Bridge, PyTorch)
#
# Build:
#   docker compose build nemotron-trainer
# =============================================================================

FROM nvcr.io/nvidia/nemo:25.11.nemotron_3_nano

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
        git \
        git-lfs \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Add user to sudoers
RUN REAL_USER=$(id -u -n ${REMOTE_UID} 2>/dev/null || echo $REMOTE_USER) && \
    echo "$REAL_USER ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$REAL_USER && \
    chmod 0440 /etc/sudoers.d/$REAL_USER

# Copy the Nemotron repo into the container
COPY --chown=$REMOTE_UID:$REMOTE_GID . /workspace/nemotron

WORKDIR /workspace/nemotron

# Install Nemotron repo (base install only — no customize extras needed)
# NeMo and Megatron-Bridge are pre-installed in the base image.
RUN pip install --no-cache-dir -e .

# Install leptonai for remote execution support (DGX Cloud)
RUN pip install --no-cache-dir leptonai>=0.20.0

# Switch to the user
USER $REMOTE_UID

CMD ["tail", "-f", "/dev/null"]
