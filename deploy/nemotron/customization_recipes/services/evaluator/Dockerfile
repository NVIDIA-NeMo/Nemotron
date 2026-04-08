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
# Nemotron Evaluator Container (nemotron-evaluator)
#
# Model evaluation container — the "sovereign container" for the
# customization pipeline. Handles:
#   - Academic benchmarks (MMLU, ARC, HellaSwag, etc.)
#   - Sovereign benchmarks via BYOB (Build Your Own Benchmark)
#   - LLM-as-judge evaluation
#   - Tool-calling evaluation (BFCL/Gorilla)
#   - NeMo-Skills benchmark evaluation
#
# Base image: NeMo Nemotron (same as trainer for model loading compatibility)
#
# Build:
#   docker compose build nemotron-evaluator
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
        wget \
        unzip \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Add user to sudoers
RUN REAL_USER=$(id -u -n ${REMOTE_UID} 2>/dev/null || echo $REMOTE_USER) && \
    echo "$REAL_USER ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$REAL_USER && \
    chmod 0440 /etc/sudoers.d/$REAL_USER

# Install NeMo-Skills for evaluation benchmarks
RUN git clone https://github.com/NVIDIA/NeMo-Skills.git /opt/nemo-skills && \
    cd /opt/nemo-skills && \
    git checkout 63cf71f4706b9c4ad959be7563ee9b88864da1eb && \
    pip install --no-cache-dir -e .
ENV PYTHONPATH="/opt/nemo-skills:${PYTHONPATH}"
ENV NEMO_SKILLS_DATA_DIR="/opt/nemo-skills/nemo_skills/dataset"

# Install evaluation benchmarks (Google Research for academic benchmarks)
RUN mkdir -p /opt/benchmarks && \
    git clone --depth=1 https://github.com/google-research/google-research.git /opt/benchmarks/google-research

# Install Gorilla benchmarks (for BFCL tool-calling evaluation)
RUN git clone https://github.com/ShishirPatil/gorilla.git /opt/gorilla && \
    cd /opt/gorilla && \
    git checkout d2177992bbba9aa228b53c0645bf8f5613a5a7c6 && \
    cd berkeley-function-call-leaderboard && \
    pip install --no-cache-dir -e .

# Install evaluation-specific dependencies
RUN pip install --no-cache-dir \
    langdetect \
    absl-py \
    immutabledict \
    nltk \
    ipython \
    sacrebleu>=2.0.0 \
    openai>=1.0.0

# Install NLTK data and spaCy models (needed for benchmark processing)
RUN python -c "\
import nltk; \
from spacy.cli import download; \
nltk.download('punkt'); \
nltk.download('punkt_tab'); \
nltk.download('stopwords'); \
nltk.download('averaged_perceptron_tagger_eng'); \
download('en_core_web_sm'); \
download('xx_sent_ud_sm')"

# Copy the Nemotron repo (for eval pipeline access)
COPY --chown=$REMOTE_UID:$REMOTE_GID . /workspace/nemotron

WORKDIR /workspace/nemotron

# Install Nemotron repo (base install for eval pipeline access)
RUN pip install --no-cache-dir -e .

# Ensure benchmark directories are world-writable
RUN chmod -R 777 /opt/benchmarks/ /opt/gorilla/ /opt/nemo-skills/

# Switch to the user
USER $REMOTE_UID

CMD ["tail", "-f", "/dev/null"]
