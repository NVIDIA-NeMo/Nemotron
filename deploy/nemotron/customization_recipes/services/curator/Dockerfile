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
# Nemotron Curator Container (nemotron-curator)
#
# Data processing container for the customization pipeline. Handles:
#   - Data acquisition, language/domain filtering
#   - Translation (Google, AWS, LLM backends)
#   - Synthetic Data Generation (SDG via DataDesigner)
#   - Build Your Own Benchmark (BYOB MCQ generation)
#   - Data quality assessment and filtering
#
# Base image: NeMo Curator (has NeMo Curator, Dask, GPU-accelerated NLP)
#
# Build:
#   docker compose build nemotron-curator
# =============================================================================

FROM nvcr.io/nvidia/nemo-curator:26.02

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

# Install additional data processing dependencies
# (NeMo Curator is pre-installed in the base image)
RUN pip install --no-cache-dir \
    data-designer>=0.2.0 \
    google-cloud-translate>=3.0.0 \
    boto3>=1.26.0 \
    openai>=1.0.0 \
    instructor>=1.0.0 \
    sacrebleu>=2.0.0 \
    pycountry>=22.0.0 \
    bcp47>=0.0.4 \
    langdetect>=1.0.9 \
    aiohttp>=3.8.0 \
    tiktoken>=0.5.0 \
    json-schema-to-pydantic>=0.2.0 \
    absl-py \
    immutabledict \
    nltk \
    ipython

# Install NeMo-Skills for evaluation benchmarks
RUN git clone https://github.com/NVIDIA/NeMo-Skills.git /opt/nemo-skills && \
    cd /opt/nemo-skills && \
    git checkout 63cf71f4706b9c4ad959be7563ee9b88864da1eb && \
    pip install --no-cache-dir -e .
ENV PYTHONPATH="/opt/nemo-skills:${PYTHONPATH}"
ENV NEMO_SKILLS_DATA_DIR="/opt/nemo-skills/nemo_skills/dataset"

# Install Gorilla benchmarks (for BFCL tool-calling evaluation)
RUN git clone https://github.com/ShishirPatil/gorilla.git /opt/gorilla && \
    cd /opt/gorilla && \
    git checkout d2177992bbba9aa228b53c0645bf8f5613a5a7c6 && \
    cd berkeley-function-call-leaderboard && \
    pip install --no-cache-dir -e .

# Install NLTK data and spaCy models
RUN python -c "\
import nltk; \
from spacy.cli import download; \
nltk.download('punkt'); \
nltk.download('punkt_tab'); \
nltk.download('stopwords'); \
nltk.download('averaged_perceptron_tagger_eng'); \
download('en_core_web_sm'); \
download('xx_sent_ud_sm')"

# Install NGC CLI (required for data-designer persona downloads)
RUN cd /tmp && \
    wget -q -O ngccli_linux.zip https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.41.4/files/ngccli_linux.zip && \
    unzip -q ngccli_linux.zip && \
    mv ngc-cli /opt/ngc-cli && \
    rm ngccli_linux.zip
ENV PATH="/opt/ngc-cli:${PATH}"

# Copy the Nemotron repo (for data_prep library access)
COPY --chown=$REMOTE_UID:$REMOTE_GID . /workspace/nemotron

WORKDIR /workspace/nemotron

# Install Nemotron repo (base install — data_prep modules are needed)
RUN pip install --no-cache-dir -e .

# Ensure benchmark directories are world-writable
RUN chmod -R 777 /opt/gorilla/ /opt/nemo-skills/

# Switch to the user
USER $REMOTE_UID

CMD ["tail", "-f", "/dev/null"]
