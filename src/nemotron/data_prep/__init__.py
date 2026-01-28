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

"""Data preparation for Megatron training.

Processes raw text data from HuggingFace, S3, or local sources into
various training formats compatible with Megatron-Bridge and Megatron-Core.

Public API:
    The primary entry points are the pipeline recipes:
    - run_pretrain_pipeline: Tokenize to Megatron bin/idx format
    - run_sft_pipeline: Chat SFT to packed Parquet format

Pipeline recipes are available in:
    - recipes/pretrain.py for pretrain (binidx) data prep
    - recipes/sft.py for SFT (packed parquet) data prep

For stage-specific data preparation, see:
    - nemotron/recipes/nano3/stage0_pretrain/data_prep.py
    - nemotron/recipes/nano3/stage1_sft/data_prep.py
    - nemotron/recipes/nano3/stage2_rl/data_prep.py

Subpackages:
    - core: Low-level shard processing functions
    - observability: W&B and stats logging utilities
    - recipes: Pipeline orchestration (pretrain, sft)
    - stages: Xenna pipeline stages
    - formats: Output format builders
    - packing: Sequence packing algorithms

Usage:
    from nemotron.data_prep import (
        DataBlend,
        run_pretrain_pipeline,
        run_sft_pipeline,
    )

    # Pretrain pipeline
    blend = DataBlend.load("pretrain_blend.json")
    result = run_pretrain_pipeline(
        blend=blend,
        output_dir="/output",
        tokenizer="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        num_shards=128,
    )
"""

from nemotron.data_prep.blend import DataBlend, Dataset
from nemotron.data_prep.config import (
    BinIdxOutputConfig,
    ChatSftOutputConfig,
    DatasetConfig,
    FileInfo,
    FormatResult,
    HfDownloadConfig,
    JsonlOutputConfig,
    ObservabilityConfig,
    PerSplitConfig,
    TokenizerConfig,
    Transform,
)

from nemotron.data_prep.utils.discovery import get_dataset_metadata
from nemotron.data_prep.formats.transforms import (
    OpenAIChatRecord,
    ShareGPTRecord,
    SftRecord,
    openai_chat,
    passthrough,
    rename,
    select,
    sft,
    sharegpt,
)
from nemotron.kit.artifact import DataBlendsArtifact, PretrainBlendsArtifact
from nemotron.kit.trackers import InputDatasetInfo, tokenizer_to_uri
from nemotron.kit.wandb import finish_wandb

# Public API - Recipe entry points
from nemotron.data_prep.api import (
    run_pretrain_pipeline,
    run_sft_pipeline,
)

# Observability utilities
from nemotron.data_prep.observability.wandb_hook import (
    log_plan_table_to_wandb,
    make_wandb_stats_hook,
)

__all__ = [
    # Public API - Recipe entry points
    "run_pretrain_pipeline",
    "run_sft_pipeline",
    # Input specification
    "DataBlend",
    "Dataset",
    "DataBlendsArtifact",
    "PretrainBlendsArtifact",
    # Configuration
    "PerSplitConfig",
    "TokenizerConfig",
    "DatasetConfig",
    "FileInfo",
    "FormatResult",
    "HfDownloadConfig",
    "ObservabilityConfig",
    # Output format configs
    "BinIdxOutputConfig",
    "JsonlOutputConfig",
    "ChatSftOutputConfig",
    "Transform",
    # Transform factories
    "sft",
    "openai_chat",
    "sharegpt",
    "passthrough",
    "select",
    "rename",
    # Transform type definitions
    "SftRecord",
    "OpenAIChatRecord",
    "ShareGPTRecord",
    # Lineage tracking
    "InputDatasetInfo",
    "tokenizer_to_uri",
    "get_dataset_metadata",
    "finish_wandb",
    # Observability utilities
    "make_wandb_stats_hook",
    "log_plan_table_to_wandb",
]
