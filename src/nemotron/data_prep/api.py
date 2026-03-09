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

"""Public API for data preparation pipelines.

Supported Pipelines:
    - run_pretrain_pipeline: Tokenize to Megatron bin/idx format
    - run_sft_pipeline: Chat SFT to packed Parquet format

Usage:
    from nemotron.data_prep.api import run_pretrain_pipeline, run_sft_pipeline
    from nemotron.data_prep import DataBlend

    # Pretrain pipeline
    blend = DataBlend.load("pretrain_blend.json")
    result = run_pretrain_pipeline(
        blend=blend,
        output_dir="/output",
        tokenizer="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        num_shards=128,
    )

    # SFT pipeline
    blend = DataBlend.load("sft_blend.json")
    result = run_sft_pipeline(
        blend=blend,
        output_dir="/output",
        tokenizer="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        num_shards=64,
        chat_template="nano3",
    )
"""

# Re-export recipe entry points as the primary public API
from nemotron.data_prep.recipes.pretrain import run_pretrain_pipeline
from nemotron.data_prep.recipes.sft import run_sft_pipeline

__all__ = [
    "run_pretrain_pipeline",
    "run_sft_pipeline",
]
