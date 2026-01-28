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

"""
Xenna pipeline recipes for data preparation.

Recipes combine reusable stages from `nemotron.data_prep.stages` into
complete end-to-end pipelines for different data formats.

Available Recipes:
    run_pretrain_pipeline: Tokenize datasets to Megatron .bin/.idx format

Usage:
    from nemotron.data_prep.recipes import run_pretrain_pipeline
    from nemotron.data_prep.blend import DataBlend

    blend = DataBlend.load("blend.json")
    result = run_pretrain_pipeline(
        blend=blend,
        output_dir="/output",
        tokenizer="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        num_shards=128,
    )
"""

from nemotron.data_prep.recipes.pretrain import run_pretrain_pipeline
from nemotron.data_prep.recipes.rl import run_rl_resolve_pipeline
from nemotron.data_prep.recipes.sft import run_sft_pipeline

__all__ = [
    "run_pretrain_pipeline",
    "run_rl_resolve_pipeline",
    "run_sft_pipeline",
]
