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
Reproducible training recipes for NVIDIA Nemotron models.

This package contains full training pipelines for various Nemotron models,
including data curation, training, and evaluation stages.

Available Recipes:
- nano3: Nemotron Nano 3 (2B parameters)
- chipnemo: ChipNeMo/ScaleRTL (Domain-adapted for RTL code generation)

Usage:
    # Run complete stage
    uv run python -m nemotron.recipes.nano3.stage0_pretrain --scale tiny

    # Run individual steps
    uv run python -m nemotron.recipes.nano3.stage0_pretrain.data_curation --scale tiny
"""
