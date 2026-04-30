#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/sft/megatron_bridge"
# image = "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
#
# [tool.runspec.run]
# launch = "torchrun"
#
# [tool.runspec.config]
# dir = "./config"
# default = "nano3"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 8
# ///

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

"""Megatron-Bridge SFT.

YAML drives everything: ``recipe._target_`` swaps the recipe; ``hf_model_path``
loads weights from HF; ``dataset:`` overrides the FinetuningDatasetConfig;
``peft:`` enables LoRA-style adapters; every other top-level section
(``train``, ``checkpoint``, ``optimizer``, …) is auto-discovered and merged
onto the recipe's ConfigContainer.
"""
from __future__ import annotations

from pathlib import Path

from megatron.bridge.training.finetune import finetune

from nemotron.steps._runners.megatron_bridge import run_megatron_bridge

DEFAULT_CONFIG = Path(__file__).parent / "config" / "nano3.yaml"
DEFAULT_RECIPE = "megatron.bridge.recipes.nemotronh.nemotron_3_nano.nemotron_3_nano_finetune_config"


def main() -> None:
    run_megatron_bridge(
        default_recipe=DEFAULT_RECIPE,
        default_config=DEFAULT_CONFIG,
        entry=finetune,
    )


if __name__ == "__main__":
    main()
