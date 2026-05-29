#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/peft/megatron_bridge"
# image = "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
#
# [tool.runspec.run]
# launch = "torchrun"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
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

"""Megatron-Bridge PEFT (LoRA / adapter fine-tuning).

Same generic runner as ``sft/megatron_bridge``, but with the PEFT block
expected. Use ``hf_model_path`` to start from HF weights or
``checkpoint.pretrained_checkpoint`` for a Megatron base. The YAML ``peft:``
block is required:

    peft:
      type: lora            # or 'dora', etc. — see Megatron-Bridge docs
      dim: 16
      alpha: 32
      target_modules: [linear_qkv, linear_proj]
"""

from __future__ import annotations

from pathlib import Path

from megatron.bridge.training.finetune import finetune

from nemotron.steps._runners.megatron_bridge import run_megatron_bridge

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"
DEFAULT_RECIPE = "megatron.bridge.recipes.nemotronh.nemotron_3_nano.nemotron_3_nano_finetune_config"


def main() -> None:
    run_megatron_bridge(
        default_recipe=DEFAULT_RECIPE,
        default_config=DEFAULT_CONFIG,
        entry=finetune,
        enable_peft=True,
        enable_hf_weights=True,
        dataset_mode="finetune",
    )


if __name__ == "__main__":
    main()
