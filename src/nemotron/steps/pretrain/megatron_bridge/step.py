#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/pretrain/megatron_bridge"
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

"""Megatron-Bridge pretraining / continued pretraining (CPT).

The same generic runner as ``sft/megatron_bridge``, but pointed at the
``pretrain`` entry point. All YAML sections are auto-discovered overrides;
``hf_model_path`` enables CPT from an HF base; ``dataset:`` is honored when
present (e.g. to swap data blends without forking the recipe).
"""

from __future__ import annotations

from pathlib import Path

from megatron.bridge.training.pretrain import pretrain

from nemotron.steps._runners.megatron_bridge import run_megatron_bridge

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"
DEFAULT_RECIPE = "megatron.bridge.recipes.nemotronh.nemotron_3_nano.nemotron_3_nano_pretrain_config"


def main() -> None:
    run_megatron_bridge(
        default_recipe=DEFAULT_RECIPE,
        default_config=DEFAULT_CONFIG,
        entry=pretrain,
        enable_peft=False,  # PEFT is for fine-tuning, not pretraining.
        enable_hf_weights=True,  # Useful for CPT from an HF base.
        dataset_mode="override",
    )


if __name__ == "__main__":
    main()
