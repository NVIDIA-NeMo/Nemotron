#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/sft/automodel"
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
# gpus_per_node = 4
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

"""SFT via NeMo-AutoModel. The recipe class is YAML-overridable.

By default uses ``TrainFinetuneRecipeForNextTokenPrediction``; set the
top-level ``_step_recipe`` key in YAML to swap in a different AutoModel
recipe class (e.g. classification, vision-language).
"""

from __future__ import annotations

from pathlib import Path

from nemotron.steps._runners.automodel import run_automodel

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"
DEFAULT_TARGET = "nemo_automodel.recipes.llm.train_ft:TrainFinetuneRecipeForNextTokenPrediction"


def main() -> None:
    run_automodel(default_target=DEFAULT_TARGET, default_config=DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
