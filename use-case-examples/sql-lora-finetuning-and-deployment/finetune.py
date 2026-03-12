# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import annotations

import os

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

# Default config: YAML in the same directory as this script (BIRD SQL PEFT for Nemotron-3-Nano, 2 GPUs)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG = os.path.join(_THIS_DIR, "bird_peft_nemotron_nano.yaml")


def main(default_config_path: str = _DEFAULT_CONFIG) -> None:
    """Main entry point for the fine-tuning recipe.

    Loads the configuration, sets up the recipe, and initiates the training loop.
    Override the config path via CLI: --config /path/to/other.yaml
    """
    cfg = parse_args_and_load_config(default_config_path)
    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
