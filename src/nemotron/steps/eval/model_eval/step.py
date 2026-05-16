#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/eval/model_eval"
#
# [tool.runspec.run]
# launch = "python"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
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

"""Entry point for the generic NeMo Evaluator Launcher step."""

from __future__ import annotations

from pathlib import Path

from nemotron.steps.eval.model_eval.runtime import run_model_eval

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"


def main() -> None:
    run_model_eval(default_config=DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
