#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/optimize/modelopt/prune"
# image = "nvcr.io/nvidia/nemo:26.02"
#
# [tool.runspec.run]
# launch = "python"
# workdir = "/opt/Model-Optimizer/examples/megatron_bridge"
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

"""Generic ModelOpt structured-pruning launcher through Megatron-Bridge."""

from __future__ import annotations

from pathlib import Path

from nemotron.steps._runners.modelopt import exec_torchrun_script

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"
UPSTREAM_SCRIPT = "/opt/Model-Optimizer/examples/megatron_bridge/prune_minitron.py"

# Backward-compatible flat config keys. New configs should put upstream script
# arguments under `args:` so users can control ModelOpt without editing Python.
LEGACY_FORWARDED_FIELDS = (
    "pp_size",
    "hf_model_name_or_path",
    "output_hf_path",
    "prune_target_params",
    "prune_export_config",
    "hparams_to_skip",
    "num_layers_in_first_pipeline_stage",
    "num_layers_in_last_pipeline_stage",
    "trust_remote_code",
)


def main() -> None:
    exec_torchrun_script(
        default_config=DEFAULT_CONFIG,
        upstream_script=UPSTREAM_SCRIPT,
        forwarded_fields=LEGACY_FORWARDED_FIELDS,
        flag_style="underscore",
        default_nproc_per_node=8,
    )


if __name__ == "__main__":
    main()
