#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/optimize/modelopt/quantize"
# image = "nvcr.io/nvidia/nemo:26.02"
#
# [tool.runspec.run]
# launch = "python"
# workdir = "/opt/Megatron-Bridge"
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

"""Generic ModelOpt quantization launcher through Megatron-Bridge."""

from __future__ import annotations

from pathlib import Path

from nemotron.steps._runners.modelopt import exec_torchrun_script

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"
UPSTREAM_SCRIPT = "/opt/Megatron-Bridge/examples/quantization/quantize.py"

# Backward-compatible flat config keys. New configs should put upstream script
# arguments under `args:` so users can control ModelOpt without editing Python.
LEGACY_FORWARDED_FIELDS = (
    "hf_model_id",
    "export_quant_cfg",
    "megatron_save_path",
    "tp",
    "pp",
    "ep",
    "etp",
    "calib_size",
    "prompts",
    "compress",
    "weight_only",
    "export_kv_cache_quant",
    "trust_remote_code",
    "disable_hf_datasets_file_lock",
)


def main() -> None:
    exec_torchrun_script(
        default_config=DEFAULT_CONFIG,
        upstream_script=UPSTREAM_SCRIPT,
        forwarded_fields=LEGACY_FORWARDED_FIELDS,
        flag_style="hyphen",
        default_nproc_per_node=8,
    )


if __name__ == "__main__":
    main()
