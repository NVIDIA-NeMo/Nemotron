#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/prep/rl_prep"
# image = "anyscale/ray:2.49.2-py312"
#
# [tool.runspec.run]
# launch = "python"
# cmd = "uv run --extra xenna python {script} --config {config}"
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

"""Thin RL data-prep wrapper. Resolves HF placeholders and shards prompt / preference JSONL for NeMo-RL."""
from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from nemotron.data_prep import DataBlend, ObservabilityConfig
from nemotron.data_prep.recipes.rl import run_rl_resolve_pipeline
from nemotron.kit.train_script import (
    apply_hydra_overrides,
    load_omegaconf_yaml,
    parse_config_and_overrides,
)

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"


def main() -> None:
    config_path, overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG)
    cfg = OmegaConf.to_container(
        apply_hydra_overrides(load_omegaconf_yaml(config_path), overrides),
        resolve=True,
    )

    run_rl_resolve_pipeline(
        blend=DataBlend.load(cfg["blend_path"]),
        output_dir=cfg["output_dir"],
        sample=cfg.get("sample"),
        force=cfg.get("force", False),
        compression=cfg.get("compression", "none"),
        num_shards_per_split=cfg.get("num_shards_per_split", 1),
        resolve_hf_placeholders=cfg.get("resolve_hf_placeholders", True),
        observability=ObservabilityConfig(**cfg.get("observability", {})),
    )


if __name__ == "__main__":
    main()
