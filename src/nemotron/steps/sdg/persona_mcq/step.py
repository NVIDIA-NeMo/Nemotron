#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/sdg/persona_mcq"
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
# gpus_per_node = 1
# ///

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the Persona MCQ pipeline."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from nemotron.kit.train_script import apply_hydra_overrides, load_omegaconf_yaml, parse_config_and_overrides
from nemotron.steps.sdg.persona_mcq.runtime.pipeline import PersonaMCQPipeline

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"


def main() -> None:
    config_path, overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG)
    raw = apply_hydra_overrides(load_omegaconf_yaml(config_path), overrides)
    config = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(config, dict):
        raise TypeError(f"{config_path}: Persona MCQ config must be a mapping")
    PersonaMCQPipeline(config).run()


if __name__ == "__main__":
    main()
