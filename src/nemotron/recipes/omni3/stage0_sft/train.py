#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "omni3/sft"
# image = "oci-archive:///home/$USER/.cache/nemotron/containers/omni3-sft.tar"
# setup = "Build the Omni SFT container with `nemotron omni3 build sft` before training."
#
# [tool.runspec.run]
# launch = "direct"
# workdir = "/workspace/Megatron-Bridge"
# cmd = "uv run torchrun --nproc-per-node=8 scripts/training/run_recipe.py --recipe {recipe} --step_func nemotron_omni_step"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 2
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

"""Thin Omni SFT launcher for Megatron-Bridge's run_recipe.py."""

from __future__ import annotations

import logging
import shlex
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from nemotron.kit.train_script import load_omegaconf_yaml, parse_config_and_overrides

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"
DEFAULT_WORKDIR = Path("/workspace/Megatron-Bridge")


def _flatten_overrides(prefix: str, value: Any) -> list[str]:
    """Flatten a nested mapping into Hydra-style key=value overrides."""
    if isinstance(value, Mapping):
        overrides: list[str] = []
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            overrides.extend(_flatten_overrides(nested_prefix, nested_value))
        return overrides
    if value is None:
        return [f"{prefix}=null"]
    return [f"{prefix}={value}"]


def _build_command(config: dict[str, Any], cli_overrides: list[str]) -> tuple[Path, list[str]]:
    """Build the Megatron-Bridge run_recipe.py command."""
    recipe_cfg = config.get("recipe", {})
    run_cfg = config.get("run", {})
    env_cfg = run_cfg.get("env", {})

    recipe_name = recipe_cfg.get("name")
    if not recipe_name:
        raise ValueError("recipe.name is required")

    command = [
        "uv",
        "run",
        "torchrun",
        f"--nproc-per-node={run_cfg.get('nproc_per_node', 8)}",
        "scripts/training/run_recipe.py",
        "--recipe",
        str(recipe_name),
        "--step_func",
        str(recipe_cfg.get("step_func", "nemotron_omni_step")),
    ]

    recipe_dataset = recipe_cfg.get("dataset")
    if recipe_dataset:
        command.extend(["--dataset", str(recipe_dataset)])

    for section in ("checkpoint", "dataset", "model", "train", "logger", "optimizer", "scheduler"):
        section_value = config.get(section)
        if section_value:
            command.extend(_flatten_overrides(section, section_value))

    command.extend(cli_overrides)
    workdir = Path(env_cfg.get("workdir", DEFAULT_WORKDIR))
    return workdir, command


def main() -> None:
    """Entry point for omni3 SFT."""
    try:
        config_path, cli_overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG_PATH)
        config = load_omegaconf_yaml(config_path)
    except FileNotFoundError as exc:
        LOGGER.error(str(exc))
        sys.exit(1)

    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        LOGGER.error("Expected mapping config, got %s", type(resolved).__name__)
        sys.exit(1)

    try:
        workdir, command = _build_command(resolved, cli_overrides)
    except Exception as exc:  # pragma: no cover - user error path
        LOGGER.error(str(exc))
        sys.exit(1)

    if not workdir.exists():
        LOGGER.error("Configured workdir does not exist: %s", workdir)
        sys.exit(1)

    print(f"Executing: {shlex.join(command)}")
    subprocess.check_call(command, cwd=workdir)


if __name__ == "__main__":
    main()
