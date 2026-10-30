#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "omni3/rl/vision"
# image = "/home/${oc.env:USER}/.cache/nemotron/containers/omni3-rl.sqsh"
# setup = "Build the Omni RL container with `nemotron omni3 build rl` before training."
#
# [tool.runspec.run]
# launch = "ray"
# workdir = "/opt/nemo-rl-omni"
# cmd = "bash scripts/nanov3_vision_rl.sh"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "omegaconf"
#
# [tool.runspec.resources]
# # Mirrors the upstream ``scripts/nanov3_vision_rl.sh`` defaults
# # (``NUM_NODES=4``, ``GPUS_PER_NODE=8``). Override per-cluster via
# # env.toml or the ``run.env.nodes`` / ``run.env.nproc_per_node`` CLI
# # dotlist if your config uses different parallelism.
# nodes = 4
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

"""Thin wrapper around the upstream Omni vision RL launcher."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from nemotron.kit.train_script import (
    apply_hydra_overrides,
    load_omegaconf_yaml,
    parse_config_and_overrides,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"
DEFAULT_COMMAND = "bash scripts/nanov3_vision_rl.sh"
DEFAULT_WORKDIR = Path("/opt/nemo-rl-omni")


def _load_resolved_config() -> dict[str, Any]:
    try:
        config_path, cli_overrides = parse_config_and_overrides(
            default_config=DEFAULT_CONFIG_PATH
        )
        config = load_omegaconf_yaml(config_path)
    except FileNotFoundError as exc:
        LOGGER.error(str(exc))
        sys.exit(1)

    if cli_overrides:
        config = apply_hydra_overrides(config, cli_overrides)

    from nemo_runspec.config.resolvers import (
        clear_artifact_cache,
        register_resolvers_from_config,
    )

    clear_artifact_cache()
    register_resolvers_from_config(
        config,
        artifacts_key="run",
        mode="pre_init",
        pre_init_patch_http_digest=False,
    )

    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        LOGGER.error("Expected mapping config, got %s", type(resolved).__name__)
        sys.exit(1)
    return resolved


def _build_command(config: dict[str, Any]) -> tuple[Path, list[str], dict[str, str]]:
    run_env = config.get("run", {}).get("env", {})
    env_vars = {
        str(key): str(value)
        for key, value in (run_env.get("env_vars") or {}).items()
        if value is not None
    }

    workdir = Path(env_vars.get("NEMORL") or run_env.get("workdir") or DEFAULT_WORKDIR)
    env_vars.setdefault("NEMORL", str(workdir))

    command = str(config.get("launcher", {}).get("command", DEFAULT_COMMAND))
    return workdir, ["bash", "-lc", command], env_vars


def main() -> None:
    """Entry point for omni3 vision RL."""
    config = _load_resolved_config()
    workdir, command, env_vars = _build_command(config)

    if not workdir.exists():
        LOGGER.error("Configured workdir does not exist: %s", workdir)
        sys.exit(1)

    print(f"Executing: {shlex.join(command)}")
    subprocess.check_call(command, cwd=workdir, env={**os.environ, **env_vars})


if __name__ == "__main__":
    main()
