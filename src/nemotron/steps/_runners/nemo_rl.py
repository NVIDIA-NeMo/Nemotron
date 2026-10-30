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

"""Shared NeMo-RL launch helpers for DPO/RLVR/RLHF steps."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def exec_nemo_rl_example(*, default_config: Path, upstream_script: str, description: str) -> None:
    """Forward ``--config`` and remaining overrides to a NeMo-RL example script."""
    args, overrides = parse_nemo_rl_args(default_config=default_config, description=description)

    cmd = [sys.executable, upstream_script, "--config", args.config, *overrides]
    os.execvp(cmd[0], cmd)


def exec_or_run_nemo_rl_grpo(
    *,
    default_config: Path,
    upstream_script: str,
    description: str,
) -> None:
    """Run GRPO through NeMo-Gym mode when requested, otherwise exec an upstream example."""
    args, overrides = parse_nemo_rl_args(default_config=default_config, description=description)
    config_path = Path(args.config)

    if should_use_nemo_gym_config(config_path, overrides):
        from nemotron.steps._runners.nemo_rl_grpo_nemo_gym import run_nemo_gym_grpo

        run_nemo_gym_grpo(config_path=config_path, overrides=overrides)
        return

    cmd = [sys.executable, upstream_script, "--config", str(config_path), *overrides]
    os.execvp(cmd[0], cmd)


def parse_nemo_rl_args(
    *,
    default_config: Path,
    description: str,
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    """Parse ``--config`` and preserve Hydra-style overrides for NeMo-RL."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=str(default_config))
    return parser.parse_known_args(argv)


def should_use_nemo_gym_config(config_path: Path, overrides: list[str] | None = None) -> bool:
    """Return whether the config requests the explicit NeMo-Gym GRPO runner."""
    config = load_nemo_rl_step_config(config_path, overrides or [])
    return bool(OmegaConf.select(config, "env.should_use_nemo_gym", default=False))


def load_nemo_rl_step_config(config_path: Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a step config, including a small local ``defaults`` convenience.

    This intentionally supports only the simple defaults form used by our step
    configs: ``defaults: base.yaml`` or a list of YAML filenames. It is not a
    full Hydra composition engine.
    """
    config = _load_config_with_local_defaults(config_path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    return config


def _load_config_with_local_defaults(config_path: Path) -> DictConfig:
    config_path = Path(config_path)
    config = OmegaConf.load(config_path)
    defaults = OmegaConf.select(config, "defaults", default=None)
    if not defaults:
        return config

    base = OmegaConf.create()
    default_items = defaults if isinstance(defaults, list) else [defaults]
    for item in default_items:
        if not isinstance(item, str):
            raise TypeError(f"{config_path}: only string YAML defaults are supported")
        default_path = Path(item)
        if not default_path.is_absolute():
            default_path = config_path.parent / default_path
        base = OmegaConf.merge(base, _load_config_with_local_defaults(default_path))

    current = OmegaConf.to_container(config, resolve=False)
    if isinstance(current, dict):
        current.pop("defaults", None)
    return OmegaConf.merge(base, OmegaConf.create(current))
