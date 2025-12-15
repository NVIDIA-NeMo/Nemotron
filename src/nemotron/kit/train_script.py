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

# Copyright (c) Nemotron Contributors
# SPDX-License-Identifier: MIT

"""
Helpers for training scripts using OmegaConf-based configuration.

This module provides utilities similar to megatron-bridge's omegaconf_utils
but without the megatron-bridge dependency, for use in data prep scripts.

Usage:
    from nemotron.kit.train_script import (
        parse_config_and_overrides,
        load_omegaconf_yaml,
        apply_hydra_overrides,
        omegaconf_to_dataclass,
    )

    config_path, overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG)
    config = load_omegaconf_yaml(config_path)
    config = apply_hydra_overrides(config, overrides)
    cfg = omegaconf_to_dataclass(config, MyConfig)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from omegaconf import DictConfig, OmegaConf

T = TypeVar("T")


def parse_config_and_overrides(
    *,
    argv: list[str] | None = None,
    default_config: str | Path,
) -> tuple[str, list[str]]:
    """Parse `--config` plus unknown args as Hydra-style overrides.

    Args:
        argv: CLI arguments. Defaults to sys.argv[1:].
        default_config: Default config file path.

    Returns:
        Tuple of (config_path, overrides).
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help="Path to the YAML config file",
    )

    args, overrides = parser.parse_known_args(argv)
    return args.config, overrides


def load_omegaconf_yaml(path: str | Path) -> DictConfig:
    """Load YAML config file into OmegaConf DictConfig.

    Args:
        path: Path to YAML config file.

    Returns:
        OmegaConf DictConfig.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    return OmegaConf.load(path)


def apply_hydra_overrides(config: DictConfig, overrides: list[str]) -> DictConfig:
    """Apply Hydra-style CLI overrides to OmegaConf config.

    Supports:
    - key=value syntax with nested keys (e.g., train.lr=0.001)
    - --flag boolean flags for config fields (e.g., --force sets force=true)
    - --no-flag to set boolean to false (e.g., --no-force sets force=false)

    Unknown keys are silently ignored.

    Args:
        config: Base OmegaConf DictConfig.
        overrides: List of CLI override strings.

    Returns:
        Modified config with overrides applied.
    """
    for override in overrides:
        # Handle --flag style boolean overrides (e.g., --force, --no-force)
        if override.startswith("--"):
            flag = override[2:]  # Remove --

            # Handle --no-flag syntax (sets to false)
            if flag.startswith("no-"):
                key = flag[3:]  # Remove no-
                parsed_value = False
            else:
                key = flag
                parsed_value = True

            # Convert dashes to underscores for config keys (e.g., --dry-run -> dry_run)
            key = key.replace("-", "_")

            # Only apply if key exists in config (ignore unknown flags)
            if key in config:
                try:
                    OmegaConf.update(config, key, parsed_value, merge=True)
                except Exception:
                    pass
            continue

        # Skip non key=value overrides
        if "=" not in override:
            continue

        key, value = override.split("=", 1)

        # Parse the value (handle booleans, numbers, null, etc.)
        parsed_value = _parse_override_value(value)

        # Apply using OmegaConf.update (handles nested keys like "train.lr")
        try:
            OmegaConf.update(config, key, parsed_value, merge=True)
        except Exception:
            # Silently ignore unknown keys (like wandb.project)
            pass

    return config


def _parse_override_value(value: str) -> Any:
    """Parse a CLI override value string into Python type.

    Args:
        value: String value from CLI.

    Returns:
        Parsed Python value (bool, int, float, None, or str).
    """
    # Handle null/None
    if value.lower() in ("null", "none", "~"):
        return None

    # Handle booleans
    if value.lower() in ("true", "yes", "on"):
        return True
    if value.lower() in ("false", "no", "off"):
        return False

    # Handle integers
    try:
        return int(value)
    except ValueError:
        pass

    # Handle floats
    try:
        return float(value)
    except ValueError:
        pass

    # Return as string
    return value


def omegaconf_to_dataclass(config: DictConfig, cls: type[T]) -> T:
    """Convert OmegaConf DictConfig to a dataclass instance.

    Handles nested dataclasses and Path fields.

    Args:
        config: OmegaConf DictConfig to convert.
        cls: Target dataclass type.

    Returns:
        Dataclass instance with values from config.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} must be a dataclass")

    # Resolve config (evaluate interpolations like ${oc.env:VAR})
    resolved = OmegaConf.to_container(config, resolve=True)

    # Convert to dataclass
    return _dict_to_dataclass(resolved, cls)


def _dict_to_dataclass(data: dict[str, Any], cls: type[T]) -> T:
    """Recursively convert dict to dataclass.

    Args:
        data: Dictionary with config values.
        cls: Target dataclass type.

    Returns:
        Dataclass instance.
    """
    if not is_dataclass(cls):
        return data  # type: ignore

    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue

        value = data[f.name]

        # Handle nested dataclasses
        if is_dataclass(f.type) and isinstance(value, dict):
            value = _dict_to_dataclass(value, f.type)
        # Handle Path fields
        elif f.type == Path and isinstance(value, str):
            value = Path(value)
        # Handle Optional[Path]
        elif hasattr(f.type, "__origin__"):
            # Check for Optional[Path] = Union[Path, None]
            args = getattr(f.type, "__args__", ())
            if Path in args and isinstance(value, str):
                value = Path(value)

        kwargs[f.name] = value

    return cls(**kwargs)


def init_wandb_from_env() -> None:
    """Initialize wandb from environment variables.

    Reads WANDB_PROJECT and WANDB_ENTITY from environment.
    If WANDB_PROJECT is set, initializes wandb and sets up the WandbTracker
    for artifact lineage tracking.

    This is used when running via nemo-run where wandb config
    is passed via environment variables to Ray workers.
    """
    project = os.environ.get("WANDB_PROJECT")
    entity = os.environ.get("WANDB_ENTITY")

    if not project:
        return

    try:
        import wandb

        # Only init if not already initialized
        if wandb.run is None:
            wandb.init(
                project=project,
                entity=entity,
                job_type="data-prep",
            )
            # Set up lineage tracker for artifact publishing
            from nemotron.kit.trackers import WandbTracker, set_lineage_tracker

            set_lineage_tracker(WandbTracker())
    except ImportError:
        pass
