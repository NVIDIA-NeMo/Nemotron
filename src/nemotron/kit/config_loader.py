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

"""pydantic_settings-based configuration loading for recipe scripts.

Uses pydantic_settings.CliSettingsSource for type-safe CLI override parsing
with support for complex types (lists, dicts), nested fields (dot notation),
and proper type coercion via Pydantic's type system.

    from nemotron.kit.config_loader import parse_config_and_overrides, load_config

    config_path, overrides = parse_config_and_overrides(default_config=DEFAULT)
    cfg = load_config(config_path, overrides, MyConfig)

Override syntax:
  - key=value        (Hydra-style, converted to --key=value for pydantic_settings)
  - --key value      (argparse-style, passed through)
  - --flag / --no-flag  (boolean toggles, converted to --flag=true/false)
  - model.lr=0.001   (nested fields via dot notation)
  - k_values=[1,2,3] (complex types parsed by Pydantic's type system)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic_settings import BaseSettings, CliSettingsSource

T = TypeVar("T", bound=BaseSettings)


class RecipeSettings(BaseSettings):
    """Base class for recipe configuration models.

    Extends pydantic_settings.BaseSettings but disables automatic env var
    loading to prevent unexpected overrides from environment variables
    (e.g., a ``BASE_MODEL`` env var overriding the ``base_model`` field).
    """

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, **kwargs):
        return (init_settings,)


def parse_config_and_overrides(
    *,
    argv: list[str] | None = None,
    default_config: str | Path,
) -> tuple[str, list[str]]:
    """Parse ``--config`` plus unknown args as overrides.

    Args:
        argv: CLI arguments. Defaults to sys.argv[1:].
        default_config: Default config file path.

    Returns:
        Tuple of (config_path, remaining_overrides).
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


def load_config(
    config_path: str | Path,
    overrides: list[str],
    model_cls: type[T],
) -> T:
    """Load YAML config, apply CLI overrides, return validated model.

    Uses pydantic_settings.CliSettingsSource for type-safe override parsing.
    Priority: CLI overrides > YAML values > model defaults.

    Args:
        config_path: Path to YAML config file.
        overrides: CLI override strings (key=value, --key value, etc.).
        model_cls: RecipeSettings subclass to validate against.

    Returns:
        Validated settings instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    path = str(config_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config_dict = yaml.safe_load(f) or {}

    # Strip 'run' key (used by nemo-run, not by config model)
    config_dict.pop("run", None)

    # Apply CLI overrides using pydantic_settings CliSettingsSource
    if overrides:
        cli_args = _hydra_to_cli_args(overrides)
        cli_source = CliSettingsSource(
            model_cls,
            cli_parse_args=cli_args,
            cli_ignore_unknown_args=True,
        )
        cli_values = cli_source()
        _deep_merge(config_dict, cli_values)

    return model_cls(**config_dict)


def _hydra_to_cli_args(overrides: list[str]) -> list[str]:
    """Convert Hydra-style overrides to argparse-style CLI args.

    Conversions:
      - ``key=value``    → ``--key=value``
      - ``--no-flag``    → ``--flag=false``
      - ``--key``        → ``--key`` (passed through; argparse knows the type)
      - ``--key value``  → passed through as-is
      - bare values      → passed through (argparse consumes them)

    Args:
        overrides: Raw CLI override strings.

    Returns:
        Argparse-compatible CLI args for CliSettingsSource.
    """
    cli_args = []
    for override in overrides:
        if override.startswith("--no-"):
            # --no-flag → --flag=false
            key = override[5:].replace("-", "_")
            cli_args.append(f"--{key}=false")
        elif override.startswith("--"):
            # Normalize dashes to underscores in key part, pass through as-is
            # Argparse handles --key=value and --key value pairs natively
            if "=" in override:
                key_part, _, val_part = override[2:].partition("=")
                cli_args.append(f"--{key_part.replace('-', '_')}={val_part}")
            else:
                cli_args.append(f"--{override[2:].replace('-', '_')}")
        elif "=" in override and not override.startswith("-"):
            # key=value → --key=value (Hydra-style)
            cli_args.append(f"--{override}")
        else:
            # Bare value (consumed by preceding --key), pass through
            cli_args.append(override)
    return cli_args


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Deep merge override into base dict (in-place).

    Args:
        base: Base dictionary (modified in-place).
        override: Override dictionary (values take precedence).
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
