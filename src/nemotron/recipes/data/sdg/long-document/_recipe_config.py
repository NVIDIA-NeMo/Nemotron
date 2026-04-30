# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared YAML + OmegaConf + Pydantic config loader for long-document SDG recipes.

The 9 long-document stages follow the same load pattern:

  1. Parse ``--config <yaml>`` from argv.
  2. Read the YAML with OmegaConf.
  3. Apply remaining argv tokens as Hydra-style ``key=value`` dotlist overrides.
  4. Validate the merged container through a stage-specific Pydantic model.

This module factors that pattern so each stage script stays small and the
behavior stays consistent.  It depends only on ``pydantic`` + ``omegaconf`` +
``pyyaml`` (all available via the recipes' PEP 723 inline ``dependencies``
list); it deliberately does **not** import ``nemo_runspec`` so that
``uv run --no-project <stage>.py`` resolves nothing beyond the script's
declared deps.

Usage from a stage script (after ``sys.path.insert(0, str(Path(__file__).parent))``):

    from _recipe_config import load_recipe_config

    DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "06-single-page-qa.yaml"

    def main(cfg: SinglePageQAConfig | None = None) -> None:
        if cfg is None:
            cfg = load_recipe_config(DEFAULT_CONFIG_PATH, SinglePageQAConfig)
        run_single_page_qa(cfg)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, TypeVar

from omegaconf import OmegaConf
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def load_recipe_config(
    default_config_path: Path,
    model_cls: type[T],
    argv: list[str] | None = None,
) -> T:
    """Parse ``--config`` + dotlist overrides from argv and validate via Pydantic.

    Args:
        default_config_path: Path to the stage's default YAML if ``--config`` is omitted.
        model_cls: Pydantic model class to validate the merged config against.
        argv: Optional argv to parse (defaults to ``sys.argv[1:]``).

    Returns:
        A validated instance of ``model_cls``.

    Exits with status 1 if the resolved config file is missing.
    """
    parser = argparse.ArgumentParser(
        description=f"Long-document SDG recipe ({model_cls.__name__}).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path,
        help=f"Path to YAML config file (default: {default_config_path}).",
    )
    args, overrides = parser.parse_known_args(argv)
    if not args.config.exists():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    yaml_cfg = OmegaConf.load(args.config)
    override_cfg = OmegaConf.from_dotlist(overrides) if overrides else OmegaConf.create({})
    merged = OmegaConf.merge(yaml_cfg, override_cfg)
    container: dict[str, Any] = OmegaConf.to_container(merged, resolve=True)  # type: ignore[assignment]
    return model_cls(**container)


__all__ = ["load_recipe_config"]
