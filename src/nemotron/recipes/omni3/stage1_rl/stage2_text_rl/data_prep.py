#!/usr/bin/env python3
"""Stage-specific wrapper for Omni text-only RL data prep."""

from __future__ import annotations

from pathlib import Path

from nemotron.recipes.omni3.stage1_rl._data_prep_base import (
    Omni3RLDataPrepConfig,
    main as _main,
)

DEFAULT_CONFIG_PATH = Path(__file__).parents[1] / "config" / "data_prep" / "text.yaml"


def main(cfg: Omni3RLDataPrepConfig | None = None):
    return _main(default_config=DEFAULT_CONFIG_PATH, cfg=cfg)


if __name__ == "__main__":
    main()
