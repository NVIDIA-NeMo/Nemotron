#!/usr/bin/env python3
"""Stage-specific wrapper for Omni MPO data prep.

Note: the runspec header (image, run.cmd, resources) lives on the parent
``omni3/stage1_rl/data_prep.py`` module; the CLI dispatcher reads it
from there for all three sub-stage shims. Don't add a runspec header
here unless you also rewire the dispatcher.
"""

from __future__ import annotations

from pathlib import Path

from nemotron.recipes.omni3.stage1_rl._data_prep_base import (
    Omni3RLDataPrepConfig,
    main as _main,
)

DEFAULT_CONFIG_PATH = Path(__file__).parents[1] / "config" / "data_prep" / "mpo.yaml"

# Module-level flag for Ray execution (used by nemotron CLI dispatcher to
# decide whether to submit via RayJob vs run locally). Mirrors the parent
# data_prep.py flag.
RAY = True


def main(cfg: Omni3RLDataPrepConfig | None = None):
    return _main(default_config=DEFAULT_CONFIG_PATH, cfg=cfg)


if __name__ == "__main__":
    main()
