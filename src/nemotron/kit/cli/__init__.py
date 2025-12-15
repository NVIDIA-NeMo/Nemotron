"""CLI utilities for nemotron.

This module provides shared CLI infrastructure built on Typer + OmegaConf.
"""

from nemotron.kit.cli.globals import GlobalContext, global_callback
from nemotron.kit.cli.recipe import recipe

__all__ = [
    "GlobalContext",
    "global_callback",
    "recipe",
]
