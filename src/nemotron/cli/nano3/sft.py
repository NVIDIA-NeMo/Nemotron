"""SFT command implementation.

This module defines the `sft` command for the nano3 recipe.
"""

from __future__ import annotations

import typer

from nemotron.kit.cli.recipe import recipe

# Paths relative to repository root
SCRIPT_PATH = "src/nemotron/recipes/nano3/stage1_sft/train.py"
CONFIG_DIR = "src/nemotron/recipes/nano3/stage1_sft/config"


@recipe(
    name="nano3/sft",
    script_path=SCRIPT_PATH,
    config_dir=CONFIG_DIR,
    default_config="default",
    packager="self_contained",
)
def sft(ctx: typer.Context) -> None:
    """Run supervised fine-tuning with Megatron-Bridge (stage1).

    Config sources merged in order:
    1. Default config (default.yaml)
    2. Named config via -c/--config
    3. env.toml profile via --run/--batch (merged into run.env)
    4. CLI dotlist overrides (e.g., train.train_iters=5000)

    Examples:
        nemotron nano3 sft -c test                       # local execution
        nemotron nano3 sft --config test --run dlw       # nemo-run attached
        nemotron nano3 sft -c test -r dlw train.train_iters=5000
        nemotron nano3 sft -c test --dry-run             # preview config
        nemotron nano3 sft -c test --batch dlw --mock    # detached + passthrough
    """
    ...
