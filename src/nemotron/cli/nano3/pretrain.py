"""Pretrain command implementation.

This module defines the `pretrain` command for the nano3 recipe.
"""

from __future__ import annotations

import typer

from nemotron.kit.cli.recipe import recipe

# Paths relative to repository root
SCRIPT_PATH = "src/nemotron/recipes/nano3/stage0_pretrain/train.py"
CONFIG_DIR = "src/nemotron/recipes/nano3/stage0_pretrain/config"


@recipe(
    name="nano3/pretrain",
    script_path=SCRIPT_PATH,
    config_dir=CONFIG_DIR,
    default_config="default",
    packager="self_contained",
    artifacts={
        "data": {
            "default": "PretrainDataArtifact-pretrain",
            "mappings": {"path": "recipe.per_split_data_args_path"},
        },
    },
)
def pretrain(ctx: typer.Context) -> None:
    """Run pretraining with Megatron-Bridge (stage0).

    Config sources merged in order:
    1. Default config (default.yaml)
    2. Named config via -c/--config
    3. env.toml profile via --run/--batch (merged into run.env)
    4. CLI dotlist overrides (e.g., train.train_iters=5000)

    Examples:
        nemotron nano3 pretrain -c test                       # local execution
        nemotron nano3 pretrain --config test --run dlw       # nemo-run attached
        nemotron nano3 pretrain -c test -r dlw train.train_iters=5000
        nemotron nano3 pretrain -c test --dry-run             # preview config
        nemotron nano3 pretrain -c test --batch dlw --mock    # detached + passthrough
    """
    ...
