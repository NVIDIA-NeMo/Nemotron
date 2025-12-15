"""Pretrain data preparation command."""

from __future__ import annotations

import typer

from nemotron.kit.cli.recipe import recipe


@recipe(
    name="nano3/data/prep/pretrain",
    script_path="src/nemotron/recipes/nano3/stage0_pretrain/data_prep.py",
    config_dir="src/nemotron/recipes/nano3/stage0_pretrain/config/data_prep",
    default_config="default",
    torchrun=False,
    ray=True,
    packager="code",
)
def pretrain(ctx: typer.Context) -> None:
    """Tokenize data for pretraining (bin/idx format).

    Config sources merged in order:
    1. Default config (default.yaml)
    2. Named config via -c/--config
    3. env.toml profile via --run/--batch (merged into run.env)
    4. CLI dotlist overrides (e.g., sample=1000)

    Examples:
        nemotron nano3 data prep pretrain                  # local execution
        nemotron nano3 data prep pretrain sample=1000      # with sampling
        nemotron nano3 data prep pretrain --config tiny    # use tiny config
        nemotron nano3 data prep pretrain --run prep       # nemo-run attached
        nemotron nano3 data prep pretrain --dry-run        # preview config
    """
    ...
