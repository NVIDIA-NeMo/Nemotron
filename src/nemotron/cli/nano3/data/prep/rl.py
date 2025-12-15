"""RL data preparation command."""

from __future__ import annotations

import typer

from nemotron.kit.cli.recipe import recipe


@recipe(
    name="nano3/data/prep/rl",
    script_path="src/nemotron/recipes/nano3/stage2_rl/data_prep.py",
    config_dir="src/nemotron/recipes/nano3/stage2_rl/config/data_prep",
    default_config="default",
    torchrun=False,
    ray=True,
    packager="code",
)
def rl(ctx: typer.Context) -> None:
    """Prepare data for RL (JSONL chat format).

    Config sources merged in order:
    1. Default config (default.yaml)
    2. Named config via -c/--config
    3. env.toml profile via --run/--batch (merged into run.env)
    4. CLI dotlist overrides (e.g., sample=1000)

    Examples:
        nemotron nano3 data prep rl                    # local execution
        nemotron nano3 data prep rl sample=1000        # with sampling
        nemotron nano3 data prep rl --config tiny      # use tiny config
        nemotron nano3 data prep rl --run prep         # nemo-run attached
        nemotron nano3 data prep rl --dry-run          # preview config
    """
    ...
