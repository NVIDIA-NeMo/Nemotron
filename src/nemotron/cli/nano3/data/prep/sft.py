"""SFT data preparation command."""

from __future__ import annotations

import typer

from nemotron.kit.cli.recipe import recipe


@recipe(
    name="nano3/data/prep/sft",
    script_path="src/nemotron/recipes/nano3/stage1_sft/data_prep.py",
    config_dir="src/nemotron/recipes/nano3/stage1_sft/config/data_prep",
    default_config="default",
    torchrun=False,
    ray=True,
    packager="code",
)
def sft(ctx: typer.Context) -> None:
    """Prepare data for SFT (packed .npy format).

    Applies chat templates to OpenAI-format messages, tokenizes with role-based
    loss masking, and outputs packed .npy files compatible with GPTSFTPackedDataset.

    Config sources merged in order:
    1. Default config (default.yaml)
    2. Named config via -c/--config
    3. env.toml profile via --run/--batch (merged into run.env)
    4. CLI dotlist overrides (e.g., sample=1000)

    Examples:
        nemotron nano3 data prep sft                   # local execution
        nemotron nano3 data prep sft sample=1000       # with sampling
        nemotron nano3 data prep sft --config tiny     # use tiny config
        nemotron nano3 data prep sft --run prep        # nemo-run attached
        nemotron nano3 data prep sft --dry-run         # preview config
    """
    ...
