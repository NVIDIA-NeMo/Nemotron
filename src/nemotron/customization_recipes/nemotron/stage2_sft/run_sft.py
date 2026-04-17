"""SFT training for customization recipes -- REMOVED.

This stage reuses the production nano3 SFT script directly:

    src/nemotron/recipes/nano3/stage1_sft/train.py

Customization is done via config overrides, not script changes.
See config/default.yaml in this directory for customization-specific settings.

The CLI command ``nemotron customize sft`` points SCRIPT_PATH to the nano3
script and overrides config_dir to use this stage's config/ directory.
See: src/nemotron/cli/commands/customize/sft.py
"""

raise ImportError(
    "run_sft.py has been removed. "
    "Use src/nemotron/recipes/nano3/stage1_sft/train.py directly, "
    "or run: nemotron customize sft -c default"
)
