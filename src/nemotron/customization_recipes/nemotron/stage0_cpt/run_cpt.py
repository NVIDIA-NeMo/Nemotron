"""CPT training for customization recipes -- REMOVED.

This stage reuses the production nano3 pretrain script directly:

    src/nemotron/recipes/nano3/stage0_pretrain/train.py

Customization is done via config overrides, not script changes.
See config/default.yaml in this directory for customization-specific settings.

The CLI command ``nemotron customize cpt`` points SCRIPT_PATH to the nano3
script and overrides config_dir to use this stage's config/ directory.
See: src/nemotron/cli/commands/customize/cpt.py
"""

raise ImportError(
    "run_cpt.py has been removed. "
    "Use src/nemotron/recipes/nano3/stage0_pretrain/train.py directly, "
    "or run: nemotron customize cpt -c default"
)
