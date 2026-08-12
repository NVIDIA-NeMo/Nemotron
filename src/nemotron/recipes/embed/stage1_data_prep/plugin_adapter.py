"""Translate Nemotron data-prep settings into the public conversion API."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_designer_retrieval_sdg import ConversionResult, ConversionRunConfig

    from nemotron.recipes.embed.stage1_data_prep.data_prep import DataPrepConfig

RETRIEVAL_SDG_SCHEMA_VERSION = 1


def build_conversion_config(cfg: DataPrepConfig) -> ConversionRunConfig:
    """Build the plugin's complete typed conversion config from a recipe profile."""
    from data_designer_retrieval_sdg import ConversionRunConfig

    if cfg.sdg_input_path is None:
        raise ValueError("sdg_input_path is required when conversion is enabled")

    return ConversionRunConfig(
        schema_version=RETRIEVAL_SDG_SCHEMA_VERSION,
        input_path=cfg.sdg_input_path.resolve(),
        corpus_id=cfg.corpus_id,
        output_dir=cfg.output_dir.resolve(),
        eval_only=False,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        seed=cfg.conversion_seed,
        quality_threshold=cfg.quality_threshold,
        max_pos_docs=cfg.max_pos_docs,
        use_group_id_in_eval=cfg.use_group_id_in_eval,
        split_strategy=cfg.split_strategy,
        groups_json=[path.resolve() for path in cfg.groups_json] if cfg.groups_json is not None else None,
    )


def execute_conversion(cfg: DataPrepConfig) -> ConversionResult:
    """Invoke the installed plugin after translating the recipe configuration."""
    from data_designer_retrieval_sdg import run_conversion_with_config

    return run_conversion_with_config(build_conversion_config(cfg))
