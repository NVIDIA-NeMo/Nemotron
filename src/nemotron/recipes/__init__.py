"""
Reproducible training recipes for NVIDIA Nemotron models.

This package contains full training pipelines for various Nemotron models,
including data curation, training, and evaluation stages.

Available Recipes:
- nano2: Nemotron Nano 2 (2B parameters)
- chipnemo: ChipNeMo/ScaleRTL (Domain-adapted for RTL code generation)

Usage:
    # Run complete stage
    uv run python -m nemotron.recipes.nano2.stage0_pretrain --scale tiny

    # Run individual steps
    uv run python -m nemotron.recipes.nano2.stage0_pretrain.data_curation --scale tiny
"""
