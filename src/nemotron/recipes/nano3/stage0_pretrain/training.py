#!/usr/bin/env python3
"""Pretrain script for Nemotron Nano3.

Uses Megatron-Bridge's ConfigContainer for full training configuration.

Usage:
    # Piped from data_prep (preferred for pipelines)
    uv run -m nemotron.recipes.nano3.stage0_pretrain.data_prep | \
    torchrun --nproc_per_node=8 -m nemotron.recipes.nano3.stage0_pretrain.training

    # With explicit data path
    torchrun --nproc_per_node=8 -m nemotron.recipes.nano3.stage0_pretrain.training \
        --config.data.data-path /path/to/blend.json

    # With mock data for testing
    torchrun --nproc_per_node=8 -m nemotron.recipes.nano3.stage0_pretrain.training \
        --config.data.mock
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer


def main(config: ConfigContainer):
    """Run Nano3 pretraining."""
    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.training.pretrain import pretrain

    return pretrain(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    from megatron.bridge.training.config import ConfigContainer

    from nemotron.kit import cli

    cli(main, parse_inputs={"data.blend_path": "config.data.data_path"})
