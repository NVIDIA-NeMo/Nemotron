#!/usr/bin/env python3
"""Nano3 training recipe CLI.

Subcommands:
    nemotron nano3 data curate   - Curate training data with NeMo Curator (coming soon)
    nemotron nano3 data prep     - Download from HF Hub and tokenize with Ray Data
    nemotron nano3 pretrain      - Run pretraining with Megatron-Bridge (stage0)
    nemotron nano3 sft           - Run supervised fine-tuning with Megatron-Bridge (coming soon)
    nemotron nano3 rl            - Run reinforcement learning with NeMo-RL (coming soon)

Examples:
    # Data preparation (use --stage/-s to select stage)
    nemotron nano3 data prep --sample 1000
    nemotron nano3 data prep --stage sft
    nemotron nano3 data prep -s rl --sample 100

    # Pretraining (requires megatron-bridge)
    nemotron nano3 pretrain --mock --max-steps 1000

    # Direct module access still works
    python -m nemotron.recipes.nano3.stage0_pretrain.data_prep --sample 1000
"""

from __future__ import annotations

from typing import Annotated, Union

import tyro
from tyro.conf import OmitArgPrefixes, subcommand

from nemotron.recipes.nano3.stage0_pretrain.data_prep import Nano3DataPrepConfig

# Import ConfigContainer for training config
# Prefer megatron-bridge if available, otherwise use local stub for CLI development
try:
    from megatron.bridge.training.config import ConfigContainer as TrainingConfig
except ImportError:
    from nemotron.config import ConfigContainer as TrainingConfig


# =============================================================================
# Placeholder Configs
# =============================================================================


from dataclasses import dataclass


@dataclass
class DataCurateConfig:
    """Curate training data. Coming soon."""

    pass


@dataclass
class SftConfig:
    """Supervised fine-tuning configuration. Coming soon."""

    pass


@dataclass
class RlConfig:
    """Reinforcement learning configuration. Coming soon."""

    pass


# =============================================================================
# Command Type Alias
# =============================================================================

Nano3Command = Union[
    Annotated[
        DataCurateConfig,
        subcommand(name="data curate", description="Curate training data with NeMo Curator (coming soon)", prefix_name=False),
    ],
    Annotated[
        Nano3DataPrepConfig,
        subcommand(
            name="data prep",
            description="Download from HF Hub and tokenize with Ray Data (use --stage/-s for stage-specific data)",
            prefix_name=False,
        ),
    ],
    Annotated[
        TrainingConfig,
        subcommand(name="pretrain", description="Run pretraining with Megatron-Bridge (stage0)", prefix_name=False),
    ],
    Annotated[
        SftConfig,
        subcommand(name="sft", description="Run supervised fine-tuning with Megatron-Bridge (coming soon)", prefix_name=False),
    ],
    Annotated[
        RlConfig,
        subcommand(name="rl", description="Run reinforcement learning with NeMo-RL (coming soon)", prefix_name=False),
    ],
]


def cli(config: Annotated[Nano3Command, OmitArgPrefixes]) -> int:
    """Nano3 training recipe.

    Run data preparation or training for the Nano3 model.
    """
    match config:
        case Nano3DataPrepConfig() as cfg:
            from nemotron.recipes.nano3.stage0_pretrain.data_prep import main

            main(cfg)
            return 0

        case DataCurateConfig():
            print("Data curation coming soon")
            return 1

        case TrainingConfig() as cfg:
            from nemotron.recipes.nano3.stage0_pretrain.training import main

            main(cfg)
            return 0

        case SftConfig():
            print("Supervised fine-tuning coming soon")
            return 1

        case RlConfig():
            print("Reinforcement learning coming soon")
            return 1

        case _:
            print(f"Unknown command: {config}")
            return 1


if __name__ == "__main__":
    import sys

    result = tyro.cli(cli)
    if isinstance(result, int):
        sys.exit(result)
