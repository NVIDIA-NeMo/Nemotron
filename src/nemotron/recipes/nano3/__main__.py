#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Nano3 training recipe CLI.

Subcommands:
    nemotron nano3 data curate        - Curate training data with NeMo Curator (coming soon)
    nemotron nano3 data prep pretrain - Tokenize data for pretraining (bin/idx format)
    nemotron nano3 data prep sft      - Prepare data for SFT (JSONL format)
    nemotron nano3 data prep rl       - Prepare data for RL (JSONL chat format)
    nemotron nano3 pretrain           - Run pretraining with Megatron-Bridge (stage0)
    nemotron nano3 sft                - Run supervised fine-tuning with Megatron-Bridge
                                      (coming soon)
    nemotron nano3 rl                 - Run reinforcement learning with NeMo-RL (coming soon)

Examples:
    # Data preparation per stage
    nemotron nano3 data prep pretrain --sample 1000
    nemotron nano3 data prep sft --sample 100
    nemotron nano3 data prep rl --sample 100

    # Pretraining (requires megatron-bridge)
    nemotron nano3 pretrain --mock --max-steps 1000

    # Direct module access still works
    python -m nemotron.recipes.nano3.stage0_pretrain.data_prep --sample 1000
"""

from __future__ import annotations

from dataclasses import dataclass

from nemotron.kit import App
from nemotron.kit.artifact import ArtifactInput
from nemotron.recipes.nano3.stage0_pretrain.data_prep import (
    PreTrainDataPrepConfig,
)
from nemotron.recipes.nano3.stage0_pretrain.data_prep import (
    main as pretrain_data_main,
)
from nemotron.recipes.nano3.stage1_sft.data_prep import (
    SFTDataPrepConfig,
)
from nemotron.recipes.nano3.stage1_sft.data_prep import (
    main as sft_data_main,
)
from nemotron.recipes.nano3.stage2_rl.data_prep import (
    RLDataPrepConfig,
)
from nemotron.recipes.nano3.stage2_rl.data_prep import (
    main as rl_data_main,
)

# Import ConfigContainer for training config
# Prefer megatron-bridge if available, otherwise use local stub for CLI development
try:
    from megatron.bridge.training.config import ConfigContainer as TrainingConfig
except ImportError:
    from nemotron.kit.megatron_stub import ConfigContainer as TrainingConfig


# =============================================================================
# Placeholder Configs and Handlers
# =============================================================================


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


def curate_main(config: DataCurateConfig) -> int:
    """Handle data curation command."""
    print("Data curation coming soon")
    return 1


def training_main(config: TrainingConfig) -> int:
    """Handle pretraining command."""
    from nemotron.recipes.nano3.stage0_pretrain.train import main

    main(config)
    return 0


def sft_main(config: SftConfig) -> int:
    """Handle SFT command."""
    print("Supervised fine-tuning coming soon")
    return 1


def rl_main(config: RlConfig) -> int:
    """Handle RL command."""
    print("Reinforcement learning coming soon")
    return 1


# =============================================================================
# Build CLI with App
# =============================================================================

app = App("nano3", description="Nano3 training recipe")

# Nested groups
data = app.group("data", description="Data curation and preparation commands")
prep = data.group("prep", description="Prepare data for training stages")

prep.command(
    "pretrain",
    PreTrainDataPrepConfig,
    pretrain_data_main,
    description="Tokenize data for pretraining (bin/idx format)",
    script_path="src/nemotron/recipes/nano3/stage0_pretrain/data_prep.py",
)
prep.command(
    "sft",
    SFTDataPrepConfig,
    sft_data_main,
    description="Prepare data for SFT (JSONL format)",
    script_path="src/nemotron/recipes/nano3/stage1_sft/data_prep.py",
)
prep.command(
    "rl",
    RLDataPrepConfig,
    rl_data_main,
    description="Prepare data for RL (JSONL chat format)",
    script_path="src/nemotron/recipes/nano3/stage2_rl/data_prep.py",
)

data.command(
    "curate",
    DataCurateConfig,
    curate_main,
    description="Curate training data with NeMo Curator (coming soon)",
)

# Top-level training commands
# script_path enables direct execution via nemo-run without pip installing nemotron
app.command(
    "pretrain",
    TrainingConfig,
    training_main,
    description="Run pretraining with Megatron-Bridge (stage0)",
    artifacts={
        "data": ArtifactInput(
            default_name="PretrainBlendsArtifact-pretrain",
            mappings={"path": "fn.per_split_data_args_path"},
        ),
    },
    script_path="src/nemotron/recipes/nano3/stage0_pretrain/train.py",
)
app.command(
    "sft",
    SftConfig,
    sft_main,
    description="Run supervised fine-tuning with Megatron-Bridge (coming soon)",
    script_path="src/nemotron/recipes/nano3/stage1_sft/train.py",
)
app.command(
    "rl",
    RlConfig,
    rl_main,
    description="Run reinforcement learning with NeMo-RL",
    artifacts={
        "model": ArtifactInput(
            default_name="ModelArtifact-sft",
            mappings={"path": "policy.model_name"},
        ),
        "data": ArtifactInput(
            default_name="DataBlendsArtifact-rl",
            mappings={
                # Use metadata fields populated by data_prep
                "metadata.train": "data.train_jsonl_fpath",
                "metadata.val": "data.validation_jsonl_fpath",
            },
        ),
    },
    script_path="src/nemotron/recipes/nano3/stage2_rl/train.py",
)

# Build the tyro-compatible CLI function (for external import)
cli = app.build()

if __name__ == "__main__":
    app.run()
