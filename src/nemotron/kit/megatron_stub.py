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

"""Stub for megatron-bridge ConfigContainer.

Provides a lightweight ConfigContainer for local CLI development
when megatron-bridge isn't installed (requires CUDA to build).

This allows the CLI to show all training options without needing
a full GPU environment.

Usage:
    The __main__.py will automatically use this stub when
    megatron-bridge import fails.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration for training."""

    data_path: Path | None = None
    """Path to blend.json or per-split data args."""

    mock: bool = False
    """Use mock/synthetic data for testing."""

    seq_length: int = 4096
    """Sequence length for training."""

    micro_batch_size: int = 1
    """Micro batch size per GPU."""

    global_batch_size: int = 8
    """Global batch size across all GPUs."""


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str = "nemotron-nano"
    """Model name/identifier."""

    num_layers: int = 32
    """Number of transformer layers."""

    hidden_size: int = 4096
    """Hidden dimension size."""

    num_attention_heads: int = 32
    """Number of attention heads."""

    ffn_hidden_size: int = 14336
    """FFN intermediate size."""

    vocab_size: int = 128256
    """Vocabulary size."""


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    lr: float = 1e-4
    """Learning rate."""

    min_lr: float = 1e-5
    """Minimum learning rate for scheduler."""

    weight_decay: float = 0.1
    """Weight decay."""

    adam_beta1: float = 0.9
    """Adam beta1."""

    adam_beta2: float = 0.95
    """Adam beta2."""


@dataclass
class TrainingConfig:
    """Training configuration."""

    max_steps: int = 1000
    """Maximum training steps."""

    log_interval: int = 10
    """Logging interval (steps)."""

    eval_interval: int = 100
    """Evaluation interval (steps)."""

    save_interval: int = 500
    """Checkpoint save interval (steps)."""

    fp16: bool = False
    """Use FP16 precision."""

    bf16: bool = True
    """Use BF16 precision."""


@dataclass
class CheckpointConfig:
    """Checkpoint configuration."""

    dir: Path = field(default_factory=lambda: Path("./checkpoints"))
    """Checkpoint directory."""

    save_on_train_end: bool = True
    """Save checkpoint at end of training."""

    resume_from: Path | None = None
    """Resume from checkpoint path."""


@dataclass
class ConfigContainer:
    """Container for all training configuration.

    This is a stub matching megatron-bridge's ConfigContainer structure
    for local CLI development without CUDA.
    """

    data: DataConfig = field(default_factory=DataConfig)
    """Data configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    """Model configuration."""

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    """Optimizer configuration."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    """Training configuration."""

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    """Checkpoint configuration."""
