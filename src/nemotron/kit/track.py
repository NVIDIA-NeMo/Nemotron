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

"""Tracking configuration for artifact resolution."""

from dataclasses import dataclass

STAGE_DIRS = {"pretrain": "stage0_pretrain", "sft": "stage1_sft", "rl": "stage2_rl"}


@dataclass
class TrackConfig:
    backend: str  # "fsspec" or "wandb"
    output_root: str | None = None  # For fsspec backend (can be any fsspec URI)


_config: TrackConfig | None = None


def init(backend: str = "fsspec", output_root: str | None = None) -> None:
    """Initialize tracking configuration.

    Args:
        backend: "fsspec" or "wandb"
        output_root: Output root for fsspec backend - can be local path or any
                     fsspec URI (hf://org/repo, s3://bucket/path, etc.)
                     Default: ./output
    """
    global _config
    _config = TrackConfig(
        backend=backend,
        output_root=output_root or "./output",
    )


def get_config() -> TrackConfig:
    """Get current config. Returns default fsspec config if init() not called."""
    if _config is None:
        return TrackConfig(backend="fsspec", output_root="./output")
    return _config
