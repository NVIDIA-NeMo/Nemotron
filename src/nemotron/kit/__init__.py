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

# Copyright (c) Nemotron Contributors
# SPDX-License-Identifier: MIT

"""
nemotron.kit - Domain-specific toolkit for Nemotron training artifacts.

This module provides Nemotron-specific building blocks:
- Artifact types (pretrain data, SFT data, RL data, model checkpoints)
- Lineage tracking (W&B and file-based backends)
- W&B integration and configuration

For generic CLI infrastructure (config loading, execution, packaging),
see the ``nemo_runspec`` package.

Quick Start:
    >>> from nemotron.kit import Artifact
    >>> from pydantic import Field
    >>>
    >>> # Artifact with validation
    >>> class Dataset(Artifact):
    ...     num_examples: int = Field(gt=0)
    >>>
    >>> dataset = Dataset(path=Path("/tmp/data"), num_examples=1000)
    >>> dataset.save()

Registry Example:
    >>> import nemotron.kit as kit
    >>>
    >>> # Initialize with fsspec backend
    >>> kit.init(backend="fsspec", root="/data/artifacts")
    >>>
    >>> # Or with W&B backend
    >>> kit.init(backend="wandb", wandb_project="my-project")
    >>>
    >>> # Save artifact to registry
    >>> dataset.save(name="my-dataset")
    >>> print(dataset.uri)  # art://my-dataset:v1
    >>>
    >>> # Load from URI
    >>> loaded = Dataset.from_uri("art://my-dataset:v1")
"""

from pathlib import Path
from typing import Any

# Artifacts
from nemotron.kit.artifact import (
    Artifact,
    DataBlendsArtifact,
    ModelArtifact,
    PretrainBlendsArtifact,
    PretrainDataArtifact,
    SFTDataArtifact,
    SplitJsonlDataArtifact,
    TrackingInfo,
    apply_scale,
    print_step_complete,
)

# Trackers
from nemotron.kit.trackers import (
    FileTracker,
    LineageTracker,
    NoOpTracker,
    WandbTracker,
    get_lineage_tracker,
    set_lineage_tracker,
    to_wandb_uri,
    tokenizer_to_uri,
)

# Wandb configuration
from nemotron.kit.wandb_kit import WandbConfig, add_run_tags, init_wandb_if_configured

__all__ = [
    # Artifacts
    "Artifact",
    "DataBlendsArtifact",
    "ModelArtifact",
    "PretrainBlendsArtifact",
    "PretrainDataArtifact",
    "SFTDataArtifact",
    "SplitJsonlDataArtifact",
    "TrackingInfo",
    "apply_scale",
    "print_step_complete",
    # Kit init
    "init",
    "is_initialized",
    # Trackers
    "LineageTracker",
    "WandbTracker",
    "FileTracker",
    "NoOpTracker",
    "set_lineage_tracker",
    "get_lineage_tracker",
    "to_wandb_uri",
    "tokenizer_to_uri",
    # Wandb configuration
    "WandbConfig",
    "init_wandb_if_configured",
    "add_run_tags",
]


def init(
    backend: str = "fsspec",
    root: str | Path | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    **kwargs: Any,
) -> None:
    """Initialize nemotron.kit with a storage backend.

    Must be called before using artifact URIs or registry features.

    Args:
        backend: Storage backend ("fsspec" or "wandb")
        root: Root path for fsspec backend (required for fsspec)
        wandb_project: W&B project name (required for wandb)
        wandb_entity: W&B entity/team name (optional for wandb)
        **kwargs: Additional backend-specific options

    Example:
        >>> import nemotron.kit as kit
        >>>
        >>> # Local filesystem
        >>> kit.init(backend="fsspec", root="/data/artifacts")
        >>>
        >>> # S3 (requires s3fs)
        >>> kit.init(backend="fsspec", root="s3://bucket/artifacts")
        >>>
        >>> # W&B
        >>> kit.init(backend="wandb", wandb_project="my-project")
    """
    # Validate backend
    if backend not in ("fsspec", "wandb"):
        raise ValueError(f"Unknown backend: {backend}. Must be 'fsspec' or 'wandb'.")

    if backend == "fsspec" and root is None:
        raise ValueError("root is required for fsspec backend")

    if backend == "wandb" and wandb_project is None:
        raise ValueError("wandb_project is required for wandb backend")

    # Initialize registry
    from nemo_runspec.artifact_registry import ArtifactRegistry, set_artifact_registry

    registry = ArtifactRegistry(
        backend=backend,
        root=root,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )
    set_artifact_registry(registry)

    # Set up lineage tracker based on backend
    if backend == "wandb":
        tracker = WandbTracker()
        set_lineage_tracker(tracker)
    elif backend == "fsspec":
        from nemotron.kit.trackers import FileTracker

        tracker = FileTracker(registry)
        set_lineage_tracker(tracker)


def is_initialized() -> bool:
    """Check if nemotron.kit has been initialized.

    Returns:
        True if init() has been called
    """
    from nemo_runspec.artifact_registry import _registry

    return _registry is not None
