# Copyright (c) Nemotron Contributors
# SPDX-License-Identifier: MIT

"""
nemotron.kit - Toolkit for building reproducible training pipelines.

This module provides everything you need to build training recipes:
- Artifact versioning, storage, and tracking
- Config file support (YAML, TOML, JSON) with CLI override
- Pipeline orchestration with subprocess piping and Slurm support
- W&B and fsspec storage backends

Quick Start:
    >>> from nemotron.kit import cli, Artifact, Step
    >>> from dataclasses import dataclass
    >>> from pydantic import Field
    >>>
    >>> # Config with file support
    >>> @dataclass
    ... class Config:
    ...     batch_size: int = 32
    >>>
    >>> config = cli(Config)  # Supports --config-file config.yaml
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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Config
from nemotron.kit.config import ConfigManager, cli

# CLI App
from nemotron.kit.app import App

# Run (nemo-run integration)
from nemotron.kit.run import RunConfig, build_executor, load_run_profile

# Artifacts
from nemotron.kit.artifact import (
    Artifact,
    DataBlendsArtifact,
    ModelArtifact,
    PretrainDataArtifact,
    SFTDataArtifact,
    SplitJsonlDataArtifact,
    TrackingInfo,
    apply_scale,
    print_step_complete,
)
from nemotron.kit.exceptions import ArtifactNotFoundError, ArtifactVersionNotFoundError
from nemotron.kit.registry import ArtifactEntry, ArtifactRegistry, ArtifactVersion

# Pipeline
from nemotron.kit.pipeline import PipelineConfig, run_pipeline
from nemotron.kit.step import Step

# Trackers
from nemotron.kit.trackers import (
    LineageTracker,
    NoOpTracker,
    WandbTracker,
    get_lineage_tracker,
    set_lineage_tracker,
    to_wandb_uri,
    tokenizer_to_uri,
)

# Track module for semantic URI resolution
from nemotron.kit import track

# Wandb configuration
from nemotron.kit.wandb import WandbConfig, init_wandb_if_configured, add_wandb_tags

__all__ = [
    # Config
    "cli",
    "ConfigManager",
    # CLI App
    "App",
    # Run (nemo-run integration)
    "RunConfig",
    "build_executor",
    "load_run_profile",
    # Artifacts
    "Artifact",
    "DataBlendsArtifact",
    "ModelArtifact",
    "PretrainDataArtifact",
    "SFTDataArtifact",
    "SplitJsonlDataArtifact",
    "TrackingInfo",
    "apply_scale",
    "print_step_complete",
    # Pipeline
    "Step",
    "PipelineConfig",
    "run_pipeline",
    # Registry
    "init",
    "get_config",
    "is_initialized",
    "ArtifactRegistry",
    "ArtifactEntry",
    "ArtifactVersion",
    # Trackers
    "LineageTracker",
    "WandbTracker",
    "NoOpTracker",
    "set_lineage_tracker",
    "get_lineage_tracker",
    "to_wandb_uri",
    "tokenizer_to_uri",
    # Exceptions
    "ArtifactNotFoundError",
    "ArtifactVersionNotFoundError",
    # Track
    "track",
    # Wandb configuration
    "WandbConfig",
    "init_wandb_if_configured",
    "add_wandb_tags",
]


@dataclass
class _KitConfig:
    """Internal configuration for nemotron.kit."""

    backend: str
    root: Path | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None


# Global configuration
_config: _KitConfig | None = None


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
    global _config

    # Validate backend
    if backend not in ("fsspec", "wandb"):
        raise ValueError(f"Unknown backend: {backend}. Must be 'fsspec' or 'wandb'.")

    if backend == "fsspec" and root is None:
        raise ValueError("root is required for fsspec backend")

    if backend == "wandb" and wandb_project is None:
        raise ValueError("wandb_project is required for wandb backend")

    # Store configuration
    _config = _KitConfig(
        backend=backend,
        root=Path(root) if root else None,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )

    # Initialize registry
    from nemotron.kit.registry import ArtifactRegistry, set_registry

    registry = ArtifactRegistry(
        backend=backend,
        root=root,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )
    set_registry(registry)

    # If using wandb backend, also set up lineage tracker
    if backend == "wandb":
        tracker = WandbTracker()
        set_lineage_tracker(tracker)


def get_config() -> _KitConfig | None:
    """Get the current kit configuration.

    Returns:
        Current configuration or None if not initialized
    """
    return _config


def is_initialized() -> bool:
    """Check if nemotron.kit has been initialized.

    Returns:
        True if init() has been called
    """
    return _config is not None


def _ensure_initialized() -> None:
    """Ensure kit.init() has been called.

    Raises:
        RuntimeError: If not initialized
    """
    if not is_initialized():
        raise RuntimeError(
            "nemotron.kit not initialized. Call kit.init() first.\n"
            "Example: kit.init(backend='fsspec', root='/data/artifacts')"
        )
