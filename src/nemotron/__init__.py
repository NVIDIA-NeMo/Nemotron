"""
Nemotron: Reproducible Training Recipes for NVIDIA Nemotron Models

Transparent • Reproducible • Production-Ready

The nemotron package provides a transparent framework for building
reproducible training pipelines. Used to create complete training recipes
for NVIDIA Nemotron models with full data preparation, training, and
evaluation stages.

Example:
    >>> from nemotron.artifact import Artifact
    >>> from pathlib import Path
    >>> from pydantic import Field
    >>>
    >>> class Dataset(Artifact):
    ...     num_examples: int = Field(gt=0)
    ...     train_path: Path
    >>>
    >>> dataset = Dataset(
    ...     path=Path("/tmp/data"),
    ...     num_examples=1000,
    ...     train_path=Path("/tmp/data/train.parquet")
    ... )
    >>> dataset.save()
"""

__version__ = "0.1.0"

from nemotron.artifact import (
    Artifact,
    LineageTracker,
    apply_scale,
    print_complete,
    set_lineage_tracker,
)

__all__ = [
    "Artifact",
    "LineageTracker",
    "apply_scale",
    "print_complete",
    "set_lineage_tracker",
]
