"""
Nemotron: Reproducible Training Recipes for NVIDIA Nemotron Models

Transparent - Reproducible - Production-Ready

The nemotron package provides a transparent framework for building
reproducible training pipelines. Used to create complete training recipes
for NVIDIA Nemotron models with full data preparation, training, and
evaluation stages.

Example:
    >>> from nemotron.kit import cli, Artifact
    >>> from dataclasses import dataclass
    >>> from pydantic import Field
    >>>
    >>> @dataclass
    ... class Config:
    ...     batch_size: int = 32
    >>>
    >>> config = cli(Config)  # Supports --config-file config.yaml
    >>>
    >>> class Dataset(Artifact):
    ...     num_examples: int = Field(gt=0)
    >>>
    >>> dataset = Dataset(path=Path("/tmp/data"), num_examples=1000)
    >>> dataset.save()
"""

__version__ = "0.1.0"
