"""Sequence packing for efficient SFT training.

Self-contained packing algorithms optimized for Ray-based data pipelines.
"""

from nemotron.data_prep.packing.algorithms import (
    PackingAlgorithm,
    SequencePacker,
    FirstFitDecreasingPacker,
    FirstFitShufflePacker,
    ConcatenativePacker,
    get_packer,
)
from nemotron.data_prep.packing.builder import PackedSequenceBuilder

__all__ = [
    # Algorithm enum and base class
    "PackingAlgorithm",
    "SequencePacker",
    # Concrete packers
    "FirstFitDecreasingPacker",
    "FirstFitShufflePacker",
    "ConcatenativePacker",
    # Factory
    "get_packer",
    # Builder
    "PackedSequenceBuilder",
]
