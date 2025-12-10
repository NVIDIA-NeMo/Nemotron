"""Sequence packing algorithms.

Self-contained implementations of common bin packing algorithms for
efficiently packing variable-length sequences into fixed-size batches.
"""

import enum
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class PackingAlgorithm(enum.Enum):
    """Available packing algorithms."""

    FIRST_FIT_DECREASING = "first_fit_decreasing"
    FIRST_FIT_SHUFFLE = "first_fit_shuffle"
    CONCATENATIVE = "concatenative"


@dataclass
class PackingMetrics:
    """Metrics from a packing operation."""

    num_sequences: int = 0
    num_bins: int = 0
    total_tokens: int = 0
    bin_capacity: int = 0
    num_truncated: int = 0

    @property
    def packing_factor(self) -> float:
        """Average sequences per bin."""
        return self.num_sequences / self.num_bins if self.num_bins > 0 else 0

    @property
    def packing_efficiency(self) -> float:
        """Percentage of bin capacity used."""
        total_capacity = self.num_bins * self.bin_capacity
        return (self.total_tokens / total_capacity * 100) if total_capacity > 0 else 0


class SequencePacker(ABC):
    """Abstract base class for sequence packers."""

    def __init__(self, bin_capacity: int):
        """Initialize packer.

        Args:
            bin_capacity: Maximum tokens per bin (pack_size).
        """
        if bin_capacity <= 0:
            raise ValueError(f"bin_capacity must be positive, got {bin_capacity}")
        self.bin_capacity = bin_capacity

    @abstractmethod
    def pack(self, sequence_lengths: list[int]) -> tuple[list[list[int]], PackingMetrics]:
        """Pack sequences into bins.

        Args:
            sequence_lengths: List of sequence lengths to pack.

        Returns:
            Tuple of (bins, metrics) where bins is a list of lists,
            each inner list contains indices into sequence_lengths.
        """
        pass


class FirstFitDecreasingPacker(SequencePacker):
    """First-fit decreasing bin packing algorithm.

    Sorts sequences by length (descending) and places each into the
    first bin that has room. Creates new bins as needed.

    Properties:
    - Deterministic
    - Good packing efficiency
    - Tends to put similar-length sequences together
    """

    def pack(self, sequence_lengths: list[int]) -> tuple[list[list[int]], PackingMetrics]:
        if not sequence_lengths:
            return [], PackingMetrics(bin_capacity=self.bin_capacity)

        # Create (length, original_index) pairs and sort by length descending
        indexed_lengths = [(length, idx) for idx, length in enumerate(sequence_lengths)]
        indexed_lengths.sort(key=lambda x: x[0], reverse=True)

        bins: list[list[int]] = []
        bin_remaining: list[int] = []
        total_tokens = 0
        num_truncated = 0

        for length, idx in indexed_lengths:
            # Truncate if needed
            if length > self.bin_capacity:
                length = self.bin_capacity
                num_truncated += 1

            total_tokens += length

            # Find first bin with room
            placed = False
            for bin_idx, remaining in enumerate(bin_remaining):
                if remaining >= length:
                    bins[bin_idx].append(idx)
                    bin_remaining[bin_idx] -= length
                    placed = True
                    break

            # Create new bin if needed
            if not placed:
                bins.append([idx])
                bin_remaining.append(self.bin_capacity - length)

        metrics = PackingMetrics(
            num_sequences=len(sequence_lengths),
            num_bins=len(bins),
            total_tokens=total_tokens,
            bin_capacity=self.bin_capacity,
            num_truncated=num_truncated,
        )

        return bins, metrics


class FirstFitShufflePacker(SequencePacker):
    """First-fit with shuffled input.

    Same algorithm as first-fit decreasing but shuffles input first.
    This provides better mixing of sequence lengths within bins.

    Properties:
    - Non-deterministic (use seed for reproducibility)
    - Good packing efficiency
    - Better diversity within bins
    """

    def __init__(self, bin_capacity: int, seed: int | None = None):
        super().__init__(bin_capacity)
        self.seed = seed
        self._rng = random.Random(seed)

    def pack(self, sequence_lengths: list[int]) -> tuple[list[list[int]], PackingMetrics]:
        if not sequence_lengths:
            return [], PackingMetrics(bin_capacity=self.bin_capacity)

        # Create (length, original_index) pairs and shuffle
        indexed_lengths = [(length, idx) for idx, length in enumerate(sequence_lengths)]
        self._rng.shuffle(indexed_lengths)

        bins: list[list[int]] = []
        bin_remaining: list[int] = []
        total_tokens = 0
        num_truncated = 0

        for length, idx in indexed_lengths:
            # Truncate if needed
            if length > self.bin_capacity:
                length = self.bin_capacity
                num_truncated += 1

            total_tokens += length

            # Find first bin with room
            placed = False
            for bin_idx, remaining in enumerate(bin_remaining):
                if remaining >= length:
                    bins[bin_idx].append(idx)
                    bin_remaining[bin_idx] -= length
                    placed = True
                    break

            # Create new bin if needed
            if not placed:
                bins.append([idx])
                bin_remaining.append(self.bin_capacity - length)

        metrics = PackingMetrics(
            num_sequences=len(sequence_lengths),
            num_bins=len(bins),
            total_tokens=total_tokens,
            bin_capacity=self.bin_capacity,
            num_truncated=num_truncated,
        )

        return bins, metrics


class ConcatenativePacker(SequencePacker):
    """Simple concatenative packing.

    Concatenates sequences in order until bin is full, then starts new bin.
    Simplest possible packing strategy.

    Properties:
    - Deterministic
    - Preserves input order
    - Simple but may have lower efficiency
    """

    def pack(self, sequence_lengths: list[int]) -> tuple[list[list[int]], PackingMetrics]:
        if not sequence_lengths:
            return [], PackingMetrics(bin_capacity=self.bin_capacity)

        bins: list[list[int]] = [[]]
        current_bin_size = 0
        total_tokens = 0
        num_truncated = 0

        for idx, length in enumerate(sequence_lengths):
            # Truncate if needed
            if length > self.bin_capacity:
                length = self.bin_capacity
                num_truncated += 1

            total_tokens += length

            # Check if fits in current bin
            if current_bin_size + length <= self.bin_capacity:
                bins[-1].append(idx)
                current_bin_size += length
            else:
                # Start new bin
                bins.append([idx])
                current_bin_size = length

        # Remove empty bins
        bins = [b for b in bins if b]

        metrics = PackingMetrics(
            num_sequences=len(sequence_lengths),
            num_bins=len(bins),
            total_tokens=total_tokens,
            bin_capacity=self.bin_capacity,
            num_truncated=num_truncated,
        )

        return bins, metrics


def get_packer(
    algorithm: PackingAlgorithm | str,
    bin_capacity: int,
    seed: int | None = None,
) -> SequencePacker:
    """Factory function to create a packer.

    Args:
        algorithm: Packing algorithm to use.
        bin_capacity: Maximum tokens per bin (pack_size).
        seed: Random seed for shuffle-based algorithms.

    Returns:
        Configured SequencePacker instance.

    Example:
        >>> packer = get_packer("first_fit_shuffle", bin_capacity=2048, seed=42)
        >>> bins, metrics = packer.pack([100, 200, 150, 300])
        >>> print(f"Packed {metrics.num_sequences} seqs into {metrics.num_bins} bins")
    """
    if isinstance(algorithm, str):
        algorithm = PackingAlgorithm(algorithm)

    if algorithm == PackingAlgorithm.FIRST_FIT_DECREASING:
        return FirstFitDecreasingPacker(bin_capacity)
    elif algorithm == PackingAlgorithm.FIRST_FIT_SHUFFLE:
        return FirstFitShufflePacker(bin_capacity, seed=seed)
    elif algorithm == PackingAlgorithm.CONCATENATIVE:
        return ConcatenativePacker(bin_capacity)
    else:
        raise ValueError(f"Unknown packing algorithm: {algorithm}")


__all__ = [
    "PackingAlgorithm",
    "PackingMetrics",
    "SequencePacker",
    "FirstFitDecreasingPacker",
    "FirstFitShufflePacker",
    "ConcatenativePacker",
    "get_packer",
]
