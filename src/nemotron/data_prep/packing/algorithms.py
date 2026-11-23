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

"""Sequence packing algorithms.

Self-contained implementations of common bin packing algorithms for
efficiently packing variable-length sequences into fixed-size batches.
"""

import enum
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass


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


class _FirstFitIndex:
    """Dynamically index remaining capacities by their maximum.

    Left-first descent preserves first-fit semantics, including bin order.
    Storage grows with the number of bins, not the number of sequences.
    """

    def __init__(self, remaining: list[int]) -> None:
        self._count = len(remaining)
        self._size = 1 << (self._count - 1).bit_length()
        self._tree: list[int | float] = [float("-inf")] * (2 * self._size)
        self._tree[self._size : self._size + self._count] = remaining
        for parent in range(self._size - 1, 0, -1):
            self._tree[parent] = max(self._tree[2 * parent], self._tree[2 * parent + 1])

    def take(self, length: int, capacity: int) -> int:
        tree = self._tree
        if tree[1] >= length:
            node = 1
            while node < self._size:
                node *= 2
                if tree[node] < length:
                    node += 1
            bin_idx = node - self._size
            tree[node] -= length
        else:
            if self._count == self._size:
                old_size = self._size
                self._size *= 2
                tree = [float("-inf")] * (2 * self._size)
                tree[self._size : self._size + self._count] = self._tree[old_size:]
                for parent in range(self._size - 1, 0, -1):
                    tree[parent] = max(tree[2 * parent], tree[2 * parent + 1])
                self._tree = tree
            bin_idx = self._count
            self._count += 1
            node = self._size + bin_idx
            tree[node] = capacity - length
        node //= 2
        while node:
            tree[node] = max(tree[2 * node], tree[2 * node + 1])
            node //= 2
        return bin_idx


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
        index: _FirstFitIndex | None = None
        total_tokens = 0
        num_truncated = 0

        for length, idx in indexed_lengths:
            # Truncate if needed
            if length > self.bin_capacity:
                length = self.bin_capacity
                num_truncated += 1

            total_tokens += length

            if index is None:
                # Keep the original scan for small bin counts: no per-sequence
                # helper call or tree maintenance in this common case.
                for bin_idx, remaining in enumerate(bin_remaining):
                    if remaining >= length:
                        bins[bin_idx].append(idx)
                        bin_remaining[bin_idx] -= length
                        break
                else:
                    bins.append([idx])
                    bin_remaining.append(self.bin_capacity - length)
                    if len(bins) == 32:
                        index = _FirstFitIndex(bin_remaining)
                        bin_remaining.clear()
            else:
                bin_idx = index.take(length, self.bin_capacity)
                if bin_idx == len(bins):
                    bins.append([idx])
                else:
                    bins[bin_idx].append(idx)

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
        index: _FirstFitIndex | None = None
        total_tokens = 0
        num_truncated = 0

        for length, idx in indexed_lengths:
            # Truncate if needed
            if length > self.bin_capacity:
                length = self.bin_capacity
                num_truncated += 1

            total_tokens += length

            if index is None:
                # Keep the original scan for small bin counts: no per-sequence
                # helper call or tree maintenance in this common case.
                for bin_idx, remaining in enumerate(bin_remaining):
                    if remaining >= length:
                        bins[bin_idx].append(idx)
                        bin_remaining[bin_idx] -= length
                        break
                else:
                    bins.append([idx])
                    bin_remaining.append(self.bin_capacity - length)
                    if len(bins) == 32:
                        index = _FirstFitIndex(bin_remaining)
                        bin_remaining.clear()
            else:
                bin_idx = index.take(length, self.bin_capacity)
                if bin_idx == len(bins):
                    bins.append([idx])
                else:
                    bins[bin_idx].append(idx)

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
