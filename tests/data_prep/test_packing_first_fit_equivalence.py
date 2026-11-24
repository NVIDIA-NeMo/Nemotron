"""First-fit assignment compatibility across index growth and repeated calls."""

import random

import pytest

from nemotron.data_prep.packing.algorithms import FirstFitDecreasingPacker, FirstFitShufflePacker, PackingMetrics


@pytest.mark.parametrize("packer_type", [FirstFitDecreasingPacker, FirstFitShufflePacker])
def test_first_fit_matches_linear_reference(packer_type) -> None:
    rng = random.Random(719)
    shuffle_rng = random.Random(42)
    packer = packer_type(64, seed=42) if packer_type is FirstFitShufflePacker else packer_type(64)
    cases = [[], [0], [64, 0], [65, 90, 64], [33] * 257 + [1] * 513]
    cases += [[rng.randrange(0, 100) for _ in range(rng.randrange(300))] for _ in range(100)]
    for lengths in cases:
        indexed = list(zip(lengths, range(len(lengths))))
        if packer_type is FirstFitShufflePacker:
            shuffle_rng.shuffle(indexed)
        else:
            indexed.sort(key=lambda item: item[0], reverse=True)
        expected = []
        remaining = []
        for length, idx in indexed:
            length = min(length, 64)
            for bin_idx, space in enumerate(remaining):
                if space >= length:
                    expected[bin_idx].append(idx)
                    remaining[bin_idx] -= length
                    break
            else:
                expected.append([idx])
                remaining.append(64 - length)
        bins, metrics = packer.pack(lengths)
        assert bins == expected
        assert metrics == PackingMetrics(
            num_sequences=len(lengths), num_bins=len(expected),
            total_tokens=sum(min(length, 64) for length in lengths), bin_capacity=64,
            num_truncated=sum(length > 64 for length in lengths),
        )
