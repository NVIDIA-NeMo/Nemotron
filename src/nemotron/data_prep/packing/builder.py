"""Packed sequence builder for GPTSFTPackedDataset format."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from nemotron.data_prep.packing.algorithms import (
    PackingAlgorithm,
    PackingMetrics,
    get_packer,
)


@dataclass
class PackedSequence:
    """A single packed sequence containing multiple original sequences.

    Format matches GPTSFTPackedDataset expectations.
    """

    input_ids: list[int]
    loss_mask: list[int]
    seq_start_id: list[int]


class PackedSequenceBuilder:
    """Builds packed sequences matching GPTSFTPackedDataset format.

    Collects tokenized sequences, then packs them using the specified
    algorithm. Output format is compatible with Megatron-Bridge's
    GPTSFTPackedDataset.

    Example:
        >>> builder = PackedSequenceBuilder(pack_size=2048, algorithm="first_fit_shuffle")
        >>> builder.add_sequence([1, 2, 3], loss_mask=[1, 1, 1])
        >>> builder.add_sequence([4, 5, 6, 7], loss_mask=[0, 1, 1, 1])
        >>> packed_data, metadata = builder.finalize()
    """

    def __init__(
        self,
        pack_size: int,
        algorithm: PackingAlgorithm | str = "first_fit_shuffle",
        seed: int | None = None,
        dtype: str = "int32",
    ):
        """Initialize packed sequence builder.

        Args:
            pack_size: Maximum tokens per packed sequence.
            algorithm: Packing algorithm to use.
            seed: Random seed for shuffle-based algorithms.
            dtype: Numpy dtype for output arrays.
        """
        self.pack_size = pack_size
        self.algorithm = algorithm
        self.seed = seed
        self.dtype = np.dtype(dtype)

        # Collect sequences before packing
        self._sequences: list[list[int]] = []
        self._loss_masks: list[list[int] | None] = []

    def add_sequence(
        self,
        input_ids: list[int],
        loss_mask: list[int] | None = None,
    ) -> None:
        """Add a tokenized sequence for packing.

        Args:
            input_ids: Token IDs for this sequence.
            loss_mask: Optional loss mask (1 = compute loss, 0 = ignore).
                If not provided, defaults to all 1s.
        """
        if not input_ids:
            return

        self._sequences.append(input_ids)
        self._loss_masks.append(loss_mask)

    def add_sequences(
        self,
        sequences: list[list[int]],
        loss_masks: list[list[int] | None] | None = None,
    ) -> None:
        """Batch add multiple sequences.

        Args:
            sequences: List of token ID sequences.
            loss_masks: Optional list of loss masks (same length as sequences).
        """
        if loss_masks is None:
            loss_masks = [None] * len(sequences)

        for input_ids, loss_mask in zip(sequences, loss_masks):
            self.add_sequence(input_ids, loss_mask)

    def finalize(self) -> tuple[list[dict], dict]:
        """Pack collected sequences and return output.

        Returns:
            Tuple of (packed_data, metadata) where:
            - packed_data: List of dicts, each with {input_ids, loss_mask, seq_start_id}
            - metadata: Dict with packing metrics

        Output format per packed item:
            {
                "input_ids": [tok1, tok2, ...],
                "loss_mask": [0, 1, 1, ...],  # Rolled by 1 for label alignment
                "seq_start_id": [0, len1, len1+len2, ...]
            }
        """
        if not self._sequences:
            return [], {
                "pack_size": self.pack_size,
                "num_sequences": 0,
                "num_packed_sequences": 0,
                "packing_factor": 0,
                "packing_efficiency": 0,
            }

        # Get sequence lengths
        lengths = [len(seq) for seq in self._sequences]

        # Pack sequences
        packer = get_packer(self.algorithm, self.pack_size, seed=self.seed)
        bins, metrics = packer.pack(lengths)

        # Build packed sequences
        packed_data: list[dict] = []
        for bin_indices in bins:
            packed = self._build_packed_sequence(bin_indices)
            packed_data.append(packed)

        metadata = {
            "pack_size": self.pack_size,
            "algorithm": str(self.algorithm),
            "num_sequences": metrics.num_sequences,
            "num_packed_sequences": metrics.num_bins,
            "packing_factor": round(metrics.packing_factor, 2),
            "packing_efficiency": round(metrics.packing_efficiency, 1),
            "num_truncated": metrics.num_truncated,
            "total_tokens": metrics.total_tokens,
        }

        return packed_data, metadata

    def _build_packed_sequence(self, indices: list[int]) -> dict:
        """Build a single packed sequence from indices.

        Args:
            indices: Indices into self._sequences for sequences to pack.

        Returns:
            Dict with input_ids, loss_mask, seq_start_id.
        """
        all_input_ids: list[int] = []
        all_loss_mask: list[int] = []
        seq_start_ids: list[int] = [0]

        for idx in indices:
            seq = self._sequences[idx]
            mask = self._loss_masks[idx]

            # Truncate if needed
            if len(seq) > self.pack_size:
                seq = seq[: self.pack_size]
                if mask is not None:
                    mask = mask[: self.pack_size]

            # Check if we have room
            current_len = len(all_input_ids)
            if current_len + len(seq) > self.pack_size:
                # Truncate to fit
                remaining = self.pack_size - current_len
                seq = seq[:remaining]
                if mask is not None:
                    mask = mask[:remaining]

            all_input_ids.extend(seq)

            # Default loss mask to all 1s if not provided
            if mask is None:
                mask = [1] * len(seq)
            all_loss_mask.extend(mask)

            # Track sequence boundary
            seq_start_ids.append(len(all_input_ids))

        # Roll loss_mask by 1 for label alignment (standard in Megatron)
        # This shifts the mask so it aligns with labels (next-token prediction)
        rolled_loss_mask = [0] + all_loss_mask[:-1] if all_loss_mask else []

        return {
            "input_ids": all_input_ids,
            "loss_mask": rolled_loss_mask,
            "seq_start_id": seq_start_ids[:-1],  # Exclude final boundary
        }

    def get_stats(self) -> dict:
        """Get current builder statistics (before finalize).

        Returns:
            Dict with num_sequences and approximate metrics.
        """
        if not self._sequences:
            return {"num_sequences": 0, "total_tokens": 0}

        total_tokens = sum(len(seq) for seq in self._sequences)
        return {
            "num_sequences": len(self._sequences),
            "total_tokens": total_tokens,
            "avg_length": total_tokens / len(self._sequences),
        }


__all__ = ["PackedSequence", "PackedSequenceBuilder"]
