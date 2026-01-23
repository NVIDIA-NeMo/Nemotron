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

"""Materialize packed samples from a SequenceSpool + BinAssignment.

This module performs the "reduce" step:
- given a bin assignment (bin_id -> sequence indices)
- and a spool providing random-access sequences (tokens + masks)
it produces packed dict items compatible with the existing .npy pickle-of-dicts
format used by GPTSFTPackedDataset.

Truncation semantics match PackedSequenceBuilder._build_packed_sequence:
- If a sequence is longer than pack_size, it is truncated to pack_size.
- If adding a sequence would exceed pack_size, it is truncated to the remaining space.
- The loss_mask is rolled by 1: [0] + mask[:-1]
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from nemotron.data_prep.packing.bin_assignment import BinAssignment
from nemotron.data_prep.packing.spool import SequenceSpoolReader


def materialize_packed_samples(
    *,
    spool_reader: SequenceSpoolReader,
    assignment: BinAssignment,
    pack_size: int,
) -> Iterator[dict]:
    """Yield packed items one bin at a time.

    Args:
        spool_reader: Reader for the SequenceSpool (random-access sequences).
        assignment: CSR-like bin assignment.
        pack_size: Maximum tokens per packed sample.

    Yields:
        Dicts with keys: input_ids, loss_mask, seq_start_id
    """
    if pack_size <= 0:
        raise ValueError(f"pack_size must be positive, got {pack_size}")

    for bin_id in range(assignment.num_bins):
        seq_indices = assignment.bin_indices(bin_id)

        all_input_ids: list[int] = []
        all_loss_mask: list[int] = []
        seq_start_ids: list[int] = [0]

        for seq_index in seq_indices:
            input_ids_arr, loss_mask_arr = spool_reader.read_sequence(int(seq_index))

            # Truncate if needed (builder truncates per-seq to pack_size).
            if input_ids_arr.shape[0] > pack_size:
                input_ids_arr = input_ids_arr[:pack_size]
                loss_mask_arr = loss_mask_arr[:pack_size]

            current_len = len(all_input_ids)
            if current_len >= pack_size:
                break

            if current_len + int(input_ids_arr.shape[0]) > pack_size:
                remaining = pack_size - current_len
                input_ids_arr = input_ids_arr[:remaining]
                loss_mask_arr = loss_mask_arr[:remaining]

            if input_ids_arr.shape[0] == 0:
                continue

            all_input_ids.extend([int(x) for x in input_ids_arr.tolist()])
            all_loss_mask.extend([int(x) for x in loss_mask_arr.tolist()])
            seq_start_ids.append(len(all_input_ids))

        rolled_loss_mask = [0] + all_loss_mask[:-1] if all_loss_mask else []

        yield {
            "input_ids": all_input_ids,
            "loss_mask": rolled_loss_mask,
            "seq_start_id": seq_start_ids[:-1],
        }


def materialize_bin_arrays(
    *,
    spool_reader: SequenceSpoolReader,
    assignment: BinAssignment,
    bin_id: int,
    pack_size: int,
    scratch_input_ids: np.ndarray,
    scratch_loss_mask: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Materialize a single bin directly to numpy arrays.

    Avoids Python list conversions by writing into preallocated buffers.

    Args:
        spool_reader: Reader for tokenized sequence spool.
        assignment: Bin assignment from packing algorithm.
        bin_id: Which bin to materialize.
        pack_size: Maximum packed sequence length.
        scratch_input_ids: Preallocated buffer of shape (pack_size,).
        scratch_loss_mask: Preallocated buffer of shape (pack_size,).

    Returns:
        packed_len: Actual length of packed tokens (excluding padding).
        seq_start_id: Array of sequence START positions within the bin (int32).
                      Invariant: seq_start_id[0] == 0 when non-empty, strictly increasing,
                      and seq_start_id[-1] < packed_len.
                      To get boundaries: list(seq_start_id) + [packed_len]
    """
    if pack_size <= 0:
        raise ValueError(f"pack_size must be positive, got {pack_size}")
    if scratch_input_ids.shape[0] < pack_size:
        raise ValueError(
            f"scratch_input_ids must have length >= pack_size, got {scratch_input_ids.shape[0]} < {pack_size}"
        )
    if scratch_loss_mask.shape[0] < pack_size:
        raise ValueError(
            f"scratch_loss_mask must have length >= pack_size, got {scratch_loss_mask.shape[0]} < {pack_size}"
        )

    seq_indices = assignment.bin_indices(int(bin_id))

    # Zero scratch buffers (padding)
    scratch_input_ids[:pack_size] = 0
    scratch_loss_mask[:pack_size] = 0

    pos = 0
    seq_start_ids: list[int] = []

    for seq_index in seq_indices:
        input_ids_arr, loss_mask_arr = spool_reader.read_sequence(int(seq_index))

        # Clamp per-seq length to pack_size first
        seq_len = int(min(int(input_ids_arr.shape[0]), pack_size))
        if pos + seq_len > pack_size:
            seq_len = pack_size - pos

        if seq_len <= 0:
            break

        seq_start_ids.append(pos)

        scratch_input_ids[pos : pos + seq_len] = input_ids_arr[:seq_len]
        scratch_loss_mask[pos : pos + seq_len] = loss_mask_arr[:seq_len]
        pos += seq_len

        if pos >= pack_size:
            break

    # Roll loss_mask by 1 for label alignment (same as materialize_packed_samples / builder)
    if pos > 0:
        scratch_loss_mask[1:pos] = scratch_loss_mask[: pos - 1].copy()
        scratch_loss_mask[0] = 0

    return pos, np.asarray(seq_start_ids, dtype=np.int32)


__all__ = ["materialize_packed_samples", "materialize_bin_arrays"]