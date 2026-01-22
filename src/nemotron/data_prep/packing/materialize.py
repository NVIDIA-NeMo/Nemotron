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


__all__ = ["materialize_packed_samples"]