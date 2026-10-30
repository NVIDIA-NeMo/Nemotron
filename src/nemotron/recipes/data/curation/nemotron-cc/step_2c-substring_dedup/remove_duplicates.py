# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
from itertools import pairwise

import numpy as np

from utils import SEPARATOR_LENGTH, div_up


def adjust_remove_and_idx(
    remove: list[tuple[int, int]],
    idx: np.ndarray,
    length_threshold: int,
    bytes_per_token: int | None = None,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    filtered_remove = []
    dedup_idx = [0]
    remove = remove[::-1]

    for doc_start, doc_end in list(pairwise(idx)):
        doc_start = int(doc_start)
        doc_end = int(doc_end)
        dedup_doc_size = doc_end - doc_start

        min_next_remove_start = doc_start + SEPARATOR_LENGTH

        while remove and doc_start <= remove[-1][0] < doc_end:
            remove_start, remove_end = remove.pop()
            remove_start = max(remove_start, min_next_remove_start)

            if remove_end > doc_end:
                remove.append((doc_end, remove_end))
                remove_end = doc_end

            if bytes_per_token is not None:
                remove_start = (
                    doc_start
                    + SEPARATOR_LENGTH
                    + div_up(
                        remove_start - (doc_start + SEPARATOR_LENGTH), bytes_per_token
                    )
                    * bytes_per_token
                )
                remove_end = (
                    doc_start
                    + SEPARATOR_LENGTH
                    + ((remove_end - (doc_start + SEPARATOR_LENGTH)) // bytes_per_token)
                    * bytes_per_token
                )

            if remove_end - remove_start < length_threshold:
                continue

            min_next_remove_start = remove_end
            filtered_remove.append((remove_start, remove_end))
            dedup_doc_size -= remove_end - remove_start

        dedup_idx.append(dedup_idx[-1] + dedup_doc_size)

    dedup_idx = np.array(dedup_idx, dtype=np.uint64)
    return filtered_remove, dedup_idx


def remove_duplicates(
    input_path: str,
    output_path: str,
    length_threshold: int,
    bytes_per_token: int | None = None,
) -> None:
    # Load byte ranges to remove from {input_path}.bin.remove.byterange
    remove: list[tuple[int, int]] = []

    with open(f"{input_path}.bin.remove.byterange") as fp:
        for line in fp:
            if "out" in line:
                break
        for line in fp:
            remove.append(tuple(map(int, line.split())))

    # Load document index from {input_path}.idx
    with open(f"{input_path}.idx", "rb") as fp_in:
        idx = np.frombuffer(fp_in.read(), dtype=np.uint64)

    # Adjust the removal ranges and compute deduped document index
    filtered_remove, idx_after_filtered_remove = adjust_remove_and_idx(
        remove, idx, length_threshold, bytes_per_token=bytes_per_token
    )

    # Write the deduped dataset to {output_path}.bin
    with open(f"{input_path}.bin", "rb") as ds, open(
        f"{output_path}.bin", "wb"
    ) as new_ds:
        start = 0
        for remove_start, remove_end in filtered_remove:
            new_ds.write(ds.read(remove_start - start))
            ds.seek(remove_end)
            start = remove_end
        new_ds.write(ds.read())

    # Write the deduped dataset document index to {output_path}.idx
    with open(f"{output_path}.idx", "wb") as fp:
        fp.write(np.array(idx_after_filtered_remove, dtype=np.uint64).tobytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to dataset.bin, dataset.idx, and dataset.bin.remove.byterange files",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to output deduped_dataset.bin and deduped_dataset.idx files",
    )
    parser.add_argument(
        "--length-threshold",
        type=int,
        required=True,
        help="Minimum length for intervals to remove",
    )
    parser.add_argument(
        "--bytes-per-token",
        type=int,
        required=False,
        help=(
            "If using a tokenizer, specify how many bytes each token uses "
            "so that removals only remove whole tokens"
        ),
    )
    args = parser.parse_args()

    remove_duplicates(
        args.input_path, args.output_path, args.length_threshold, args.bytes_per_token
    )
