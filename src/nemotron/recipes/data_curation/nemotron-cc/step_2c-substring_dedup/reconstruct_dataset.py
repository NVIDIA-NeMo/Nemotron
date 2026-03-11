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
import json
import os
import struct
from itertools import pairwise

import numpy as np
import pandas as pd
import tiktoken

from utils import SEPARATOR_LENGTH, SEPARATOR_MAGIC


def decode_separator(separator: bytes) -> tuple[int, int]:
    if len(separator) != SEPARATOR_LENGTH:
        raise RuntimeError(
            f"Separator must have {SEPARATOR_LENGTH} bytes, but got "
            f"{separator} of length {len(separator)} bytes."
        )
    if separator[: len(SEPARATOR_MAGIC)] != SEPARATOR_MAGIC:
        raise RuntimeError(
            f"Separator must start with {SEPARATOR_MAGIC}, but got {separator}."
        )
    file_id, doc_id_within_file = struct.unpack(
        "<II", separator[len(SEPARATOR_MAGIC) :]
    )
    return file_id, doc_id_within_file


def reconstruct_dataset(
    input_path: str,
    output_path: str,
    original_dataset_path: str,
    id_to_filename_path: str,
    text_field: str = "text",
) -> None:
    os.makedirs(output_path, exist_ok=True)

    tokenizer = tiktoken.get_encoding("gpt2")

    with open(id_to_filename_path, "r") as fp:
        id_to_filename = {int(k): v for k, v in json.load(fp).items()}

    with open(f"{input_path}.idx", "rb") as fp:
        idx_bytes = fp.read()
    idx = np.frombuffer(idx_bytes, dtype=np.uint64)

    with open(f"{input_path}.bin", "rb") as fp:
        fp.seek(idx[0])
        cur_file_id = None
        cur_dataset = None
        cur_dataset_updated = []

        # Iterate over the document index to reconstruct the dataset
        for doc_start, doc_end in list(pairwise(idx)):
            doc_start = int(doc_start)
            doc_end = int(doc_end)
            doc_size = doc_end - doc_start - SEPARATOR_LENGTH

            separator = fp.read(SEPARATOR_LENGTH)
            file_id, doc_id_within_file = decode_separator(separator)

            if file_id != cur_file_id:
                if cur_dataset_updated:
                    filename = id_to_filename[cur_file_id]
                    out_filename = os.path.join(output_path, filename)
                    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
                    df = pd.DataFrame(cur_dataset_updated)

                    if not out_filename.endswith(".jsonl"):
                        out_filename += ".jsonl"

                    df.to_json(out_filename + ".partial", orient="records", lines=True, force_ascii=False)
                    os.rename(out_filename + ".partial", out_filename)

                filename = id_to_filename[file_id]
                orig_path = os.path.join(original_dataset_path, filename)

                if not orig_path.endswith(".jsonl"):
                    orig_path += ".jsonl"

                cur_dataset = pd.read_json(orig_path, lines=True).to_dict("records")
                cur_file_id = file_id
                cur_dataset_updated = []

            cur_doc = cur_dataset[doc_id_within_file].copy()
            doc_bytes = fp.read(doc_size)
            tokens = np.frombuffer(doc_bytes, dtype=np.uint16).tolist()
            cur_doc[text_field] = tokenizer.decode(tokens)
            cur_dataset_updated.append(cur_doc)

        if cur_dataset_updated:
            filename = id_to_filename[cur_file_id]
            out_filename = os.path.join(output_path, filename)
            os.makedirs(os.path.dirname(out_filename), exist_ok=True)
            df = pd.DataFrame(cur_dataset_updated)

            if not out_filename.endswith(".jsonl"):
                out_filename += ".jsonl"

            df.to_json(out_filename + ".partial", orient="records", lines=True, force_ascii=False)
            os.rename(out_filename + ".partial", out_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to deduped_dataset.bin and deduped_dataset.idx files",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to output deduplicated JSONL dataset",
    )
    parser.add_argument(
        "--original-dataset-path",
        type=str,
        required=True,
        help="Path to the original JSONL dataset",
    )
    parser.add_argument(
        "--id-to-filename-path",
        type=str,
        required=True,
        help="Path to id_to_filename.json file",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Name of the text field in the original dataset",
    )
    args = parser.parse_args()

    reconstruct_dataset(
        args.input_path,
        args.output_path,
        args.original_dataset_path,
        args.id_to_filename_path,
        args.text_field,
    )
