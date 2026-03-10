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
import os
import shutil
import struct

import numpy as np
import tiktoken

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.backends.experimental.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.tasks import DocumentBatch, FileGroupTask
from nemo_curator.utils.file_utils import get_all_file_paths_under

from utils import SEPARATOR_MAGIC, div_up


def make_separator(file_id: int, doc_id_within_file: int):
    return SEPARATOR_MAGIC + struct.pack("<II", file_id, doc_id_within_file)


class PrepareSinglePartition(ProcessingStage[DocumentBatch, FileGroupTask]):
    def __init__(self, output_path: str, text_field: str = "text", filename_to_id: dict[str, str] = None):
        self.output_path = output_path
        self.text_field = text_field
        self.filename_to_id = filename_to_id
        self.chunk_size = 8192

    def setup(self, _: WorkerMetadata) -> None:
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def process(self, batch: DocumentBatch) -> FileGroupTask:
        df = batch.to_pandas()

        # Extract filename and file_id
        filename = batch._metadata["source_files"]
        assert len(filename) == 1  # assumes files_per_partition=1
        filename = filename[0]
        filename = os.path.basename(filename)
        file_id = int(self.filename_to_id[filename])
        output_path = os.path.join(self.output_path, "prepare_single", filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write bin file for this batch
        with open(f"{output_path}.bin", "wb") as fp:
            idx = [0]
            for i, row in df.iterrows():
                text = row[self.text_field]
                num_chunks = div_up(len(text), self.chunk_size)
                tokens = []
                for j in range(0, num_chunks):
                    this_text = text[j * self.chunk_size : (j + 1) * self.chunk_size]
                    tokens += self.tokenizer.encode(this_text, disallowed_special=())
                bytes_ = np.array(tokens, dtype=np.uint16).view(np.uint8).tobytes()

                next_line = make_separator(file_id, i) + bytes_
                fp.write(next_line)
                idx.append(idx[-1] + len(next_line))

        # Write idx file for this batch
        with open(f"{output_path}.idx", "wb") as fp:
            fp.write(np.array(idx, dtype=np.uint64).tobytes())

        return FileGroupTask(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=[output_path],
        )


def write_id_to_filename(input_file_paths: list[str], output_path: str) -> dict[str, str]:
    import json

    input_file_list = [os.path.basename(filename) for filename in input_file_paths]
    id_to_filename = {str(i): filename for i, filename in enumerate(input_file_list)}

    with open(f"{output_path}/id_to_filename.json", "w") as fp:
        json.dump(id_to_filename, fp)

    # Return filename_to_id dictionary
    return {v: k for k, v in id_to_filename.items()}


def prepare_dataset(input_path: str, output_path: str, text_field: str = "text", file_limit: int = None):
    input_file_paths = get_all_file_paths_under(
        input_path,
        recurse_subdirectories=True,
        keep_extensions=["jsonl"],
    )
    if file_limit is not None:
        input_file_paths = input_file_paths[:file_limit]

    filename_to_id = write_id_to_filename(input_file_paths, output_path)

    # Execute Curator pipeline
    pipeline = Pipeline(name="prepare_dataset")
    pipeline.add_stage(JsonlReader(input_file_paths))
    pipeline.add_stage(PrepareSinglePartition(output_path, text_field, filename_to_id))
    results = pipeline.run(RayDataExecutor())

    # Gather all the absolute paths to the bin and idx files
    filenames = [result.data[0] for result in results]
    output_dir = f"{output_path}/data"
    os.makedirs(output_dir, exist_ok=True)

    # Write bin file for the entire dataset
    with open(f"{output_dir}/dataset.bin", "wb") as fp_out:
        for filename in filenames:
            bin_path = f"{filename}.bin"
            if os.path.getsize(bin_path) == 0:
                continue
            with open(bin_path, "rb") as fp_in:
                shutil.copyfileobj(fp_in, fp_out)

    # Write idx file for the entire dataset
    with open(f"{output_dir}/dataset.idx", "wb") as fp_out:
        offset = None
        for filename in filenames:
            idx_path = f"{filename}.idx"
            if os.path.getsize(idx_path) == 0:
                continue
            with open(idx_path, "rb") as fp_in:
                idx = np.frombuffer(fp_in.read(), dtype=np.uint64)
            if offset is None:
                shifted_idx = idx
            else:
                shifted_idx = idx[1:] + offset
            fp_out.write(np.array(shifted_idx, dtype=np.uint64).tobytes())
            offset = shifted_idx[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path", type=str, required=True, help="Path to input JSONL dataset"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to place output prepare_single directory, id_to_filename.json file, and data directory",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Name of the text field in the original dataset",
    )
    parser.add_argument(
        "--file-limit",
        type=int,
        default=None,
        help="Maximum number of input files to process",
    )
    args = parser.parse_args()

    client = RayClient()
    client.start()

    prepare_dataset(args.input_path, args.output_path, text_field=args.text_field, file_limit=args.file_limit)

    client.stop()
