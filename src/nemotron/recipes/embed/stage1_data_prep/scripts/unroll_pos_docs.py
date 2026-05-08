# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""
Unroll training data with multiple positive documents into records with single positive documents.

This script takes training data where each question has multiple positive documents
and creates a new dataset where each (question, pos_doc) pair becomes its own record.

Input Format:
    Expects NeMo Retriever training data format (train.json):

    {
        "corpus": {"path": "/path/to/corpus/"},
        "data": [
            {
                "question_id": "q0",
                "question": "...",
                "corpus_id": "corpus_name",
                "pos_doc": [{"id": "d1"}, {"id": "d2"}],
                "neg_doc": []
            }
        ]
    }

Output:
    Same format, but each record has exactly one positive document:
    {"question_id": "q0_0", "question": "...", "pos_doc": [{"id": "d1"}]}
    {"question_id": "q0_1", "question": "...", "pos_doc": [{"id": "d2"}]}

Usage:
    python scripts/unroll_pos_docs.py data/nv_pp_dd_sdg_train_eval/train.json
    python scripts/unroll_pos_docs.py data/nv_pp_dd_sdg_train_eval/train.json --output data/nv_pp_dd_sdg_train_eval/train_unrolled.json
    python scripts/unroll_pos_docs.py data/nv_pp_dd_sdg_train_eval/train.json --suffix _unrolled
"""

import argparse
import json
from pathlib import Path
from typing import Any


def unroll_training_data(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Unroll training records with multiple positive docs into individual records.

    Args:
        data: List of training records with potentially multiple pos_doc entries

    Returns:
        List of training records where each has exactly one pos_doc
    """
    unrolled = []

    for record in data:
        pos_docs = record.get("pos_doc", [])

        if len(pos_docs) <= 1:
            # Already has single (or zero) pos_doc, keep as-is
            unrolled.append(record)
        else:
            # Unroll into multiple records
            base_question_id = record["question_id"]

            for idx, pos_doc in enumerate(pos_docs):
                new_record = {
                    "question_id": f"{base_question_id}_{idx}",
                    "question": record["question"],
                    "corpus_id": record["corpus_id"],
                    "pos_doc": [pos_doc],
                    "neg_doc": record.get("neg_doc", []),
                }
                unrolled.append(new_record)

    return unrolled


def main():
    parser = argparse.ArgumentParser(
        description="Unroll training data with multiple positive docs into single-pos-doc records"
    )
    parser.add_argument("input_file", type=str, help="Path to input training JSON file")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output JSON file. If not specified, uses input filename with '_unrolled' suffix",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_unrolled",
        help="Suffix to add to input filename for output (default: '_unrolled')",
    )

    args = parser.parse_args()

    input_path = Path(args.input_file)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}{args.suffix}{input_path.suffix}"

    print(f"Reading input: {input_path}")

    with open(input_path, "r") as f:
        training_data = json.load(f)

    # Extract corpus info and data
    corpus_info = training_data.get("corpus", {})
    data = training_data.get("data", [])

    original_count = len(data)
    original_pos_doc_count = sum(len(r.get("pos_doc", [])) for r in data)

    print(f"Original records: {original_count:,}")
    print(f"Total pos_doc entries: {original_pos_doc_count:,}")

    # Unroll the data
    unrolled_data = unroll_training_data(data)

    unrolled_count = len(unrolled_data)

    print(f"Unrolled records: {unrolled_count:,}")
    print(f"Expansion ratio: {unrolled_count / original_count:.2f}x")

    # Create output structure
    output_data = {"corpus": corpus_info, "data": unrolled_data}

    print(f"Writing output: {output_path}")

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
