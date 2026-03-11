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

"""
Text2SQL instruction dataset: JSONL with "input" (prompt) and "output" (answer)
columns, compatible with step1-dataprep training.jsonl. Uses
ColumnMappedTextInstructionDataset so behavior matches other AutoModel LLM datasets.
"""

import os
from pathlib import Path
from typing import Optional, Union

from nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset import (
    ColumnMappedTextInstructionDataset,
)

TEXT2SQL_COLUMN_MAPPING = {"question": "input", "answer": "output"}


def make_text2sql_dataset(
    tokenizer,
    path_or_dataset: Optional[Union[str, list]] = None,
    path_or_dataset_id: Optional[Union[str, list]] = None,
    split: str = "train",
    seq_length: Optional[int] = None,
    limit_dataset_samples: Optional[int] = None,
    answer_only_loss_mask: bool = True,
    padding: Union[str, bool] = "do_not_pad",
    truncation: Union[str, bool] = True,
    use_hf_chat_template: bool = False,
    **kwargs,
) -> ColumnMappedTextInstructionDataset:
    """Build a Text2SQL dataset from JSONL with 'input' and 'output' columns.

    Override path with DATASET_DIR env if set. If path is a directory,
    uses training.jsonl inside it.
    """
    path = path_or_dataset_id or path_or_dataset
    if path is None:
        path = "training.jsonl"
    if os.environ.get("DATASET_DIR"):
        base = os.environ.get("DATASET_DIR").rstrip("/")
        if isinstance(path, str) and not os.path.isabs(path) and path != "training.jsonl":
            path = os.path.join(base, path)
        else:
            path = base if Path(base).is_file() else os.path.join(base, "training.jsonl")
    if isinstance(path, str) and Path(path).is_dir():
        path = os.path.join(path, "training.jsonl")
    return ColumnMappedTextInstructionDataset(
        path_or_dataset_id=path,
        column_mapping=TEXT2SQL_COLUMN_MAPPING.copy(),
        tokenizer=tokenizer,
        split=split,
        seq_length=seq_length,
        limit_dataset_samples=limit_dataset_samples,
        answer_only_loss_mask=answer_only_loss_mask,
        padding=padding,
        truncation=truncation,
        use_hf_chat_template=use_hf_chat_template,
        **kwargs,
    )
