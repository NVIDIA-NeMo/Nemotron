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

"""ChatSftShardProcessor Ray actor for parallel chat-templated SFT output processing."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import ray
from fsspec import filesystem

from nemotron.data_prep.chat_sft_shard_core import process_chat_sft_shard_core
from nemotron.data_prep.config import FileInfo

logger = logging.getLogger(__name__)


@ray.remote
class ChatSftShardProcessor:
    """Ray actor for chat-templated SFT output with loss masking."""

    def __init__(
        self,
        resolved_tokenizer: dict,
        messages_field: str,
        tools_field: str,
        pack_size: int,
        algorithm: str,
        dtype: str,
        chat_template: str | None = None,
        max_doc_tokens: int | None = None,
        max_rows: int | None = None,
        seed: int | None = None,
        used_in_filter: str | None = None,
        used_in_field: str = "used_in",
    ):
        from transformers import AutoTokenizer

        self.messages_field = messages_field
        self.tools_field = tools_field
        self.pack_size = pack_size
        self.algorithm = algorithm
        self.dtype = np.dtype(dtype)
        self.max_doc_tokens = max_doc_tokens
        self.max_rows = max_rows
        self.seed = seed
        self.used_in_filter = used_in_filter
        self.used_in_field = used_in_field

        self._tokenizer = AutoTokenizer.from_pretrained(
            resolved_tokenizer["model"],
            revision=resolved_tokenizer.get("resolved_revision"),
            trust_remote_code=resolved_tokenizer.get("trust_remote_code", False),
        )

        if chat_template:
            if chat_template == "nano3":
                template_path = Path(__file__).parent / "templates" / "nano3.jinja"
                with open(template_path) as f:
                    self._tokenizer.chat_template = f.read()
            elif Path(chat_template).exists():
                with open(chat_template) as f:
                    self._tokenizer.chat_template = f.read()
            else:
                self._tokenizer.chat_template = chat_template

    def process_shard(
        self,
        shard_index: int,
        files: list[dict],  # FileInfo as dicts for Ray serialization
        output_dir: str,
        receipts_dir: str,
        fs_protocol: str,
    ) -> dict:
        """Process files to a single packed shard with loss masks."""
        fs = filesystem(fs_protocol)

        return process_chat_sft_shard_core(
            shard_index=shard_index,
            files=[FileInfo(**f) for f in files],
            output_dir=output_dir,
            receipts_dir=receipts_dir,
            output_fs=fs,
            tokenizer=self._tokenizer,
            messages_field=self.messages_field,
            tools_field=self.tools_field,
            pack_size=self.pack_size,
            algorithm=self.algorithm,
            dtype=self.dtype,
            chat_template=None,
            max_doc_tokens=self.max_doc_tokens,
            max_rows=self.max_rows,
            seed=self.seed,
            used_in_filter=self.used_in_filter,
            used_in_field=self.used_in_field,
        )
