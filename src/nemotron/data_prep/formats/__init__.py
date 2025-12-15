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

"""Output format implementations."""

from nemotron.data_prep.formats.indexed_dataset import IndexedDatasetBuilder
from nemotron.data_prep.formats.jsonl_dataset import JsonlDatasetBuilder
from nemotron.data_prep.formats.transforms import (
    Conversation,
    Message,
    OpenAIChatRecord,
    SftRecord,
    SftRecordWithSystem,
    ShareGPTRecord,
    Transform,
    openai_chat,
    passthrough,
    rename,
    select,
    sft,
    sharegpt,
)

__all__ = [
    # Writers
    "IndexedDatasetBuilder",
    "JsonlDatasetBuilder",
    # Transform types
    "Transform",
    "SftRecord",
    "SftRecordWithSystem",
    "Message",
    "OpenAIChatRecord",
    "Conversation",
    "ShareGPTRecord",
    # Transform factories
    "sft",
    "openai_chat",
    "sharegpt",
    "passthrough",
    "select",
    "rename",
]
