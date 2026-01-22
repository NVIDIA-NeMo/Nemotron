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

"""Xenna integration for Nemotron data prep.

Architecture:
- executor.py: run_xenna() - format-agnostic pipeline executor
- pipeline_specs.py: build_*_pipeline_spec() - PipelineSpec factories
- observability.py: wandb callback and polling helpers
- stages.py: Xenna Stage implementations
- work_items.py: Work item dataclasses
"""

from nemotron.data_prep.xenna.executor import run_xenna
from nemotron.data_prep.xenna.observability import (
    start_download_poller,
    start_receipt_poller,
)
from nemotron.data_prep.xenna.pipeline_specs import (
    build_chat_sft_pipeline_spec,
    build_jsonl_pipeline_spec,
    build_pretrain_pipeline_spec,
)

# Stage and work item exports
from nemotron.data_prep.xenna.stages import (
    ChatSftCentralPackStage,
    ChatSftSpoolStage,
    HfPredownloadStage,
    JsonlShardStage,
    PretrainShardStage,
)
from nemotron.data_prep.xenna.work_items import (
    ChatSftShardWorkItem,
    ChatSftSpoolWorkItem,
    JsonlShardWorkItem,
    ShardWorkItem,
)

__all__ = [
    # Core
    "run_xenna",
    "build_pretrain_pipeline_spec",
    "build_jsonl_pipeline_spec",
    "build_chat_sft_pipeline_spec",
    "start_receipt_poller",
    "start_download_poller",
    # Stages
    "HfPredownloadStage",
    "PretrainShardStage",
    "JsonlShardStage",
    "ChatSftSpoolStage",
    "ChatSftCentralPackStage",
    # Work items
    "ShardWorkItem",
    "JsonlShardWorkItem",
    "ChatSftShardWorkItem",
    "ChatSftSpoolWorkItem",
]
