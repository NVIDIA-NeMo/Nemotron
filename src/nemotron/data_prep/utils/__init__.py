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

"""Utility modules for data preparation.

This package provides:
- Filesystem utilities (get_filesystem, read_json, write_json)
- Input file discovery (discover_input_files, fetch_hf_dataset_metadata)
- Size formatting utilities (parse_byte_size, format_byte_size)
- HuggingFace environment utilities (detect_hf_env_vars, make_hf_runtime_env)
- HuggingFace placeholder resolution for RL datasets
- Train/valid/test split utilities (distribute_shards_to_splits)
"""

from nemotron.data_prep.utils.hf_env import detect_hf_env_vars, make_hf_runtime_env
from nemotron.data_prep.utils.size import format_byte_size, parse_byte_size

# Filesystem utilities
from nemotron.data_prep.utils.filesystem import (
    ensure_dir,
    file_exists,
    get_filesystem,
    read_json,
    write_json,
)

# Input file discovery
from nemotron.data_prep.utils.discovery import (
    DatasetMetadata,
    discover_input_files,
    fetch_hf_dataset_metadata,
)

# HF placeholder resolution
from nemotron.data_prep.utils.hf_placeholder import (
    HFPlaceholderResolver,
    PlaceholderConfig,
    TARGET_DATASETS,
    is_placeholder_record,
)

# Split utilities
from nemotron.data_prep.utils.splits import distribute_shards_to_splits

__all__ = [
    # Size utilities
    "parse_byte_size",
    "format_byte_size",
    # HF environment
    "detect_hf_env_vars",
    "make_hf_runtime_env",
    # Filesystem
    "get_filesystem",
    "read_json",
    "write_json",
    "ensure_dir",
    "file_exists",
    # Discovery
    "discover_input_files",
    "fetch_hf_dataset_metadata",
    "DatasetMetadata",
    # HF placeholder
    "HFPlaceholderResolver",
    "PlaceholderConfig",
    "TARGET_DATASETS",
    "is_placeholder_record",
    # Splits
    "distribute_shards_to_splits",
]
