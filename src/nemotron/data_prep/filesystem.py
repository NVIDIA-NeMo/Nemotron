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

"""Filesystem utilities using fsspec for cloud-native operations."""

import json
import os
from typing import Any

from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs


def get_filesystem(path: str) -> tuple[AbstractFileSystem, str]:
    """Get filesystem and normalized path from a URI.

    For local paths, resolves '..' and '.' components to produce absolute paths.
    This ensures paths in blend.json are fully resolved and portable.
    """
    # For local paths (no scheme or file://), normalize first
    # This resolves '..' and '.' components
    if not path.startswith(("s3://", "gs://", "gcs://", "hdfs://", "http://", "https://")):
        # Handle file:// scheme
        if path.startswith("file://"):
            local_path = path[7:]
            resolved = os.path.realpath(local_path)
            path = f"file://{resolved}"
        else:
            # Plain local path - resolve it
            path = os.path.realpath(path)

    fs, normalized = url_to_fs(path)
    return fs, normalized


def read_json(fs: AbstractFileSystem, path: str) -> Any:
    """Read JSON file from filesystem."""
    with fs.open(path, "r") as f:
        return json.load(f)


def write_json(fs: AbstractFileSystem, path: str, data: Any, indent: int = 2) -> None:
    """Write JSON file to filesystem."""
    with fs.open(path, "w") as f:
        json.dump(data, f, indent=indent)


def ensure_dir(fs: AbstractFileSystem, path: str) -> None:
    """Ensure directory exists, creating it if necessary."""
    fs.makedirs(path, exist_ok=True)


def file_exists(fs: AbstractFileSystem, path: str) -> bool:
    """Check if a file exists."""
    try:
        return fs.exists(path)
    except Exception:
        return False
