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

"""Tests for filesystem input discovery."""

from __future__ import annotations

import os

import fsspec

from nemotron.data_prep.config import DatasetConfig
from nemotron.data_prep.utils.discovery import discover_filesystem_files


def test_discover_local_directory_does_not_double_paths(tmp_path) -> None:
    # A local dataset directory with a couple of parquet files.
    for name in ("a.parquet", "b.parquet"):
        (tmp_path / name).write_bytes(b"data")

    fs = fsspec.filesystem("file")
    cfg = DatasetConfig(name="ds", path=str(tmp_path), text_field="text")

    files = discover_filesystem_files(cfg, fs)

    found = sorted(f.path for f in files)
    expected = sorted(str(tmp_path / name) for name in ("a.parquet", "b.parquet"))
    assert found == expected
    # Regression for path doubling: every discovered path must actually exist.
    for f in files:
        assert os.path.exists(f.path)
