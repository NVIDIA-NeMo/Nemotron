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

"""Tests for nemotron.kit.tracker helper functions."""

from pathlib import Path

import pytest

from nemotron.kit.trackers import (
    _parse_ref,
    _uri_to_artifact_name,
    to_wandb_uri,
    tokenizer_to_uri,
)


@pytest.mark.parametrize(
    ("uri", "subset", "expected"),
    [
        ("hf://nvidia/Nemotron-CC", None, "nvidia-Nemotron-CC"),
        ("hf://nvidia/Nemotron-CC", "en", "nvidia-Nemotron-CC-en"),
        ("s3://bucket/key", None, "bucket-key"),
        ("gs://bucket/key", None, "bucket-key"),
        ("local/path", None, "local-path"),
        ("path/with:invalid@chars#here", None, "path-with-invalid-chars-here"),
        ("", None, "dataset"),
    ],
)
def test_uri_to_artifact_name(uri, subset, expected):
    assert _uri_to_artifact_name(uri, subset=subset) == expected


def test_uri_to_artifact_name_truncates_to_128_chars():
    long_uri = "hf://" + "a" * 200
    assert len(_uri_to_artifact_name(long_uri)) == 128


@pytest.mark.parametrize(
    ("path", "expected_prefix"),
    [
        ("hf://nvidia/Nemotron-CC", "https://huggingface.co/datasets/nvidia/Nemotron-CC"),
        ("s3://bucket/key", "s3://bucket/key"),
        ("gs://bucket/key", "gs://bucket/key"),
        ("http://example.com/data", "http://example.com/data"),
        ("https://example.com/data", "https://example.com/data"),
        ("file:///absolute/path", "file:///absolute/path"),
    ],
)
def test_to_wandb_uri_known_protocols(path, expected_prefix):
    assert to_wandb_uri(path) == expected_prefix


def test_to_wandb_uri_local_path_is_resolved_and_file_prefixed(tmp_path):
    local = tmp_path / "subdir" / "data.json"
    result = to_wandb_uri(str(local))
    assert result.startswith("file://")
    assert Path(result.replace("file://", "")).resolve() == local.resolve()


@pytest.mark.parametrize(
    ("model", "revision", "expected"),
    [
        (
            "meta-llama/Llama-3.2-1B",
            None,
            "https://huggingface.co/meta-llama/Llama-3.2-1B",
        ),
        (
            "meta-llama/Llama-3.2-1B",
            "abc123",
            "https://huggingface.co/meta-llama/Llama-3.2-1B/tree/abc123",
        ),
    ],
)
def test_tokenizer_to_uri_hf_model(model, revision, expected):
    assert tokenizer_to_uri(model, revision=revision) == expected


def test_tokenizer_to_uri_local_path(tmp_path):
    local = tmp_path / "tokenizer"
    result = tokenizer_to_uri(str(local))
    assert result.startswith("file://")
    assert Path(result.replace("file://", "")).resolve() == local.resolve()


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("Name:v5", ("Name", 5)),
        ("Name:v0", ("Name", 0)),
        ("Name:latest", ("Name", "latest")),
        ("Name:10", ("Name", 10)),
        ("Name:rc1", ("Name", "rc1")),
        ("Name", ("Name", None)),
        ("entity/project/Name:v5", ("entity/project/Name", 5)),
    ],
)
def test_parse_ref(ref, expected):
    assert _parse_ref(ref) == expected
