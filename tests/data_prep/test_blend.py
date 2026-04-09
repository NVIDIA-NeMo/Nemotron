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

"""Tests for DataBlend and Dataset models."""

import json

import pytest
from pydantic import ValidationError

from nemotron.data_prep.blend import DataBlend, Dataset


# =============================================================================
# Dataset
# =============================================================================


class TestDataset:
    def test_minimal(self) -> None:
        ds = Dataset(name="test", path="hf://org/repo")
        assert ds.name == "test"
        assert ds.path == "hf://org/repo"
        assert ds.weight == 1.0
        assert ds.text_field == "text"
        assert ds.split is None
        assert ds.subset is None

    def test_all_fields(self) -> None:
        ds = Dataset(
            name="pile",
            path="hf://EleutherAI/pile",
            weight=2.5,
            split="train",
            subset="en",
            text_field="content",
        )
        assert ds.weight == 2.5
        assert ds.split == "train"
        assert ds.subset == "en"
        assert ds.text_field == "content"

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            Dataset(name="test")  # type: ignore[call-arg]

        with pytest.raises(ValidationError):
            Dataset(path="hf://repo")  # type: ignore[call-arg]


# =============================================================================
# DataBlend — Single Blend Mode
# =============================================================================


class TestDataBlendSingleMode:
    def test_basic(self) -> None:
        ds = Dataset(name="a", path="/data/a")
        blend = DataBlend(datasets=[ds])
        assert not blend.is_per_split
        assert blend.datasets == [ds]

    def test_splits_property(self) -> None:
        ds = Dataset(name="a", path="/data/a")
        blend = DataBlend(datasets=[ds])
        assert blend.splits == {"all": [ds]}

    def test_from_datasets(self) -> None:
        d1 = Dataset(name="a", path="/a")
        d2 = Dataset(name="b", path="/b")
        blend = DataBlend.from_datasets(d1, d2)
        assert blend.datasets == [d1, d2]
        assert not blend.is_per_split

    def test_multiple_datasets(self) -> None:
        datasets = [
            Dataset(name="a", path="/a", weight=1.0),
            Dataset(name="b", path="/b", weight=2.0),
            Dataset(name="c", path="/c", weight=0.5),
        ]
        blend = DataBlend(datasets=datasets)
        assert len(blend.datasets) == 3


# =============================================================================
# DataBlend — Per-Split Mode
# =============================================================================


class TestDataBlendPerSplitMode:
    def test_train_only(self) -> None:
        ds = Dataset(name="t", path="/t")
        blend = DataBlend(train=[ds])
        assert blend.is_per_split
        assert blend.splits == {"train": [ds]}

    def test_train_and_valid(self) -> None:
        t = Dataset(name="t", path="/t")
        v = Dataset(name="v", path="/v")
        blend = DataBlend(train=[t], valid=[v])
        assert blend.is_per_split
        assert "train" in blend.splits
        assert "valid" in blend.splits
        assert "test" not in blend.splits

    def test_all_splits(self) -> None:
        t = Dataset(name="t", path="/t")
        v = Dataset(name="v", path="/v")
        te = Dataset(name="te", path="/te")
        blend = DataBlend(train=[t], valid=[v], test=[te])
        assert len(blend.splits) == 3


# =============================================================================
# DataBlend — Validation
# =============================================================================


class TestDataBlendValidation:
    def test_empty_fails(self) -> None:
        with pytest.raises(ValidationError, match="Must specify"):
            DataBlend()

    def test_mixed_mode_fails(self) -> None:
        ds = Dataset(name="a", path="/a")
        with pytest.raises(ValidationError, match="Cannot mix"):
            DataBlend(datasets=[ds], train=[ds])

    def test_datasets_with_valid_fails(self) -> None:
        ds = Dataset(name="a", path="/a")
        with pytest.raises(ValidationError, match="Cannot mix"):
            DataBlend(datasets=[ds], valid=[ds])


# =============================================================================
# DataBlend — Load from JSON
# =============================================================================


class TestDataBlendLoad:
    def test_load_single_blend(self, tmp_path) -> None:
        data = {
            "datasets": [
                {"name": "pile", "path": "hf://EleutherAI/pile", "split": "train"},
            ]
        }
        path = tmp_path / "blend.json"
        path.write_text(json.dumps(data))

        blend = DataBlend.load(path)
        assert not blend.is_per_split
        assert len(blend.datasets) == 1
        assert blend.datasets[0].name == "pile"

    def test_load_per_split(self, tmp_path) -> None:
        data = {
            "train": [{"name": "train_ds", "path": "/train"}],
            "valid": [{"name": "valid_ds", "path": "/valid"}],
        }
        path = tmp_path / "blend.json"
        path.write_text(json.dumps(data))

        blend = DataBlend.load(path)
        assert blend.is_per_split
        assert "train" in blend.splits
        assert "valid" in blend.splits

    def test_roundtrip(self, tmp_path) -> None:
        original = DataBlend(
            datasets=[
                Dataset(name="a", path="/a", weight=2.0),
                Dataset(name="b", path="/b", weight=0.5),
            ]
        )
        path = tmp_path / "blend.json"
        path.write_text(original.model_dump_json())

        loaded = DataBlend.load(path)
        assert loaded.datasets == original.datasets
