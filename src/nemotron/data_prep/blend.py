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

"""Data blend specification - pure data manifest."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, model_validator


class Dataset(BaseModel):
    """Single dataset entry in a blend.

    Attributes:
        name: Unique identifier for this dataset
        path: Data location (hf://repo/name, s3://bucket/prefix, /local/path)
        weight: Relative weight in the blend (default: 1.0)
        split: HuggingFace split name (required for hf:// paths)
        subset: HuggingFace config/subset name
        text_field: Field containing text to tokenize (default: "text")
    """

    name: str
    path: str
    weight: float = 1.0
    split: str | None = None
    subset: str | None = None
    text_field: str = "text"


class DataBlend(BaseModel):
    """Data blend specification.

    Supports two modes:

    1. Single blend mode:
       {"datasets": [...]}
       Megatron-Bridge splits by ratio at training time.

    2. Per-split mode:
       {"train": [...], "valid": [...], "test": [...]}
       Separate tokenized outputs for each split.

    Examples:
        # Single blend
        blend = DataBlend(datasets=[
            Dataset(name="pile", path="hf://EleutherAI/pile", split="train"),
        ])

        # Per-split
        blend = DataBlend(
            train=[Dataset(name="train", path="hf://...", split="train")],
            valid=[Dataset(name="valid", path="hf://...", split="validation")],
        )

        # From file
        blend = DataBlend.load("data_blend.json")
    """

    # Single blend mode
    datasets: list[Dataset] | None = None

    # Per-split mode
    train: list[Dataset] | None = None
    valid: list[Dataset] | None = None
    test: list[Dataset] | None = None

    @model_validator(mode="after")
    def check_mode(self) -> DataBlend:
        """Validate exactly one mode is specified."""
        has_single = self.datasets is not None
        has_splits = any([self.train, self.valid, self.test])

        if has_single and has_splits:
            raise ValueError(
                "Cannot mix 'datasets' with 'train'/'valid'/'test'. "
                "Use either single blend mode or per-split mode."
            )
        if not has_single and not has_splits:
            raise ValueError(
                "Must specify either 'datasets' or at least one of 'train'/'valid'/'test'."
            )
        return self

    @property
    def is_per_split(self) -> bool:
        """True if using per-split mode."""
        return self.datasets is None

    @property
    def splits(self) -> dict[str, list[Dataset]]:
        """Return datasets grouped by split name.

        For single blend mode, returns {"all": datasets}.
        For per-split mode, returns {"train": [...], "valid": [...], ...}.
        """
        if not self.is_per_split:
            return {"all": self.datasets}
        return {
            k: v
            for k, v in [
                ("train", self.train),
                ("valid", self.valid),
                ("test", self.test),
            ]
            if v is not None
        }

    @classmethod
    def load(cls, path: str | Path) -> DataBlend:
        """Load blend specification from JSON file."""
        with open(path) as f:
            return cls.model_validate(json.load(f))

    @classmethod
    def from_datasets(cls, *datasets: Dataset) -> DataBlend:
        """Create single-blend from dataset list."""
        return cls(datasets=list(datasets))
