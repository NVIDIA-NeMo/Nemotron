# Copyright (c) 2025, NVIDIA CORPORATION.
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
Tests for nemotron artifact functionality.
"""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import Field

from nemotron.artifact import Artifact, apply_scale


class SampleDataset(Artifact):
    """Sample artifact for testing."""

    num_examples: int = Field(gt=0)
    quality: float = Field(ge=0.0, le=1.0)


def test_artifact_save_and_load():
    """Test saving and loading an artifact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_artifact"

        # Create artifact
        artifact = SampleDataset(
            path=output_dir,
            num_examples=100,
            quality=0.85,
            metrics={"num_examples": 100, "quality": 0.85},
        )

        # Save
        artifact.save()

        # Verify metadata.json exists
        metadata_path = output_dir / "metadata.json"
        assert metadata_path.exists()

        # Load and verify
        loaded = SampleDataset.load(path=output_dir)
        assert loaded.num_examples == 100
        assert loaded.quality == 0.85
        assert loaded.metrics["num_examples"] == 100


def test_artifact_validation():
    """Test Pydantic validation works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_validation"

        # Valid artifact
        valid = SampleDataset(path=output_dir, num_examples=100, quality=0.5, metrics={})
        assert valid.num_examples == 100

        # Invalid: negative num_examples
        with pytest.raises(Exception):  # Pydantic ValidationError
            SampleDataset(path=output_dir, num_examples=-1, quality=0.5, metrics={})

        # Invalid: quality out of range
        with pytest.raises(Exception):  # Pydantic ValidationError
            SampleDataset(path=output_dir, num_examples=100, quality=1.5, metrics={})


def test_artifact_metadata_format():
    """Test that metadata.json has correct format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_format"

        artifact = SampleDataset(
            path=output_dir,
            num_examples=50,
            quality=0.75,
            metrics={"accuracy": 0.9},
            attrs={"source": "test"},
        )
        artifact.save()

        # Load metadata.json
        with open(output_dir / "metadata.json") as f:
            metadata = json.load(f)

        # Verify required fields
        assert metadata["schema_version"] == 1
        assert metadata["type"] == "sampledataset"
        assert metadata["num_examples"] == 50
        assert metadata["quality"] == 0.75
        assert metadata["metrics"]["accuracy"] == 0.9
        assert metadata["attrs"]["source"] == "test"
        assert metadata["producer"] == "local"


def test_apply_scale():
    """Test scale factor utility."""
    assert apply_scale(100_000, "tiny") == 1_000  # 1%
    assert apply_scale(100_000, "small") == 10_000  # 10%
    assert apply_scale(100_000, "medium") == 30_000  # 30%
    assert apply_scale(100_000, "full") == 100_000  # 100%

    # Minimum 1 even for tiny scale
    assert apply_scale(10, "tiny") == 1

    # Cap tiny at 10k rows
    assert apply_scale(2_000_000, "tiny") == 10_000  # Would be 20k, capped at 10k
    assert apply_scale(500_000, "tiny") == 5_000  # Under cap, not affected

    # Invalid scale
    with pytest.raises(ValueError):
        apply_scale(100, "invalid")


def test_artifact_type_inference():
    """Test that artifact type is inferred from class name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_type"

        artifact = SampleDataset(path=output_dir, num_examples=10, quality=0.5, metrics={})

        # Type should be lowercase class name
        assert artifact.type == "sampledataset"
