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

"""Tests for nemotron.kit.registry module."""

import tempfile
from pathlib import Path

import pytest

from nemotron.kit.exceptions import ArtifactNotFoundError, ArtifactVersionNotFoundError
from nemotron.kit.registry import (
    ArtifactEntry,
    ArtifactRegistry,
    ArtifactVersion,
    get_registry,
    set_registry,
)


class TestArtifactVersion:
    """Tests for ArtifactVersion dataclass."""

    def test_create_version(self):
        """Test creating an artifact version."""
        version = ArtifactVersion(
            version=1,
            path="/data/artifacts/my-artifact/v1",
            created_at="2025-01-01T00:00:00+00:00",
            metadata={"tokens": 1000},
        )
        assert version.version == 1
        assert version.path == "/data/artifacts/my-artifact/v1"
        assert version.created_at == "2025-01-01T00:00:00+00:00"
        assert version.metadata == {"tokens": 1000}

    def test_default_metadata(self):
        """Test that metadata defaults to empty dict."""
        version = ArtifactVersion(
            version=1,
            path="/path",
            created_at="2025-01-01T00:00:00+00:00",
        )
        assert version.metadata == {}


class TestArtifactEntry:
    """Tests for ArtifactEntry dataclass."""

    def test_create_entry(self):
        """Test creating an artifact entry."""
        entry = ArtifactEntry(name="my-artifact")
        assert entry.name == "my-artifact"
        assert entry.versions == []
        assert entry.aliases == {}

    def test_latest_version_empty(self):
        """Test latest_version with no versions."""
        entry = ArtifactEntry(name="test")
        assert entry.latest_version() is None

    def test_latest_version(self):
        """Test latest_version returns highest version number."""
        entry = ArtifactEntry(
            name="test",
            versions=[
                ArtifactVersion(version=1, path="/v1", created_at="2025-01-01"),
                ArtifactVersion(version=3, path="/v3", created_at="2025-01-03"),
                ArtifactVersion(version=2, path="/v2", created_at="2025-01-02"),
            ],
        )
        latest = entry.latest_version()
        assert latest is not None
        assert latest.version == 3

    def test_get_version_found(self):
        """Test get_version returns correct version."""
        entry = ArtifactEntry(
            name="test",
            versions=[
                ArtifactVersion(version=1, path="/v1", created_at="2025-01-01"),
                ArtifactVersion(version=2, path="/v2", created_at="2025-01-02"),
            ],
        )
        v1 = entry.get_version(1)
        assert v1 is not None
        assert v1.path == "/v1"

    def test_get_version_not_found(self):
        """Test get_version returns None for missing version."""
        entry = ArtifactEntry(name="test")
        assert entry.get_version(1) is None

    def test_to_dict(self):
        """Test serialization to dict."""
        entry = ArtifactEntry(
            name="test",
            versions=[
                ArtifactVersion(
                    version=1,
                    path="/v1",
                    created_at="2025-01-01",
                    metadata={"key": "value"},
                ),
            ],
            aliases={"latest": 1, "prod": 1},
        )
        data = entry.to_dict()
        assert data["name"] == "test"
        assert len(data["versions"]) == 1
        assert data["versions"][0]["version"] == 1
        assert data["versions"][0]["metadata"] == {"key": "value"}
        assert data["aliases"] == {"latest": 1, "prod": 1}

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "name": "test",
            "versions": [
                {
                    "version": 1,
                    "path": "/v1",
                    "created_at": "2025-01-01",
                    "metadata": {"key": "value"},
                },
            ],
            "aliases": {"latest": 1},
        }
        entry = ArtifactEntry.from_dict(data)
        assert entry.name == "test"
        assert len(entry.versions) == 1
        assert entry.versions[0].version == 1
        assert entry.versions[0].metadata == {"key": "value"}
        assert entry.aliases == {"latest": 1}

    def test_from_dict_missing_metadata(self):
        """Test deserialization handles missing metadata."""
        data = {
            "name": "test",
            "versions": [
                {
                    "version": 1,
                    "path": "/v1",
                    "created_at": "2025-01-01",
                    # No metadata field
                },
            ],
        }
        entry = ArtifactEntry.from_dict(data)
        assert entry.versions[0].metadata == {}


class TestArtifactRegistryFsspec:
    """Tests for ArtifactRegistry with fsspec backend."""

    def test_init_creates_root(self):
        """Test that init creates root directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "artifacts"
            registry = ArtifactRegistry(backend="fsspec", root=root)
            assert root.exists()
            assert registry.backend == "fsspec"

    def test_init_fsspec_requires_root(self):
        """Test that fsspec backend requires root."""
        with pytest.raises(ValueError, match="root is required"):
            ArtifactRegistry(backend="fsspec")

    def test_init_wandb_requires_project(self):
        """Test that wandb backend requires project."""
        with pytest.raises(ValueError, match="wandb_project is required"):
            ArtifactRegistry(backend="wandb")

    def test_publish_and_resolve(self):
        """Test publishing and resolving an artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "registry"
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "data.txt").write_text("test data")

            registry = ArtifactRegistry(backend="fsspec", root=root)
            version = registry.publish("my-artifact", source, metadata={"tokens": 1000})

            assert version.version == 1
            assert version.metadata == {"tokens": 1000}

            # Resolve should return path to artifact
            resolved = registry.resolve("my-artifact")
            assert resolved.exists()
            assert (resolved / "data.txt").read_text() == "test data"

    def test_publish_increments_version(self):
        """Test that publishing increments version number."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "registry"
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "data.txt").write_text("v1")

            registry = ArtifactRegistry(backend="fsspec", root=root)

            v1 = registry.publish("my-artifact", source)
            assert v1.version == 1

            (source / "data.txt").write_text("v2")
            v2 = registry.publish("my-artifact", source)
            assert v2.version == 2

            (source / "data.txt").write_text("v3")
            v3 = registry.publish("my-artifact", source)
            assert v3.version == 3

    def test_resolve_specific_version(self):
        """Test resolving a specific version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "registry"
            source = Path(tmpdir) / "source"
            source.mkdir()

            registry = ArtifactRegistry(backend="fsspec", root=root)

            (source / "data.txt").write_text("v1")
            registry.publish("my-artifact", source)

            (source / "data.txt").write_text("v2")
            registry.publish("my-artifact", source)

            # Resolve version 1
            resolved = registry.resolve("my-artifact", version=1)
            assert (resolved / "data.txt").read_text() == "v1"

            # Resolve version 2
            resolved = registry.resolve("my-artifact", version=2)
            assert (resolved / "data.txt").read_text() == "v2"

    def test_resolve_latest_alias(self):
        """Test resolving 'latest' alias."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "registry"
            source = Path(tmpdir) / "source"
            source.mkdir()

            registry = ArtifactRegistry(backend="fsspec", root=root)

            (source / "data.txt").write_text("v1")
            registry.publish("my-artifact", source)

            (source / "data.txt").write_text("v2")
            registry.publish("my-artifact", source)

            # Resolve latest (should be v2)
            resolved = registry.resolve("my-artifact", version="latest")
            assert (resolved / "data.txt").read_text() == "v2"

    def test_resolve_artifact_not_found(self):
        """Test that resolving nonexistent artifact raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "registry"
            registry = ArtifactRegistry(backend="fsspec", root=root)

            with pytest.raises(ArtifactNotFoundError):
                registry.resolve("nonexistent")

    def test_resolve_version_not_found(self):
        """Test that resolving nonexistent version raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "registry"
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "data.txt").write_text("test")

            registry = ArtifactRegistry(backend="fsspec", root=root)
            registry.publish("my-artifact", source)

            with pytest.raises(ArtifactVersionNotFoundError):
                registry.resolve("my-artifact", version=999)

    def test_alias_create_and_resolve(self):
        """Test creating and resolving an alias."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "registry"
            source = Path(tmpdir) / "source"
            source.mkdir()

            registry = ArtifactRegistry(backend="fsspec", root=root)

            (source / "data.txt").write_text("v1")
            registry.publish("my-artifact", source)

            (source / "data.txt").write_text("v2")
            registry.publish("my-artifact", source)

            # Create alias for v1
            registry.alias("my-artifact", "production", 1)

            # Resolve via alias
            resolved = registry.resolve("my-artifact", version="production")
            assert (resolved / "data.txt").read_text() == "v1"

    def test_alias_artifact_not_found(self):
        """Test aliasing nonexistent artifact raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "registry"
            registry = ArtifactRegistry(backend="fsspec", root=root)

            with pytest.raises(ArtifactNotFoundError):
                registry.alias("nonexistent", "prod", 1)

    def test_alias_version_not_found(self):
        """Test aliasing nonexistent version raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "registry"
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "data.txt").write_text("test")

            registry = ArtifactRegistry(backend="fsspec", root=root)
            registry.publish("my-artifact", source)

            with pytest.raises(ArtifactVersionNotFoundError):
                registry.alias("my-artifact", "prod", 999)

    def test_get_artifact(self):
        """Test getting artifact entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "registry"
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "data.txt").write_text("test")

            registry = ArtifactRegistry(backend="fsspec", root=root)
            registry.publish("my-artifact", source)

            entry = registry.get("my-artifact")
            assert entry is not None
            assert entry.name == "my-artifact"
            assert len(entry.versions) == 1

    def test_get_artifact_not_found(self):
        """Test getting nonexistent artifact returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "registry"
            registry = ArtifactRegistry(backend="fsspec", root=root)

            assert registry.get("nonexistent") is None

    def test_list_artifacts(self):
        """Test listing all artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "registry"
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "data.txt").write_text("test")

            registry = ArtifactRegistry(backend="fsspec", root=root)
            registry.publish("artifact-a", source)
            registry.publish("artifact-b", source)
            registry.publish("artifact-c", source)

            artifacts = registry.list()
            assert sorted(artifacts) == ["artifact-a", "artifact-b", "artifact-c"]

    def test_persistence(self):
        """Test that registry persists across restarts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "registry"
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "data.txt").write_text("test data")

            # First registry instance
            registry1 = ArtifactRegistry(backend="fsspec", root=root)
            registry1.publish("my-artifact", source, metadata={"key": "value"})
            registry1.alias("my-artifact", "prod", 1)

            # Second registry instance (simulating restart)
            registry2 = ArtifactRegistry(backend="fsspec", root=root)

            # Should find the artifact
            entry = registry2.get("my-artifact")
            assert entry is not None
            assert len(entry.versions) == 1
            assert entry.versions[0].metadata == {"key": "value"}
            assert entry.aliases["prod"] == 1

    def test_publish_single_file(self):
        """Test publishing a single file instead of directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "registry"
            source_file = Path(tmpdir) / "model.bin"
            source_file.write_bytes(b"model data")

            registry = ArtifactRegistry(backend="fsspec", root=root)
            version = registry.publish("my-model", source_file)

            resolved = registry.resolve("my-model")
            assert (resolved / "model.bin").read_bytes() == b"model data"


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_registry_not_initialized(self):
        """Test that get_registry raises when not initialized."""
        set_registry(None)
        with pytest.raises(RuntimeError, match="not initialized"):
            get_registry()

    def test_set_and_get_registry(self):
        """Test setting and getting global registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "registry"
            registry = ArtifactRegistry(backend="fsspec", root=root)

            set_registry(registry)
            assert get_registry() is registry

            # Cleanup
            set_registry(None)
