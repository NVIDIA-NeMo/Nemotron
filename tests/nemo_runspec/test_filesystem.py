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

"""Tests for nemo_runspec.filesystem module."""

import tempfile
from pathlib import Path

import pytest

from nemo_runspec.filesystem import ArtifactFileSystem
from nemo_runspec.artifact_registry import ArtifactRegistry, set_artifact_registry


@pytest.fixture
def setup_registry_with_artifact():
    """Setup a registry with a test artifact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup registry
        root = Path(tmpdir) / "registry"
        registry = ArtifactRegistry(backend="fsspec", root=root)
        set_artifact_registry(registry)

        # Create source artifact
        source = Path(tmpdir) / "source"
        source.mkdir()
        (source / "data.json").write_text('{"key": "value"}')
        (source / "subdir").mkdir()
        (source / "subdir" / "nested.txt").write_text("nested content")

        # Publish artifact
        registry.publish("test-artifact", source, metadata={"test": True})

        yield {
            "registry": registry,
            "source": source,
            "tmpdir": tmpdir,
        }

        # Cleanup
        set_artifact_registry(None)


class TestArtifactFileSystemParseUri:
    """Tests for URI parsing functionality."""

    def test_parse_simple_name_with_version(self):
        """Test parsing simple name:version format."""
        fs = ArtifactFileSystem()
        name, version, file_path = fs._parse_uri("art://my-artifact:v1")
        assert name == "my-artifact"
        assert version == 1
        assert file_path == ""

    def test_parse_simple_name_with_version_and_file(self):
        """Test parsing name:version/file format."""
        fs = ArtifactFileSystem()
        name, version, file_path = fs._parse_uri("art://my-artifact:v1/data.json")
        assert name == "my-artifact"
        assert version == 1
        assert file_path == "data.json"

    def test_parse_simple_name_with_version_and_nested_file(self):
        """Test parsing name:version/path/to/file format."""
        fs = ArtifactFileSystem()
        name, version, file_path = fs._parse_uri("art://my-artifact:v2/path/to/file.txt")
        assert name == "my-artifact"
        assert version == 2
        assert file_path == "path/to/file.txt"

    def test_parse_latest_version(self):
        """Test parsing name:latest format."""
        fs = ArtifactFileSystem()
        name, version, file_path = fs._parse_uri("art://my-artifact:latest")
        assert name == "my-artifact"
        assert version is None  # latest maps to None
        assert file_path == ""

    def test_parse_numeric_version(self):
        """Test parsing name:N (numeric without v prefix)."""
        fs = ArtifactFileSystem()
        name, version, file_path = fs._parse_uri("art://my-artifact:5")
        assert name == "my-artifact"
        assert version == 5

    def test_parse_no_version(self):
        """Test parsing name without version (implies latest)."""
        fs = ArtifactFileSystem()
        name, version, file_path = fs._parse_uri("art://my-artifact")
        assert name == "my-artifact"
        assert version is None

    def test_parse_wandb_full_path(self):
        """Test parsing entity/project/name:version format."""
        fs = ArtifactFileSystem()
        name, version, file_path = fs._parse_uri("art://romeyn/nemotron/DataBlendsArtifact:v10")
        assert name == "romeyn/nemotron/DataBlendsArtifact"
        assert version == 10
        assert file_path == ""

    def test_parse_wandb_full_path_with_file(self):
        """Test parsing entity/project/name:version/file format."""
        fs = ArtifactFileSystem()
        name, version, file_path = fs._parse_uri(
            "art://romeyn/nemotron/DataBlendsArtifact:v10/blends.json"
        )
        assert name == "romeyn/nemotron/DataBlendsArtifact"
        assert version == 10
        assert file_path == "blends.json"

    def test_parse_without_protocol(self):
        """Test parsing without art:// prefix."""
        fs = ArtifactFileSystem()
        name, version, file_path = fs._parse_uri("my-artifact:v1/data.json")
        assert name == "my-artifact"
        assert version == 1
        assert file_path == "data.json"


class TestArtifactFileSystemOperations:
    """Tests for filesystem operations."""

    def test_open_file(self, setup_registry_with_artifact):
        """Test opening a file from artifact."""
        fs = ArtifactFileSystem()

        with fs._open("art://test-artifact:v1/data.json", "r") as f:
            content = f.read()
        assert content == '{"key": "value"}'

    def test_open_binary_file(self, setup_registry_with_artifact):
        """Test opening a file in binary mode."""
        fs = ArtifactFileSystem()

        with fs._open("art://test-artifact:v1/data.json", "rb") as f:
            content = f.read()
        assert content == b'{"key": "value"}'

    def test_open_nested_file(self, setup_registry_with_artifact):
        """Test opening a nested file."""
        fs = ArtifactFileSystem()

        with fs._open("art://test-artifact:v1/subdir/nested.txt", "r") as f:
            content = f.read()
        assert content == "nested content"

    def test_open_write_mode_fails(self, setup_registry_with_artifact):
        """Test that write mode raises error."""
        fs = ArtifactFileSystem()

        with pytest.raises(ValueError, match="read-only"):
            fs._open("art://test-artifact:v1/data.json", "w")

    def test_ls_artifact_root(self, setup_registry_with_artifact):
        """Test listing artifact root."""
        fs = ArtifactFileSystem()

        result = fs.ls("art://test-artifact:v1", detail=False)
        # Should contain data.json and subdir
        file_names = [p.split("/")[-1] for p in result]
        assert "data.json" in file_names
        assert "subdir" in file_names

    def test_ls_with_detail(self, setup_registry_with_artifact):
        """Test listing with detail=True."""
        fs = ArtifactFileSystem()

        result = fs.ls("art://test-artifact:v1", detail=True)
        assert isinstance(result, list)
        assert all(isinstance(item, dict) for item in result)

        # Find the data.json entry
        data_json = next((item for item in result if "data.json" in item["name"]), None)
        assert data_json is not None
        assert data_json["type"] == "file"
        assert data_json["size"] > 0

    def test_ls_subdirectory(self, setup_registry_with_artifact):
        """Test listing a subdirectory."""
        fs = ArtifactFileSystem()

        result = fs.ls("art://test-artifact:v1/subdir", detail=False)
        file_names = [p.split("/")[-1] for p in result]
        assert "nested.txt" in file_names

    def test_info_file(self, setup_registry_with_artifact):
        """Test getting info for a file."""
        fs = ArtifactFileSystem()

        info = fs.info("art://test-artifact:v1/data.json")
        assert info["type"] == "file"
        assert info["size"] > 0

    def test_info_directory(self, setup_registry_with_artifact):
        """Test getting info for a directory."""
        fs = ArtifactFileSystem()

        info = fs.info("art://test-artifact:v1/subdir")
        assert info["type"] == "directory"

    def test_exists_true(self, setup_registry_with_artifact):
        """Test exists returns True for existing path."""
        fs = ArtifactFileSystem()

        assert fs.exists("art://test-artifact:v1/data.json") is True
        assert fs.exists("art://test-artifact:v1/subdir") is True

    def test_exists_false(self, setup_registry_with_artifact):
        """Test exists returns False for non-existing path."""
        fs = ArtifactFileSystem()

        # Non-existing file within existing artifact
        assert fs.exists("art://test-artifact:v1/nonexistent.txt") is False
        # Non-existing version of existing artifact
        assert fs.exists("art://test-artifact:v999/data.json") is False

    def test_cat_file(self, setup_registry_with_artifact):
        """Test reading entire file content."""
        fs = ArtifactFileSystem()

        content = fs.cat_file("art://test-artifact:v1/data.json")
        assert content == b'{"key": "value"}'

    def test_cat_file_with_range(self, setup_registry_with_artifact):
        """Test reading file with byte range."""
        fs = ArtifactFileSystem()

        content = fs.cat_file("art://test-artifact:v1/data.json", start=0, end=5)
        assert content == b'{"key'

    def test_get_file(self, setup_registry_with_artifact):
        """Test copying file to local path."""
        fs = ArtifactFileSystem()

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "local_copy.json"
            fs.get_file("art://test-artifact:v1/data.json", str(dest))

            assert dest.exists()
            assert dest.read_text() == '{"key": "value"}'

    def test_isfile_true(self, setup_registry_with_artifact):
        """Test isfile returns True for file."""
        fs = ArtifactFileSystem()

        assert fs.isfile("art://test-artifact:v1/data.json") is True

    def test_isfile_false_for_dir(self, setup_registry_with_artifact):
        """Test isfile returns False for directory."""
        fs = ArtifactFileSystem()

        assert fs.isfile("art://test-artifact:v1/subdir") is False

    def test_isfile_false_for_nonexistent(self, setup_registry_with_artifact):
        """Test isfile returns False for nonexistent."""
        fs = ArtifactFileSystem()

        assert fs.isfile("art://test-artifact:v1/nonexistent.txt") is False

    def test_isdir_true(self, setup_registry_with_artifact):
        """Test isdir returns True for directory."""
        fs = ArtifactFileSystem()

        assert fs.isdir("art://test-artifact:v1/subdir") is True

    def test_isdir_false_for_file(self, setup_registry_with_artifact):
        """Test isdir returns False for file."""
        fs = ArtifactFileSystem()

        assert fs.isdir("art://test-artifact:v1/data.json") is False


class TestArtifactFileSystemProtocol:
    """Tests for filesystem protocol registration."""

    def test_protocol_attribute(self):
        """Test that protocol is 'art'."""
        fs = ArtifactFileSystem()
        assert fs.protocol == "art"
