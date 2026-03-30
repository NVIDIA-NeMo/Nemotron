"""Unit tests for hf:// corpus URI parsing and download."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from nemotron.recipes.embed.stage0_sdg.data_prep import _resolve_corpus_dir

# Patch target: the function does `from huggingface_hub import snapshot_download`
# inside the function body, so we patch it on the huggingface_hub module.
_PATCH_TARGET = "huggingface_hub.snapshot_download"


class TestResolveCorpusDirLocal:
    """Tests for local path handling (no hf:// prefix)."""

    def test_relative_path_resolves_to_absolute(self, tmp_path):
        d = tmp_path / "corpus"
        d.mkdir()
        result = _resolve_corpus_dir(str(d))
        assert result.is_absolute()
        assert result == d.resolve()

    def test_absolute_path_unchanged(self, tmp_path):
        d = tmp_path / "corpus"
        d.mkdir()
        result = _resolve_corpus_dir(str(d))
        assert result == d


class TestResolveCorpusDirHF:
    """Tests for hf:// URI parsing and download."""

    @patch(_PATCH_TARGET)
    def test_basic_uri(self, mock_download, tmp_path):
        mock_download.return_value = str(tmp_path)
        result = _resolve_corpus_dir("hf://nvidia/my-dataset/subdir/path")

        mock_download.assert_called_once_with(
            repo_id="nvidia/my-dataset",
            repo_type="dataset",
            allow_patterns="subdir/path/**",
        )
        assert result == tmp_path / "subdir" / "path"

    @patch(_PATCH_TARGET)
    def test_uri_with_revision(self, mock_download, tmp_path):
        mock_download.return_value = str(tmp_path)
        result = _resolve_corpus_dir("hf://nvidia/my-dataset@abc123/subdir")

        mock_download.assert_called_once_with(
            repo_id="nvidia/my-dataset",
            repo_type="dataset",
            revision="abc123",
            allow_patterns="subdir/**",
        )
        assert result == tmp_path / "subdir"

    @patch(_PATCH_TARGET)
    def test_uri_with_long_sha_revision(self, mock_download, tmp_path):
        sha = "4d3874eb983d6c653cc478cba53f3de42a8f0d24"
        mock_download.return_value = str(tmp_path)
        _resolve_corpus_dir(f"hf://nvidia/my-dataset@{sha}/data")

        mock_download.assert_called_once_with(
            repo_id="nvidia/my-dataset",
            repo_type="dataset",
            revision=sha,
            allow_patterns="data/**",
        )

    @patch(_PATCH_TARGET)
    def test_uri_no_subdir(self, mock_download, tmp_path):
        mock_download.return_value = str(tmp_path)
        result = _resolve_corpus_dir("hf://nvidia/my-dataset")

        mock_download.assert_called_once_with(
            repo_id="nvidia/my-dataset",
            repo_type="dataset",
        )
        assert result == tmp_path

    @patch(_PATCH_TARGET)
    def test_uri_revision_no_subdir(self, mock_download, tmp_path):
        mock_download.return_value = str(tmp_path)
        result = _resolve_corpus_dir("hf://nvidia/my-dataset@main")

        mock_download.assert_called_once_with(
            repo_id="nvidia/my-dataset",
            repo_type="dataset",
            revision="main",
        )
        assert result == tmp_path

    def test_invalid_uri_missing_org(self):
        with pytest.raises(SystemExit):
            _resolve_corpus_dir("hf://just-a-name")

    @patch(_PATCH_TARGET)
    def test_default_config_uri(self, mock_download, tmp_path):
        """Test the actual default config URI parses correctly."""
        mock_download.return_value = str(tmp_path)
        uri = "hf://nvidia/Retrieval-Synthetic-NVDocs-v1@1c0d1856f3fb595b2dda98d4b61061fa6d782d51/sample_corpus/nv_pp_random"
        result = _resolve_corpus_dir(uri)

        mock_download.assert_called_once_with(
            repo_id="nvidia/Retrieval-Synthetic-NVDocs-v1",
            repo_type="dataset",
            revision="1c0d1856f3fb595b2dda98d4b61061fa6d782d51",
            allow_patterns="sample_corpus/nv_pp_random/**",
        )
        assert result == tmp_path / "sample_corpus" / "nv_pp_random"
