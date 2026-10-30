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

"""Tests for nemotron.data_prep.recipes.rl_omni.ensure_raw_dir."""

from __future__ import annotations

from pathlib import Path

import pytest

from nemotron.data_prep.recipes.rl_omni import (
    _normalize_hf_repo_id,
    _raw_dir_is_present,
    ensure_raw_dir,
)


class TestNormalizeHfRepoId:
    def test_strips_hf_prefix(self):
        assert _normalize_hf_repo_id("hf://OpenGVLab/MMPR") == "OpenGVLab/MMPR"

    def test_passes_bare_repo_id_through(self):
        assert _normalize_hf_repo_id("OpenGVLab/MMPR") == "OpenGVLab/MMPR"


class TestRawDirIsPresent:
    def test_tiny_requires_zip_and_parquet(self, tmp_path: Path):
        assert not _raw_dir_is_present(tmp_path, "tiny")
        (tmp_path / "images.zip").write_bytes(b"")
        assert not _raw_dir_is_present(tmp_path, "tiny")
        (tmp_path / "mmpr_tiny.parquet").write_bytes(b"")
        assert _raw_dir_is_present(tmp_path, "tiny")

    def test_mpo_treats_any_non_empty_dir_as_staged(self, tmp_path: Path):
        # MPO's required files aren't declarative — non-empty is enough.
        assert not _raw_dir_is_present(tmp_path, "mpo")
        (tmp_path / "anything.json").write_text("{}")
        assert _raw_dir_is_present(tmp_path, "mpo")


class TestEnsureRawDirSkipsWhenStaged:
    def test_tiny_pre_staged_no_download(self, tmp_path: Path, monkeypatch):
        (tmp_path / "images.zip").write_bytes(b"")
        (tmp_path / "mmpr_tiny.parquet").write_bytes(b"")
        # Trip-wire: if the helper tries to import snapshot_download we
        # know it didn't take the early-return path.
        import nemotron.data_prep.recipes.rl_omni as mod

        original_import = __import__

        def hostile_import(name, *args, **kwargs):
            if name == "huggingface_hub":
                pytest.fail("ensure_raw_dir should not download when raw_dir is pre-staged")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", hostile_import)
        ensure_raw_dir(flavor="tiny", raw_dir=tmp_path, source_uri="hf://OpenGVLab/MMPR-Tiny")

    def test_mpo_pre_staged_no_download(self, tmp_path: Path, monkeypatch):
        (tmp_path / "meta_public.json").write_text("{}")
        original_import = __import__

        def hostile_import(name, *args, **kwargs):
            if name == "huggingface_hub":
                pytest.fail("ensure_raw_dir should not download when raw_dir is pre-staged")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", hostile_import)
        ensure_raw_dir(flavor="mpo", raw_dir=tmp_path, source_uri="hf://OpenGVLab/MMPR")


class TestEnsureRawDirFailsWithoutSourceUri:
    def test_tiny_missing_files_no_source_uri(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="images.zip"):
            ensure_raw_dir(flavor="tiny", raw_dir=tmp_path, source_uri=None)

    def test_tiny_missing_files_empty_source_uri(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="images.zip"):
            ensure_raw_dir(flavor="tiny", raw_dir=tmp_path, source_uri="")

    def test_mpo_empty_dir_no_source_uri(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            ensure_raw_dir(flavor="mpo", raw_dir=tmp_path, source_uri=None)


class TestEnsureRawDirDownloads:
    def test_tiny_calls_snapshot_download_with_normalized_repo(
        self, tmp_path: Path, monkeypatch
    ):
        calls: list[dict] = []

        def fake_snapshot_download(*, repo_id, repo_type, local_dir):
            calls.append({"repo_id": repo_id, "repo_type": repo_type, "local_dir": local_dir})
            # Simulate the real download by populating the expected files.
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "images.zip").write_bytes(b"")
            (Path(local_dir) / "mmpr_tiny.parquet").write_bytes(b"")
            return local_dir

        # Stub the lazy import. Build a fake module with the function.
        import sys
        import types

        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.snapshot_download = fake_snapshot_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

        ensure_raw_dir(
            flavor="tiny",
            raw_dir=tmp_path / "raw",
            source_uri="hf://OpenGVLab/MMPR-Tiny",
        )

        assert len(calls) == 1
        assert calls[0]["repo_id"] == "OpenGVLab/MMPR-Tiny"  # hf:// stripped
        assert calls[0]["repo_type"] == "dataset"
        assert calls[0]["local_dir"] == str((tmp_path / "raw").resolve())

    def test_raises_when_download_doesnt_satisfy_requirements(
        self, tmp_path: Path, monkeypatch
    ):
        def fake_snapshot_download(*, repo_id, repo_type, local_dir):
            # Simulate a download that succeeds but doesn't produce the
            # expected files (e.g. dataset layout changed).
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "README.md").write_text("ignored")
            return local_dir

        import sys
        import types

        fake_hf = types.ModuleType("huggingface_hub")
        fake_hf.snapshot_download = fake_snapshot_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

        with pytest.raises(FileNotFoundError, match="snapshot_download"):
            ensure_raw_dir(
                flavor="tiny",
                raw_dir=tmp_path / "raw",
                source_uri="OpenGVLab/MMPR-Tiny",
            )
