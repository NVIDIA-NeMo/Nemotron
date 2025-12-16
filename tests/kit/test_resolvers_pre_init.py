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

import sys
import types

from omegaconf import OmegaConf

from nemotron.kit import resolvers


class TestGetJobId:
    """Tests for _get_job_id() function."""

    def test_uses_slurm_job_id_when_available(self, monkeypatch):
        """Should use SLURM_JOB_ID environment variable when available."""
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        job_id = resolvers._get_job_id()
        assert job_id == "slurm_12345"

    def test_uses_torchelastic_run_id_when_available(self, monkeypatch):
        """Should use TORCHELASTIC_RUN_ID when SLURM_JOB_ID is not available."""
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setenv("TORCHELASTIC_RUN_ID", "abc123")
        job_id = resolvers._get_job_id()
        assert "abc123" in job_id

    def test_fallback_to_hostname_and_ppid(self, monkeypatch):
        """Should fall back to hostname and parent PID."""
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
        job_id = resolvers._get_job_id()
        # Should contain hostname and some numeric ID
        assert "_" in job_id


class TestMarkerPathUniqueness:
    """Tests for marker file path uniqueness across jobs."""

    def test_different_slurm_jobs_get_different_marker_paths(self, monkeypatch, tmp_path):
        """Different SLURM_JOB_IDs should result in different marker paths."""
        monkeypatch.setenv("NEMO_RUN_DIR", str(tmp_path))
        artifacts = {"data": "TestArtifact:latest"}

        monkeypatch.setenv("SLURM_JOB_ID", "job_1")
        path1 = resolvers._get_marker_path(artifacts)

        monkeypatch.setenv("SLURM_JOB_ID", "job_2")
        path2 = resolvers._get_marker_path(artifacts)

        assert path1 != path2
        assert "job_1" in str(path1)
        assert "job_2" in str(path2)

    def test_same_job_same_artifacts_get_same_marker_path(self, monkeypatch, tmp_path):
        """Same job ID and artifacts should produce the same marker path."""
        monkeypatch.setenv("NEMO_RUN_DIR", str(tmp_path))
        monkeypatch.setenv("SLURM_JOB_ID", "job_1")
        artifacts = {"data": "TestArtifact:latest"}

        path1 = resolvers._get_marker_path(artifacts)
        path2 = resolvers._get_marker_path(artifacts)

        assert path1 == path2

    def test_different_artifacts_get_different_marker_paths(self, monkeypatch, tmp_path):
        """Different artifact configs should produce different marker paths."""
        monkeypatch.setenv("NEMO_RUN_DIR", str(tmp_path))
        monkeypatch.setenv("SLURM_JOB_ID", "job_1")

        path1 = resolvers._get_marker_path({"data": "ArtifactA:latest"})
        path2 = resolvers._get_marker_path({"data": "ArtifactB:latest"})

        assert path1 != path2


def test_register_resolvers_from_config_pre_init(monkeypatch, tmp_path):
    resolvers.clear_artifact_cache()

    monkeypatch.setenv("WANDB_ENTITY", "ent")
    monkeypatch.setenv("WANDB_PROJECT", "proj")

    downloaded_dir = tmp_path / "artifact"
    downloaded_dir.mkdir()

    class FakeArtifact:
        def __init__(self, ref: str):
            self.qualified_name = ref
            self.version = "v5"
            self.name = "DataBlendsArtifact-pretrain"
            self.type = "dataset"

        def download(self, skip_cache: bool = True):
            return str(downloaded_dir)

    class FakeApi:
        def __init__(self):
            self.last_ref = None

        def artifact(self, ref: str):
            self.last_ref = ref
            return FakeArtifact(ref)

    fake_api = FakeApi()

    class FakeWandb(types.SimpleNamespace):
        def Api(self):  # noqa: N802
            return fake_api

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb())

    cfg = OmegaConf.create(
        {
            "run": {"data": "DataBlendsArtifact-pretrain:v5"},
            "recipe": {"per_split_data_args_path": "${art:data,path}"},
        }
    )

    qualified_names = resolvers.register_resolvers_from_config(cfg, mode="pre_init")
    assert qualified_names == ["ent/proj/DataBlendsArtifact-pretrain:v5"]
    assert fake_api.last_ref == "ent/proj/DataBlendsArtifact-pretrain:v5"

    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["recipe"]["per_split_data_args_path"] == str(downloaded_dir)
