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

"""Tests for nemo_runspec.execution module.

Tests executor creation for all supported backends:
- local, docker, slurm (existing)
- dgxcloud (run:ai), lepton (new)

Also tests Ray executor helpers and the cloud Ray backend patching.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import base64
import json
import re
from dataclasses import dataclass, field
from omegaconf import OmegaConf

from nemo_runspec.execution import (
    _derive_cloud_workspace,
    _get_env,
    _git_mount_commands,
    _to_plain,
    build_env_vars,
    create_executor,
    get_executor_type,
    get_startup_commands,
    prepend_startup_to_cmd,
    _parse_netrc,
    materialize_podman_auth_from_enroot,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(**kwargs) -> dict:
    """Create a plain dict env config for testing."""
    return kwargs


def _make_env_omegaconf(**kwargs):
    """Create an OmegaConf DictConfig env config for testing."""
    return OmegaConf.create(kwargs)


class _FakePackager:
    """Minimal packager stub for executor creation tests."""
    pass


# ---------------------------------------------------------------------------
# _get_env tests
# ---------------------------------------------------------------------------


class TestGetEnv:
    def test_dict_config(self):
        env = {"key": "value"}
        assert _get_env(env, "key") == "value"
        assert _get_env(env, "missing", "default") == "default"

    def test_omegaconf_config(self):
        env = OmegaConf.create({"key": "value"})
        assert _get_env(env, "key") == "value"
        assert _get_env(env, "missing", "default") == "default"

    def test_none_config(self):
        assert _get_env(None, "key", "default") == "default"


# ---------------------------------------------------------------------------
# Startup commands tests
# ---------------------------------------------------------------------------


class TestStartupCommands:
    def test_empty_env(self):
        assert get_startup_commands(None) == []
        assert get_startup_commands({}) == []

    def test_valid_commands(self):
        env = {"startup_commands": ["echo hello", "pip install foo"]}
        assert get_startup_commands(env) == ["echo hello", "pip install foo"]

    def test_prepend_startup_to_cmd(self):
        result = prepend_startup_to_cmd(["cmd1", "cmd2"], "main_cmd")
        assert "cmd1" in result
        assert "cmd2" in result
        assert "main_cmd" in result

    def test_prepend_empty(self):
        assert prepend_startup_to_cmd([], "main_cmd") == "main_cmd"


# ---------------------------------------------------------------------------
# create_executor tests — Local
# ---------------------------------------------------------------------------


class TestCreateExecutorLocal:
    def test_local_executor(self):
        env = _make_env(executor="local")
        executor = create_executor(
            env=env,
            env_vars={"KEY": "VALUE"},
            packager=_FakePackager(),
        )
        import nemo_run as run

        assert isinstance(executor, run.LocalExecutor)

    def test_local_executor_default(self):
        """When no executor type specified, defaults to local."""
        executor = create_executor(
            env={},
            env_vars={},
            packager=_FakePackager(),
        )
        import nemo_run as run

        assert isinstance(executor, run.LocalExecutor)


# ---------------------------------------------------------------------------
# create_executor tests — Docker
# ---------------------------------------------------------------------------


class TestCreateExecutorDocker:
    def test_docker_executor(self):
        env = _make_env(
            executor="docker",
            container_image="nvcr.io/nvidia/nemo:latest",
            gpus_per_node=8,
        )
        executor = create_executor(
            env=env,
            env_vars={"KEY": "VALUE"},
            packager=_FakePackager(),
        )
        import nemo_run as run

        assert isinstance(executor, run.DockerExecutor)
        assert executor.container_image == "nvcr.io/nvidia/nemo:latest"

    def test_docker_executor_requires_image(self):
        env = _make_env(executor="docker")
        with pytest.raises(ValueError, match="container_image required"):
            create_executor(env=env, env_vars={}, packager=_FakePackager())

    def test_docker_executor_fallback_image(self):
        env = _make_env(executor="docker")
        executor = create_executor(
            env=env,
            env_vars={},
            packager=_FakePackager(),
            default_image="nvcr.io/fallback:latest",
        )
        import nemo_run as run

        assert isinstance(executor, run.DockerExecutor)
        assert executor.container_image == "nvcr.io/fallback:latest"


# ---------------------------------------------------------------------------
# create_executor tests — DGXCloud
# ---------------------------------------------------------------------------


class TestCreateExecutorDGXCloud:
    @pytest.fixture
    def dgxcloud_env(self):
        return _make_env(
            executor="dgxcloud",
            base_url="https://dgx.example.com/api/v1",
            kube_apiserver_url="https://dgx.example.com/k8s",
            client_id="test-client-id",
            client_secret="test-client-secret",
            project_name="test-project",
            container_image="nvcr.io/nvidia/nemo:latest",
            pvc_nemo_run_dir="/pvc/nemo_run",
            nodes=2,
            gpus_per_node=8,
            nprocs_per_node=8,
        )

    def test_dgxcloud_executor_creation(self, dgxcloud_env):
        executor = create_executor(
            env=dgxcloud_env,
            env_vars={"HF_TOKEN": "test"},
            packager=_FakePackager(),
        )
        import nemo_run as run

        assert isinstance(executor, run.DGXCloudExecutor)
        assert executor.base_url == "https://dgx.example.com/api/v1"
        assert executor.client_id == "test-client-id"
        assert executor.client_secret == "test-client-secret"
        assert executor.project_name == "test-project"
        assert executor.container_image == "nvcr.io/nvidia/nemo:latest"
        assert executor.pvc_nemo_run_dir == "/pvc/nemo_run"
        assert executor.nodes == 2
        assert executor.gpus_per_node == 8

    def test_dgxcloud_with_legacy_app_id_fields(self):
        """Backward compat: legacy app_id/app_secret names still map to client_id."""
        env = _make_env(
            executor="dgxcloud",
            base_url="https://dgx.example.com/api/v1",
            kube_apiserver_url="https://dgx.example.com/k8s",
            app_id="legacy-app-id",
            app_secret="legacy-app-secret",
            project_name="test-project",
            container_image="nvcr.io/nvidia/nemo:latest",
            pvc_nemo_run_dir="/pvc/nemo_run",
        )
        executor = create_executor(env=env, env_vars={}, packager=_FakePackager())
        import nemo_run as run

        assert isinstance(executor, run.DGXCloudExecutor)
        assert executor.client_id == "legacy-app-id"

    def test_dgxcloud_requires_base_url(self):
        env = _make_env(
            executor="dgxcloud",
            kube_apiserver_url="https://dgx.example.com/k8s",
            client_id="id",
            client_secret="secret",
            project_name="proj",
            container_image="img",
            pvc_nemo_run_dir="/pvc",
        )
        with pytest.raises(ValueError, match="base_url required"):
            create_executor(env=env, env_vars={}, packager=_FakePackager())

    def test_dgxcloud_requires_credentials(self):
        env = _make_env(
            executor="dgxcloud",
            base_url="https://dgx.example.com",
            kube_apiserver_url="https://dgx.example.com/k8s",
            project_name="proj",
            container_image="img",
            pvc_nemo_run_dir="/pvc",
        )
        with pytest.raises(ValueError, match="client_id/client_secret"):
            create_executor(env=env, env_vars={}, packager=_FakePackager())

    def test_dgxcloud_requires_project_name(self):
        env = _make_env(
            executor="dgxcloud",
            base_url="https://dgx.example.com",
            kube_apiserver_url="https://dgx.example.com/k8s",
            client_id="id",
            client_secret="secret",
            container_image="img",
            pvc_nemo_run_dir="/pvc",
        )
        with pytest.raises(ValueError, match="project_name required"):
            create_executor(env=env, env_vars={}, packager=_FakePackager())

    def test_dgxcloud_requires_pvc_dir(self):
        env = _make_env(
            executor="dgxcloud",
            base_url="https://dgx.example.com",
            kube_apiserver_url="https://dgx.example.com/k8s",
            client_id="id",
            client_secret="secret",
            project_name="proj",
            container_image="img",
        )
        with pytest.raises(ValueError, match="pvc_nemo_run_dir required"):
            create_executor(env=env, env_vars={}, packager=_FakePackager())

    def test_dgxcloud_requires_image(self):
        env = _make_env(
            executor="dgxcloud",
            base_url="https://dgx.example.com",
            kube_apiserver_url="https://dgx.example.com/k8s",
            client_id="id",
            client_secret="secret",
            project_name="proj",
            pvc_nemo_run_dir="/pvc",
        )
        with pytest.raises(ValueError, match="container_image required"):
            create_executor(env=env, env_vars={}, packager=_FakePackager())

    def test_dgxcloud_requires_kube_apiserver_url(self):
        env = _make_env(
            executor="dgxcloud",
            base_url="https://dgx.example.com",
            client_id="id",
            client_secret="secret",
            project_name="proj",
            container_image="img",
            pvc_nemo_run_dir="/pvc",
        )
        with pytest.raises(ValueError, match="kube_apiserver_url required"):
            create_executor(env=env, env_vars={}, packager=_FakePackager())

    def test_dgxcloud_with_pvcs(self, dgxcloud_env):
        dgxcloud_env["pvcs"] = [
            {"claimName": "my-pvc", "path": "/data", "readOnly": False}
        ]
        executor = create_executor(
            env=dgxcloud_env, env_vars={}, packager=_FakePackager()
        )
        assert executor.pvcs == [{"claimName": "my-pvc", "path": "/data", "readOnly": False}]

    def test_dgxcloud_with_custom_spec(self, dgxcloud_env):
        dgxcloud_env["custom_spec"] = {"schedulerName": "runai-scheduler"}
        executor = create_executor(
            env=dgxcloud_env, env_vars={}, packager=_FakePackager()
        )
        assert executor.custom_spec == {"schedulerName": "runai-scheduler"}

    def test_dgxcloud_with_omegaconf(self):
        env = _make_env_omegaconf(
            executor="dgxcloud",
            base_url="https://dgx.example.com/api/v1",
            kube_apiserver_url="https://dgx.example.com/k8s",
            client_id="test-id",
            client_secret="test-secret",
            project_name="proj",
            container_image="nvcr.io/nvidia/nemo:latest",
            pvc_nemo_run_dir="/pvc/nemo_run",
            nodes=4,
            gpus_per_node=8,
        )
        executor = create_executor(env=env, env_vars={}, packager=_FakePackager())
        import nemo_run as run

        assert isinstance(executor, run.DGXCloudExecutor)
        assert executor.nodes == 4


# ---------------------------------------------------------------------------
# create_executor tests — Lepton
# ---------------------------------------------------------------------------


class TestCreateExecutorLepton:
    @pytest.fixture
    def lepton_env(self):
        return _make_env(
            executor="lepton",
            container_image="nvcr.io/nvidia/nemo:latest",
            nemo_run_dir="/nemo_run",
            nodes=2,
            gpus_per_node=8,
            resource_shape="gpu.h100.8",
            node_group="my-node-group",
        )

    def test_lepton_executor_creation(self, lepton_env):
        executor = create_executor(
            env=lepton_env,
            env_vars={"HF_TOKEN": "test"},
            packager=_FakePackager(),
        )
        import nemo_run as run

        assert isinstance(executor, run.LeptonExecutor)
        assert executor.container_image == "nvcr.io/nvidia/nemo:latest"
        assert executor.nemo_run_dir == "/nemo_run"
        assert executor.nodes == 2
        assert executor.gpus_per_node == 8
        assert executor.resource_shape == "gpu.h100.8"
        assert executor.node_group == "my-node-group"

    def test_lepton_requires_nemo_run_dir(self):
        env = _make_env(
            executor="lepton",
            container_image="nvcr.io/nvidia/nemo:latest",
        )
        with pytest.raises(ValueError, match="nemo_run_dir required"):
            create_executor(env=env, env_vars={}, packager=_FakePackager())

    def test_lepton_requires_image(self):
        env = _make_env(
            executor="lepton",
            nemo_run_dir="/nemo_run",
        )
        with pytest.raises(ValueError, match="container_image required"):
            create_executor(env=env, env_vars={}, packager=_FakePackager())

    def test_lepton_with_mounts(self, lepton_env):
        lepton_env["mounts"] = [
            {"path": "/data", "mount_path": "/data"},
        ]
        executor = create_executor(
            env=lepton_env, env_vars={}, packager=_FakePackager()
        )
        assert executor.mounts == [{"path": "/data", "mount_path": "/data"}]

    def test_lepton_with_shared_memory(self, lepton_env):
        lepton_env["shared_memory_size"] = 131072
        executor = create_executor(
            env=lepton_env, env_vars={}, packager=_FakePackager()
        )
        assert executor.shared_memory_size == 131072

    def test_lepton_with_pre_launch_commands(self, lepton_env):
        lepton_env["pre_launch_commands"] = ["pip install foo", "echo ready"]
        executor = create_executor(
            env=lepton_env, env_vars={}, packager=_FakePackager()
        )
        assert executor.pre_launch_commands == ["pip install foo", "echo ready"]

    def test_lepton_with_node_reservation(self, lepton_env):
        lepton_env["node_reservation"] = "reserved-pool-123"
        executor = create_executor(
            env=lepton_env, env_vars={}, packager=_FakePackager()
        )
        assert executor.node_reservation == "reserved-pool-123"

    def test_lepton_with_image_pull_secrets(self, lepton_env):
        lepton_env["image_pull_secrets"] = ["nvcr-secret"]
        executor = create_executor(
            env=lepton_env, env_vars={}, packager=_FakePackager()
        )
        assert executor.image_pull_secrets == ["nvcr-secret"]

    def test_lepton_with_omegaconf(self):
        env = _make_env_omegaconf(
            executor="lepton",
            container_image="nvcr.io/nvidia/nemo:latest",
            nemo_run_dir="/nemo_run",
            nodes=4,
            gpus_per_node=8,
            resource_shape="gpu.h100.8",
        )
        executor = create_executor(env=env, env_vars={}, packager=_FakePackager())
        import nemo_run as run

        assert isinstance(executor, run.LeptonExecutor)
        assert executor.nodes == 4

    def test_lepton_default_image_fallback(self):
        env = _make_env(
            executor="lepton",
            nemo_run_dir="/nemo_run",
        )
        executor = create_executor(
            env=env,
            env_vars={},
            packager=_FakePackager(),
            default_image="nvcr.io/fallback:latest",
        )
        import nemo_run as run

        assert isinstance(executor, run.LeptonExecutor)
        assert executor.container_image == "nvcr.io/fallback:latest"


# ---------------------------------------------------------------------------
# create_executor tests — Unknown type
# ---------------------------------------------------------------------------


class TestCreateExecutorUnknown:
    def test_unknown_executor_type(self):
        env = _make_env(executor="kubernetes")
        with pytest.raises(ValueError, match="Unknown executor type"):
            create_executor(env=env, env_vars={}, packager=_FakePackager())


# ---------------------------------------------------------------------------
# create_executor tests — script_resources defaults
# ---------------------------------------------------------------------------


class TestCreateExecutorScriptResources:
    def test_dgxcloud_uses_script_resources(self):
        """When env doesn't specify nodes/gpus, uses script_resources defaults."""

        class FakeResources:
            nodes = 4
            gpus_per_node = 8

        env = _make_env(
            executor="dgxcloud",
            base_url="https://dgx.example.com",
            kube_apiserver_url="https://dgx.example.com/k8s",
            client_id="id",
            client_secret="secret",
            project_name="proj",
            container_image="img",
            pvc_nemo_run_dir="/pvc",
        )
        executor = create_executor(
            env=env,
            env_vars={},
            packager=_FakePackager(),
            script_resources=FakeResources(),
        )
        assert executor.nodes == 4
        assert executor.gpus_per_node == 8

    def test_lepton_uses_script_resources(self):
        class FakeResources:
            nodes = 2
            gpus_per_node = 4

        env = _make_env(
            executor="lepton",
            container_image="img",
            nemo_run_dir="/nemo_run",
        )
        executor = create_executor(
            env=env,
            env_vars={},
            packager=_FakePackager(),
            script_resources=FakeResources(),
        )
        assert executor.nodes == 2
        assert executor.gpus_per_node == 4


# ---------------------------------------------------------------------------
# build_env_vars tests (sanity check existing functionality)
# ---------------------------------------------------------------------------


class TestBuildEnvVars:
    def test_remote_job_dir(self):
        job_config = OmegaConf.create({"run": {"wandb": {}}})
        env_config = {"remote_job_dir": "/lustre/jobs"}
        env_vars = build_env_vars(job_config, env_config)
        assert env_vars.get("NEMO_RUN_DIR") == "/lustre/jobs"
        assert env_vars.get("HF_HOME") == "/lustre/jobs/hf"

    def test_explicit_env_vars_override(self):
        job_config = OmegaConf.create({"run": {"wandb": {}}})
        env_config = {
            "remote_job_dir": "/lustre/jobs",
            "env_vars": {"NEMO_RUN_DIR": "/custom/path"},
        }
        env_vars = build_env_vars(job_config, env_config)
        # Explicit env_vars should override auto-detected ones
        assert env_vars["NEMO_RUN_DIR"] == "/custom/path"


# ---------------------------------------------------------------------------
# get_executor_type tests — the helper commands import
# ---------------------------------------------------------------------------


class TestGetExecutorType:
    def test_default_slurm(self):
        assert get_executor_type(None) == "slurm"
        assert get_executor_type({}) == "slurm"

    def test_custom_default(self):
        assert get_executor_type(None, default="local") == "local"

    def test_explicit_dict(self):
        assert get_executor_type({"executor": "lepton"}) == "lepton"
        assert get_executor_type({"executor": "dgxcloud"}) == "dgxcloud"

    def test_explicit_omegaconf(self):
        env = OmegaConf.create({"executor": "lepton"})
        assert get_executor_type(env) == "lepton"


# ---------------------------------------------------------------------------
# _to_plain tests — OmegaConf → plain Python conversion
# ---------------------------------------------------------------------------


class TestToPlain:
    def test_passes_through_primitives(self):
        assert _to_plain(None) is None
        assert _to_plain(42) == 42
        assert _to_plain("s") == "s"

    def test_converts_omegaconf_dict(self):
        cfg = OmegaConf.create({"a": 1, "b": {"c": 2}})
        plain = _to_plain(cfg)
        assert isinstance(plain, dict) and not OmegaConf.is_config(plain)
        assert plain == {"a": 1, "b": {"c": 2}}

    def test_converts_omegaconf_list(self):
        cfg = OmegaConf.create([{"x": 1}, {"x": 2}])
        plain = _to_plain(cfg)
        assert plain == [{"x": 1}, {"x": 2}]
        assert all(isinstance(item, dict) for item in plain)

    def test_recurses_into_plain_dicts_with_nested_omegaconf(self):
        nested = OmegaConf.create({"k": "v"})
        result = _to_plain({"outer": nested, "other": [nested]})
        assert result == {"outer": {"k": "v"}, "other": [{"k": "v"}]}
        assert not OmegaConf.is_config(result["outer"])

    def test_converts_tuples_to_lists(self):
        assert _to_plain((1, 2, 3)) == [1, 2, 3]


# ---------------------------------------------------------------------------
# _derive_cloud_workspace tests — workspace selection policy
# ---------------------------------------------------------------------------


class TestDeriveCloudWorkspace:
    def test_explicit_workspace_wins(self):
        env = {"executor": "lepton", "workspace": "/mnt/explicit"}
        assert _derive_cloud_workspace(env) == "/mnt/explicit"

    def test_lepton_uses_first_mount_path(self):
        env = {
            "executor": "lepton",
            "mounts": [{"path": "/src", "mount_path": "/mnt/lustre-shared"}],
        }
        assert _derive_cloud_workspace(env) == "/mnt/lustre-shared"

    def test_lepton_skips_auto_mount_strings(self):
        """auto_mount sentinels aren't dicts — must be skipped."""
        env = {
            "executor": "lepton",
            "mounts": [
                "__auto_mount__:git+https://example.com/repo@main",
                {"path": "/src", "mount_path": "/mnt/shared"},
            ],
        }
        assert _derive_cloud_workspace(env) == "/mnt/shared"

    def test_dgxcloud_uses_first_pvc_path(self):
        env = {
            "executor": "dgxcloud",
            "pvcs": [{"claimName": "my-pvc", "path": "/pvc/data"}],
        }
        assert _derive_cloud_workspace(env) == "/pvc/data"

    def test_fallback_to_tmp_when_no_mounts(self, caplog):
        import logging

        env = {"executor": "lepton"}
        with caplog.at_level(logging.WARNING, logger="nemo_runspec.execution"):
            assert _derive_cloud_workspace(env) == "/tmp"
        # A warning must be emitted so users know /tmp is ephemeral
        assert any("ephemeral" in rec.message for rec in caplog.records)

    def test_omegaconf_env_supported(self):
        env = OmegaConf.create({
            "executor": "lepton",
            "mounts": [{"path": "/src", "mount_path": "/mnt/omega"}],
        })
        assert _derive_cloud_workspace(env) == "/mnt/omega"


# ---------------------------------------------------------------------------
# _git_mount_commands tests — auto_mount → shell git clones for cloud
# ---------------------------------------------------------------------------


class TestGitMountCommands:
    """Tests for the cloud-executor git auto_mount translation.

    The resolver registers repos as side-effects of OmegaConf resolution,
    so we patch ``get_git_mounts`` directly to keep tests hermetic.
    """

    def test_no_mounts_returns_empty(self):
        with patch("nemo_runspec.config.resolvers.get_git_mounts", return_value={}):
            assert _git_mount_commands() == []

    def test_skips_entries_without_target(self):
        mounts = {"repo1": {"url": "https://example.com/r.git", "ref": "main", "target": ""}}
        with patch("nemo_runspec.config.resolvers.get_git_mounts", return_value=mounts):
            assert _git_mount_commands() == []

    def test_branch_ref_uses_clone_depth_one(self):
        mounts = {
            "megatron": {
                "url": "https://github.com/NVIDIA/Megatron-LM.git",
                "ref": "main",
                "target": "/opt/megatron-lm",
            }
        }
        with patch("nemo_runspec.config.resolvers.get_git_mounts", return_value=mounts):
            commands = _git_mount_commands()
        assert len(commands) == 1
        assert "git clone --depth 1 -b main" in commands[0]
        assert commands[0].startswith("rm -rf /opt/megatron-lm")
        assert "/opt/megatron-lm" in commands[0]

    def test_commit_sha_uses_fetch_fallback(self):
        """40-char hex ref = commit SHA; needs init+fetch (clone -b won't work on SHAs)."""
        sha = "a" * 40
        mounts = {
            "bridge": {
                "url": "https://github.com/NVIDIA/Megatron-Bridge.git",
                "ref": sha,
                "target": "/opt/bridge",
            }
        }
        with patch("nemo_runspec.config.resolvers.get_git_mounts", return_value=mounts):
            commands = _git_mount_commands()
        assert len(commands) == 1
        cmd = commands[0]
        assert "git init /tmp/_git_bridge" in cmd
        assert f"fetch --depth 1 origin {sha}" in cmd
        assert "checkout FETCH_HEAD" in cmd
        # Must have a fallback so container built-in survives if SHA can't be fetched
        assert "|| echo" in cmd and "container built-in" in cmd

    def test_sha_detection_case_insensitive(self):
        """SHAs in upper-case hex should also be detected."""
        sha = "ABCDEF0123456789" + "0" * 24  # 40 hex chars
        mounts = {
            "r": {"url": "u", "ref": sha, "target": "/opt/r"},
        }
        with patch("nemo_runspec.config.resolvers.get_git_mounts", return_value=mounts):
            commands = _git_mount_commands()
        # Case-insensitive SHA detection → should use fetch path, not clone -b
        assert "git init" in commands[0]

    def test_non_sha_hex_like_ref_treated_as_branch(self):
        """Short hex refs (not 40 chars) are treated as branch/tag names."""
        mounts = {"r": {"url": "u", "ref": "abc123", "target": "/opt/r"}}
        with patch("nemo_runspec.config.resolvers.get_git_mounts", return_value=mounts):
            commands = _git_mount_commands()
        assert "clone --depth 1 -b abc123" in commands[0]

    def test_multiple_mounts_produces_independent_commands(self):
        mounts = {
            "a": {"url": "ua", "ref": "main", "target": "/opt/a"},
            "b": {"url": "ub", "ref": "develop", "target": "/opt/b"},
        }
        with patch("nemo_runspec.config.resolvers.get_git_mounts", return_value=mounts):
            commands = _git_mount_commands()
        assert len(commands) == 2
        assert any("/opt/a" in c and "-b main" in c for c in commands)
        assert any("/opt/b" in c and "-b develop" in c for c in commands)


# ---------------------------------------------------------------------------
# Lepton executor: mounts and auto_mount filtering
# ---------------------------------------------------------------------------


class TestLeptonAutoMountFiltering:
    def test_auto_mount_string_filtered_out(self):
        """`__auto_mount__:...` sentinels are Slurm-only; must not reach Lepton."""
        env = _make_env(
            executor="lepton",
            container_image="img",
            nemo_run_dir="/nemo_run",
            mounts=[
                "__auto_mount__:git+https://example.com/repo@main",
                {"path": "/data", "mount_path": "/data"},
            ],
        )
        executor = create_executor(env=env, env_vars={}, packager=_FakePackager())
        assert executor.mounts == [{"path": "/data", "mount_path": "/data"}]

    def test_pre_launch_includes_auto_mount_commands(self):
        """Auto_mount registered repos become pre_launch git clones."""
        env = _make_env(
            executor="lepton",
            container_image="img",
            nemo_run_dir="/nemo_run",
            pre_launch_commands=["echo hello"],
        )
        mounts = {"r": {"url": "u", "ref": "main", "target": "/opt/r"}}
        with patch("nemo_runspec.config.resolvers.get_git_mounts", return_value=mounts):
            executor = create_executor(env=env, env_vars={}, packager=_FakePackager())
        # user commands preserved + git clone appended
        assert "echo hello" in executor.pre_launch_commands
        assert any("clone" in c and "/opt/r" in c for c in executor.pre_launch_commands)


# ---------------------------------------------------------------------------
# DGXCloud: _to_plain applied to pvcs / custom_spec
# ---------------------------------------------------------------------------


class TestDGXCloudSerialization:
    def test_omegaconf_pvcs_are_plain_dicts(self):
        """fiddle can't serialize DictConfig; _to_plain must convert."""
        env = _make_env_omegaconf(
            executor="dgxcloud",
            base_url="https://dgx.example.com",
            kube_apiserver_url="https://dgx.example.com/k8s",
            client_id="id",
            client_secret="secret",
            project_name="proj",
            container_image="img",
            pvc_nemo_run_dir="/pvc",
            pvcs=[{"claimName": "my-pvc", "path": "/data", "readOnly": False}],
            custom_spec={"schedulerName": "runai"},
        )
        executor = create_executor(env=env, env_vars={}, packager=_FakePackager())
        # Must be plain dicts, not OmegaConf containers
        assert isinstance(executor.pvcs, list)
        assert isinstance(executor.pvcs[0], dict)
        assert not OmegaConf.is_config(executor.pvcs[0])
        assert isinstance(executor.custom_spec, dict)
        assert not OmegaConf.is_config(executor.custom_spec)


@dataclass
class _Result:
    ok: bool = True
    stdout: str = ""
    stderr: str = ""


@dataclass
class _ScriptedTunnel:
    """Fake tunnel that returns scripted file contents and records writes.

    ``files`` maps an enroot-credentials path to its contents (or ``None``
    if it should be reported missing). The tunnel intercepts the
    ``test -f X && cat X || true`` read pattern emitted by
    ``materialize_podman_auth_from_enroot`` and returns the corresponding
    content, and intercepts the ``printf %s <b64> | base64 -d > X`` write
    pattern to capture decoded contents in ``writes``.
    """

    files: dict[str, str | None]
    commands: list[str] = field(default_factory=list)
    writes: dict[str, str] = field(default_factory=dict)

    def run(self, cmd: str, hide: bool = False, warn: bool = False) -> _Result:
        self.commands.append(cmd)
        cat_match = re.search(r'test -f "([^"]+)" && cat "[^"]+" \|\| true', cmd)
        if cat_match:
            path = cat_match.group(1)
            content = self.files.get(path)
            if content is None:
                # Match real shell behaviour: with `|| true` the command
                # succeeds with empty stdout.
                return _Result(ok=True, stdout="")
            return _Result(ok=True, stdout=content)

        # ``shlex.quote`` only adds single quotes when needed (e.g. paths
        # with shell metachars), so accept both quoted and unquoted forms.
        write_match = re.search(
            r"printf %s (\S+) \| base64 -d > (\S+)",
            cmd,
        )
        if write_match:
            encoded = _unwrap_quotes(write_match.group(1))
            path = _unwrap_quotes(write_match.group(2))
            self.writes[path] = base64.b64decode(encoded).decode()

        return _Result(ok=True)


def _unwrap_quotes(s: str) -> str:
    """Strip a single matching pair of single or double quotes."""
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


# ---------------------------------------------------------------------------
# _parse_netrc
# ---------------------------------------------------------------------------


class TestParseNetrc:
    def test_single_machine(self):
        creds = _parse_netrc("machine nvcr.io login $oauthtoken password nvapi-secret")
        assert creds == {"nvcr.io": ("$oauthtoken", "nvapi-secret")}

    def test_multiple_machines_on_separate_lines(self):
        content = (
            "machine gitlab.example.com login alice password gl-token\n"
            "machine nvcr.io login $oauthtoken password nvapi-secret\n"
        )
        assert _parse_netrc(content) == {
            "gitlab.example.com": ("alice", "gl-token"),
            "nvcr.io": ("$oauthtoken", "nvapi-secret"),
        }

    def test_default_block_is_skipped(self):
        # ``default`` is netrc's catch-all entry; we don't surface it
        # because podman auth.json is keyed per-registry.
        content = (
            "machine nvcr.io login $oauthtoken password nvapi-secret\n"
            "default login fallback password fbk-token\n"
        )
        assert _parse_netrc(content) == {
            "nvcr.io": ("$oauthtoken", "nvapi-secret"),
        }

    def test_empty_string_returns_empty_dict(self):
        assert _parse_netrc("") == {}

    def test_ignores_extra_whitespace(self):
        content = "  machine\tnvcr.io  login   $oauthtoken\n  password   secret\n"
        assert _parse_netrc(content) == {"nvcr.io": ("$oauthtoken", "secret")}


# ---------------------------------------------------------------------------
# materialize_podman_auth_from_enroot
# ---------------------------------------------------------------------------


class TestMaterializePodmanAuth:
    NETRC = (
        "machine gitlab.example.com login alice password gl-token\n"
        "machine nvcr.io login $oauthtoken password nvapi-secret\n"
    )
    DEFAULT_PATH = "$HOME/.config/enroot/.credentials"

    def test_returns_none_when_credentials_missing(self):
        tunnel = _ScriptedTunnel(files={self.DEFAULT_PATH: None})
        assert materialize_podman_auth_from_enroot(tunnel, "/lustre/cache/.auth") is None
        assert tunnel.writes == {}

    def test_returns_none_when_no_target_registry_present(self):
        tunnel = _ScriptedTunnel(
            files={self.DEFAULT_PATH: "machine gitlab.example.com login a password b\n"},
        )
        assert (
            materialize_podman_auth_from_enroot(tunnel, "/lustre/cache/.auth") is None
        )
        assert tunnel.writes == {}

    def test_writes_auth_json_for_default_registry(self):
        tunnel = _ScriptedTunnel(files={self.DEFAULT_PATH: self.NETRC})

        path = materialize_podman_auth_from_enroot(tunnel, "/lustre/cache/.auth")

        assert path == "/lustre/cache/.auth/auth.json"
        assert path in tunnel.writes
        payload = json.loads(tunnel.writes[path])
        # Only nvcr.io should appear by default — other entries (gitlab)
        # must not leak into the build container.
        assert set(payload["auths"].keys()) == {"nvcr.io"}
        decoded = base64.b64decode(payload["auths"]["nvcr.io"]["auth"]).decode()
        assert decoded == "$oauthtoken:nvapi-secret"

    def test_respects_explicit_registry_allowlist(self):
        tunnel = _ScriptedTunnel(files={self.DEFAULT_PATH: self.NETRC})

        path = materialize_podman_auth_from_enroot(
            tunnel,
            "/lustre/cache/.auth",
            registries=("nvcr.io", "gitlab.example.com"),
        )

        assert path == "/lustre/cache/.auth/auth.json"
        payload = json.loads(tunnel.writes[path])
        assert set(payload["auths"].keys()) == {"nvcr.io", "gitlab.example.com"}

    def test_registry_match_is_case_insensitive(self):
        netrc = "machine NVCR.IO login $oauthtoken password nvapi-secret\n"
        tunnel = _ScriptedTunnel(files={self.DEFAULT_PATH: netrc})

        path = materialize_podman_auth_from_enroot(tunnel, "/lustre/cache/.auth")

        # Case-folding is applied to the match so the configured allowlist
        # tolerates the netrc entry's casing.
        payload = json.loads(tunnel.writes[path])
        assert set(payload["auths"].keys()) == {"NVCR.IO"}

    def test_writes_with_mode_0600(self):
        tunnel = _ScriptedTunnel(files={self.DEFAULT_PATH: self.NETRC})
        materialize_podman_auth_from_enroot(tunnel, "/lustre/cache/.auth")
        # The chmod is part of the same shell command as the write; assert
        # at least one issued command sets 0600 on the auth file.
        assert any(
            "chmod 600" in c and "/lustre/cache/.auth/auth.json" in c
            for c in tunnel.commands
        ), tunnel.commands

    def test_creates_out_dir(self):
        tunnel = _ScriptedTunnel(files={self.DEFAULT_PATH: self.NETRC})
        materialize_podman_auth_from_enroot(tunnel, "/lustre/cache/.auth")
        assert any(
            "mkdir -p" in c and "/lustre/cache/.auth" in c for c in tunnel.commands
        ), tunnel.commands

    def test_custom_credentials_path(self):
        custom = "/etc/enroot/credentials"
        tunnel = _ScriptedTunnel(files={custom: self.NETRC})
        path = materialize_podman_auth_from_enroot(
            tunnel,
            "/lustre/cache/.auth",
            enroot_credentials_path=custom,
        )
        assert path == "/lustre/cache/.auth/auth.json"
