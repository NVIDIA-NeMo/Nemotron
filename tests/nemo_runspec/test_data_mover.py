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

"""Tests for :mod:`nemo_runspec.data_mover`."""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

import pytest

from nemo_runspec.data_mover import Plan, SourcePackager, _auto_includes, plan_for

# ── _auto_includes ───────────────────────────────────────────────────────────


def _write_fake_repo(root: Path) -> None:
    """Build a synthetic repo layout mirroring this project."""
    src = root / "src"
    # nemo_runspec package (no recipes/)
    (src / "nemo_runspec").mkdir(parents=True)
    (src / "nemo_runspec" / "__init__.py").write_text("")
    (src / "nemo_runspec" / "foo.py").write_text("x=1")
    # nemotron package with two recipe families
    (src / "nemotron").mkdir()
    (src / "nemotron" / "__init__.py").write_text("")
    (src / "nemotron" / "kit").mkdir()
    (src / "nemotron" / "kit" / "__init__.py").write_text("")
    recipes = src / "nemotron" / "recipes"
    recipes.mkdir()
    (recipes / "__init__.py").write_text("")
    (recipes / "nano3").mkdir()
    (recipes / "nano3" / "__init__.py").write_text("")
    (recipes / "super3").mkdir()
    (recipes / "super3" / "__init__.py").write_text("")
    # Noise that must be filtered from the packager
    (src / "nemotron" / "__pycache__").mkdir()
    (src / "nemotron" / "__pycache__" / "x.pyc").write_text("bytecode")


def test_auto_includes_scopes_to_active_recipe_family(tmp_path):
    _write_fake_repo(tmp_path)
    includes = _auto_includes(tmp_path, script_path="src/nemotron/recipes/nano3/stage1_sft/data_prep.py")
    # Active family included, inactive one omitted.
    assert "src/nemotron/recipes/nano3" in includes
    assert "src/nemotron/recipes/super3" not in includes
    # Sibling package always shipped.
    assert "src/nemo_runspec" in includes
    # Non-recipes children of nemotron shipped.
    assert "src/nemotron/kit" in includes


def test_auto_includes_ships_all_families_without_hint(tmp_path):
    _write_fake_repo(tmp_path)
    includes = _auto_includes(tmp_path, script_path=None)
    # No family hint → ship both.
    assert "src/nemotron/recipes/nano3" in includes
    assert "src/nemotron/recipes/super3" in includes


def test_auto_includes_scopes_to_active_step_subtree(tmp_path):
    _write_fake_repo(tmp_path)
    steps = tmp_path / "src" / "nemotron" / "steps"
    (steps / "_runners").mkdir(parents=True)
    (steps / "_runners" / "__init__.py").write_text("")
    (steps / "index.py").write_text("")
    (steps / "sft" / "automodel").mkdir(parents=True)
    (steps / "sft" / "__init__.py").write_text("")
    (steps / "sft" / "automodel" / "step.py").write_text("")
    (steps / "prep").mkdir()
    (steps / "prep" / "__init__.py").write_text("")
    (steps / "prep" / "_common.py").write_text("")
    (steps / "prep" / "sft_packing").mkdir()
    (steps / "prep" / "sft_packing" / "step.py").write_text("")
    (steps / "rl" / "nemo_rl").mkdir(parents=True)
    (steps / "rl" / "nemo_rl" / "step.py").write_text("")

    includes = _auto_includes(tmp_path, script_path="src/nemotron/steps/sft/automodel/step.py")

    assert "src/nemotron/steps/index.py" in includes
    assert "src/nemotron/steps/sft/__init__.py" in includes
    assert "src/nemotron/steps/sft/automodel" in includes
    assert "src/nemotron/steps/_runners" in includes
    assert "src/nemotron/steps/rl" not in includes
    assert "src/nemotron/recipes/nano3" not in includes


def test_auto_includes_ships_active_step_ancestor_helpers(tmp_path):
    _write_fake_repo(tmp_path)
    steps = tmp_path / "src" / "nemotron" / "steps"
    (steps / "prep").mkdir(parents=True)
    (steps / "prep" / "__init__.py").write_text("")
    (steps / "prep" / "_common.py").write_text("")
    (steps / "prep" / "sft_packing").mkdir()
    (steps / "prep" / "sft_packing" / "step.py").write_text("")
    (steps / "sft" / "automodel").mkdir(parents=True)
    (steps / "sft" / "automodel" / "step.py").write_text("")

    includes = _auto_includes(tmp_path, script_path="src/nemotron/steps/prep/sft_packing/step.py")

    assert "src/nemotron/steps/prep/__init__.py" in includes
    assert "src/nemotron/steps/prep/_common.py" in includes
    assert "src/nemotron/steps/prep/sft_packing" in includes
    assert "src/nemotron/steps/sft" not in includes


def test_auto_includes_raises_when_src_missing(tmp_path):
    with pytest.raises(ValueError, match="No src/"):
        _auto_includes(tmp_path, script_path=None)


# ── SourcePackager ──────────────────────────────────────────────────────────


def test_source_packager_filters_pycache_and_pyc(tmp_path):
    _write_fake_repo(tmp_path)
    pkg = SourcePackager(
        repo_root=str(tmp_path),
        script_path="src/nemotron/recipes/nano3/x.py",
    )
    out = pkg.package(None, str(tmp_path), "test")
    assert out is not None and os.path.exists(out)
    with tarfile.open(out) as tf:
        names = tf.getnames()
    # Pyc + __pycache__ stripped.
    assert not any(n.endswith(".pyc") for n in names)
    assert not any("__pycache__" in n for n in names)
    # Real package files present.
    assert any(n.endswith("src/nemo_runspec/__init__.py") for n in names)


# ── plan_for ─────────────────────────────────────────────────────────────────


def test_plan_for_lepton_chunks_source_into_env_vars(tmp_path, monkeypatch):
    _write_fake_repo(tmp_path)
    # Skip the nemo-run patch side-effects — not relevant here.
    monkeypatch.setattr("nemo_runspec.run.patch_cloud_data_mover_skip_configs", lambda: None)
    monkeypatch.setattr(
        "nemo_runspec.run.patch_dgxcloud_strip_source_chunks_from_exports",
        lambda: None,
    )
    env_vars: dict[str, str] = {}
    plan = plan_for(
        executor_type="lepton",
        env_vars=env_vars,
        script_path="src/nemotron/recipes/nano3/x.py",
        pod_nemotron_home="/mnt/foo/_nemotron",
        repo_root=tmp_path,
    )
    assert isinstance(plan, Plan)
    assert plan.pod_src_root == "/mnt/foo/_nemotron/src"
    # Env vars populated with chunk count + at least one chunk.
    n = int(env_vars["_NEMOTRON_SRC_CHUNKS"])
    assert n >= 1
    assert "_NEMOTRON_SRC_CHUNK_0" in env_vars
    # NODE_RANK gate is present so multi-pod NFS runs don't race.
    script = plan.pre_script_cmds[0]
    assert "NODE_RANK" in script and "tar -xz" in script
    assert plan.source_ready_marker is not None
    assert plan.source_ready_marker in script
    assert not plan.needs_pwd_symlinks


def test_plan_for_cloud_ready_marker_is_unique_per_submission(tmp_path, monkeypatch):
    _write_fake_repo(tmp_path)
    monkeypatch.setattr("nemo_runspec.run.patch_cloud_data_mover_skip_configs", lambda: None)
    monkeypatch.setattr(
        "nemo_runspec.run.patch_dgxcloud_strip_source_chunks_from_exports",
        lambda: None,
    )

    env_a: dict[str, str] = {}
    env_b: dict[str, str] = {}
    plan_a = plan_for(
        executor_type="lepton",
        env_vars=env_a,
        script_path="src/nemotron/recipes/nano3/x.py",
        pod_nemotron_home="/mnt/foo/_nemotron",
        repo_root=tmp_path,
    )
    plan_b = plan_for(
        executor_type="lepton",
        env_vars=env_b,
        script_path="src/nemotron/recipes/nano3/x.py",
        pod_nemotron_home="/mnt/foo/_nemotron",
        repo_root=tmp_path,
    )

    assert env_a["_NEMOTRON_SRC_SHA256"] == env_b["_NEMOTRON_SRC_SHA256"]
    assert ".nemotron-src-ready-${_NEMOTRON_SRC_CHUNKS}" not in plan_a.pre_script_cmds[0]
    assert plan_a.source_ready_marker != plan_b.source_ready_marker
    assert plan_a.pre_script_cmds[0] != plan_b.pre_script_cmds[0]


def test_plan_for_dgxcloud_chunks_source_into_env_vars(tmp_path, monkeypatch):
    """DGXCloud uses the same env-var chunking as Lepton now — env vars travel
    via ``environmentVariables`` in the Job spec (bypassing the 10 KiB Args cap)
    and a companion patch strips them from the ``torchrun_job.sh`` exports so
    the launcher file stays small."""
    _write_fake_repo(tmp_path)
    monkeypatch.setattr("nemo_runspec.run.patch_cloud_data_mover_skip_configs", lambda: None)
    monkeypatch.setattr(
        "nemo_runspec.run.patch_dgxcloud_strip_source_chunks_from_exports",
        lambda: None,
    )
    env_vars: dict[str, str] = {}
    plan = plan_for(
        executor_type="dgxcloud",
        env_vars=env_vars,
        script_path="src/nemotron/recipes/nano3/x.py",
        pod_nemotron_home="/workspace/_nemotron",
        repo_root=tmp_path,
    )
    assert plan.pod_src_root == "/workspace/_nemotron/src"
    # Env vars populated, no file-based PVC path.
    assert int(env_vars["_NEMOTRON_SRC_CHUNKS"]) >= 1
    assert "_NEMOTRON_SRC_CHUNK_0" in env_vars
    assert not plan.needs_pwd_symlinks


def test_plan_for_fallback_uses_native_packager_path(tmp_path, monkeypatch):
    _write_fake_repo(tmp_path)
    monkeypatch.setattr("nemo_runspec.run.patch_cloud_data_mover_skip_configs", lambda: None)
    monkeypatch.setattr(
        "nemo_runspec.run.patch_dgxcloud_strip_source_chunks_from_exports",
        lambda: None,
    )
    plan = plan_for(
        executor_type="slurm",
        env_vars={},
        script_path=None,
        pod_nemotron_home="/unused",
        repo_root=tmp_path,
    )
    # Native extraction path; caller must symlink under $PWD/src.
    assert plan.pod_src_root == "/nemo_run/code/src"
    assert plan.needs_pwd_symlinks is True
    assert plan.pre_script_cmds == []
