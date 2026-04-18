# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    includes = _auto_includes(
        tmp_path, script_path="src/nemotron/recipes/nano3/stage1_sft/data_prep.py"
    )
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


def test_source_packager_fixed_output_returns_none(tmp_path):
    """DGXCloud flow: ``None`` return signals nemo-run to skip auto-extract."""
    _write_fake_repo(tmp_path)
    pkg = SourcePackager(
        repo_root=str(tmp_path),
        script_path=None,
        fixed_output_name="nemotron-src.tgz",
    )
    out = pkg.package(None, str(tmp_path), "unused-name")
    assert out is None
    # File still written to job_dir under the fixed name.
    assert (tmp_path / "nemotron-src.tgz").exists()


# ── plan_for ─────────────────────────────────────────────────────────────────


def test_plan_for_lepton_chunks_source_into_env_vars(tmp_path, monkeypatch):
    _write_fake_repo(tmp_path)
    # Skip the nemo-run data-mover patch side-effect — not relevant here.
    monkeypatch.setattr(
        "nemo_runspec.run.patch_cloud_data_mover_skip_configs", lambda: None
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
    assert not plan.needs_pwd_symlinks


def test_plan_for_dgxcloud_uses_fixed_filename(tmp_path, monkeypatch):
    _write_fake_repo(tmp_path)
    monkeypatch.setattr(
        "nemo_runspec.run.patch_cloud_data_mover_skip_configs", lambda: None
    )
    plan = plan_for(
        executor_type="dgxcloud",
        env_vars={},
        script_path="src/nemotron/recipes/nano3/x.py",
        pod_nemotron_home="/workspace/_nemotron",
        repo_root=tmp_path,
    )
    assert plan.packager.fixed_output_name == "nemotron-src.tgz"
    assert plan.pod_src_root == "/workspace/_nemotron/src"
    # Entrypoint extracts from /nemo_run/ (the pvc_job_dir symlink target).
    assert any("/nemo_run/nemotron-src.tgz" in c for c in plan.pre_script_cmds)
    assert not plan.needs_pwd_symlinks


def test_plan_for_fallback_uses_native_packager_path(tmp_path, monkeypatch):
    _write_fake_repo(tmp_path)
    monkeypatch.setattr(
        "nemo_runspec.run.patch_cloud_data_mover_skip_configs", lambda: None
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
