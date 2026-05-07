"""Tests for runspec-aware local UV execution helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from nemo_runspec import execution, parse


def test_embed_finetune_runspec_defaults_to_all_local_gpus() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "src" / "nemotron" / "recipes" / "embed" / "stage2_finetune" / "train.py"

    spec = parse(script_path)

    assert spec.resources.gpus_per_node == "gpu"


def test_execute_uv_local_uses_stage_project_lock(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    stage_dir = repo_root / "src" / "nemotron" / "recipes" / "embed" / "stage2_finetune"
    script_path = stage_dir / "train.py"
    train_path = Path("/tmp/train.yaml")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(execution.subprocess, "run", fake_run)
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/ambient-venv")

    with pytest.raises(typer.Exit) as exc_info:
        execution.execute_uv_local(
            script_path=str(script_path),
            stage_dir=stage_dir,
            repo_root=repo_root,
            train_path=train_path,
            passthrough=["model.foo=bar"],
            extra_with=["demo-extra"],
            extras=["tensorrt"],
            pre_script_args=[
                "-m",
                "torch.distributed.run",
                "--nproc_per_node=gpu",
            ],
        )

    assert exc_info.value.exit_code == 0
    assert len(calls) == 1
    assert calls[0][0] == [
        "/usr/bin/uv",
        "run",
        "--with",
        str(repo_root),
        "--with",
        "demo-extra",
        "--project",
        str(stage_dir),
        "--extra",
        "tensorrt",
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=gpu",
        str(script_path),
        "--config",
        str(train_path),
        "model.foo=bar",
    ]
    assert "VIRTUAL_ENV" not in calls[0][1]["env"]


def test_execute_uv_local_preserves_nested_relative_script_path(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    stage_dir = repo_root / "src" / "nemotron" / "recipes" / "embed" / "stage2_finetune"
    train_path = Path("/tmp/train.yaml")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(execution.subprocess, "run", fake_run)

    with pytest.raises(typer.Exit):
        execution.execute_uv_local(
            script_path="scripts/train.py",
            stage_dir=stage_dir,
            repo_root=repo_root,
            train_path=train_path,
            passthrough=[],
        )

    assert calls[0][0][-3:] == [
        str(stage_dir / "scripts" / "train.py"),
        "--config",
        str(train_path),
    ]


def test_execute_uv_local_from_spec_uses_torchrun_launch(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "src" / "nemotron" / "recipes" / "embed" / "stage2_finetune" / "train.py"
    train_path = Path("/tmp/train.yaml")
    captured = {}

    def fake_execute_uv_local(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(execution, "execute_uv_local", fake_execute_uv_local)

    spec = SimpleNamespace(
        script_path=script_path,
        run=SimpleNamespace(launch="torchrun"),
        resources=SimpleNamespace(gpus_per_node="gpu"),
    )

    execution.execute_uv_local_from_spec(
        spec=spec,
        train_path=train_path,
        passthrough=["model.foo=bar"],
    )

    assert captured["script_path"] == str(script_path)
    assert captured["stage_dir"] == script_path.parent
    assert captured["repo_root"] == repo_root
    assert captured["train_path"] == train_path
    assert captured["passthrough"] == ["model.foo=bar"]
    assert captured["pre_script_args"] == [
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=gpu",
    ]


def test_execute_uv_local_from_spec_uses_numeric_runspec_resource(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "src" / "nemotron" / "recipes" / "embed" / "stage2_finetune" / "train.py"
    captured = {}

    def fake_execute_uv_local(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(execution, "execute_uv_local", fake_execute_uv_local)

    spec = SimpleNamespace(
        script_path=script_path,
        run=SimpleNamespace(launch="torchrun"),
        resources=SimpleNamespace(gpus_per_node=1),
    )

    execution.execute_uv_local_from_spec(
        spec=spec,
        train_path=Path("/tmp/train.yaml"),
        passthrough=[],
    )

    assert captured["pre_script_args"] == [
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
    ]


def test_execute_uv_local_from_spec_allows_torchrun_nproc_override(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "src" / "nemotron" / "recipes" / "embed" / "stage2_finetune" / "train.py"
    captured = {}

    def fake_execute_uv_local(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(execution, "execute_uv_local", fake_execute_uv_local)

    spec = SimpleNamespace(
        script_path=script_path,
        run=SimpleNamespace(launch="torchrun"),
        resources=SimpleNamespace(gpus_per_node=1),
    )

    execution.execute_uv_local_from_spec(
        spec=spec,
        train_path=Path("/tmp/train.yaml"),
        passthrough=[],
        torchrun_nproc_per_node="gpu",
    )

    assert captured["pre_script_args"] == [
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=gpu",
    ]


def test_execute_uv_local_from_spec_forwards_direct_launch_extras(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "src" / "nemotron" / "recipes" / "embed" / "stage4_export" / "export.py"
    captured = {}

    def fake_execute_uv_local(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(execution, "execute_uv_local", fake_execute_uv_local)

    spec = SimpleNamespace(
        script_path=script_path,
        run=SimpleNamespace(launch="direct"),
    )

    execution.execute_uv_local_from_spec(
        spec=spec,
        train_path=Path("/tmp/export.yaml"),
        passthrough=[],
        extra_with=["demo"],
        extras=["tensorrt"],
    )

    assert captured["repo_root"] == repo_root
    assert captured["extra_with"] == ["demo"]
    assert captured["extras"] == ["tensorrt"]
    assert captured["pre_script_args"] == []
