"""Tests for the programmatic model-eval launch boundary."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
from omegaconf import OmegaConf

from nemotron.steps.eval.model_eval import runtime


def test_programmatic_launch_uses_configured_tasks_and_returns_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    launcher = ModuleType("nemo_evaluator_launcher")
    launcher.__path__ = []  # type: ignore[attr-defined]
    api = ModuleType("nemo_evaluator_launcher.api")
    api.__path__ = []  # type: ignore[attr-defined]
    functional = ModuleType("nemo_evaluator_launcher.api.functional")

    def run_eval(config, *, dry_run, tasks):  # type: ignore[no-untyped-def]
        calls.append((config, dry_run, tasks))
        return "invocation"

    functional.run_eval = run_eval  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nemo_evaluator_launcher", launcher)
    monkeypatch.setitem(sys.modules, "nemo_evaluator_launcher.api", api)
    monkeypatch.setitem(
        sys.modules,
        "nemo_evaluator_launcher.api.functional",
        functional,
    )
    saved = tmp_path / "saved-launcher.yaml"
    monkeypatch.setattr(
        runtime,
        "_save_launcher_config",
        lambda config_path, cfg, launcher_cfg: saved,
    )
    cfg = OmegaConf.create(
        {
            "dry_run": True,
            "task_filters": ["bfcl.task"],
            "execution": {"type": "local", "mode": "sequential"},
            "deployment": {"type": "none"},
            "target": {"api_endpoint": {}},
            "evaluation": {"tasks": [{"name": "bfcl.task"}]},
        }
    )

    result = runtime.launch_model_eval_config(
        config_path=tmp_path / "materialized.yaml",
        cfg=cfg,
    )

    assert result.launcher_config_path == saved
    assert result.invocation_id == "invocation"
    assert calls[0][1:] == (True, ["bfcl.task"])


def test_a_missing_launcher_raises_instead_of_exiting_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "nemo_evaluator_launcher.api.functional", None)
    written: list[Path] = []
    monkeypatch.setattr(
        runtime,
        "_save_launcher_config",
        lambda config_path, cfg, launcher_cfg: written.append(config_path),
    )

    # SystemExit here would be a BaseException that an embedding caller cannot map
    # into its own error contract, and a config saved before the check would look
    # like a submission that never happened.
    with pytest.raises(runtime.ModelEvalDependencyError):
        runtime.launch_model_eval_config(
            config_path=tmp_path / "materialized.yaml",
            cfg=OmegaConf.create({"execution": {"type": "local"}}),
        )

    assert written == []
