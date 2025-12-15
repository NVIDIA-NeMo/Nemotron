from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from nemotron.kit.artifact import ArtifactInput
from nemotron.kit.run import list_run_profiles
from nemotron.kit.tui import maybe_run_stage_tui
from nemotron.kit.tui.form_utils import coerce_value, flatten_dataclass
from nemotron.kit.tui.widgets.artifact_picker import (
    _build_wandb_version_options,
    _resolve_wandb_project_path,
    _wandb_artifact_urls,
)


@dataclass
class DummyConfig:
    x: int = 1


class DummyApp:
    def __init__(self) -> None:
        self._commands = [
            (
                "pretrain",
                DummyConfig,
                lambda cfg: cfg,
                "desc",
                {"data": ArtifactInput(default_name="X", mappings={"path": "x"})},
                None,
                None,
            ),
            (
                "sft",
                DummyConfig,
                lambda cfg: cfg,
                "desc",
                None,
                None,
                None,
                None,
            ),
            (
                "curate",
                DummyConfig,
                lambda cfg: cfg,
                "desc",
                None,
                None,
                None,
            ),
        ]


def test_maybe_run_stage_tui_only_intercepts_single_stage_arg() -> None:
    app = DummyApp()

    with patch.object(sys, "argv", ["nemotron", "pretrain"]):
        with patch("nemotron.kit.tui.run_stage_tui") as run_tui:
            assert maybe_run_stage_tui(app) is True
            run_tui.assert_called_once()

    with patch.object(sys, "argv", ["nemotron", "data", "prep", "pretrain"]):
        with patch("nemotron.kit.tui.run_stage_tui") as run_tui:
            assert maybe_run_stage_tui(app) is False
            run_tui.assert_not_called()

    with patch.object(sys, "argv", ["nemotron", "curate"]):
        with patch("nemotron.kit.tui.run_stage_tui") as run_tui:
            assert maybe_run_stage_tui(app) is False
            run_tui.assert_not_called()


def test_maybe_run_stage_tui_handles_extended_command_tuple() -> None:
    app = DummyApp()

    with patch.object(sys, "argv", ["nemotron", "sft"]):
        with patch("nemotron.kit.tui.run_stage_tui") as run_tui:
            assert maybe_run_stage_tui(app) is True
            run_tui.assert_called_once()


def test_maybe_run_stage_tui_returns_false_when_command_missing() -> None:
    app = DummyApp()
    app._commands = []
    with patch.object(sys, "argv", ["nemotron", "pretrain"]):
        with patch("nemotron.kit.tui.run_stage_tui") as run_tui:
            assert maybe_run_stage_tui(app) is False
            run_tui.assert_not_called()


def test_list_run_profiles_excludes_wandb() -> None:
    content = """
[wandb]
project = "p"

[local]
executor = "local"

[pretrain]
extends = "local"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
        f.write(content)
        path = Path(f.name)

    try:
        profiles = list_run_profiles(config_path=path)
        assert profiles == ["local", "pretrain"]
    finally:
        path.unlink()


def test_flatten_dataclass_and_coerce_value() -> None:
    from nemotron.kit.megatron_stub import ConfigContainer

    cfg = ConfigContainer()
    flat = flatten_dataclass(cfg)

    assert "data.seq_length" in flat
    assert "training.max_steps" in flat

    _val, ann = flat["training.max_steps"]
    assert coerce_value("123", ann) == 123


def test_build_wandb_version_options() -> None:
    class A:
        def __init__(self, version: str, created_at: str) -> None:
            self.version = version
            self.created_at = created_at

    latest = A("v13", "2025-12-11T19:09:29Z")
    versions = [
        A("v13", "2025-12-11T19:09:29Z"),
        A("v12", "2025-12-10T10:00:00Z"),
    ]

    options = _build_wandb_version_options(latest=latest, versions=versions)
    assert options[0][1] == "latest"
    assert any(v == "v12" for _label, v in options)


def test_resolve_wandb_project_path_prefers_run_toml_config(monkeypatch) -> None:
    class Cfg:
        project = "proj"
        entity = "ent"

    monkeypatch.setenv("WANDB_PROJECT", "envproj")
    monkeypatch.setenv("WANDB_ENTITY", "envent")

    import nemotron.kit.tui.widgets.artifact_picker as ap

    monkeypatch.setattr(ap, "load_wandb_config", lambda: Cfg())
    assert _resolve_wandb_project_path() == "ent/proj"


def test_wandb_artifact_urls() -> None:
    meta, lineage = _wandb_artifact_urls(
        project_path="romeyn/nemotron",
        type_name="DataBlendsArtifact",
        artifact_name="DataBlendsArtifact-sft",
        version="v3",
    )
    assert meta.endswith("/metadata")
    assert lineage.endswith("/lineage")
    assert "wandb.ai/romeyn/nemotron" in meta
