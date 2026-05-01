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

"""Tests for the agentic translation CLI command."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import typer
from typer.testing import CliRunner

import nemotron.cli.commands.steps.translation as translation_module
from nemotron.cli.bin.nemotron import app


def _fake_cfg(**overrides):
    defaults = {
        "mode": "local",
        "passthrough": [],
        "dry_run": False,
        "ctx": SimpleNamespace(),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_root_cli_registers_steps_translation_command() -> None:
    result = CliRunner().invoke(app, ["steps", "--help"])

    assert result.exit_code == 0
    assert "translation" in result.output


def test_translation_cli_runs_checked_in_step(monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "input_path": "/data/source.jsonl",
        "output_dir": "/data/translated",
        "source_language": "en",
        "target_language": "hi",
    }
    run_mock = Mock(return_value=Path("/data/translated"))

    monkeypatch.setattr(translation_module, "parse_recipe_config", lambda ctx: _fake_cfg())
    monkeypatch.setattr(translation_module, "_load_translation_config", lambda cfg: config)
    monkeypatch.setattr(translation_module, "_run_translation_step", run_mock)

    translation_module.translation(ctx=Mock())

    run_mock.assert_called_once_with(config)


def test_translation_cli_rejects_remote_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        translation_module,
        "parse_recipe_config",
        lambda ctx: _fake_cfg(mode="run"),
    )

    with pytest.raises(typer.Exit) as exc_info:
        translation_module.translation(ctx=Mock())

    assert exc_info.value.exit_code == 1


def test_translation_cli_dry_run_skips_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    run_mock = Mock()

    monkeypatch.setattr(
        translation_module,
        "parse_recipe_config",
        lambda ctx: _fake_cfg(dry_run=True),
    )
    monkeypatch.setattr(
        translation_module,
        "_load_translation_config",
        lambda cfg: {"input_path": "/data/source.jsonl", "output_dir": "/data/translated"},
    )
    monkeypatch.setattr(translation_module, "_run_translation_step", run_mock)

    translation_module.translation(ctx=Mock())

    run_mock.assert_not_called()
