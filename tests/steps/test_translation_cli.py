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

"""Smoke tests for `nemotron steps` CLI surface for translation.

The bespoke ``nemotron steps translation`` recipe command and the top-level
``nemotron byob`` command were removed in favour of the generic
``nemotron steps run <id>`` dispatcher. These tests guard the new contract.
"""

from __future__ import annotations

from typer.testing import CliRunner

from nemotron.cli.bin.nemotron import app


def test_steps_help_lists_catalog_commands() -> None:
    result = CliRunner().invoke(app, ["steps", "--help"])

    assert result.exit_code == 0
    assert "list" in result.output
    assert "show" in result.output
    assert "run" in result.output


def test_steps_help_does_not_expose_bespoke_translation_command() -> None:
    """`nemotron steps translation` has been collapsed into `steps run translate/nemo_curator`."""

    result = CliRunner().invoke(app, ["steps", "translation", "--help"])

    assert result.exit_code != 0


def test_root_does_not_register_step_alias() -> None:
    result = CliRunner().invoke(app, ["step", "--help"])

    assert result.exit_code != 0
    assert "No such command" in result.output


def test_root_does_not_register_top_level_byob_command() -> None:
    """`nemotron byob` has been collapsed into `steps run byob/mcq`."""

    result = CliRunner().invoke(app, ["byob", "--help"])

    assert result.exit_code != 0


def test_steps_show_resolves_curator_step() -> None:
    result = CliRunner().invoke(app, ["steps", "show", "translate/nemo_curator"])

    assert result.exit_code == 0, result.output
    assert "translate/nemo_curator" in result.output


def test_steps_show_resolves_byob_mcq_step() -> None:
    result = CliRunner().invoke(app, ["steps", "show", "byob/mcq"])

    assert result.exit_code == 0, result.output
    assert "byob/mcq" in result.output


def test_steps_show_rejects_legacy_translation_id() -> None:
    """The legacy `translate/translation` id no longer resolves."""

    result = CliRunner().invoke(app, ["steps", "show", "translate/translation"])

    assert result.exit_code != 0
    combined = (result.output or "") + (result.stderr or "")
    assert "Unknown step id" in combined or "Did you mean" in combined


def test_steps_show_rejects_legacy_byob_id() -> None:
    """The legacy single-segment `byob` id no longer resolves.

    The directory-tail short-form would have matched ``byob`` to a folder named
    ``byob``, but the new layout has no such folder — the step lives at
    ``byob/mcq``.
    """

    result = CliRunner().invoke(app, ["steps", "show", "byob"])

    assert result.exit_code != 0
    combined = (result.output or "") + (result.stderr or "")
    assert "Unknown step id" in combined or "Did you mean" in combined
