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

"""CLI structure tests for super3 commands.

Uses ``typer.testing.CliRunner`` for in-process, fast CLI testing.
These tests verify that all super3 subcommands are importable and registered
correctly — catching import errors (e.g. missing subpackages) before they
reach users.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from nemotron.cli.bin.nemotron import app

runner = CliRunner()

# Top-level super3 commands
SUPER3_TOP_COMMANDS = ["pretrain", "sft", "rl", "eval", "pipe", "data", "model"]

# super3 rl subcommands
SUPER3_RL_COMMANDS = ["rlvr", "swe1", "swe2", "rlhf"]

# super3 data prep subcommands
SUPER3_DATA_PREP_COMMANDS = ["pretrain", "sft", "rl"]

# super3 data import subcommands
SUPER3_DATA_IMPORT_COMMANDS = ["pretrain", "sft", "rl"]


class TestSuper3AppStructure:
    def test_help_succeeds(self):
        result = runner.invoke(app, ["super3", "--help"])
        assert result.exit_code == 0, f"super3 --help failed: {result.output}"

    @pytest.mark.parametrize("command", SUPER3_TOP_COMMANDS)
    def test_top_command_listed(self, command):
        result = runner.invoke(app, ["super3", "--help"])
        assert result.exit_code == 0
        assert command in result.output, (
            f"'{command}' not found in super3 --help output"
        )

    @pytest.mark.parametrize("command", SUPER3_TOP_COMMANDS)
    def test_top_command_help_succeeds(self, command):
        result = runner.invoke(app, ["super3", command, "--help"])
        assert result.exit_code == 0, (
            f"super3 {command} --help failed: {result.output}\n{result.exception}"
        )


class TestSuper3RlStructure:
    @pytest.mark.parametrize("command", SUPER3_RL_COMMANDS)
    def test_rl_subcommand_listed(self, command):
        result = runner.invoke(app, ["super3", "rl", "--help"])
        assert result.exit_code == 0
        assert command in result.output, (
            f"'{command}' not found in super3 rl --help output"
        )

    @pytest.mark.parametrize("command", SUPER3_RL_COMMANDS)
    def test_rl_subcommand_help_succeeds(self, command):
        result = runner.invoke(app, ["super3", "rl", command, "--help"])
        assert result.exit_code == 0, (
            f"super3 rl {command} --help failed: {result.output}\n{result.exception}"
        )


class TestSuper3DataStructure:
    def test_data_subcommands_listed(self):
        result = runner.invoke(app, ["super3", "data", "--help"])
        assert result.exit_code == 0
        assert "prep" in result.output
        assert "import" in result.output

    def test_data_prep_help_succeeds(self):
        result = runner.invoke(app, ["super3", "data", "prep", "--help"])
        assert result.exit_code == 0, (
            f"super3 data prep --help failed: {result.output}"
        )

    @pytest.mark.parametrize("command", SUPER3_DATA_PREP_COMMANDS)
    def test_data_prep_subcommand_listed(self, command):
        result = runner.invoke(app, ["super3", "data", "prep", "--help"])
        assert result.exit_code == 0
        assert command in result.output, (
            f"'{command}' not found in super3 data prep --help output"
        )

    @pytest.mark.parametrize("command", SUPER3_DATA_PREP_COMMANDS)
    def test_data_prep_subcommand_help_succeeds(self, command):
        result = runner.invoke(app, ["super3", "data", "prep", command, "--help"])
        assert result.exit_code == 0, (
            f"super3 data prep {command} --help failed: {result.output}\n{result.exception}"
        )

    def test_data_import_help_succeeds(self):
        result = runner.invoke(app, ["super3", "data", "import", "--help"])
        assert result.exit_code == 0, (
            f"super3 data import --help failed: {result.output}"
        )

    @pytest.mark.parametrize("command", SUPER3_DATA_IMPORT_COMMANDS)
    def test_data_import_subcommand_listed(self, command):
        result = runner.invoke(app, ["super3", "data", "import", "--help"])
        assert result.exit_code == 0
        assert command in result.output, (
            f"'{command}' not found in super3 data import --help output"
        )

    @pytest.mark.parametrize("command", SUPER3_DATA_IMPORT_COMMANDS)
    def test_data_import_subcommand_help_succeeds(self, command):
        result = runner.invoke(app, ["super3", "data", "import", command, "--help"])
        assert result.exit_code == 0, (
            f"super3 data import {command} --help failed: {result.output}\n{result.exception}"
        )
