"""Fast CLI smoke tests for the super3 recipe."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from nemotron.cli.bin.nemotron import app

runner = CliRunner()
RL_PREP_SUBCOMMANDS = ["rlvr", "swe1", "swe2", "rlhf"]


class TestSuper3DataPrepStructure:
    def test_data_prep_help_succeeds(self):
        result = runner.invoke(app, ["super3", "data", "prep", "--help"])
        assert result.exit_code == 0
        assert "rl" in result.output

    def test_data_prep_rl_help_succeeds(self):
        result = runner.invoke(app, ["super3", "data", "prep", "rl", "--help"])
        assert result.exit_code == 0
        for command in RL_PREP_SUBCOMMANDS:
            assert command in result.output

    @pytest.mark.parametrize("command", RL_PREP_SUBCOMMANDS)
    def test_data_prep_rl_subcommand_help_succeeds(self, command):
        result = runner.invoke(app, ["super3", "data", "prep", "rl", command, "--help"])
        assert result.exit_code == 0
