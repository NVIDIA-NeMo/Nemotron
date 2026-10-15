"""CLI structure tests for super3.

Regression guard: the `data prep rl` group once shadowed itself (a legacy
rl.py module next to the rl/ package), which silently dropped the whole
super3 CLI -- and omni3/super35 RL, which reuse super3's RL backend.
"""

from __future__ import annotations

import sys

import pytest
from typer.testing import CliRunner

from nemotron.cli.bin.nemotron import app

runner = CliRunner()

RL_STAGES = ("rlvr", "swe1", "swe2", "rlhf")


def _invoke(*args: str):
    result = runner.invoke(app, ["super3", *args])
    assert result.exit_code == 0, f"{' '.join(args)} failed: {result.output}\n{result.exception}"
    return result


def test_super3_lists_data_group():
    assert "data" in _invoke("--help").output


def test_data_prep_rl_lists_every_stage():
    output = _invoke("data", "prep", "rl", "--help").output
    for stage in RL_STAGES:
        assert stage in output


@pytest.mark.parametrize("stage", RL_STAGES)
def test_data_prep_rl_stage_matches_training_stage(stage):
    _invoke("data", "prep", "rl", stage, "--help")
    _invoke("rl", stage, "--help")


@pytest.mark.parametrize("stage", RL_STAGES)
def test_data_prep_rl_stage_dry_run(monkeypatch, stage):
    monkeypatch.setattr(sys, "argv", ["nemotron", "super3", "data", "prep", "rl", stage, "-d"])
    _invoke("data", "prep", "rl", stage, "-d")
