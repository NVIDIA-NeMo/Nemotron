"""The public CLI must dispatch without importing training frameworks."""

import json
from pathlib import Path

from typer.testing import CliRunner

from nemotron.cli.bin.nemotron import app

runner = CliRunner()


def test_public_cli_discovers_parse():
    result = runner.invoke(app, ["parse", "--help"])
    assert result.exit_code == 0, result.output
    assert all(name in result.output for name in ("data", "sft", "rlvr", "eval"))


def test_dry_run_resolves_profile_and_overrides_without_runtime(tmp_path):
    output = tmp_path / "missing-runtime"
    result = runner.invoke(app, ["parse", "sft", "--dry-run", f"runtime_dir={output}",
                                "training.step_scheduler.max_steps=2"])
    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    assert value["training"]["step_scheduler"]["max_steps"] == 2
    assert not output.exists()


def test_cli_passes_explicit_resolved_config_to_isolated_worker(monkeypatch, tmp_path):
    calls = []

    def run(module, args, **kwargs):
        calls.append((module, json.loads(Path(args[1]).read_text()), kwargs))

    monkeypatch.setattr("nemotron.cli.commands.parse.commands.run_module", run)
    result = runner.invoke(app, ["parse", "rlvr", f"data_dir={tmp_path / 'data'}", "selection_seed=42"])
    assert result.exit_code == 0, result.output
    module, config, options = calls[0]
    assert module == "nemotron.recipes.parse.stage2_rlvr.train"
    assert config["data_dir"] == str(tmp_path / "data")
    assert config["candidate_contexts"] == 512
    assert config["evaluation_samples"] == 8
    assert options["runtime_dir"] == Path(".nemotron/parse-runtime")


def test_invalid_options_cannot_launch(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Must not start a stage")

    monkeypatch.setattr("nemotron.cli.commands.parse.commands.run_module", forbidden)
    for arguments in (["--run", "cluster"], ["--unknown"], ["unknown_setting=1"]):
        result = runner.invoke(app, ["parse", "sft", *arguments])
        assert result.exit_code != 0
