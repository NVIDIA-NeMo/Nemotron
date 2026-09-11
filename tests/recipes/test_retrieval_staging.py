from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import nemo_run
import pytest
from typer.testing import CliRunner

from nemotron.cli.bin.nemotron import app

COMMAND_MODULES = [
    ("embed", "prep", "nemotron.cli.commands.embed.prep"),
    ("embed", "finetune", "nemotron.cli.commands.embed.finetune"),
    ("rerank", "prep", "nemotron.cli.commands.rerank.prep"),
    ("rerank", "finetune", "nemotron.cli.commands.rerank.finetune"),
]

runner = CliRunner()


@pytest.mark.parametrize(("recipe", "command", "module_name"), COMMAND_MODULES)
def test_stage_reports_missing_profile(recipe, command, module_name, monkeypatch, tmp_path) -> None:
    del module_name
    env_file = tmp_path / "env.toml"
    env_file.write_text('[known]\nexecutor = "slurm"\n', encoding="utf-8")
    monkeypatch.setenv("NEMOTRON_ENV_FILE", str(env_file))
    argv = ["nemotron", recipe, command, "--run", "missing", "--stage"]
    monkeypatch.setattr(sys, "argv", argv)

    result = runner.invoke(app, argv[1:])

    assert result.exit_code == 1
    assert "Error: Profile 'missing' not found" in result.output


@pytest.mark.parametrize(("recipe", "command", "module_name"), COMMAND_MODULES)
def test_stage_dry_run_returns_before_validation(recipe, command, module_name, monkeypatch, tmp_path) -> None:
    env_file = tmp_path / "env.toml"
    env_file.write_text('[local-docker]\nexecutor = "docker"\n', encoding="utf-8")
    monkeypatch.setenv("NEMOTRON_ENV_FILE", str(env_file))
    module = import_module(module_name)
    monkeypatch.setattr(
        module,
        "validate_staging_target",
        lambda _env: pytest.fail("dry-run must return before staging validation"),
    )
    monkeypatch.setattr(
        module,
        "_execute_remote",
        lambda **_kwargs: pytest.fail("dry-run must return before remote execution"),
    )
    argv = ["nemotron", recipe, command, "--run", "local-docker", "--stage", "--dry-run"]
    monkeypatch.setattr(sys, "argv", argv)

    result = runner.invoke(app, argv[1:])

    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(("recipe", "command", "module_name"), COMMAND_MODULES)
def test_remote_stage_uses_staging_path(recipe, command, module_name, monkeypatch, tmp_path) -> None:
    del recipe, command
    module = import_module(module_name)
    executor = object()
    experiments = []
    selected_paths = []

    class FakeExperiment:
        def __init__(self, name: str) -> None:
            self.name = name
            self.add_calls = []
            self.run_calls = []
            experiments.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def add(self, script, *, executor, name: str):
            self.add_calls.append((script, executor, name))

        def run(self, **kwargs) -> None:
            self.run_calls.append(kwargs)

    monkeypatch.setattr(nemo_run, "Experiment", FakeExperiment)
    monkeypatch.setattr("nemo_runspec.execution.create_executor", lambda **_kwargs: executor)
    monkeypatch.setattr(
        "nemo_runspec.run.patch_nemo_run_rsync_accept_new_host_keys",
        lambda: None,
    )
    monkeypatch.setattr(
        "nemo_runspec.run.patch_nemo_run_ray_template_for_cpu",
        lambda: None,
    )
    monkeypatch.setattr(
        module,
        "run_or_stage_experiment",
        lambda experiment, *, stage, attached: selected_paths.append((experiment, stage, attached)),
    )

    module._execute_remote(
        train_path=Path(tmp_path / "train.yaml"),
        env={},
        passthrough=[],
        attached=True,
        env_vars={},
        force_squash=False,
        stage=True,
    )

    assert len(experiments) == 1
    experiment = experiments[0]
    assert len(experiment.add_calls) == 1
    assert experiment.run_calls == []
    assert selected_paths == [(experiment, True, True)]
