from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from nemo_runspec.cli_context import GlobalContext
from nemo_runspec.staging import (
    parse_env_for_stage,
    run_or_stage_experiment,
    stage_experiment,
    validate_staging_target,
)


class FakeSSHTunnel:
    def __init__(self, remote_dir: str) -> None:
        self.job_dir = remote_dir
        self.key = "user@cluster.example.com"
        self.session = object()
        self.packaging_jobs = {}
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def run(self, command: str) -> None:
        raise AssertionError(f"unexpected tunnel command: {command}")


class FakeExperiment:
    def __init__(self, local_dir: Path, tunnel: FakeSSHTunnel | None = None) -> None:
        executor = SimpleNamespace(
            experiment_dir=str(local_dir),
            job_dir=str(local_dir / "embed-finetune"),
            job_name="embed-finetune",
            env_vars={"NEMO_RUN_DIR": "/remote/jobs", "PLAIN_VALUE": "two words"},
            setup_lines="source /remote/private.env",
        )
        self.jobs = [SimpleNamespace(executor=executor)]
        self.tunnels = {} if tunnel is None else {tunnel.key: tunnel}
        self.dryrun_calls: list[dict[str, bool]] = []
        self.run_calls: list[dict[str, bool]] = []

    def dryrun(self, **kwargs: bool) -> None:
        self.dryrun_calls.append(kwargs)
        local_dir = Path(self.jobs[0].executor.experiment_dir)
        (local_dir / "embed-finetune_sbatch.sh").write_text(
            "#!/usr/bin/env bash\npython main.py\n",
            encoding="utf-8",
        )

    def run(self, **kwargs: bool) -> None:
        self.run_calls.append(kwargs)


def test_stage_experiment_prepares_and_transfers_without_running(monkeypatch, tmp_path, capsys) -> None:
    local_dir = tmp_path / "experiments" / "embed-finetune" / "experiment-id"
    code_dir = local_dir / "embed-finetune" / "code"
    code_dir.mkdir(parents=True)
    tunnel = FakeSSHTunnel("/remote/jobs/embed-finetune/experiment-id")
    experiment = FakeExperiment(local_dir, tunnel)
    transfers = []

    def fake_rsync(session, source: str, target: str) -> None:
        transfers.append((session, source, target))

    monkeypatch.setattr(
        "nemo_runspec.staging._load_nemo_run_staging_api",
        lambda: (FakeSSHTunnel, fake_rsync),
    )

    staged_dirs = stage_experiment(experiment)

    assert experiment.dryrun_calls == [{"log": False, "delete_exp_dir": False}]
    assert experiment.run_calls == []
    assert tunnel.connected is True
    assert transfers == [(tunnel.session, str(local_dir), "/remote/jobs/embed-finetune")]
    assert staged_dirs == ["/remote/jobs/embed-finetune/experiment-id"]
    output = capsys.readouterr().out
    assert "Staged experiment files to user@cluster.example.com" in output
    assert "cd /remote/jobs/embed-finetune/experiment-id/embed-finetune/code" in output
    assert "./run.sh" in output
    launcher = code_dir / "run.sh"
    launcher_text = launcher.read_text(encoding="utf-8")
    assert 'exec bash "$script_dir"/../../embed-finetune_sbatch.sh "$@"' in launcher_text
    assert "source /remote/private.env" not in launcher_text
    assert "PLAIN_VALUE" not in launcher_text
    assert launcher.stat().st_mode & 0o111


def test_stage_experiment_requires_an_ssh_target(monkeypatch, tmp_path, capsys) -> None:
    (tmp_path / "embed-finetune" / "code").mkdir(parents=True)
    experiment = FakeExperiment(tmp_path)
    monkeypatch.setattr(
        "nemo_runspec.staging._load_nemo_run_staging_api",
        lambda: (FakeSSHTunnel, lambda *_args: None),
    )

    with pytest.raises(typer.Exit):
        stage_experiment(experiment)

    assert experiment.run_calls == []
    assert "did not produce an SSH staging target" in capsys.readouterr().err


def test_run_or_stage_experiment_never_runs_when_staging(monkeypatch, tmp_path) -> None:
    experiment = FakeExperiment(tmp_path)
    monkeypatch.setattr(
        "nemo_runspec.staging.stage_experiment",
        lambda exp: ["/remote/staged"] if exp is experiment else [],
    )

    result = run_or_stage_experiment(experiment, stage=True, attached=True)

    assert result == ["/remote/staged"]
    assert experiment.run_calls == []


@pytest.mark.parametrize(
    ("attached", "expected"),
    [
        (True, {"detach": False, "tail_logs": True}),
        (False, {"detach": True, "tail_logs": False}),
    ],
)
def test_run_or_stage_experiment_preserves_normal_execution(attached, expected, tmp_path) -> None:
    experiment = FakeExperiment(tmp_path)

    result = run_or_stage_experiment(experiment, stage=False, attached=attached)

    assert result is None
    assert experiment.run_calls == [expected]


def test_validate_staging_target_accepts_slurm_ssh_profile() -> None:
    validate_staging_target(
        {
            "executor": "slurm",
            "tunnel": "ssh",
            "host": "cluster.example.com",
            "user": "user",
            "remote_job_dir": "/remote/jobs",
        }
    )


@pytest.mark.parametrize(
    "profile",
    [
        {"executor": "docker"},
        {"executor": "slurm", "tunnel": "ssh", "host": "cluster.example.com", "user": "user"},
        {
            "executor": "slurm",
            "tunnel": "local",
            "host": "cluster.example.com",
            "user": "user",
            "remote_job_dir": "/remote/jobs",
        },
    ],
)
def test_validate_staging_target_rejects_unsupported_profiles(profile, capsys) -> None:
    with pytest.raises(typer.Exit):
        validate_staging_target(profile)

    assert "Error: --stage" in capsys.readouterr().err


def test_parse_env_for_stage_reports_missing_profile(monkeypatch, tmp_path, capsys) -> None:
    env_file = tmp_path / "env.toml"
    env_file.write_text('[known]\nexecutor = "slurm"\n', encoding="utf-8")
    monkeypatch.setenv("NEMOTRON_ENV_FILE", str(env_file))

    with pytest.raises(typer.Exit):
        parse_env_for_stage(GlobalContext(run="missing"), stage=True)

    assert "Profile 'missing' not found" in capsys.readouterr().err
