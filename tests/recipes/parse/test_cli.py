"""The public CLI must dispatch without importing training frameworks."""

import json

import pytest
from typer.testing import CliRunner

from nemotron.cli.bin.nemotron import app

runner = CliRunner()


def test_public_cli_discovers_parse():
    result = runner.invoke(app, ["parse", "--help"])
    assert result.exit_code == 0, result.output
    assert all(name in result.output for name in ("data", "sft", "rlvr", "eval"))


def test_dry_run_resolves_profile_and_overrides_without_runtime(tmp_path):
    output = tmp_path / "missing-runtime"
    result = runner.invoke(
        app, ["parse", "sft", "--dry-run", f"output_dir={output}", "training.step_scheduler.max_steps=2"]
    )
    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    assert value["training"]["step_scheduler"]["max_steps"] == 2
    assert not output.exists()


def test_cli_passes_resolved_config_to_shared_uv_runner(monkeypatch, tmp_path):
    from omegaconf import OmegaConf

    calls = []
    monkeypatch.setattr("nemo_runspec.config.generate_job_dir", lambda *a: tmp_path / "job")

    def run(**kwargs):
        calls.append((OmegaConf.to_container(OmegaConf.load(kwargs["train_path"])), kwargs))

    monkeypatch.setattr("nemo_runspec.execution.execute_uv_local_from_spec", run)
    result = runner.invoke(app, ["parse", "rlvr", f"data_dir={tmp_path / 'data'}", "selection_seed=42"])
    assert result.exit_code == 0, result.exception
    config, options = calls[0]
    assert config["data_dir"] == str(tmp_path / "data")
    assert config["candidate_contexts"] == 512
    assert config["evaluation_samples"] == 8
    assert options["locked"] is True
    assert options["isolate_environment"] is True
    assert options["spec"].name == "parse/rlvr"
    assert options["env_vars"]["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert options["env_vars"]["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"


def test_rl_docker_uses_only_profile_mounts(monkeypatch, tmp_path):
    from nemotron.cli.commands.parse import commands

    profile = tmp_path / "env.toml"
    profile.write_text('[parse-docker]\nexecutor="docker"\nmounts=["/data:/data"]\n')
    monkeypatch.setenv("NEMOTRON_ENV_FILE", str(profile))
    monkeypatch.setattr("nemo_runspec.config.generate_job_dir", lambda *a: tmp_path / "job")
    calls = []
    monkeypatch.setattr(commands, "_execute_remote", lambda spec, path, env, *args: calls.append(env))
    result = runner.invoke(
        app,
        [
            "parse",
            "rlvr",
            "--run",
            "parse-docker",
            "data_dir=/data/data",
            "sft_output=/data/sft",
            "output_dir=/data/rlvr",
        ],
    )
    assert result.exit_code == 0, result.exception
    assert list(calls[0].mounts) == ["/data:/data"]


def test_invalid_options_cannot_launch(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Must not start a stage")

    monkeypatch.setattr("nemo_runspec.execution.execute_uv_local_from_spec", forbidden)
    for arguments in (["--stage"], ["--unknown"], ["unknown_setting=1"], ["runtime_dir=old"]):
        result = runner.invoke(app, ["parse", "sft", *arguments])
        assert result.exit_code != 0


def test_docker_dispatch_uses_profile_and_portable_training_config(monkeypatch, tmp_path):
    from omegaconf import OmegaConf

    from nemotron.cli.commands.parse import commands

    env = tmp_path / "env.toml"
    env.write_text('[parse-docker]\nexecutor="docker"\ncontainer="parse:test"\nmounts=["/data:/data"]\n')
    monkeypatch.setenv("NEMOTRON_ENV_FILE", str(env))
    monkeypatch.setattr("nemo_runspec.config.generate_job_dir", lambda *a: tmp_path / "job")
    calls = []

    def remote(spec, train_path, env, env_vars, request, directory):
        calls.append((OmegaConf.load(train_path), env, directory))

    monkeypatch.setattr(commands, "_execute_remote", remote)
    result = runner.invoke(
        app, ["parse", "sft", "--run", "parse-docker", "data_dir=/data/data", "output_dir=/data/sft"]
    )
    assert result.exit_code == 0, result.exception
    config, execution, directory = calls[0]
    assert execution.container == "parse:test"
    assert directory == "stage1_sft"
    assert config.training_config == "src/nemotron/recipes/parse/stage1_sft/config/automodel.yaml"
    assert "run" not in config


@pytest.mark.parametrize("entrypoint", [None, "/custom-wrapper"])
def test_eval_docker_runs_the_worker_and_preserves_profile_settings(monkeypatch, tmp_path, entrypoint):
    from docker.types import ContainerConfig

    from nemotron.cli.commands.parse import commands

    profile = (
        '[parse-docker]\nexecutor="docker"\nmounts=["/data:/data"]\n'
        '[parse-docker.additional_kwargs]\nuser="1000:1000"\n'
    )
    if entrypoint is not None:
        profile += f'entrypoint="{entrypoint}"\n'
    env = tmp_path / "env.toml"
    env.write_text(profile)
    monkeypatch.setenv("NEMOTRON_ENV_FILE", str(env))
    monkeypatch.setattr("nemo_runspec.config.generate_job_dir", lambda *a: tmp_path / "job")
    calls = []
    monkeypatch.setattr(commands, "_execute_remote", lambda spec, path, env, *args: calls.append(env))
    result = runner.invoke(
        app,
        [
            "parse",
            "eval",
            "--run",
            "parse-docker",
            "data_dir=/data/data",
            "model_artifact=/data/rlvr/artifact.json",
            "output_dir=/data/eval",
        ],
    )
    assert result.exit_code == 0, result.exception
    kwargs = dict(calls[0].additional_kwargs)
    assert kwargs["user"] == "1000:1000"
    assert kwargs["entrypoint"] == (entrypoint or "")
    if entrypoint is None:
        # Exercise Docker's actual API serialization: [] clears the image's
        # server entrypoint instead of prepending it to the stage command.
        payload = ContainerConfig(version="1.45", image="vllm", command="bash", **kwargs)
        assert payload["Entrypoint"] == []


def test_attached_remote_failure_returns_nonzero(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import nemo_run
    import pytest
    import typer

    from nemotron.cli.commands.parse import commands

    class Experiment:
        def __init__(self, name):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.finished = True
            return False

        def add(self, *args, **kwargs):
            # The stock vLLM image supplies python3, without a python alias.
            assert args[0].entrypoint == "python3"

        def run(self, **kwargs):
            pass

        def status(self, **kwargs):
            assert self.finished
            return {"parse-data": {"status": "FAILED"}}

    monkeypatch.setattr(nemo_run, "Experiment", Experiment)
    monkeypatch.setattr("nemo_runspec.execution.create_executor", lambda **kwargs: SimpleNamespace())
    spec = SimpleNamespace(name="parse/data", image="parse:test", resources=None)
    request = SimpleNamespace(attached=True, force_squash=False)
    with pytest.raises(typer.Exit) as error:
        commands._execute_remote(spec, tmp_path / "config.yaml", {}, {}, request, "stage0_data")
    assert error.value.exit_code == 1


@pytest.mark.parametrize("exit_code", [0, 7, None])
def test_docker_exit_receipt_uses_launched_executor_after_wait(monkeypatch, tmp_path, exit_code):
    from types import SimpleNamespace

    import nemo_run

    from nemotron.cli.commands.parse import commands

    original = nemo_run.DockerExecutor(container_image="parse:test")
    launched = nemo_run.DockerExecutor(container_image="parse:test", job_dir=str(tmp_path))

    class Experiment:
        def __init__(self, name):
            self.jobs = [SimpleNamespace(executor=launched)]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            (tmp_path / "code").mkdir()
            if exit_code is not None:
                (tmp_path / "code/.parse-stage-exit-code").write_text(str(exit_code))
            return False

        def add(self, *args, **kwargs):
            pass

        def run(self, **kwargs):
            pass

        def status(self, **kwargs):
            return {"parse-data": {"status": "SUCCEEDED"}}

    monkeypatch.setattr(nemo_run, "Experiment", Experiment)
    monkeypatch.setattr("nemo_runspec.execution.create_executor", lambda **kwargs: original)
    spec = SimpleNamespace(name="parse/data", image="parse:test", resources=None)
    from contextlib import nullcontext

    import typer

    outcome = nullcontext() if exit_code == 0 else pytest.raises(typer.Exit)
    with outcome as error:
        commands._execute_remote(
            spec, tmp_path / "config.yaml", {}, {}, SimpleNamespace(attached=True, force_squash=False), "stage0_data"
        )
    if exit_code != 0:
        assert error.value.exit_code == (exit_code or 1)


def test_wheel_execution_is_rejected_before_environment_preparation(monkeypatch, tmp_path):
    from nemotron.cli.commands.parse import commands

    monkeypatch.setattr(commands, "ROOT", tmp_path / "site-packages/nemotron/recipes/parse")
    result = runner.invoke(app, ["parse", "sft"])
    assert result.exit_code != 0
    assert "source checkout" in result.output


def test_detached_docker_is_rejected_before_launch(tmp_path):
    from types import SimpleNamespace

    import typer

    from nemotron.cli.commands.parse import commands

    with pytest.raises(typer.BadParameter, match="Docker --batch is unavailable"):
        commands._execute_remote(
            None, tmp_path / "config.yaml", {"executor": "docker"}, {}, SimpleNamespace(attached=False), "stage0_data"
        )


@pytest.mark.parametrize(
    "setting", ["nodes=2", "ntasks_per_node=2", "gpus_per_node=2", "nprocs_per_node=2", "nproc_per_node=2"]
)
def test_unsupported_resources_fail_before_submission(monkeypatch, tmp_path, setting):
    profile = tmp_path / "env.toml"
    profile.write_text('[cluster]\nexecutor="slurm"\n' + setting + "\n")
    monkeypatch.setenv("NEMOTRON_ENV_FILE", str(profile))
    result = runner.invoke(app, ["parse", "sft", "--run", "cluster", "--dry-run"])
    assert result.exit_code != 0
    assert "one node, one task, and one GPU" in result.output


def test_local_profile_uses_isolated_uv_and_preserves_environment(monkeypatch, tmp_path):
    profile = tmp_path / "env.toml"
    profile.write_text('[workstation]\nexecutor="local"\n[workstation.env_vars]\nCUSTOM_SETTING="value"\n')
    monkeypatch.setenv("NEMOTRON_ENV_FILE", str(profile))
    monkeypatch.setattr("nemo_runspec.config.generate_job_dir", lambda *a: tmp_path / "job")
    calls = []
    monkeypatch.setattr("nemo_runspec.execution.execute_uv_local_from_spec", lambda **kw: calls.append(kw))
    result = runner.invoke(app, ["parse", "sft", "--run", "workstation"])
    assert result.exit_code == 0, result.exception
    assert calls[0]["isolate_environment"] is True
    assert calls[0]["env_vars"]["CUSTOM_SETTING"] == "value"


@pytest.mark.parametrize("data", ["relative", "/unmounted/data", "/data/../outside"])
def test_remote_artifacts_must_be_persistent(monkeypatch, tmp_path, data):
    profile = tmp_path / "env.toml"
    profile.write_text('[container]\nexecutor="docker"\nmounts=["/data:/data"]\n')
    monkeypatch.setenv("NEMOTRON_ENV_FILE", str(profile))
    result = runner.invoke(app, ["parse", "data", "--run", "container", "--dry-run", f"data_dir={data}"])
    assert result.exit_code != 0
    assert "absolute and beneath a profile mount" in result.output


@pytest.mark.parametrize("mount", ["/data:/data:ro", "/data:/data:ro,z"])
def test_remote_outputs_cannot_use_readonly_mount(monkeypatch, tmp_path, mount):
    profile = tmp_path / "env.toml"
    profile.write_text('[container]\nexecutor="docker"\nmounts=["' + mount + '"]\n')
    monkeypatch.setenv("NEMOTRON_ENV_FILE", str(profile))
    result = runner.invoke(app, ["parse", "data", "--run", "container", "--dry-run", "data_dir=/data/output"])
    assert result.exit_code != 0
    assert "absolute and beneath a profile mount" in result.output


@pytest.mark.parametrize("host_available", [False, True])
def test_remote_artifacts_use_resolved_docker_mounts(monkeypatch, tmp_path, host_available):
    from omegaconf import OmegaConf

    from nemo_runspec.execution import create_executor

    host = tmp_path / "shared-data"
    host.mkdir()
    if host_available:
        monkeypatch.setenv("PARSE_DATA_ROOT", str(host))
    else:
        monkeypatch.delenv("PARSE_DATA_ROOT", raising=False)
    profile = tmp_path / "env.toml"
    profile.write_text('[container]\nexecutor="docker"\nmounts=["$PARSE_DATA_ROOT:/work"]\n')
    monkeypatch.setenv("NEMOTRON_ENV_FILE", str(profile))
    result = runner.invoke(app, ["parse", "data", "--run", "container", "--dry-run", "data_dir=/work/data"])
    executor = create_executor(
        env=OmegaConf.create({"executor": "docker", "mounts": ["$PARSE_DATA_ROOT:/work"]}),
        env_vars={},
        packager=None,
        default_image="parse:test",
    )
    if host_available:
        assert result.exit_code == 0, result.output
        assert executor.volumes == [f"{host}:/work"]
    else:
        assert result.exit_code != 0
        assert "absolute and beneath a profile mount" in result.output
        assert executor.volumes == []


@pytest.mark.parametrize("setting", ["model=/unmounted/model", "training_config=/unmounted/config.yaml"])
def test_remote_custom_inputs_must_be_mounted(monkeypatch, tmp_path, setting):
    profile = tmp_path / "env.toml"
    profile.write_text('[container]\nexecutor="docker"\nmounts=["/data:/data"]\n')
    monkeypatch.setenv("NEMOTRON_ENV_FILE", str(profile))
    result = runner.invoke(
        app,
        ["parse", "sft", "--run", "container", "--dry-run", "data_dir=/data/data", "output_dir=/data/output", setting],
    )
    assert result.exit_code != 0
    assert "absolute" in result.output
    assert setting.split("=")[0] in result.output


def test_nested_readonly_mount_wins_over_writable_parent(monkeypatch, tmp_path):
    profile = tmp_path / "env.toml"
    profile.write_text('[container]\nexecutor="docker"\nmounts=["/data:/data","/readonly:/data/reports:ro"]\n')
    monkeypatch.setenv("NEMOTRON_ENV_FILE", str(profile))
    result = runner.invoke(app, ["parse", "data", "--run", "container", "--dry-run", "data_dir=/data/reports/output"])
    assert result.exit_code != 0
    assert "absolute and beneath a profile mount" in result.output
