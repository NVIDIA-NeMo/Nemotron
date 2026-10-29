"""Contracts `eval/model_eval` relies on from the layers around it.

These are integration seams, not unit behaviour: a change on either side is
silent unless something pins it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

CONFIG_DIR = Path("src/nemotron/steps/eval/model_eval/config")


# --- outer scheduler vs launcher scheduler -----------------------------------


def _default_cfg():
    cfg = OmegaConf.load(CONFIG_DIR / "default.yaml")
    # Avoid the artifact resolver; the checkpoint path is irrelevant here.
    cfg.deployment.checkpoint_path = "/mnt/ckpt/iter_0001000"
    return cfg


def test_launcher_executor_is_independent_of_the_outer_executor():
    """`--batch` sets `run.env.executor`. When `execution.type` read that same
    field, submitting the step to Slurm made the LAUNCHER submit again from
    inside the allocation, and "outer local, launcher Slurm" -- the normal way
    to use launcher mode on a cluster -- could not be expressed at all."""
    from nemotron.steps.eval.model_eval.runtime import _build_launcher_config

    cfg = _default_cfg()
    cfg.run.env.executor = "local"
    cfg.run.env.launcher_executor = "slurm"
    launcher_cfg, _, _ = _build_launcher_config(cfg)
    assert launcher_cfg.execution.type == "slurm"

    # ... and the outer executor does not leak into the launcher either way.
    cfg = _default_cfg()
    cfg.run.env.executor = "lepton"
    launcher_cfg, _, _ = _build_launcher_config(cfg)
    assert launcher_cfg.execution.type == "local"


def test_default_config_ships_a_local_launcher_executor():
    cfg = OmegaConf.load(CONFIG_DIR / "default.yaml")
    assert cfg.run.env.launcher_executor == "local"
    assert cfg.execution.type == "local"


def test_auto_squash_follows_the_launcher_executor(monkeypatch):
    """Squashing prepares the images the LAUNCHER runs. Keyed to the outer
    executor, outer-local/launcher-Slurm silently skipped it, which breaks
    air-gapped and private-registry clusters.

    Asserted at the executor gate: reaching image collection is as far as this
    can go without a real SSH tunnel.
    """
    import nemo_runspec.evaluator as ev

    class _ReachedError(Exception):
        pass

    def _boom(_cfg):
        raise _ReachedError

    monkeypatch.setattr(ev, "collect_evaluator_images", _boom)

    def _run(executor: str, launcher_executor: str | None) -> bool:
        env = {"executor": executor, "tunnel": "ssh", "host": "h", "remote_job_dir": "/d", "user": "u"}
        if launcher_executor is not None:
            env["launcher_executor"] = launcher_executor
        cfg = OmegaConf.create({"run": {"env": env}, "deployment": {"image": "example/img:1"}})
        try:
            ev.maybe_auto_squash_evaluator(cfg, mode="batch", dry_run=False, force_squash=False)
        except _ReachedError:
            return True
        return False

    assert _run("local", "slurm") is True, "outer-local + launcher_executor=slurm skipped the squash path"
    # Steps with no launcher_executor keep the original behaviour.
    assert _run("slurm", None) is True
    assert _run("local", None) is False
    # An explicit local launcher wins over an outer slurm executor: the
    # launcher, not the outer runner, is what consumes these images.
    assert _run("slurm", "local") is False


# --- Lepton platform secrets --------------------------------------------------


def test_secret_vars_reach_the_lepton_executor_and_stay_out_of_env_vars():
    """Values in `env_vars` are stored verbatim in the submitted job spec.
    A token must travel as a secret reference instead."""
    pytest.importorskip("nemo_run")
    from nemo_runspec.execution import _create_lepton_executor

    env = {
        "nemo_run_dir": "/mnt/shared/nemo_run",
        "container_image": "example/harness:1",
        "node_group": "ng",
        "secret_vars": {"ENDPOINT_TOKEN": "eval-endpoint-token"},
    }
    env_vars = {"EVAL_MODEL_HANDLE": "my-model"}
    ex = _create_lepton_executor(env, env_vars, packager=None, default_image=None, script_resources=None)

    assert ex.secret_vars == {"ENDPOINT_TOKEN": "eval-endpoint-token"}
    # The name maps to a secret; the value never appears in the spec's env vars.
    assert "ENDPOINT_TOKEN" not in ex.env_vars
    assert "eval-endpoint-token" not in ex.env_vars.values()


def test_lepton_executor_without_secret_vars_is_unchanged():
    pytest.importorskip("nemo_run")
    from nemo_runspec.execution import _create_lepton_executor

    ex = _create_lepton_executor(
        {"nemo_run_dir": "/d", "container_image": "example/harness:1"},
        {},
        packager=None,
        default_image=None,
        script_resources=None,
    )
    assert ex.secret_vars == {}


# --- W&B is opt-in ------------------------------------------------------------


def test_eval_default_config_does_not_request_wandb(monkeypatch):
    """A stale `wandb login` on the submitting host aborted steps that never
    asked for W&B, with a 401 about artifact resolution they do not perform."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    from nemo_runspec.execution import _job_uses_wandb

    assert _job_uses_wandb(OmegaConf.load(CONFIG_DIR / "default.yaml")) is False
    assert _job_uses_wandb(OmegaConf.load(CONFIG_DIR / "direct.yaml")) is False


@pytest.mark.parametrize(
    "cfg",
    [
        {"run": {"wandb": {"project": "p", "entity": None}}},
        {"run": {"wandb": {"project": None, "entity": "e"}}},
        {"export": {"wandb": {"project": "p"}}},
        {"execution": {"auto_export": {"destinations": ["wandb"]}}},
    ],
)
def test_configs_that_do_use_wandb_still_validate_credentials(cfg, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    from nemo_runspec.execution import _job_uses_wandb

    assert _job_uses_wandb(OmegaConf.create(cfg)) is True


def test_explicit_wandb_api_key_is_an_opt_in(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "k")
    from nemo_runspec.execution import _job_uses_wandb

    assert _job_uses_wandb(OmegaConf.load(CONFIG_DIR / "default.yaml")) is True


# --- provenance survives config extraction ------------------------------------


def test_extract_train_config_strips_run_env():
    """The precondition the two provenance fixes exist for. If this ever stops
    holding, `harness_image` and the EVAL_HARNESS_IMAGE forwarding can be
    simplified -- and the README/step.toml auto-squash notes are stale."""
    from nemo_runspec.config.loader import extract_train_config

    full = OmegaConf.load(CONFIG_DIR / "direct.yaml")
    assert not (extract_train_config(full).get("run") or {}).get("env")


def test_harness_image_is_the_single_source_of_truth():
    """`run.env.container_image` interpolates from `harness_image`, so the
    executor and the manifest can never disagree about which image ran."""
    full = OmegaConf.load(CONFIG_DIR / "direct.yaml")
    assert full.run.env.container_image == full.harness_image


def _submit_then_run(monkeypatch, tmp_path, *, host_image, cli_image=None):
    """Simulate the host -> container transition for real.

    The config is serialised UNRESOLVED, so every `${oc.env:...}` in it is
    re-evaluated inside the container against only the variables the executor
    forwarded. Anything the test leaves set in the parent process would hide
    exactly the bug this is here to catch.
    """
    from nemo_runspec.config.loader import extract_train_config
    from nemotron.steps.eval.model_eval.runtime import run_direct

    # --- on the submitting host ---
    monkeypatch.setenv("EVAL_HARNESS_IMAGE", host_image)
    full = OmegaConf.load(CONFIG_DIR / "direct.yaml")
    if cli_image is not None:
        # What `run.env.container_image=<image>` on the command line does. A
        # PROFILE cannot do this -- see the test below.
        full.run.env.container_image = cli_image
    executor_image = full.run.env.container_image
    # This is exactly what `steps run` hands to build_env_vars().
    forwarded = OmegaConf.to_container(full.run.env, resolve=True)["env_vars"]
    runtime_cfg = extract_train_config(full)

    # --- inside the container ---
    monkeypatch.delenv("EVAL_HARNESS_IMAGE", raising=False)
    for key, value in forwarded.items():
        monkeypatch.setenv(str(key), str(value))
    monkeypatch.setenv("EVAL_ENDPOINT_URL", "https://host/v1/completions")
    monkeypatch.setenv("EVAL_MODEL_HANDLE", "my-model")
    monkeypatch.setenv("EVAL_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("ENDPOINT_TOKEN", "t")
    runtime_cfg.dry_run = True
    run_direct(runtime_cfg, task_filters=["hellaswag"])

    manifest = json.loads((tmp_path / "results" / "run_manifest.dry-run.json").read_text())
    return executor_image, manifest["harness"]


def test_pinned_image_reaches_the_container(tmp_path, monkeypatch):
    """`harness_image` survives extraction, but only as the literal string
    `${oc.env:EVAL_HARNESS_IMAGE,<default-tag>}`. A container that is not sent
    that variable falls back to the default tag, so the executor ran the pinned
    image while the manifest certified a different, unpinned one."""
    digest = "example/harness@sha256:" + "a" * 64
    executor_image, harness = _submit_then_run(monkeypatch, tmp_path, host_image=digest)
    assert executor_image == digest
    assert harness["image"] == digest
    assert harness["image_pinned_by_digest"] is True


def test_a_batch_profile_cannot_change_the_harness_image(tmp_path, monkeypatch):
    """`build_job_config()` re-applies YAML-owned resource keys AFTER merging
    the env.toml profile, so a profile's `container_image` never reaches a
    config that sets one -- as `direct.yaml` does. Documented as such; this
    pins the behaviour the documentation now claims."""
    from nemo_runspec.cli_context import GlobalContext
    from nemo_runspec.config.loader import build_job_config

    host_image = "host/harness@sha256:" + "a" * 64
    monkeypatch.setenv("EVAL_HARNESS_IMAGE", host_image)
    train_cfg = OmegaConf.load(CONFIG_DIR / "direct.yaml")
    profile = OmegaConf.create({"executor": "slurm", "container_image": "profile/harness@sha256:" + "b" * 64})

    job = build_job_config(
        train_cfg,
        GlobalContext(config="direct", batch="p"),
        recipe_name="eval/model_eval",
        script_path="step.py",
        argv=[],
        env_profile=profile,
    )
    assert OmegaConf.to_container(job.run.env, resolve=True)["container_image"] == host_image


def test_cli_override_selects_the_image_and_is_recorded(tmp_path, monkeypatch):
    """The supported per-run escape hatch: CLI overrides land in the YAML
    before the profile is merged, so they do win -- and the manifest has to
    follow the executor, not the host variable."""
    cli_image = "cli/harness@sha256:" + "c" * 64
    executor_image, harness = _submit_then_run(
        monkeypatch,
        tmp_path,
        host_image="host/harness@sha256:" + "a" * 64,
        cli_image=cli_image,
    )
    assert executor_image == cli_image
    assert harness["image"] == cli_image
    assert harness["image_pinned_by_digest"] is True


def test_unpinned_image_is_reported_as_unpinned_end_to_end(tmp_path, monkeypatch):
    _, harness = _submit_then_run(monkeypatch, tmp_path, host_image="example/harness:latest")
    assert harness["image"] == "example/harness:latest"
    assert harness["image_pinned_by_digest"] is False


def test_auto_squash_does_not_run_for_a_submitted_launcher_job():
    """Pins the documented limitation rather than leaving it assumed: the
    runner strips `run.env` before invoking the step, so the squash path is
    unreachable in a submitted run no matter which executor is configured.
    If this ever starts failing, the README and step.toml notes are stale."""
    import nemo_runspec.evaluator as ev
    from nemo_runspec.config.loader import extract_train_config
    from nemotron.steps.eval.model_eval.runtime import _maybe_auto_squash

    full = OmegaConf.load(CONFIG_DIR / "default.yaml")
    full.deployment.checkpoint_path = "/mnt/ckpt/iter_0001000"
    full.run.env.launcher_executor = "slurm"
    full.run.env.tunnel = "ssh"
    runtime_cfg = extract_train_config(full)
    assert not (runtime_cfg.get("run") or {}).get("env")

    original = ev.collect_evaluator_images
    reached = []
    ev.collect_evaluator_images = lambda cfg: reached.append(cfg) or []
    try:
        _maybe_auto_squash(runtime_cfg, dry_run=False)
    finally:
        ev.collect_evaluator_images = original
    assert not reached
