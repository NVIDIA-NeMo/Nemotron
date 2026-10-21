"""Lepton jobs must keep their logs after they terminate.

`LeptonExecutor.create_lepton_job` hardcodes `log=None`, so a job that dies
during startup left `lep job log` with nothing but "Connection stopped." —
recovering a traceback meant re-creating the job byte-for-byte with `-lg true`.
"""

from __future__ import annotations

import pytest


def _lepton_module():
    return pytest.importorskip("nemo_run.core.execution.lepton")


def _patch_fresh(lep_mod):
    """Re-apply the patch over a spy, so the spy becomes the wrapped call."""
    from nemo_runspec.run import patch_lepton_enable_log_collection

    lep_mod.LeptonExecutor._nemotron_log_collection_patched = False
    patch_lepton_enable_log_collection()


def test_log_collection_is_enabled_on_the_job_spec(monkeypatch):
    lep_mod = _lepton_module()
    pytest.importorskip("leptonai.api.v1.types.deployment")

    captured = {}

    def spy(self, name):
        # Built with whatever `LeptonJobUserSpec` is bound at call time --
        # which is what the patch swaps.
        captured["spec"] = lep_mod.LeptonJobUserSpec(resource_shape="cpu.small")
        return captured["spec"]

    monkeypatch.setattr(lep_mod.LeptonExecutor, "create_lepton_job", spy, raising=False)
    _patch_fresh(lep_mod)

    ex = lep_mod.LeptonExecutor(container_image="img", nemo_run_dir="/d", node_group="ng")
    ex.create_lepton_job("job-name")

    log = captured["spec"].log
    assert log is not None, "job spec still has log=None; logs vanish on termination"
    assert log.enable_collection is True
    assert log.save_termination_logs is True


def test_an_explicit_log_setting_is_not_overwritten(monkeypatch):
    """If a future nemo-run release sets `log` itself, it keeps winning."""
    lep_mod = _lepton_module()
    deployment = pytest.importorskip("leptonai.api.v1.types.deployment")

    captured = {}
    explicit = deployment.LeptonLog(enable_collection=False)

    def spy(self, name):
        captured["spec"] = lep_mod.LeptonJobUserSpec(resource_shape="cpu.small", log=explicit)
        return captured["spec"]

    monkeypatch.setattr(lep_mod.LeptonExecutor, "create_lepton_job", spy, raising=False)
    _patch_fresh(lep_mod)

    ex = lep_mod.LeptonExecutor(container_image="img", nemo_run_dir="/d", node_group="ng")
    ex.create_lepton_job("job-name")
    assert captured["spec"].log.enable_collection is False


def test_patch_is_idempotent():
    """`plan_for()` runs on every submission; double-wrapping would nest."""
    from nemo_runspec.run import patch_lepton_enable_log_collection

    lep_mod = _lepton_module()
    patch_lepton_enable_log_collection()
    first = lep_mod.LeptonExecutor.create_lepton_job
    patch_lepton_enable_log_collection()
    assert lep_mod.LeptonExecutor.create_lepton_job is first
