"""Validate super35 RL configs against NeMo-RL's GRPO MasterConfig schema.

Uses NeMo-RL's own resolvers and Pydantic validation (as its
tests/unit/test_config_validation.py does) on the config the shared runner
hands off, with ``${art:...}`` stubbed. Catches keys or types the pinned
NeMo-RL rejects at job start.

Runs only where the pinned NeMo-RL (super-v3.5-posttraining @ f299847f1) is
importable; it installs on CPU with
``pip install --ignore-requires-python -e <nemo-rl checkout>``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

nemo_rl_config = pytest.importorskip("nemo_rl.utils.config")
grpo = pytest.importorskip("nemo_rl.algorithms.grpo")
pydantic = pytest.importorskip("pydantic")

from nemotron.recipes.super35.stage2_rl import _nemo_rl  # noqa: E402

RL_ROOT = Path(__file__).resolve().parents[3] / "src/nemotron/recipes/super35/stage2_rl"
RL_CONFIGS = sorted(RL_ROOT.glob("**/config/*.yaml"))


@pytest.fixture(autouse=True)
def _stub_artifacts(monkeypatch, tmp_path):
    def fake_register(config, **_):
        if OmegaConf.has_resolver("art"):
            OmegaConf.clear_resolver("art")
        OmegaConf.register_new_resolver("art", lambda key, field: str(tmp_path / key / field))

    import nemo_runspec.config.resolvers as resolvers

    monkeypatch.setattr(resolvers, "register_resolvers_from_config", fake_register)
    monkeypatch.setattr(resolvers, "clear_artifact_cache", lambda: None)
    for var in ("NVIDIA_API_KEY", "USER"):
        monkeypatch.setenv(var, "stub")
    nemo_rl_config.register_omegaconf_resolvers()


@pytest.mark.parametrize("config", RL_CONFIGS, ids=[str(p.relative_to(RL_ROOT)) for p in RL_CONFIGS])
def test_handed_off_config_matches_grpo_schema(config):
    handed_off, _ = _nemo_rl.load_config(str(config), [])
    resolved = OmegaConf.to_container(OmegaConf.create(handed_off), resolve=True)
    pydantic.TypeAdapter(grpo.MasterConfig).validate_python(resolved)
