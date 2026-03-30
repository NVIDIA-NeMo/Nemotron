"""Contract tests for embed CLI module public API.

Ensures each stage module exports the expected constants and META object,
that config directories contain YAML files, and that script paths exist.
"""

from __future__ import annotations

import importlib

import pytest
from pydantic import BaseModel

from .conftest import STAGES

# ---------------------------------------------------------------------------
# Per-stage expected exports
# ---------------------------------------------------------------------------
# All stages export: SCRIPT_PATH, SPEC, META
# Most also export SCRIPT_REMOTE; deploy does not
_STANDARD_CONSTANTS = ["SCRIPT_PATH", "SCRIPT_REMOTE", "SPEC", "META"]
_DEPLOY_CONSTANTS = ["SCRIPT_PATH", "SPEC", "META"]


class TestModuleExports:
    @pytest.mark.parametrize(
        "stage",
        STAGES,
        ids=[s["name"] for s in STAGES],
    )
    def test_has_expected_constants(self, stage):
        mod = importlib.import_module(stage["cli_module"])
        expected = _DEPLOY_CONSTANTS if stage["name"] == "deploy" else _STANDARD_CONSTANTS
        for name in expected:
            assert hasattr(mod, name), f"{stage['cli_module']} missing constant '{name}'"

    @pytest.mark.parametrize(
        "stage",
        STAGES,
        ids=[s["name"] for s in STAGES],
    )
    def test_meta_has_config_model(self, stage):
        mod = importlib.import_module(stage["cli_module"])
        meta = mod.META
        assert hasattr(meta, "config_model"), f"{stage['cli_module']} META missing config_model"
        assert issubclass(meta.config_model, BaseModel)

    @pytest.mark.parametrize(
        "stage",
        STAGES,
        ids=[s["name"] for s in STAGES],
    )
    def test_meta_has_config_dir(self, stage):
        from pathlib import Path

        mod = importlib.import_module(stage["cli_module"])
        meta = mod.META
        config_dir = Path(meta.config_dir)
        assert config_dir.is_dir(), f"META.config_dir not a directory: {config_dir}"
        yamls = list(config_dir.glob("*.yaml"))
        assert len(yamls) > 0, f"No .yaml files in {config_dir}"

    @pytest.mark.parametrize(
        "stage",
        [s for s in STAGES if s["name"] != "deploy"],
        ids=[s["name"] for s in STAGES if s["name"] != "deploy"],
    )
    def test_script_path_exists(self, stage):
        from pathlib import Path

        mod = importlib.import_module(stage["cli_module"])
        script = Path(mod.SCRIPT_PATH)
        assert script.exists(), f"SCRIPT_PATH not found: {script}"
        assert script.suffix == ".py"
