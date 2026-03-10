"""Contract tests for embed CLI module public API.

Ensures each stage module exports the expected constants and functions,
that CONFIG_DIR points to a real directory with YAML files, and that
CONFIG_MODEL is a Pydantic BaseModel subclass.
"""

from __future__ import annotations

import importlib

import pytest
from pydantic import BaseModel

from .conftest import STAGES

# ---------------------------------------------------------------------------
# Per-stage expected exports
# ---------------------------------------------------------------------------
# Most stages have: SCRIPT_LOCAL, SCRIPT_REMOTE, CONFIG_DIR, CONFIG_MODEL, DEPENDENCIES
# Deploy is special: SCRIPT (no _LOCAL/_REMOTE), no DEPENDENCIES
_STANDARD_CONSTANTS = ["SCRIPT_LOCAL", "SCRIPT_REMOTE", "CONFIG_DIR", "CONFIG_MODEL", "DEPENDENCIES"]
_DEPLOY_CONSTANTS = ["SCRIPT", "CONFIG_DIR", "CONFIG_MODEL"]

MODULE_EXPORTS = [
    (s["cli_module"], _DEPLOY_CONSTANTS if s["name"] == "deploy" else _STANDARD_CONSTANTS)
    for s in STAGES
]


class TestModuleExports:
    @pytest.mark.parametrize(
        "module_path,expected_constants",
        MODULE_EXPORTS,
        ids=[s["name"] for s in STAGES],
    )
    def test_has_expected_constants(self, module_path, expected_constants):
        mod = importlib.import_module(module_path)
        for name in expected_constants:
            assert hasattr(mod, name), f"{module_path} missing constant '{name}'"

    @pytest.mark.parametrize(
        "module_path",
        [s["cli_module"] for s in STAGES],
        ids=[s["name"] for s in STAGES],
    )
    def test_has_callable_command_function(self, module_path):
        mod = importlib.import_module(module_path)
        # Each module exposes a function matching the stage command name
        # e.g. sdg.sdg, prep.prep, deploy.deploy
        func_name = module_path.rsplit(".", 1)[-1]
        assert hasattr(mod, func_name), f"{module_path} missing function '{func_name}'"
        assert callable(getattr(mod, func_name))

    @pytest.mark.parametrize(
        "module_path",
        [s["cli_module"] for s in STAGES],
        ids=[s["name"] for s in STAGES],
    )
    def test_config_dir_exists_with_yaml(self, module_path):
        mod = importlib.import_module(module_path)
        config_dir = mod.CONFIG_DIR
        assert config_dir.is_dir(), f"CONFIG_DIR not a directory: {config_dir}"
        yamls = list(config_dir.glob("*.yaml"))
        assert len(yamls) > 0, f"No .yaml files in {config_dir}"

    @pytest.mark.parametrize(
        "module_path",
        [s["cli_module"] for s in STAGES if s["name"] != "deploy"],
        ids=[s["name"] for s in STAGES if s["name"] != "deploy"],
    )
    def test_script_local_exists(self, module_path):
        mod = importlib.import_module(module_path)
        script = mod.SCRIPT_LOCAL
        assert script.exists(), f"SCRIPT_LOCAL not found: {script}"
        assert script.suffix == ".py"

    @pytest.mark.parametrize(
        "module_path",
        [s["cli_module"] for s in STAGES],
        ids=[s["name"] for s in STAGES],
    )
    def test_config_model_is_pydantic(self, module_path):
        mod = importlib.import_module(module_path)
        assert issubclass(mod.CONFIG_MODEL, BaseModel)
