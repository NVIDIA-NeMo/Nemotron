"""Shared fixtures and stage registry for embed recipe tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import nemotron

# Repo root: up from src/nemotron/__init__.py → src → repo root
REPO_ROOT = Path(nemotron.__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Stage registry — single source of truth for all cross-stage tests
# ---------------------------------------------------------------------------
STAGES = [
    {
        "name": "sdg",
        "cli_module": "nemotron.cli.embed.sdg",
        "config_model_path": "nemotron.recipes.embed.stage0_sdg.data_prep",
        "config_model_name": "SDGConfig",
        "yaml_path": REPO_ROOT / "src/nemotron/recipes/embed/stage0_sdg/config/default.yaml",
        "command": "sdg",
    },
    {
        "name": "prep",
        "cli_module": "nemotron.cli.embed.prep",
        "config_model_path": "nemotron.recipes.embed.stage1_data_prep.data_prep",
        "config_model_name": "DataPrepConfig",
        "yaml_path": REPO_ROOT / "src/nemotron/recipes/embed/stage1_data_prep/config/default.yaml",
        "command": "prep",
    },
    {
        "name": "finetune",
        "cli_module": "nemotron.cli.embed.finetune",
        "config_model_path": "nemotron.recipes.embed.stage2_finetune.train",
        "config_model_name": "FinetuneConfig",
        "yaml_path": REPO_ROOT / "src/nemotron/recipes/embed/stage2_finetune/config/default.yaml",
        "command": "finetune",
    },
    {
        "name": "eval",
        "cli_module": "nemotron.cli.embed.eval",
        "config_model_path": "nemotron.recipes.embed.stage3_eval.eval",
        "config_model_name": "EvalConfig",
        "yaml_path": REPO_ROOT / "src/nemotron/recipes/embed/stage3_eval/config/default.yaml",
        "command": "eval",
    },
    {
        "name": "export",
        "cli_module": "nemotron.cli.embed.export",
        "config_model_path": "nemotron.recipes.embed.stage4_export.export",
        "config_model_name": "ExportConfig",
        "yaml_path": REPO_ROOT / "src/nemotron/recipes/embed/stage4_export/config/default.yaml",
        "command": "export",
    },
    {
        "name": "deploy",
        "cli_module": "nemotron.cli.embed.deploy",
        "config_model_path": "nemotron.recipes.embed.stage5_deploy.deploy",
        "config_model_name": "DeployConfig",
        "yaml_path": REPO_ROOT / "src/nemotron/recipes/embed/stage5_deploy/config/default.yaml",
        "command": "deploy",
    },
]


@pytest.fixture()
def repo_root() -> Path:
    """Resolved repository root."""
    return REPO_ROOT


@pytest.fixture(params=STAGES, ids=[s["name"] for s in STAGES])
def stage_info(request: pytest.FixtureRequest) -> dict:
    """Yield one stage dict per parametrized test invocation."""
    return request.param


def _import_config_class(stage: dict):
    """Import and return the Pydantic config class for a stage."""
    import importlib

    mod = importlib.import_module(stage["config_model_path"])
    return getattr(mod, stage["config_model_name"])


def _load_yaml_dict(stage: dict) -> dict:
    """Load a stage's default.yaml, stripping the ``run:`` key."""
    with open(stage["yaml_path"]) as f:
        raw = yaml.safe_load(f)
    raw.pop("run", None)
    return raw


@pytest.fixture(params=STAGES, ids=[s["name"] for s in STAGES])
def config_model_and_yaml(request: pytest.FixtureRequest):
    """Yield (ConfigClass, yaml_dict, stage_name) for each stage."""
    stage = request.param
    cls = _import_config_class(stage)
    yaml_dict = _load_yaml_dict(stage)
    return cls, yaml_dict, stage["name"]
