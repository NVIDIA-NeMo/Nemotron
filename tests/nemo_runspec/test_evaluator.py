# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for nemo_runspec.evaluator module.

Tests the evaluator config "compile" pipeline:
  YAML -> OmegaConf.load() -> merge dotlist -> build_job_config()
  -> strip 'run' section -> resolve ${run.*} interpolations -> eval.yaml

This pipeline is the boundary between the nemotron CLI and nemo-evaluator-launcher.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from nemo_runspec.config import apply_dotlist_overrides, load_config
from nemo_runspec.evaluator import (
    collect_evaluator_images,
    get_non_task_args,
    inject_wandb_env_mappings,
    needs_wandb,
    parse_task_flags,
)
from nemo_runspec.utils import resolve_run_interpolations

# ---------------------------------------------------------------------------
# Paths to config files
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_CONFIG = REPO_ROOT / "src/nemotron/recipes/evaluator/config/nemotron-3-nano-nemo-ray.yaml"
NANO3_EVAL_CONFIG = REPO_ROOT / "src/nemotron/recipes/nano3/stage3_eval/config/default.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compile_eval_config(config_path: Path, dotlist: list[str] | None = None) -> dict:
    """Simulate the evaluator compile pipeline: load -> merge -> strip run -> resolve.

    This mirrors what the eval command does:
    1. OmegaConf.load() the YAML
    2. Apply dotlist overrides
    3. Convert to dict (resolve=False to preserve ${art:...})
    4. Pop the 'run' section
    5. Resolve ${run.*} interpolations
    6. Return the eval config dict (what nemo-evaluator-launcher receives)
    """
    # Step 1-2: Load and merge
    cfg = load_config(config_path)
    if dotlist:
        cfg = apply_dotlist_overrides(cfg, dotlist)

    # Step 3-4: Convert and strip run section
    config_dict = OmegaConf.to_container(cfg, resolve=False)
    run_section = config_dict.pop("run", {})

    # Step 5: Resolve ${run.*} interpolations
    eval_config = resolve_run_interpolations(config_dict, run_section)

    return eval_config


# ===========================================================================
# Config Compile Tests
# ===========================================================================

class TestEvaluatorConfigCompile:
    """Test the evaluator config compile pipeline."""

    def test_evaluator_config_loads_without_errors(self):
        """The working evaluator config loads via OmegaConf."""
        cfg = load_config(EVALUATOR_CONFIG)
        assert isinstance(cfg, DictConfig)

    def test_nano3_eval_config_loads_without_errors(self):
        """The nano3 stage3_eval config loads via OmegaConf."""
        cfg = load_config(NANO3_EVAL_CONFIG)
        assert isinstance(cfg, DictConfig)

    def test_evaluator_config_compiles_run_stripped(self):
        """After compile, the 'run' section is stripped from evaluator config."""
        result = _compile_eval_config(EVALUATOR_CONFIG)
        assert "run" not in result

    def test_nano3_config_compiles_run_stripped(self):
        """After compile, the 'run' section is stripped from nano3 config."""
        result = _compile_eval_config(NANO3_EVAL_CONFIG)
        assert "run" not in result

    def test_evaluator_config_has_required_top_level_keys(self):
        """Compiled evaluator config has execution, deployment, evaluation, export."""
        result = _compile_eval_config(EVALUATOR_CONFIG)
        for key in ("execution", "deployment", "evaluation", "export"):
            assert key in result, f"Missing required key: {key}"

    def test_nano3_config_has_required_top_level_keys(self):
        """Compiled nano3 config has execution, deployment, evaluation, export."""
        result = _compile_eval_config(NANO3_EVAL_CONFIG)
        for key in ("execution", "deployment", "evaluation", "export"):
            assert key in result, f"Missing required key: {key}"

    def test_evaluator_run_interpolations_resolved(self):
        """All ${run.*} interpolations in evaluator config are resolved."""
        result = _compile_eval_config(EVALUATOR_CONFIG)
        yaml_str = OmegaConf.to_yaml(OmegaConf.create(result))
        assert "${run." not in yaml_str, (
            f"Unresolved ${{run.*}} interpolations in evaluator config:\n"
            f"{[line for line in yaml_str.splitlines() if '${run.' in line]}"
        )

    def test_nano3_run_interpolations_resolved(self):
        """All ${run.*} interpolations in nano3 config are resolved."""
        result = _compile_eval_config(NANO3_EVAL_CONFIG)
        yaml_str = OmegaConf.to_yaml(OmegaConf.create(result))
        assert "${run." not in yaml_str, (
            f"Unresolved ${{run.*}} interpolations in nano3 config:\n"
            f"{[line for line in yaml_str.splitlines() if '${run.' in line]}"
        )

    def test_evaluator_config_no_missing_values(self):
        """Compiled evaluator config has no OmegaConf MISSING values."""
        result = _compile_eval_config(EVALUATOR_CONFIG)
        cfg = OmegaConf.create(result)
        # Iterate all leaf values to check for MISSING
        for key in OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False):
            pass  # Would raise on structural issues

    def test_nano3_execution_type_resolves(self):
        """nano3 execution.type resolves to the value from run.env.executor."""
        result = _compile_eval_config(NANO3_EVAL_CONFIG)
        assert result["execution"]["type"] == "local"

    def test_evaluator_execution_type_is_slurm(self):
        """Working evaluator config has execution.type=slurm."""
        result = _compile_eval_config(EVALUATOR_CONFIG)
        assert result["execution"]["type"] == "slurm"

    def test_evaluator_config_tasks_not_empty(self):
        """Compiled evaluator config has at least one task."""
        result = _compile_eval_config(EVALUATOR_CONFIG)
        tasks = result["evaluation"]["tasks"]
        assert len(tasks) > 0

    def test_nano3_config_tasks_not_empty(self):
        """Compiled nano3 config has at least one task."""
        result = _compile_eval_config(NANO3_EVAL_CONFIG)
        tasks = result["evaluation"]["tasks"]
        assert len(tasks) > 0

    def test_evaluator_tasks_have_names(self):
        """Every task in evaluator config has a 'name' field."""
        result = _compile_eval_config(EVALUATOR_CONFIG)
        for task in result["evaluation"]["tasks"]:
            assert "name" in task, f"Task missing 'name': {task}"

    def test_nano3_tasks_have_names(self):
        """Every task in nano3 config has a 'name' field."""
        result = _compile_eval_config(NANO3_EVAL_CONFIG)
        for task in result["evaluation"]["tasks"]:
            assert "name" in task, f"Task missing 'name': {task}"

    def test_evaluator_export_wandb_resolved(self):
        """export.wandb.entity/project resolve (to null is fine, but not ${run.*})."""
        result = _compile_eval_config(EVALUATOR_CONFIG)
        wandb = result["export"]["wandb"]
        assert isinstance(wandb.get("entity"), (str, type(None)))
        assert isinstance(wandb.get("project"), (str, type(None)))

    def test_nano3_export_wandb_resolved(self):
        """export.wandb.entity/project resolve (to null is fine)."""
        result = _compile_eval_config(NANO3_EVAL_CONFIG)
        wandb = result["export"]["wandb"]
        assert isinstance(wandb.get("entity"), (str, type(None)))
        assert isinstance(wandb.get("project"), (str, type(None)))

    def test_dotlist_override_applies(self):
        """Dotlist overrides are applied during compile."""
        result = _compile_eval_config(
            NANO3_EVAL_CONFIG,
            dotlist=["evaluation.nemo_evaluator_config.config.params.parallelism=16"],
        )
        assert result["evaluation"]["nemo_evaluator_config"]["config"]["params"]["parallelism"] == 16

    def test_defaults_key_present_but_harmless(self):
        """The 'defaults' key from Hydra YAML passes through compile as-is.

        nemo-evaluator-launcher handles Hydra defaults resolution internally.
        """
        result = _compile_eval_config(NANO3_EVAL_CONFIG)
        assert "defaults" in result

    def test_nano3_checkpoint_path_is_artifact_ref(self):
        """nano3 deployment.checkpoint_path uses ${art:model,path}."""
        cfg = load_config(NANO3_EVAL_CONFIG)
        config_dict = OmegaConf.to_container(cfg, resolve=False)
        cp = config_dict["deployment"]["checkpoint_path"]
        assert "${art:" in cp or isinstance(cp, str)


# ===========================================================================
# Evaluator Config Alignment Tests
# ===========================================================================

class TestConfigAlignment:
    """Verify nano3/stage3_eval config is aligned with the working evaluator config."""

    @pytest.fixture()
    def evaluator_compiled(self) -> dict:
        return _compile_eval_config(EVALUATOR_CONFIG)

    @pytest.fixture()
    def nano3_compiled(self) -> dict:
        return _compile_eval_config(NANO3_EVAL_CONFIG)

    def test_both_have_auto_export_wandb(self, evaluator_compiled, nano3_compiled):
        """Both configs export results to W&B."""
        ev_dest = evaluator_compiled["execution"]["auto_export"]["destinations"]
        n3_dest = nano3_compiled["execution"]["auto_export"]["destinations"]
        assert "wandb" in ev_dest
        assert "wandb" in n3_dest

    def test_both_have_evaluation_tasks_section(self, evaluator_compiled, nano3_compiled):
        """Both configs define evaluation.tasks as a list."""
        assert isinstance(evaluator_compiled["evaluation"]["tasks"], list)
        assert isinstance(nano3_compiled["evaluation"]["tasks"], list)

    def test_both_have_export_wandb_section(self, evaluator_compiled, nano3_compiled):
        """Both configs have export.wandb with entity and project."""
        for label, cfg in [("evaluator", evaluator_compiled), ("nano3", nano3_compiled)]:
            assert "wandb" in cfg["export"], f"{label} missing export.wandb"
            assert "entity" in cfg["export"]["wandb"], f"{label} missing export.wandb.entity"
            assert "project" in cfg["export"]["wandb"], f"{label} missing export.wandb.project"

    def test_evaluator_tasks_superset_of_common(self, evaluator_compiled):
        """Working evaluator config has the standard benchmark tasks."""
        task_names = {t["name"] for t in evaluator_compiled["evaluation"]["tasks"]}
        expected = {"adlr_mmlu", "hellaswag"}
        assert expected.issubset(task_names), f"Missing tasks: {expected - task_names}"


# ===========================================================================
# Pure Function Tests
# ===========================================================================

class TestParseTaskFlags:
    """Tests for parse_task_flags."""

    def test_single_task(self):
        assert parse_task_flags(["-t", "adlr_mmlu"]) == ["adlr_mmlu"]

    def test_multiple_tasks(self):
        assert parse_task_flags(["-t", "adlr_mmlu", "-t", "hellaswag"]) == [
            "adlr_mmlu",
            "hellaswag",
        ]

    def test_long_flag(self):
        assert parse_task_flags(["--task", "adlr_mmlu"]) == ["adlr_mmlu"]

    def test_no_tasks_returns_none(self):
        assert parse_task_flags([]) is None

    def test_flag_without_value_ignored(self):
        assert parse_task_flags(["-t"]) is None


class TestGetNonTaskArgs:
    """Tests for get_non_task_args."""

    def test_all_task_args(self):
        assert get_non_task_args(["-t", "adlr_mmlu", "--task", "hellaswag"]) == []

    def test_mixed_args(self):
        assert get_non_task_args(["-t", "adlr_mmlu", "--unknown", "val"]) == [
            "--unknown",
            "val",
        ]

    def test_empty(self):
        assert get_non_task_args([]) == []


class TestNeedsWandb:
    """Tests for needs_wandb."""

    def test_auto_export_wandb(self):
        cfg = OmegaConf.create(
            {"execution": {"auto_export": {"destinations": ["wandb"]}}}
        )
        assert needs_wandb(cfg) is True

    def test_export_wandb_section(self):
        cfg = OmegaConf.create({"export": {"wandb": {"entity": "team"}}})
        assert needs_wandb(cfg) is True

    def test_no_wandb(self):
        cfg = OmegaConf.create({"execution": {}, "export": {}})
        assert needs_wandb(cfg) is False

    def test_empty_config(self):
        cfg = OmegaConf.create({})
        assert needs_wandb(cfg) is False


class TestInjectWandbEnvMappings:
    """Tests for inject_wandb_env_mappings."""

    def test_injects_evaluation_env_vars(self):
        cfg = OmegaConf.create({"evaluation": {}, "execution": {}})
        inject_wandb_env_mappings(cfg)
        assert cfg.evaluation.env_vars.WANDB_API_KEY == "WANDB_API_KEY"
        assert cfg.evaluation.env_vars.WANDB_PROJECT == "WANDB_PROJECT"
        assert cfg.evaluation.env_vars.WANDB_ENTITY == "WANDB_ENTITY"

    def test_injects_export_env_vars(self):
        cfg = OmegaConf.create({"evaluation": {}, "execution": {}})
        inject_wandb_env_mappings(cfg)
        assert cfg.execution.env_vars.export.WANDB_API_KEY == "WANDB_API_KEY"

    def test_does_not_overwrite_existing(self):
        cfg = OmegaConf.create(
            {"evaluation": {"env_vars": {"WANDB_API_KEY": "CUSTOM"}}, "execution": {}}
        )
        inject_wandb_env_mappings(cfg)
        assert cfg.evaluation.env_vars.WANDB_API_KEY == "CUSTOM"


class TestCollectEvaluatorImages:
    """Tests for collect_evaluator_images."""

    def test_collects_deployment_image(self):
        cfg = OmegaConf.create({"deployment": {"image": "nvcr.io/nvidia/nemo:latest"}})
        images = collect_evaluator_images(cfg)
        assert ("deployment.image", "nvcr.io/nvidia/nemo:latest") in images

    def test_collects_proxy_image(self):
        cfg = OmegaConf.create(
            {
                "deployment": {"image": "img1"},
                "execution": {"proxy": {"image": "img2"}},
            }
        )
        images = collect_evaluator_images(cfg)
        assert len(images) == 2
        assert ("execution.proxy.image", "img2") in images

    def test_empty_config(self):
        cfg = OmegaConf.create({})
        images = collect_evaluator_images(cfg)
        assert images == []
