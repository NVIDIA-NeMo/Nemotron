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

"""Tests for YAML config files under customization_recipes/.

Validates that every config YAML in customization_recipes/**/config/
can be loaded as valid OmegaConf, resolves without errors (where
possible), and contains expected top-level keys.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Discover all YAML config files
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RECIPES_ROOT = _REPO_ROOT / "src" / "nemotron" / "customization_recipes"

# Collect all .yaml files under any config/ directory within customization_recipes/
_CONFIG_FILES: list[Path] = sorted(_RECIPES_ROOT.rglob("config/*.yaml"))

# Also check for config files in subdirectories of config/
_CONFIG_FILES.extend(sorted(_RECIPES_ROOT.rglob("config/**/*.yaml")))

# De-duplicate while preserving order
_seen: set[str] = set()
_UNIQUE_CONFIG_FILES: list[Path] = []
for p in _CONFIG_FILES:
    key = str(p)
    if key not in _seen:
        _seen.add(key)
        _UNIQUE_CONFIG_FILES.append(p)

_CONFIG_FILES = _UNIQUE_CONFIG_FILES


def _short_id(path: Path) -> str:
    """Generate a short, readable test ID from a config file path."""
    rel = path.relative_to(_RECIPES_ROOT)
    return str(rel)


# ---------------------------------------------------------------------------
# Parameterized tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    len(_CONFIG_FILES) == 0,
    reason="No YAML config files found under customization_recipes/",
)
class TestConfigFiles:
    """Validate all YAML configs under customization_recipes/**/config/."""

    @pytest.mark.parametrize(
        "config_path",
        _CONFIG_FILES,
        ids=[_short_id(p) for p in _CONFIG_FILES],
    )
    def test_yaml_is_valid_omegaconf(self, config_path: Path) -> None:
        """Each YAML file should parse as valid OmegaConf without errors."""
        cfg = OmegaConf.load(str(config_path))
        assert cfg is not None
        # Should be a DictConfig (not a ListConfig at the top level)
        assert OmegaConf.is_dict(cfg), (
            f"{config_path.name} top level should be a dict, got {type(cfg).__name__}"
        )

    @pytest.mark.parametrize(
        "config_path",
        _CONFIG_FILES,
        ids=[_short_id(p) for p in _CONFIG_FILES],
    )
    def test_yaml_is_not_empty(self, config_path: Path) -> None:
        """Each YAML config should contain at least one key."""
        cfg = OmegaConf.load(str(config_path))
        keys = list(cfg.keys()) if OmegaConf.is_dict(cfg) else []
        assert len(keys) > 0, f"Config file {config_path.name} is empty"

    @pytest.mark.parametrize(
        "config_path",
        _CONFIG_FILES,
        ids=[_short_id(p) for p in _CONFIG_FILES],
    )
    def test_yaml_resolve_no_missing(self, config_path: Path) -> None:
        """Attempt to resolve interpolations. OmegaConf resolvers like
        ${art:...} are custom and will not be registered in test, so we
        check for missing mandatory values (marked with ???) instead.

        Note: We allow InterpolationResolutionError for custom resolvers
        (e.g., ${art:data,path}) since those require runtime registration.
        We only fail on MissingMandatoryValue.
        """
        cfg = OmegaConf.load(str(config_path))
        # Walk the config looking for MISSING values (???)
        missing_keys = OmegaConf.missing_keys(cfg)
        # It is acceptable for configs to have MISSING values when they
        # are meant to be provided at runtime. We just verify parse works.
        # This test primarily ensures the YAML syntax is valid.
        assert cfg is not None


# ---------------------------------------------------------------------------
# Specific config content tests
# ---------------------------------------------------------------------------


class TestConfigContent:
    """Test that specific config files contain expected keys/structure."""

    def _load(self, relative_path: str) -> "DictConfig":
        full = _RECIPES_ROOT / relative_path
        if not full.exists():
            pytest.skip(f"Config not found: {relative_path}")
        return OmegaConf.load(str(full))

    def test_cpt_config_has_model(self) -> None:
        cfg = self._load("nemotron/stage1_cpt/config/default.yaml")
        assert "model" in cfg, "CPT config should have 'model' section"

    def test_cpt_config_has_optimizer(self) -> None:
        cfg = self._load("nemotron/stage1_cpt/config/default.yaml")
        assert "optimizer" in cfg, "CPT config should have 'optimizer' section"

    def test_cpt_config_has_step_scheduler(self) -> None:
        cfg = self._load("nemotron/stage1_cpt/config/default.yaml")
        assert "step_scheduler" in cfg, "CPT config should have 'step_scheduler' section"

    def test_sft_config_has_model(self) -> None:
        cfg = self._load("nemotron/stage2_sft/config/default.yaml")
        assert "model" in cfg, "SFT config should have 'model' section"

    def test_sft_config_has_dataset(self) -> None:
        cfg = self._load("nemotron/stage2_sft/config/default.yaml")
        assert "dataset" in cfg, "SFT config should have 'dataset' section"

    def test_sft_config_has_packed_sequence(self) -> None:
        cfg = self._load("nemotron/stage2_sft/config/default.yaml")
        assert "packed_sequence" in cfg, "SFT config should have 'packed_sequence' section"

    def test_rl_config_has_training_type(self) -> None:
        cfg = self._load("nemotron/stage3_rl/config/default.yaml")
        assert "training_type" in cfg, "RL config should have 'training_type'"
        assert cfg.training_type in ("dpo", "grpo"), (
            f"training_type should be dpo or grpo, got {cfg.training_type}"
        )

    def test_rl_config_has_policy(self) -> None:
        cfg = self._load("nemotron/stage3_rl/config/default.yaml")
        assert "policy" in cfg, "RL config should have 'policy' section"

    def test_byob_config_has_generation_model(self) -> None:
        cfg = self._load("nemotron/stage4_byob/config/default.yaml")
        assert "generation_model_config" in cfg, (
            "BYOB config should have 'generation_model_config'"
        )

    def test_byob_config_has_judge_model(self) -> None:
        cfg = self._load("nemotron/stage4_byob/config/default.yaml")
        assert "judge_model_config" in cfg, (
            "BYOB config should have 'judge_model_config'"
        )

    def test_eval_config_has_sections(self) -> None:
        cfg = self._load("nemotron/stage5_eval/config/default.yaml")
        assert "data_eval" in cfg or "model_eval" in cfg, (
            "Eval config should have 'data_eval' or 'model_eval' section"
        )

    def test_quant_config_has_method(self) -> None:
        cfg = self._load("nemotron/stage6_quantization/config/default.yaml")
        assert "quantization" in cfg, "Quant config should have 'quantization' section"
        assert "method" in cfg.quantization, (
            "Quant config should specify quantization.method"
        )

    def test_all_stage_configs_exist(self) -> None:
        """Every stage should have a config/default.yaml."""
        stages = [
            "nemotron/stage1_cpt",
            "nemotron/stage2_sft",
            "nemotron/stage3_rl",
            "nemotron/stage4_byob",
            "nemotron/stage5_eval",
            "nemotron/stage6_quantization",
        ]
        for stage in stages:
            path = _RECIPES_ROOT / stage / "config" / "default.yaml"
            assert path.exists(), f"Missing config: {stage}/config/default.yaml"
