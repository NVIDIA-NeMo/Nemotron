"""Integration tests: YAML config files vs Pydantic model schemas.

These are the highest-value tests in the suite. The ``extra="forbid"`` on
every config model means that loading YAML through the model catches field
renames, additions, typos, and type mismatches — the #1 refactoring bug.
"""

from __future__ import annotations

import yaml


class TestYAMLConfigCompat:
    """Parametrized across all 6 stages via the config_model_and_yaml fixture."""

    def test_default_yaml_parses_through_model(self, config_model_and_yaml):
        cls, yaml_dict, stage_name = config_model_and_yaml
        instance = cls(**yaml_dict)
        assert instance is not None

    def test_yaml_values_accessible_as_attributes(self, config_model_and_yaml):
        cls, yaml_dict, stage_name = config_model_and_yaml
        instance = cls(**yaml_dict)
        for key in yaml_dict:
            assert hasattr(instance, key), f"Stage {stage_name}: attribute '{key}' missing"

    def test_yaml_files_exist(self, stage_info):
        assert stage_info["yaml_path"].exists(), (
            f"Stage {stage_info['name']}: default.yaml not found at {stage_info['yaml_path']}"
        )

    def test_yaml_is_valid_yaml(self, stage_info):
        with open(stage_info["yaml_path"]) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), (
            f"Stage {stage_info['name']}: default.yaml did not parse as dict"
        )
