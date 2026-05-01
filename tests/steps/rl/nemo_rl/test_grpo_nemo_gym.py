# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Checks for the RLVR/RLHF NeMo-Gym GRPO dispatch layer."""

from pathlib import Path

from omegaconf import OmegaConf

from nemotron.steps._runners.nemo_rl import (
    load_nemo_rl_step_config,
    should_use_nemo_gym_config,
)
from nemotron.steps._runners.nemo_rl_grpo_nemo_gym import resolve_nemo_gym_data_paths

REPO_ROOT = Path(__file__).resolve().parents[4]
RLVR_CONFIG = REPO_ROOT / "src/nemotron/steps/rl/nemo_rl/rlvr/config"
RLHF_CONFIG = REPO_ROOT / "src/nemotron/steps/rl/nemo_rl/rlhf/config"


def test_rlvr_default_stays_on_upstream_math_example() -> None:
    assert should_use_nemo_gym_config(RLVR_CONFIG / "default.yaml") is False


def test_rlvr_nemo_gym_config_uses_generic_runner() -> None:
    assert should_use_nemo_gym_config(RLVR_CONFIG / "nemo_gym.yaml") is True


def test_rlhf_default_uses_nemo_gym_genrm_runner() -> None:
    assert should_use_nemo_gym_config(RLHF_CONFIG / "default.yaml") is True
    cfg = load_nemo_rl_step_config(RLHF_CONFIG / "default.yaml")
    assert OmegaConf.select(cfg, "env.use_genrm_compare") is True
    assert OmegaConf.select(cfg, "env.nemo_gym.genrm_model.responses_api_models.vllm_model.model")


def test_cli_override_can_disable_nemo_gym_dispatch() -> None:
    assert (
        should_use_nemo_gym_config(
            RLHF_CONFIG / "default.yaml",
            ["env.should_use_nemo_gym=false"],
        )
        is False
    )


def test_local_defaults_loader_merges_relative_yaml(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        "env:\n  should_use_nemo_gym: true\npolicy:\n  model_name: base\n",
        encoding="utf-8",
    )
    (tmp_path / "child.yaml").write_text(
        "defaults: base.yaml\npolicy:\n  model_name: child\n",
        encoding="utf-8",
    )

    cfg = load_nemo_rl_step_config(tmp_path / "child.yaml")
    assert OmegaConf.select(cfg, "env.should_use_nemo_gym") is True
    assert OmegaConf.select(cfg, "policy.model_name") == "child"


def test_resolve_nemo_gym_data_paths_supports_nested_and_flat_forms() -> None:
    nested = {
        "data": {
            "train": {"data_path": "/data/train.jsonl", "num_repeats": 2},
            "validation": {"data_path": "/data/val.jsonl"},
        }
    }
    assert resolve_nemo_gym_data_paths(nested) == ("/data/train.jsonl", "/data/val.jsonl", 2, None)

    flat = {
        "data": {
            "train_jsonl_fpath": "/flat/train.jsonl",
            "validation_jsonl_fpath": "/flat/val.jsonl",
        }
    }
    assert resolve_nemo_gym_data_paths(flat) == ("/flat/train.jsonl", "/flat/val.jsonl", None, None)
