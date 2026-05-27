# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Focused checks for the NeMo-Gym GRPO runner and configs."""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemotron.steps._runners.nemo_rl import (
    load_nemo_rl_step_config,
    should_use_nemo_gym_config,
)
from nemotron.steps._runners.nemo_rl_grpo_nemo_gym import (
    materialize_nemo_gym_data_manifest,
    materialize_nemo_gym_response_data,
    materialize_nemo_gym_runtime_config,
    set_nemo_gym_validation_size,
    setup_initial_checkpoint,
    validate_async_grpo_config,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
RLVR_CONFIG = REPO_ROOT / "src/nemotron/steps/rl/nemo_rl/rlvr/config"
RLHF_CONFIG = REPO_ROOT / "src/nemotron/steps/rl/nemo_rl/rlhf/config"


def test_nemo_gym_dispatch_configs() -> None:
    assert should_use_nemo_gym_config(RLVR_CONFIG / "default.yaml") is False
    assert should_use_nemo_gym_config(RLVR_CONFIG / "nemo_gym.yaml") is True
    assert should_use_nemo_gym_config(RLHF_CONFIG / "default.yaml") is True
    assert (
        should_use_nemo_gym_config(
            RLHF_CONFIG / "default.yaml",
            ["env.should_use_nemo_gym=false"],
        )
        is False
    )


def test_rlhf_genrm_config_contract() -> None:
    for path in (RLHF_CONFIG / "default.yaml", RLHF_CONFIG / "tiny.yaml"):
        cfg = load_nemo_rl_step_config(path)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        nemo_gym = resolved["env"]["nemo_gym"]

        genrm_impls = nemo_gym["genrm_model"]["responses_api_models"]
        assert resolved["env"]["use_genrm_compare"] is True
        assert list(genrm_impls) == ["genrm_model"]
        assert nemo_gym["genrm_model_name"]
        assert (
            nemo_gym["genrm_compare"]["resources_servers"]["genrm_compare"]["num_rollouts_per_prompt"]
            == resolved["grpo"]["num_generations_per_prompt"]
        )
        assert resolved["data"]["validation"]["repeat"] == resolved["grpo"]["num_generations_per_prompt"]

    tiny = OmegaConf.to_container(load_nemo_rl_step_config(RLHF_CONFIG / "tiny.yaml"), resolve=True)
    assert tiny["cluster"]["num_nodes"] == 1
    assert tiny["policy"]["generation"]["colocated"]["resources"]["num_nodes"] == 1
    assert tiny["env"]["nemo_gym"]["num_gpu_nodes"] == 1


def test_nemo_gym_configs_follow_upstream_runtime_contract() -> None:
    for path in (RLHF_CONFIG / "default.yaml", RLVR_CONFIG / "nemo_gym.yaml"):
        cfg = load_nemo_rl_step_config(path)
        assert OmegaConf.select(cfg, "grpo.max_val_samples") is None
        assert OmegaConf.select(cfg, "grpo.val_batch_size") is None
        assert OmegaConf.select(cfg, "data.max_input_seq_length") is None
        assert OmegaConf.select(cfg, "data.num_workers") == 0
        assert OmegaConf.select(cfg, "env.should_log_nemo_gym_responses") is True
        assert OmegaConf.select(cfg, "env.nemo_gym.rollout_max_attempts_to_avoid_lp_nan") == 1
        assert OmegaConf.select(cfg, "env.nemo_gym.is_trajectory_collection") is False
        assert OmegaConf.select(cfg, "policy.generation.vllm_cfg.async_engine") is False
        assert OmegaConf.select(cfg, "policy.generation.vllm_cfg.enforce_eager") is True
        assert (
            OmegaConf.select(
                cfg,
                "env.nemo_gym.policy_model.responses_api_models.vllm_model.uses_reasoning_parser",
            )
            is False
        )
        assert (
            OmegaConf.select(
                cfg,
                "env.nemo_gym.policy_model.responses_api_models.vllm_model.extra_body.chat_template_kwargs.enable_thinking",
            )
            is False
        )

    rlhf_config_paths = OmegaConf.select(
        load_nemo_rl_step_config(RLHF_CONFIG / "default.yaml"),
        "env.nemo_gym.config_paths",
    )
    assert (
        "resources_servers/single_step_tool_use_with_argument_comparison/configs/"
        "single_step_tool_use_with_argument_comparison.yaml" not in rlhf_config_paths
    )


@pytest.mark.parametrize(
    ("manifest", "allow_train_as_validation", "expected_validation"),
    [
        ({"train": "/data/train.jsonl", "validation": "/data/val.jsonl"}, False, "/data/val.jsonl"),
        ({"train": "/data/train.jsonl"}, True, "/data/train.jsonl"),
    ],
)
def test_materialize_nemo_gym_data_manifest(
    tmp_path: Path,
    manifest: dict[str, str],
    allow_train_as_validation: bool,
    expected_validation: str,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = {
        "data": {
            "manifest_path": str(manifest_path),
            "allow_train_as_validation": allow_train_as_validation,
            "train": {"split": "train"},
            "validation": {"split": "validation"},
        }
    }

    materialize_nemo_gym_data_manifest(config)

    assert config["data"]["train"] == {"split": "train", "data_path": "/data/train.jsonl"}
    assert config["data"]["validation"] == {
        "split": "validation",
        "data_path": expected_validation,
    }
    assert "manifest_path" not in config["data"]
    assert "allow_train_as_validation" not in config["data"]


def test_materialize_nemo_gym_response_data_normalizes_rows(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        json.dumps({"question": "What is 2+2?", "tools": [{"type": "function"}]}) + "\n",
        encoding="utf-8",
    )
    val_path = tmp_path / "val.jsonl"
    val_path.write_text(
        json.dumps({"extra_env_info": {"responses_create_params": {"input": [{"role": "user", "content": "hello"}]}}})
        + "\n",
        encoding="utf-8",
    )
    config = {
        "data": {
            "normalized_data_dir": str(tmp_path / "normalized"),
            "default_agent_ref": "genrm_simple_agent",
            "train": {"data_path": str(train_path)},
            "validation": {"data_path": str(val_path)},
        }
    }

    materialize_nemo_gym_response_data(config)

    normalized_train_path = Path(config["data"]["train"]["data_path"])
    normalized_val_path = Path(config["data"]["validation"]["data_path"])
    train_row = json.loads(normalized_train_path.read_text(encoding="utf-8"))
    val_row = json.loads(normalized_val_path.read_text(encoding="utf-8"))

    assert normalized_train_path.parent == tmp_path / "normalized"
    assert train_row["responses_create_params"] == {
        "input": [{"role": "user", "content": "What is 2+2?"}],
        "tools": [{"type": "function"}],
    }
    assert val_row["responses_create_params"]["input"][0]["content"] == "hello"
    assert train_row["agent_ref"] == {
        "type": "responses_api_agents",
        "name": "genrm_simple_agent",
    }
    assert "default_agent_ref" not in config["data"]


def test_materialize_nemo_gym_response_data_requires_agent_ref(tmp_path: Path) -> None:
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(json.dumps({"question": "hello"}) + "\n", encoding="utf-8")
    config = {"data": {"train": {"data_path": str(data_path)}}}

    with pytest.raises(ValueError, match="agent_ref"):
        materialize_nemo_gym_response_data(config)


def test_set_nemo_gym_validation_size() -> None:
    config = {"grpo": {"max_val_samples": None, "val_batch_size": 2}}
    set_nemo_gym_validation_size(config, [object(), object(), object()])
    assert config["grpo"] == {"max_val_samples": 3, "val_batch_size": 2}

    with pytest.raises(ValueError, match="max_val_samples"):
        set_nemo_gym_validation_size(
            {"grpo": {"max_val_samples": 2, "val_batch_size": 1}},
            [object()],
        )


def test_validate_async_grpo_config_rejects_unsupported_features() -> None:
    config = {
        "grpo": {
            "use_dynamic_sampling": True,
            "reward_scaling": {"enabled": False},
            "reward_shaping": {"enabled": False},
        },
        "data": {"use_multiple_dataloader": False},
    }

    with pytest.raises(NotImplementedError, match="use_dynamic_sampling"):
        validate_async_grpo_config(config)


def test_materialize_nemo_gym_runtime_config_normalizes_genrm_and_policy_alias(
    tmp_path: Path,
) -> None:
    inherited_config = tmp_path / "genrm_compare.yaml"
    inherited_config.write_text(
        """
genrm_simple_agent_reasoning_off:
  responses_api_agents:
    simple_agent:
      model:
        type: responses_api_models
        name: policy_model_reasoning_off
genrm_model:
  responses_api_models:
    genrm_model:
      entrypoint: app.py
      model: ${genrm_model_name}
""",
        encoding="utf-8",
    )
    config = {
        "config_paths": [str(inherited_config)],
        "genrm_model_name": "reward",
        "genrm_model": {
            "responses_api_models": {
                "vllm_model": {
                    "entrypoint": "app.py",
                    "model": "stale",
                    "spinup_server": True,
                }
            }
        },
    }

    materialized = materialize_nemo_gym_runtime_config(
        config,
        policy_model_name="policy",
        policy_base_urls=["http://policy-0/v1"],
    )

    assert "config_paths" not in materialized
    assert "policy_model_reasoning_off" not in config
    assert materialized["genrm_model"]["responses_api_models"] == {
        "genrm_model": {
            "entrypoint": "app.py",
            "model": "reward",
        }
    }

    policy_alias = materialized["policy_model_reasoning_off"]["responses_api_models"]["vllm_model"]
    assert policy_alias["base_url"] == ["http://policy-0/v1"]
    assert policy_alias["model"] == "policy"
    assert policy_alias["return_token_id_information"] is True
    assert policy_alias["uses_reasoning_parser"] is False
    assert policy_alias["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_materialize_nemo_gym_runtime_config_does_not_add_unused_policy_alias() -> None:
    materialized = materialize_nemo_gym_runtime_config(
        {"policy_model": {"responses_api_models": {"vllm_model": {"model": "policy"}}}},
        policy_model_name="policy",
        policy_base_urls=["http://policy-0/v1"],
    )

    assert "policy_model_reasoning_off" not in materialized


def test_materialize_nemo_gym_runtime_config_collapses_generic_duplicate_model_server() -> None:
    materialized = materialize_nemo_gym_runtime_config(
        {
            "reward_model_name": "reward",
            "reward_model": {
                "responses_api_models": {
                    "reward_model": {
                        "entrypoint": "app.py",
                        "model": "${reward_model_name}",
                    },
                    "vllm_model": {
                        "entrypoint": "app.py",
                        "model": "stale",
                    },
                }
            },
        },
        policy_model_name="policy",
        policy_base_urls=["http://policy-0/v1"],
    )

    assert materialized["reward_model"]["responses_api_models"] == {
        "reward_model": {
            "entrypoint": "app.py",
            "model": "reward",
        }
    }


def test_runner_uses_upstream_nemo_rl_gym_primitives() -> None:
    source = (REPO_ROOT / "src/nemotron/steps/_runners/nemo_rl_grpo_nemo_gym.py").read_text(encoding="utf-8")

    assert "setup_response_data" in source
    assert "create_env" in source
    assert "async_grpo_train" in source
    assert "_should_use_async_rollouts" not in source
    assert "CompatNemoGym" not in source
    assert "RolloutCollectionHelper" not in source
    assert "setup_nemo_gym_jsonl_dataset" not in source


def test_setup_initial_checkpoint_writes_complete_marker(tmp_path: Path) -> None:
    source = tmp_path / "source" / "iter_0000001"
    source.mkdir(parents=True)
    (source / "model_optim_rng.pt").write_text("weights", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoints"

    setup_initial_checkpoint(str(source.parent), str(checkpoint_dir))

    step_dir = checkpoint_dir / "step_0"
    target = step_dir / "policy" / "weights" / "model_optim_rng.pt"
    assert target.is_symlink()
    assert target.resolve() == source / "model_optim_rng.pt"
    assert (step_dir / ".complete").read_text(encoding="utf-8") == "ok\n"
    info = json.loads((step_dir / "training_info.json").read_text(encoding="utf-8"))
    assert info["initial_checkpoint"] == str(source.parent)

    setup_initial_checkpoint(str(source.parent), str(checkpoint_dir))


def test_setup_initial_checkpoint_refuses_incomplete_existing_view(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    checkpoint_dir = tmp_path / "checkpoints"
    (checkpoint_dir / "step_0" / "policy" / "weights").mkdir(parents=True)

    with pytest.raises(RuntimeError, match=r"\.complete is missing"):
        setup_initial_checkpoint(str(source), str(checkpoint_dir))
