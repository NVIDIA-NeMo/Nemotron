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

"""Lean NeMo-Gym GRPO runner for RLVR/RLHF steps.

This runner intentionally follows NeMo-RL's upstream
``examples/nemo_gym/run_grpo_nemo_gym.py`` flow. Nemotron owns config/artifact
resolution and small launch preflight; NeMo-RL owns datasets, environment
creation, rollout collection, and GRPO training.
"""

from __future__ import annotations

import hashlib
import json
import os
import pprint
from copy import deepcopy
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from nemotron.steps._runners.nemo_rl import load_nemo_rl_step_config


def run_nemo_gym_grpo(*, config_path: Path, overrides: list[str] | None = None) -> None:
    """Run NeMo-RL GRPO with a NeMo-Gym environment."""
    _register_mul_resolver()
    config_omega = load_nemo_rl_step_config(Path(config_path), overrides or [])
    print(f"Loaded configuration from: {config_path}")
    if overrides:
        print(f"Overrides: {overrides}")

    from nemo_runspec.config.resolvers import clear_artifact_cache, register_resolvers_from_config

    clear_artifact_cache()
    register_resolvers_from_config(
        config_omega,
        artifacts_key="run",
        mode="pre_init",
        pre_init_patch_http_digest=False,
    )

    config: dict[str, Any] = OmegaConf.to_container(config_omega, resolve=True)
    print("Applied CLI overrides")

    _maybe_chdir_to_nemo_rl_workdir(config)
    materialize_nemo_gym_data_manifest(config)
    materialize_nemo_gym_response_data(config)
    _patch_wandb(config)

    import ray
    from nemo_rl.algorithms.grpo import (
        _should_use_nemo_gym,
        async_grpo_train,
        grpo_train,
        setup,
    )
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.utils import setup_response_data
    from nemo_rl.distributed.virtual_cluster import init_ray
    from nemo_rl.environments.nemo_gym import NemoGymConfig, setup_nemo_gym_config
    from nemo_rl.environments.utils import create_env
    from nemo_rl.models.generation import configure_generation_config
    from nemo_rl.utils.logger import get_next_experiment_dir

    _setup_initial_policy(config)
    _setup_log_dirs(config, get_next_experiment_dir)

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"].get("generation") is not None, "A generation config is required for GRPO"
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"],
        tokenizer,
    )

    setup_nemo_gym_config(config, tokenizer)
    assert _should_use_nemo_gym(config), "Set env.should_use_nemo_gym=true for this runner"

    print("\nSetting up NeMo-Gym response data...")
    train_dataset, val_dataset = setup_response_data(
        tokenizer=tokenizer,
        data_config=config["data"],
        env_configs=None,
    )
    set_nemo_gym_validation_size(config, val_dataset)

    print("Final config:")
    pprint.pprint(config)

    init_ray()
    (
        policy,
        policy_generation,
        _cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    is_trajectory_collection = config["env"]["nemo_gym"].pop("is_trajectory_collection", False) or False
    initial_global_config_dict = materialize_nemo_gym_runtime_config(
        config["env"]["nemo_gym"],
        policy_model_name=policy_generation.cfg["model_name"],
        policy_base_urls=policy_generation.dp_openai_server_base_urls,
    )
    _log_nemo_gym_runtime_summary(config, initial_global_config_dict)
    nemo_gym_config = NemoGymConfig(
        model_name=policy_generation.cfg["model_name"],
        base_urls=policy_generation.dp_openai_server_base_urls,
        initial_global_config_dict=initial_global_config_dict,
    )
    nemo_gym = create_env(
        env_name="nemo_gym",
        env_config=nemo_gym_config,
    )
    ray.get(nemo_gym.health_check.remote())

    task_to_env = {"nemo_gym": nemo_gym}
    val_task_to_env = task_to_env

    if is_trajectory_collection:
        _collect_trajectories(
            policy=policy,
            policy_generation=policy_generation,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            val_task_to_env=val_task_to_env,
            logger=logger,
            master_config=master_config,
        )
        return

    if config["grpo"].get("async_grpo", {}).get("enabled", False):
        validate_async_grpo_config(config)
        print("Running async GRPO training")
        async_config = config["grpo"]["async_grpo"]
        async_grpo_train(
            policy=policy,
            policy_generation=policy_generation,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            task_to_env=task_to_env,
            val_task_to_env=val_task_to_env,
            logger=logger,
            checkpointer=checkpointer,
            grpo_save_state=grpo_state,
            master_config=master_config,
            max_trajectory_age_steps=async_config["max_trajectory_age_steps"],
        )
        return

    print("Running synchronous GRPO training")
    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        val_task_to_env,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


def set_nemo_gym_validation_size(config: dict[str, Any], val_dataset: Any) -> None:
    """Use the validation dataset directly, matching NeMo-RL's gym example."""
    if config["grpo"]["max_val_samples"] is not None:
        raise ValueError(
            "A non-null `grpo.max_val_samples` parameter is not supported for "
            "NeMo-Gym response data. The validation set is consumed directly. "
            "Use `grpo.val_batch_size` to cap validation rollout batches."
        )

    if val_dataset is None:
        return

    val_size = len(val_dataset)
    val_batch_size = config["grpo"].get("val_batch_size")
    if val_batch_size is None:
        val_batch_size = val_size
    if val_batch_size <= 0:
        raise ValueError("`grpo.val_batch_size` must be positive when set")
    val_batch_size = min(val_batch_size, val_size)

    print(
        "Setting `grpo.max_val_samples` to the validation dataset length "
        f"({val_size}) and `grpo.val_batch_size` to {val_batch_size}"
    )
    config["grpo"]["max_val_samples"] = val_size
    config["grpo"]["val_batch_size"] = val_batch_size


def validate_async_grpo_config(config: dict[str, Any]) -> None:
    """Reject async GRPO combinations that upstream NeMo-RL does not support."""
    unsupported_features = [
        "use_dynamic_sampling",
        "reward_scaling",
        "reward_shaping",
    ]
    for feature in unsupported_features:
        if feature not in config["grpo"]:
            continue
        if feature == "use_dynamic_sampling":
            if config["grpo"][feature]:
                raise NotImplementedError(f"{feature} is not supported with async GRPO")
            continue
        if config["grpo"][feature]["enabled"]:
            raise NotImplementedError(f"{feature} is not supported with async GRPO")

    if config["data"].get("use_multiple_dataloader", False):
        raise NotImplementedError("use_multiple_dataloader is not supported with async GRPO")


def materialize_nemo_gym_runtime_config(
    nemo_gym_config: dict[str, Any],
    *,
    policy_model_name: str,
    policy_base_urls: str | list[str],
) -> dict[str, Any]:
    """Materialize Gym config-paths and fill runtime policy model aliases.

    NeMo-RL allocates the policy OpenAI-compatible server at runtime. Some Gym
    resource configs reference a second policy server with reasoning disabled;
    when that reference is present, materialize it from the runtime policy URL.
    """
    materialized = materialize_nemo_gym_config_paths(nemo_gym_config)
    _normalize_responses_api_model_servers(materialized)
    _ensure_policy_model_alias(
        materialized,
        alias_name="policy_model_reasoning_off",
        policy_model_name=policy_model_name,
        policy_base_urls=policy_base_urls,
    )
    return materialized


def _ensure_policy_model_alias(
    nemo_gym_config: dict[str, Any],
    *,
    alias_name: str,
    policy_model_name: str,
    policy_base_urls: str | list[str],
) -> None:
    if alias_name in nemo_gym_config:
        return
    if alias_name not in _collect_responses_api_model_refs(nemo_gym_config):
        return

    nemo_gym_config[alias_name] = {
        "responses_api_models": {
            "vllm_model": {
                "entrypoint": "app.py",
                "base_url": policy_base_urls,
                "api_key": "dummy_key",
                "model": policy_model_name,
                "return_token_id_information": True,
                "uses_reasoning_parser": False,
                "extra_body": {
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                    }
                },
            }
        }
    }
    print(
        f"Materialized runtime policy model alias `{alias_name}` from policy_generation",
        flush=True,
    )


def _normalize_responses_api_model_servers(nemo_gym_config: dict[str, Any]) -> None:
    """Keep one concrete implementation per ``responses_api_models`` server.

    Gym config-paths can define a server implementation while the step config
    overrides the same top-level server with a different implementation name.
    The merged shape has two implementations under one server and fails Gym
    validation, so collapse each server to one deterministic implementation.
    """
    for server_name, server_config in list(nemo_gym_config.items()):
        if not isinstance(server_config, dict):
            continue
        model_impls = server_config.get("responses_api_models")
        if not isinstance(model_impls, dict):
            continue
        impl_name = _select_responses_api_model_impl(server_name, model_impls)
        if impl_name is None:
            continue

        model_config = model_impls[impl_name]
        raw_model = model_config.get("model")
        model = _resolve_exact_root_interpolation(raw_model, nemo_gym_config)
        if model != raw_model:
            model_config["model"] = model
        model_name_key = f"{server_name}_name"
        if model and model_name_key not in nemo_gym_config:
            nemo_gym_config[model_name_key] = model

        if list(model_impls) != [impl_name]:
            print(
                f"Using `{impl_name}` for `{server_name}` responses_api_models "
                f"and dropping merged implementations {sorted(set(model_impls) - {impl_name})}",
                flush=True,
            )
            model_impls.clear()
            model_impls[impl_name] = model_config


def _select_responses_api_model_impl(
    server_name: str,
    model_impls: dict[str, Any],
) -> str | None:
    valid_impl_names = [name for name, value in model_impls.items() if isinstance(value, dict)]
    if not valid_impl_names:
        return None

    for preferred_name in (server_name, "genrm_model", "vllm_model"):
        if preferred_name in valid_impl_names:
            return preferred_name
    return valid_impl_names[0]


def _resolve_exact_root_interpolation(value: Any, root: dict[str, Any]) -> Any:
    if not isinstance(value, str) or not value.startswith("${") or not value.endswith("}"):
        return value

    key_path = value[2:-1]
    if ":" in key_path:
        return value

    current: Any = root
    for key in key_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return value
        current = current[key]
    return current


def _collect_responses_api_model_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        if value.get("type") == "responses_api_models" and value.get("name"):
            refs.add(str(value["name"]))
        for child in value.values():
            refs.update(_collect_responses_api_model_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.update(_collect_responses_api_model_refs(child))
    return refs


def _log_nemo_gym_runtime_summary(
    config: dict[str, Any],
    initial_global_config_dict: dict[str, Any],
) -> None:
    """Print the few Gym settings that explain rollout liveness."""
    data_cfg = config.get("data", {})
    genrm_compare_cfg = (
        initial_global_config_dict.get("genrm_compare", {}).get("resources_servers", {}).get("genrm_compare", {})
    )
    genrm_model_impls = initial_global_config_dict.get("genrm_model", {}).get(
        "responses_api_models",
        {},
    )
    genrm_impl_name = next(iter(genrm_model_impls), None)
    genrm_impl = genrm_model_impls.get(genrm_impl_name, {}) if genrm_impl_name else {}
    print(
        "\nNeMo-Gym runtime summary:\n"
        f"  train_data: {data_cfg.get('train')}\n"
        f"  validation_data: {data_cfg.get('validation')}\n"
        f"  genrm_impl: {genrm_impl_name}\n"
        f"  genrm_model: {genrm_impl.get('model')}\n"
        f"  num_rollouts_per_prompt: {genrm_compare_cfg.get('num_rollouts_per_prompt')}\n"
        f"  uv_venv_dir: {initial_global_config_dict.get('uv_venv_dir')}",
        flush=True,
    )


def materialize_nemo_gym_config_paths(nemo_gym_config: dict[str, Any]) -> dict[str, Any]:
    """Inline NeMo-Gym config paths so we can safely normalize merged servers."""
    materialized = deepcopy(nemo_gym_config)
    config_paths = list(materialized.get("config_paths") or [])
    if not config_paths:
        return materialized

    extra_configs = []
    discovered_paths = list(config_paths)
    index = 0
    while index < len(discovered_paths):
        config_path = discovered_paths[index]
        index += 1
        resolved_path = _resolve_nemo_gym_config_path(config_path)
        if resolved_path is None:
            return materialized

        extra_config = OmegaConf.load(resolved_path)
        extra_configs.append(extra_config)
        for nested_path in extra_config.get("config_paths") or []:
            if nested_path not in discovered_paths:
                discovered_paths.append(nested_path)

    merged = OmegaConf.merge(*extra_configs, OmegaConf.create(materialized))
    materialized = OmegaConf.to_container(merged, resolve=False)
    assert isinstance(materialized, dict)
    materialized.pop("config_paths", None)
    return materialized


def _resolve_nemo_gym_config_path(config_path: str) -> Path | None:
    path = Path(config_path)
    if path.is_absolute():
        return path if path.exists() else None

    candidates = [Path.cwd() / path]
    try:
        from nemo_gym import PARENT_DIR

        candidates.append(Path(PARENT_DIR) / path)
    except ImportError:
        pass
    candidates.append(Path("/opt/nemo-rl/3rdparty/Gym-workspace/Gym") / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def materialize_nemo_gym_data_manifest(config: dict[str, Any]) -> None:
    """Expand Nemotron prep manifests into NeMo-RL's native data fields."""
    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        return

    manifest_path = data_cfg.get("manifest_path")
    if not manifest_path:
        return

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    train_path = manifest.get("train")
    val_path = manifest.get("val") or manifest.get("validation")
    if train_path and not val_path and data_cfg.get("allow_train_as_validation", False):
        val_path = train_path
    if not train_path or not val_path:
        manifest_keys = ", ".join(sorted(str(key) for key in manifest.keys()))
        raise ValueError(
            f"{manifest_path} must contain non-empty train and val paths (keys: {manifest_keys or '<empty>'})"
        )

    train_cfg = dict(data_cfg.get("train") or {})
    val_cfg = dict(data_cfg.get("validation") or {})
    train_cfg["data_path"] = str(train_path)
    val_cfg["data_path"] = str(val_path)
    data_cfg["train"] = train_cfg
    data_cfg["validation"] = val_cfg
    data_cfg.pop("manifest_path", None)
    data_cfg.pop("allow_train_as_validation", None)


def materialize_nemo_gym_response_data(config: dict[str, Any]) -> None:
    """Ensure NeMo-Gym JSONL rows expose ``responses_create_params``."""
    data_cfg = config.get("data")
    if not isinstance(data_cfg, dict):
        return

    default_agent_ref = _default_agent_ref(config)
    data_cfg.pop("default_agent_ref", None)
    for split in ("train", "validation"):
        split_cfg = data_cfg.get(split)
        if not isinstance(split_cfg, dict):
            continue
        data_path = split_cfg.get("data_path")
        if not data_path:
            continue
        split_cfg["data_path"] = _materialize_response_jsonl(
            Path(str(data_path)),
            output_dir=_normalized_response_data_dir(data_cfg),
            default_agent_ref=default_agent_ref,
        )


def _materialize_response_jsonl(
    source_path: Path,
    *,
    output_dir: Path,
    default_agent_ref: dict[str, str] | None,
) -> str:
    rows: list[dict[str, Any]] = []
    changed = False
    with source_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{source_path}:{line_number} must contain a JSON object")
            normalized = _normalize_nemo_gym_row(
                row,
                source_path,
                line_number,
                default_agent_ref=default_agent_ref,
            )
            changed = changed or normalized != row
            rows.append(normalized)

    if not changed:
        return str(source_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    stat = source_path.stat()
    digest = hashlib.sha1(f"{source_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest()[:12]
    target_path = output_dir / f"{source_path.stem}.{digest}.nemo_gym.jsonl"
    with target_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Materialized NeMo-Gym response data: {source_path} -> {target_path}")
    return str(target_path)


def _normalized_response_data_dir(data_cfg: dict[str, Any]) -> Path:
    configured = data_cfg.get("normalized_data_dir")
    if configured:
        return Path(str(configured))
    output_dir = os.environ.get("RL_OUTPUT_DIR")
    if output_dir:
        return Path(output_dir) / "nemo_gym_data"
    return Path("/tmp/nemotron_nemo_gym_data")


def _normalize_nemo_gym_row(
    row: dict[str, Any],
    source_path: Path,
    line_number: int,
    *,
    default_agent_ref: dict[str, str] | None,
) -> dict[str, Any]:
    normalized = row
    if isinstance(row.get("responses_create_params"), dict):
        pass
    else:
        extra_env_info = row.get("extra_env_info")
        if isinstance(extra_env_info, dict) and isinstance(
            extra_env_info.get("responses_create_params"),
            dict,
        ):
            normalized = dict(row)
            normalized["responses_create_params"] = extra_env_info["responses_create_params"]
        else:
            messages = _extract_response_messages(row)
            if not messages:
                raise ValueError(
                    f"{source_path}:{line_number} is missing `responses_create_params` and "
                    "does not contain prompt/messages/question/problem/input fields to normalize"
                )

            normalized = dict(row)
            params: dict[str, Any] = {"input": messages}
            if row.get("tools"):
                params["tools"] = row["tools"]
            normalized["responses_create_params"] = params

    if _is_agent_ref(normalized.get("agent_ref")):
        return normalized
    if default_agent_ref is None:
        raise ValueError(
            f"{source_path}:{line_number} is missing `agent_ref`; set "
            "`data.default_agent_ref` or provide agent_ref in the JSONL row"
        )
    if normalized is row:
        normalized = dict(row)
    normalized["agent_ref"] = dict(default_agent_ref)
    return normalized


def _default_agent_ref(config: dict[str, Any]) -> dict[str, str] | None:
    data_cfg = config.get("data")
    if isinstance(data_cfg, dict):
        agent_ref = data_cfg.get("default_agent_ref")
        if isinstance(agent_ref, str):
            return {"type": "responses_api_agents", "name": agent_ref}
        if _is_agent_ref(agent_ref):
            return {"type": str(agent_ref["type"]), "name": str(agent_ref["name"])}

    env_cfg = config.get("env")
    if isinstance(env_cfg, dict) and env_cfg.get("use_genrm_compare"):
        genrm_agent_names = env_cfg.get("genrm_agent_names") or ["genrm_simple_agent"]
        if genrm_agent_names:
            return {"type": "responses_api_agents", "name": str(genrm_agent_names[0])}

    return None


def _is_agent_ref(value: Any) -> bool:
    return isinstance(value, dict) and bool(value.get("type")) and bool(value.get("name"))


def _extract_response_messages(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    for field in ("messages", "prompt"):
        messages = _coerce_response_messages(row.get(field))
        if messages:
            return messages

    for field in ("question", "problem", "input"):
        value = row.get(field)
        if value is not None:
            return [{"role": "user", "content": str(value)}]

    return None


def _coerce_response_messages(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if not isinstance(value, list):
        return None

    messages = []
    for item in value:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            return None
        content = item.get("content", item.get("value", item.get("text")))
        if content is None:
            return None
        role = item.get("role") or _message_role_from_sharegpt(item.get("from")) or "user"
        messages.append({"role": str(role), "content": str(content)})
    return messages or None


def _message_role_from_sharegpt(value: Any) -> str | None:
    role = str(value).lower() if value is not None else ""
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant", "model"}:
        return "assistant"
    if role == "system":
        return "system"
    return None


def _setup_log_dirs(config: dict[str, Any], get_next_experiment_dir) -> None:
    logger_cfg = config.setdefault("logger", {})
    if logger_cfg.get("log_dir"):
        logger_cfg["log_dir"] = get_next_experiment_dir(logger_cfg["log_dir"])
        print(f"Using log directory: {logger_cfg['log_dir']}")
    if config.get("checkpointing", {}).get("enabled"):
        print(f"Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}")


def _setup_initial_policy(config: dict[str, Any]) -> None:
    initial_checkpoint = config.get("initial_checkpoint")
    if not initial_checkpoint:
        return

    if config.get("convert_initial_checkpoint_to_hf", False):
        hf_checkpoint_path = convert_megatron_to_hf(
            megatron_checkpoint_path=initial_checkpoint,
            hf_model_id=config["policy"]["model_name"],
            output_dir=config.get("converted_checkpoint_dir"),
        )
        config["policy"]["model_name"] = hf_checkpoint_path
        config["policy"]["tokenizer"]["name"] = hf_checkpoint_path
        print(f"Updated model_name to converted checkpoint: {hf_checkpoint_path}")
        return

    if config.get("checkpointing", {}).get("enabled"):
        setup_initial_checkpoint(initial_checkpoint, config["checkpointing"]["checkpoint_dir"])


def convert_megatron_to_hf(
    *,
    megatron_checkpoint_path: str,
    hf_model_id: str,
    output_dir: str | None = None,
) -> str:
    """Convert a Megatron checkpoint to Hugging Face format using Megatron-Bridge."""
    megatron_path = Path(megatron_checkpoint_path)
    if megatron_path.is_dir():
        iter_dirs = [d for d in megatron_path.iterdir() if d.is_dir() and d.name.startswith("iter_")]
        if iter_dirs:
            iter_dirs.sort(key=lambda x: int(x.name.split("_")[1]))
            megatron_path = iter_dirs[-1]
            print(f"Using checkpoint iteration: {megatron_path.name}")

    output_path = Path(output_dir) if output_dir else megatron_path.parent / f"{megatron_path.name}_hf"
    if (output_path / "config.json").exists():
        print(f"HF checkpoint already exists at {output_path}, skipping conversion")
        return str(output_path)

    print("Converting Megatron checkpoint to Hugging Face format...")
    print(f"  Source: {megatron_path}")
    print(f"  HF model ID: {hf_model_id}")
    print(f"  Output: {output_path}")

    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(hf_model_id, trust_remote_code=True)
    bridge.export_ckpt(megatron_path=str(megatron_path), hf_path=str(output_path))
    print(f"Conversion complete: {output_path}")
    return str(output_path)


def setup_initial_checkpoint(initial_checkpoint_path: str, checkpoint_dir: str) -> None:
    """Create a NeMo-RL step_0 checkpoint view over an initial Megatron checkpoint."""
    checkpoint_dir_path = Path(checkpoint_dir)
    initial_path = Path(initial_checkpoint_path)
    step_dir = checkpoint_dir_path / "step_0"
    complete_marker = step_dir / ".complete"

    if complete_marker.exists():
        print(f"Found existing checkpoints in {checkpoint_dir_path}, skipping initial checkpoint setup")
        return

    existing_checkpoints = list(checkpoint_dir_path.glob("step_*"))
    if existing_checkpoints:
        raise RuntimeError(
            f"Found existing checkpoints in {checkpoint_dir_path}, but {complete_marker} is missing. "
            "Remove the incomplete checkpoint view or restore the marker before retrying."
        )

    weights_dir = step_dir / "policy" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    if not initial_path.exists():
        raise ValueError(f"Initial checkpoint path does not exist: {initial_path}")
    if not initial_path.is_dir():
        raise ValueError(f"Initial checkpoint path is not a directory: {initial_path}")

    iter_dirs = [d for d in initial_path.iterdir() if d.is_dir() and d.name.startswith("iter_")]
    if iter_dirs:
        iter_dirs.sort(key=lambda x: int(x.name.split("_")[1]))
        source_dir = iter_dirs[-1]
        print(f"Using checkpoint iteration: {source_dir.name}")
    else:
        source_dir = initial_path

    for item in source_dir.iterdir():
        target = weights_dir / item.name
        if not target.exists():
            target.symlink_to(item)

    training_info = {
        "step": 0,
        "epoch": 0,
        "global_step": 0,
        "initial_checkpoint": str(initial_path),
    }
    (step_dir / "training_info.json").write_text(json.dumps(training_info, indent=2), encoding="utf-8")
    complete_marker.write_text("ok\n", encoding="utf-8")
    print(f"Set up initial checkpoint at {step_dir}")


def _collect_trajectories(
    *,
    policy: Any,
    policy_generation: Any,
    val_dataloader: Any,
    tokenizer: Any,
    val_task_to_env: dict[str, Any],
    logger: Any,
    master_config: dict[str, Any],
) -> None:
    from nemo_rl.algorithms.grpo import refit_policy_generation
    from nemo_rl.experience.rollouts import run_async_nemo_gym_rollout
    from wandb import Table

    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]
    refit_policy_generation(policy, policy_generation, colocated_inference)

    log_filename = "trajectory_collection.jsonl"
    print("\nRunning trajectory collection...", flush=True)
    generation_config = master_config["policy"]["generation"]

    for val_batch in val_dataloader:
        nemo_gym_rollout_result = run_async_nemo_gym_rollout(
            policy_generation=policy_generation,
            input_batch=val_batch,
            tokenizer=tokenizer,
            task_to_env=val_task_to_env,
            max_seq_len=None,
            generation_config=generation_config,
            max_rollout_turns=None,
            greedy=False,
        )

        rows_to_log: list[str] = []
        for key, value in nemo_gym_rollout_result.rollout_metrics.items():
            if "full_result" not in key:
                continue
            value: Table
            rows_to_log.extend(row[0] for row in value.data)

        logger.log_string_list_as_jsonl(rows_to_log, log_filename)

    policy_generation.finish_generation()


def _register_mul_resolver() -> None:
    if not OmegaConf.has_resolver("mul"):
        OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def _maybe_chdir_to_nemo_rl_workdir(config: dict[str, Any]) -> None:
    workdir = config.get("nemo_rl_workdir") or config.get("run", {}).get("workdir") or "/opt/nemo-rl"
    if workdir and Path(workdir).is_dir():
        os.chdir(workdir)


def _patch_wandb(config: dict[str, Any]) -> None:
    try:
        from nemotron.kit.wandb_kit import (
            patch_nemo_rl_checkpoint_logging,
            patch_wandb_http_handler_skip_digest_verification,
            patch_wandb_local_file_handler_skip_digest_verification,
            patch_wandb_runid_for_seeded_random,
        )
    except ImportError as exc:
        print(f"[wandb] patches disabled: {type(exc).__name__}: {exc}", flush=True)
        return

    patch_wandb_http_handler_skip_digest_verification()
    patch_wandb_local_file_handler_skip_digest_verification()
    patch_wandb_runid_for_seeded_random()
    artifact_name = config.get("checkpointing", {}).get("artifact_name")
    if artifact_name:
        patch_nemo_rl_checkpoint_logging(artifact_name=artifact_name)

    try:
        import wandb.util

        wandb.util.VALUE_BYTES_LIMIT = 10_000_000
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] could not raise VALUE_BYTES_LIMIT: {type(exc).__name__}: {exc}", flush=True)
