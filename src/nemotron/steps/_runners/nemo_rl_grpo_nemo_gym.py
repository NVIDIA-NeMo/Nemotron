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

"""Generic NeMo-Gym GRPO runner used by RLVR/RLHF steps."""

from __future__ import annotations

import inspect
import json
import os
import pprint
from collections.abc import Mapping
from itertools import chain, repeat
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from nemotron.steps._runners.nemo_rl import load_nemo_rl_step_config


def run_nemo_gym_grpo(*, config_path: Path, overrides: list[str] | None = None) -> None:
    """Run NeMo-RL GRPO with a NeMo-Gym environment.

    The code is intentionally close to the Super3 RLVR/RLHF runner, but keeps
    recipe-specific artifact names, resource-server choices, and model paths in
    config files.
    """
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
    _patch_wandb(config)

    import ray
    from nemo_rl.algorithms.grpo import (
        _should_use_nemo_gym,
        grpo_train,
        setup,
    )
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
    from nemo_rl.distributed.virtual_cluster import init_ray
    from nemo_rl.environments.nemo_gym import (
        NemoGymConfig,
        setup_nemo_gym_config,
    )
    from nemo_rl.models.generation import configure_generation_config
    from nemo_rl.utils.logger import get_next_experiment_dir

    _setup_initial_policy(config)

    logger_cfg = config.setdefault("logger", {})
    if logger_cfg.get("log_dir"):
        logger_cfg["log_dir"] = get_next_experiment_dir(logger_cfg["log_dir"])
        print(f"Using log directory: {logger_cfg['log_dir']}")
    if config.get("checkpointing", {}).get("enabled"):
        print(f"Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}")

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"].get("generation") is not None, "A generation config is required for GRPO"
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"],
        tokenizer,
    )

    setup_nemo_gym_config(config, tokenizer)
    assert _should_use_nemo_gym(config), "Set env.should_use_nemo_gym=true for this runner"

    print("\nSetting up NeMo-Gym data...")
    train_path, val_path, train_repeats, val_repeats = resolve_nemo_gym_data_paths(config)
    train_dataset = setup_nemo_gym_jsonl_dataset(
        jsonl_fpath=train_path,
        tokenizer=tokenizer,
        num_repeats=train_repeats,
    )
    val_dataset = setup_nemo_gym_jsonl_dataset(
        jsonl_fpath=val_path,
        tokenizer=tokenizer,
        num_repeats=val_repeats,
    )

    if config["grpo"]["max_val_samples"] is not None:
        raise ValueError(
            "A non-null `grpo.max_val_samples` parameter is not supported for "
            "NeMo-Gym JSONL. The validation set is consumed directly."
        )
    config["grpo"]["max_val_samples"] = len(val_dataset)
    config["grpo"]["val_batch_size"] = len(val_dataset)

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
    ) = unpack_setup_result_compat(setup(config, tokenizer, train_dataset, val_dataset))

    is_trajectory_collection = (
        config["env"]["nemo_gym"].pop("is_trajectory_collection", False) or False
    )
    nemo_gym_runtime_defaults = {
        "rollout_max_attempts_to_avoid_lp_nan": 1,
    }
    for key, value in nemo_gym_runtime_defaults.items():
        config["env"]["nemo_gym"].setdefault(key, value)
    nemo_gym_config = NemoGymConfig(
        model_name=policy_generation.cfg["model_name"],
        base_urls=policy_generation.dp_openai_server_base_urls,
        initial_global_config_dict=config["env"]["nemo_gym"],
    )
    nemo_gym_cls = _nemo_gym_compat_actor_class(ray)
    nemo_gym = nemo_gym_cls.options(
        runtime_env={
            "env_vars": _ray_actor_env_overrides(),
            "py_executable": get_actor_python_env("nemo_rl.environments.nemo_gym.NemoGym"),
        }
    ).remote(nemo_gym_config)
    try:
        ray.get(nemo_gym.health_check.remote())
    except AttributeError:
        print("Skipping NemoGym health_check; actor API does not expose it")

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

    call_grpo_train_compat(
        grpo_train,
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
        grpo_state=grpo_state,
        master_config=master_config,
    )


def _nemo_gym_compat_actor_class(ray_module):
    """Create a NemoGym-compatible actor for NeMo-RL/Gym API drift."""

    @ray_module.remote
    class CompatNemoGym:
        def __init__(self, cfg):
            os.environ.update(_ray_actor_env_overrides())
            _log_ray_actor_env_cleanup()

            import ray
            from nemo_gym.cli import GlobalConfigDictParserConfig, RunHelper
            from nemo_gym.rollout_collection import RolloutCollectionHelper
            from nemo_gym.server_utils import HEAD_SERVER_KEY_NAME, BaseServerConfig
            from nemo_rl.distributed.virtual_cluster import _get_free_port_local, _get_node_ip_local
            from omegaconf import DictConfig

            self.cfg = cfg
            self.node_ip = _get_node_ip_local()
            self.head_server_port = _get_free_port_local()

            initial_global_config_dict = dict(self.cfg.get("initial_global_config_dict") or {})
            initial_global_config_dict["policy_model_name"] = self.cfg["model_name"]
            initial_global_config_dict["policy_api_key"] = "dummy_key"
            initial_global_config_dict["policy_base_url"] = self.cfg["base_urls"]
            initial_global_config_dict.setdefault("global_aiohttp_connector_limit_per_host", 16_384)
            initial_global_config_dict.setdefault("global_aiohttp_connector_limit", 65_536)

            assert ray.is_initialized(), "Ray must be initialized before using NeMo-Gym environment"
            ray_context = ray.get_runtime_context()
            assert ray_context.gcs_address, "Ray must have a GCS address"
            initial_global_config_dict["ray_head_node_address"] = ray_context.gcs_address
            initial_global_config_dict[HEAD_SERVER_KEY_NAME] = {
                "host": "0.0.0.0",
                "port": self.head_server_port,
            }
            self.rollout_max_attempts_to_avoid_lp_nan = initial_global_config_dict.pop(
                "rollout_max_attempts_to_avoid_lp_nan",
                1,
            )
            assert self.rollout_max_attempts_to_avoid_lp_nan >= 1, (
                "`rollout_max_attempts_to_avoid_lp_nan` must be at least 1"
            )

            _patch_nemo_gym_gpu_helper_start(ray)
            self.rh = RunHelper()
            self.rh.start(
                global_config_dict_parser_config=GlobalConfigDictParserConfig(
                    initial_global_config_dict=DictConfig(initial_global_config_dict),
                    skip_load_from_cli=True,
                )
            )
            self.head_server_config = BaseServerConfig(
                host=self.node_ip,
                port=self.head_server_port,
            )
            self.rch = RolloutCollectionHelper()

        async def health_check(self):
            return True

        async def run_rollouts(self, *args, **kwargs):
            return await _nemo_gym_run_rollouts_compat(self, *args, **kwargs)

        async def shutdown(self):
            self.rh.shutdown()
            return None

    return CompatNemoGym


def _ray_actor_env_overrides() -> dict[str, str]:
    """Mask one-shot transport env vars before NeMo-Gym spawns child actors."""
    env_vars = {
        "_NEMOTRON_SRC_CHUNKS": "0",
        "_NEMOTRON_SRC_SHA256": "",
        "_NEMOTRON_CONFIG_B64": "",
    }
    chunk_keys = {key for key in os.environ if key.startswith("_NEMOTRON_SRC_CHUNK_")}
    try:
        n_chunks = int(os.environ.get("_NEMOTRON_SRC_CHUNKS", "0") or "0")
    except ValueError:
        n_chunks = 0
    for idx in range(n_chunks):
        chunk_keys.add(f"_NEMOTRON_SRC_CHUNK_{idx}")
    for key in chunk_keys:
        env_vars[key] = ""
    if os.environ.get("PYTHONPATH"):
        env_vars["PYTHONPATH"] = os.environ["PYTHONPATH"]
    return env_vars


def _log_ray_actor_env_cleanup() -> None:
    chunk_keys = [key for key in os.environ if key.startswith("_NEMOTRON_SRC_CHUNK_")]
    nonempty_chunks = sum(1 for key in chunk_keys if os.environ.get(key))
    print(
        "NemoGym actor env scrubbed: "
        f"_NEMOTRON_SRC_CHUNK_* keys={len(chunk_keys)}, "
        f"nonempty={nonempty_chunks}, "
        f"_NEMOTRON_SRC_CHUNKS={os.environ.get('_NEMOTRON_SRC_CHUNKS', '')!r}",
        flush=True,
    )


def _patch_nemo_gym_gpu_helper_start(ray_module) -> None:
    import nemo_gym.cli as nemo_gym_cli
    import nemo_gym.ray_utils as nemo_gym_ray_utils

    if getattr(nemo_gym_ray_utils, "_nemotron_gpu_helper_patch", False):
        return

    original_start = nemo_gym_ray_utils._start_global_ray_gpu_scheduling_helper

    def start_or_reuse_global_ray_gpu_scheduling_helper(*args, **kwargs):
        try:
            return original_start(*args, **kwargs)
        except ValueError as exc:
            message = str(exc)
            helper_name = "_NeMoGymRayGPUSchedulingHelper"
            if helper_name not in message or "already taken" not in message:
                raise
            print(
                f"Reusing existing NeMo-Gym Ray GPU scheduling helper actor {helper_name}",
                flush=True,
            )
            try:
                return ray_module.get_actor(helper_name)
            except Exception:
                return ray_module.get_actor(helper_name, namespace=None)

    nemo_gym_ray_utils._start_global_ray_gpu_scheduling_helper = (
        start_or_reuse_global_ray_gpu_scheduling_helper
    )
    nemo_gym_cli._start_global_ray_gpu_scheduling_helper = (
        start_or_reuse_global_ray_gpu_scheduling_helper
    )
    nemo_gym_ray_utils._nemotron_gpu_helper_patch = True


async def _nemo_gym_run_rollouts_compat(
    nemo_gym,
    nemo_gym_examples: list[dict],
    tokenizer,
    timer_prefix: str,
) -> tuple[list[dict], dict]:
    import torch
    from nemo_rl.utils.timer import Timer

    timer = Timer()
    timer.start("_run_rollouts_total")
    max_attempts, trial = nemo_gym.rollout_max_attempts_to_avoid_lp_nan, 0
    while trial < max_attempts:
        nemo_gym_num_rows = len(nemo_gym_examples)
        nemo_gym_result_iterator = nemo_gym.rch.run_examples(
            examples=nemo_gym_examples,
            head_server_config=nemo_gym.head_server_config,
        )
        nemo_rl_rowidxs = []
        nemo_rl_results = []
        for task in nemo_gym_result_iterator:
            with timer.time(label=f"{timer_prefix}/await_results"):
                nemo_gym_row, nemo_gym_result = await task
            with timer.time(label=f"{timer_prefix}/postprocess_results"):
                nemo_rl_result = _postprocess_nemo_gym_to_nemo_rl_result(
                    nemo_gym_result,
                    tokenizer,
                )
            nemo_rl_rowidxs.append(nemo_gym_row["_rowidx"])
            nemo_rl_results.append(nemo_rl_result)

        logprob_contains_nan = False
        for nemo_rl_result in nemo_rl_results:
            for message in nemo_rl_result["message_log"]:
                generation_logprobs = message.get("generation_logprobs")
                if generation_logprobs is not None and torch.isnan(generation_logprobs).any():
                    logprob_contains_nan = True
                    break
            if logprob_contains_nan:
                break
        if logprob_contains_nan:
            trial += 1
            print(
                f"Generation logprobs contain NaN; retrying... (trial {trial}/{max_attempts})",
                flush=True,
            )
            continue
        break

    nemo_rl_sort_results = [None] * nemo_gym_num_rows
    for rowidx, result in zip(nemo_rl_rowidxs, nemo_rl_results):
        nemo_rl_sort_results[rowidx] = result
    timer.stop("_run_rollouts_total")
    timing_metrics = timer.get_timing_metrics("sum")
    total_time = timing_metrics.pop("_run_rollouts_total")
    if total_time:
        timing_metrics[f"{timer_prefix}/postprocess_results_pct"] = (
            100 * timing_metrics[f"{timer_prefix}/postprocess_results"] / total_time
        )
    return nemo_rl_sort_results, timing_metrics


def _postprocess_nemo_gym_to_nemo_rl_result(nemo_gym_result: dict, tokenizer) -> dict:
    import torch

    if not isinstance(nemo_gym_result, dict):
        raise TypeError(f"Hit a non-successful response when querying NeMo Gym: {nemo_gym_result}")
    nemo_rl_message_log = []
    seen_token_ids: list[int] = []
    for output_item_dict in nemo_gym_result["response"]["output"]:
        if "generation_token_ids" not in output_item_dict:
            continue
        if seen_token_ids != output_item_dict["prompt_token_ids"][: len(seen_token_ids)]:
            raise ValueError("Non-contiguous messages found in NeMo Gym rollout result")
        nemo_rl_message_log.append(
            {
                "role": "user",
                "content": "",
                "token_ids": torch.tensor(output_item_dict["prompt_token_ids"][len(seen_token_ids) :]),
            }
        )
        nemo_rl_message_log.append(
            {
                "role": "assistant",
                "content": "",
                "token_ids": torch.tensor(output_item_dict["generation_token_ids"]),
                "generation_logprobs": torch.tensor(output_item_dict["generation_log_probs"]),
            }
        )
        seen_token_ids.extend(nemo_rl_message_log[-2]["token_ids"])
        seen_token_ids.extend(nemo_rl_message_log[-1]["token_ids"])
        output_item_dict["prompt_str"] = tokenizer.decode(output_item_dict.pop("prompt_token_ids"))
        output_item_dict["generation_str"] = tokenizer.decode(
            output_item_dict.pop("generation_token_ids")
        )
        output_item_dict.pop("generation_log_probs")

    if not nemo_rl_message_log:
        input_messages = nemo_gym_result["responses_create_params"]["input"]
        prompt_token_ids = tokenizer.apply_chat_template(input_messages, tokenize=True)
        raise ValueError(
            "NeMo Gym returned a result with no generation data. "
            "This typically means the prompt for the first turn already exceeds "
            f"the vLLM max_model_len. Prompt length: {len(prompt_token_ids)} tokens."
        )
    return {
        "message_log": nemo_rl_message_log,
        "input_message_log": nemo_rl_message_log[:1],
        "full_result": nemo_gym_result,
    }


def call_grpo_train_compat(grpo_train_fn, **available):
    """Call NeMo-RL grpo_train across minor API signature drift."""
    signature = inspect.signature(grpo_train_fn)
    aliases = {
        "grpo_save_state": "grpo_state",
    }
    kwargs = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        source_name = name if name in available else aliases.get(name)
        if source_name in available:
            kwargs[name] = available[source_name]
    return grpo_train_fn(**kwargs)


def unpack_setup_result_compat(setup_result):
    """Unpack NeMo-RL setup() across minor return tuple drift."""
    policy, policy_generation, *items = setup_result
    checkpointer = None
    cluster = None
    dataloader = None
    val_dataloader = None
    loss_fn = None
    logger = None
    grpo_state = None
    master_config = None
    extra_setup_outputs = []

    unknown_items = []
    for item in items:
        if grpo_state is None and _is_grpo_state(item):
            grpo_state = item
        elif master_config is None and _is_master_config(item):
            master_config = item
        elif checkpointer is None and _is_checkpoint_manager(item):
            checkpointer = item
        elif cluster is None and _is_virtual_cluster(item):
            cluster = item
        elif logger is None and _is_logger(item):
            logger = item
        elif loss_fn is None and _is_loss_fn(item):
            loss_fn = item
        else:
            unknown_items.append(item)

    dataloader_candidates = [
        candidate
        for candidate in (_unwrap_dataloader_candidate(item) for item in unknown_items)
        if candidate is not None and _is_dataloader_candidate(candidate)
    ]
    if dataloader_candidates:
        dataloader = dataloader_candidates.pop(0)
    if dataloader_candidates:
        val_dataloader = dataloader_candidates.pop(0)
    extra_setup_outputs.extend(dataloader_candidates)
    extra_setup_outputs.extend(
        item for item in unknown_items if item is not dataloader and item is not val_dataloader
    )

    required = {
        "cluster": cluster,
        "dataloader": dataloader,
        "loss_fn": loss_fn,
        "logger": logger,
        "checkpointer": checkpointer,
        "grpo_state": grpo_state,
        "master_config": master_config,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        item_types = ", ".join(type(item).__name__ for item in items)
        raise TypeError(
            f"Could not unpack NeMo-RL setup() return values; missing {missing}; "
            f"item types: {item_types}"
        )

    if extra_setup_outputs:
        extra_types = ", ".join(type(item).__name__ for item in extra_setup_outputs)
        print(f"Ignoring {len(extra_setup_outputs)} extra NeMo-RL setup return value(s): {extra_types}")

    return (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


def _is_checkpoint_manager(value) -> bool:
    return type(value).__name__ == "CheckpointManager" or hasattr(value, "finalize_checkpoint")


def _is_virtual_cluster(value) -> bool:
    return type(value).__name__ in {"RayVirtualCluster", "ActorHandle"}


def _is_logger(value) -> bool:
    return type(value).__name__ == "Logger"


def _is_loss_fn(value) -> bool:
    return "Loss" in type(value).__name__ or callable(value)


def _is_dataloader_candidate(value) -> bool:
    if isinstance(value, (tuple, list)):
        return _unwrap_dataloader_candidate(value) is not None
    if isinstance(value, (str, bytes, Mapping)):
        return False
    if _is_checkpoint_manager(value) or _is_virtual_cluster(value) or _is_logger(value) or _is_loss_fn(value):
        return False
    return hasattr(value, "__iter__")


def _unwrap_dataloader_candidate(value):
    if not isinstance(value, (tuple, list)):
        return value
    for item in value:
        if _is_dataloader_candidate(item) and not isinstance(item, (tuple, list)):
            print(f"Using nested {type(item).__name__} from {type(value).__name__} dataloader wrapper")
            return item
    return None


def _is_grpo_state(value) -> bool:
    return isinstance(value, Mapping) and "current_step" in value


def _is_master_config(value) -> bool:
    return isinstance(value, Mapping) and "policy" in value and "grpo" in value


def resolve_nemo_gym_data_paths(config: dict[str, Any]) -> tuple[str, str, int | None, int | None]:
    """Resolve train/validation JSONL paths from flat or nested NeMo-RL config."""
    data_cfg = config["data"]
    manifest_path = data_cfg.get("manifest_path")
    if manifest_path:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        train_path = manifest.get("train")
        val_path = manifest.get("val") or manifest.get("validation")
        if train_path and not val_path and data_cfg.get("allow_train_as_validation", False):
            val_path = train_path
        if not train_path or not val_path:
            manifest_keys = ", ".join(sorted(str(key) for key in manifest.keys()))
            raise ValueError(
                f"{manifest_path} must contain non-empty train and val paths "
                f"(keys: {manifest_keys or '<empty>'})"
            )
        train_repeats = data_cfg.get("train", {}).get("num_repeats") if isinstance(data_cfg.get("train"), dict) else None
        val_repeats = (
            data_cfg.get("validation", {}).get("num_repeats")
            if isinstance(data_cfg.get("validation"), dict)
            else None
        )
        return str(train_path), str(val_path), train_repeats, val_repeats

    if isinstance(data_cfg.get("train"), dict):
        train_cfg = data_cfg["train"]
        val_cfg = data_cfg.get("validation", {})
        train_path = train_cfg.get("data_path") or train_cfg.get("path")
        val_path = val_cfg.get("data_path") or val_cfg.get("path")
        train_repeats = train_cfg.get("num_repeats") or data_cfg.get("num_repeats")
        val_repeats = val_cfg.get("num_repeats")
    else:
        train_path = data_cfg.get("train_jsonl_fpath") or data_cfg.get("train_data_path")
        val_path = data_cfg.get("validation_jsonl_fpath") or data_cfg.get("val_data_path")
        train_repeats = data_cfg.get("num_repeats")
        val_repeats = None

    if not train_path or not val_path:
        raise ValueError(
            "NeMo-Gym GRPO requires data.train.data_path/data.validation.data_path "
            "or data.train_jsonl_fpath/data.validation_jsonl_fpath."
        )
    return str(train_path), str(val_path), train_repeats, val_repeats


def setup_nemo_gym_jsonl_dataset(
    *,
    jsonl_fpath: str,
    tokenizer: Any,
    num_repeats: int | None = None,
) -> Any:
    """Load NeMo-Gym JSONL and adapt records to NeMo-RL ``DatumSpec`` objects."""
    from nemo_rl.data.datasets import AllTaskProcessedDataset
    from nemo_rl.data.interfaces import DatumSpec

    try:
        from nemo_rl.environments.nemo_gym import nemo_gym_example_to_nemo_rl_datum_spec
    except ImportError:
        import torch

        def nemo_gym_example_to_nemo_rl_datum_spec(
            nemo_gym_example: dict[str, Any],
            idx: int,
        ) -> DatumSpec:
            return DatumSpec(
                message_log=[{"role": "user", "content": "", "token_ids": torch.tensor([])}],
                length=0,
                extra_env_info=nemo_gym_example,
                loss_multiplier=1.0,
                idx=idx,
                task_name="nemo_gym",
                stop_strings=None,
                token_ids=[],
            )

    with open(jsonl_fpath, encoding="utf-8") as f:
        nemo_gym_examples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded data at {jsonl_fpath}. Found {len(nemo_gym_examples)} examples")
    nemo_gym_examples = [_ensure_responses_create_params(example) for example in nemo_gym_examples]
    if num_repeats:
        previous_length = len(nemo_gym_examples)
        nemo_gym_examples = list(
            chain.from_iterable(
                repeat(nemo_gym_example, num_repeats) for nemo_gym_example in nemo_gym_examples
            )
        )
        print(f"Repeated {jsonl_fpath} from {previous_length} to {len(nemo_gym_examples)} examples")

    nemo_rl_examples: list[DatumSpec] = [
        nemo_gym_example_to_nemo_rl_datum_spec(nemo_gym_example, idx)
        for idx, nemo_gym_example in enumerate(nemo_gym_examples)
    ]

    def passthrough_task_processor(datum_dict, *args, **kwargs):
        if "responses_create_params" not in datum_dict:
            extra_env_info = datum_dict.get("extra_env_info")
            if isinstance(extra_env_info, dict) and "responses_create_params" in extra_env_info:
                datum_dict["responses_create_params"] = extra_env_info["responses_create_params"]
        return datum_dict

    return AllTaskProcessedDataset(
        nemo_rl_examples,
        tokenizer,
        None,
        passthrough_task_processor,
    )


def _ensure_responses_create_params(example: dict[str, Any]) -> dict[str, Any]:
    if "responses_create_params" in example:
        return example

    prompt = example.get("prompt") or example.get("messages")
    if isinstance(prompt, list):
        messages = prompt
    else:
        question = example.get("question") or example.get("problem") or example.get("prompt") or ""
        messages = [{"role": "user", "content": str(question)}]

    normalized = dict(example)
    normalized["responses_create_params"] = {"input": messages}
    return normalized


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

    existing_checkpoints = list(checkpoint_dir_path.glob("step_*"))
    if existing_checkpoints:
        print(f"Found existing checkpoints in {checkpoint_dir_path}, skipping initial checkpoint setup")
        return

    step_dir = checkpoint_dir_path / "step_0"
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
    except Exception:
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
    except Exception:
        pass
