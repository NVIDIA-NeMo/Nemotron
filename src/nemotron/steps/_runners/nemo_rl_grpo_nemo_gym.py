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

import json
import os
import pprint
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
        NemoGym,
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
    ) = setup(config, tokenizer, train_dataset, val_dataset)

    is_trajectory_collection = (
        config["env"]["nemo_gym"].pop("is_trajectory_collection", False) or False
    )
    nemo_gym_config = NemoGymConfig(
        model_name=policy_generation.cfg["model_name"],
        base_urls=policy_generation.dp_openai_server_base_urls,
        initial_global_config_dict=config["env"]["nemo_gym"],
    )
    nemo_gym = NemoGym.options(
        runtime_env={
            "py_executable": get_actor_python_env("nemo_rl.environments.nemo_gym.NemoGym"),
        }
    ).remote(nemo_gym_config)
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


def resolve_nemo_gym_data_paths(config: dict[str, Any]) -> tuple[str, str, int | None, int | None]:
    """Resolve train/validation JSONL paths from flat or nested NeMo-RL config."""
    data_cfg = config["data"]
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
        return datum_dict

    return AllTaskProcessedDataset(
        nemo_rl_examples,
        tokenizer,
        None,
        passthrough_task_processor,
    )


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
