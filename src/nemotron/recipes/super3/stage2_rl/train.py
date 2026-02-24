#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "super3/rl"
# image = "nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_super"
# setup = """
# nemo-rl is pre-installed in the image at /opt/nemo-rl.
# The training script and config.yaml must be placed in /opt/nemo-rl/ before execution.
# """
#
# [tool.runspec.run]
# launch = "ray"
# cmd = "uv run python {script} --config {config}"
# workdir = "/opt/nemo-rl"
#
# [tool.runspec.config]
# dir = "./config"
# default = "tiny"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 8
# ///

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

"""RL (Reinforcement Learning) script for Nemotron Super3.

Uses NeMo-RL's GRPO algorithm for reinforcement learning training.
This script is designed to run inside a container with NeMo-RL installed.

CLI:
    nemotron super3 rl              # local execution
    nemotron super3 rl --run dgx    # submit to cluster

Execution logic: src/nemotron/cli/commands/super3/rl.py

Direct usage:
    python /path/to/train.py --config /path/to/grpo_config.yaml
    python /path/to/train.py --config /path/to/grpo_config.yaml \
        grpo.num_iterations=100 policy.generation.temperature=0.7
"""

from __future__ import annotations

# Flag to indicate this module requires Ray execution
RAY = True

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import pprint  # noqa: E402
from itertools import chain, repeat  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from nemo_rl.algorithms.grpo import MasterConfig


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run GRPO training for Nemotron Super3",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )

    args, overrides = parser.parse_known_args()
    return args, overrides


def convert_megatron_to_hf(
    megatron_checkpoint_path: str,
    hf_model_id: str,
    output_dir: str | None = None,
) -> str:
    """Convert a Megatron checkpoint to HuggingFace format using Megatron-Bridge."""
    from pathlib import Path

    megatron_path = Path(megatron_checkpoint_path)

    if megatron_path.is_dir():
        iter_dirs = [d for d in megatron_path.iterdir() if d.is_dir() and d.name.startswith("iter_")]
        if iter_dirs:
            iter_dirs.sort(key=lambda x: int(x.name.split("_")[1]))
            megatron_path = iter_dirs[-1]
            print(f"Using checkpoint iteration: {megatron_path.name}")

    if output_dir is None:
        output_dir = megatron_path.parent / f"{megatron_path.name}_hf"
    output_path = Path(output_dir)

    if (output_path / "config.json").exists():
        print(f"HF checkpoint already exists at {output_path}, skipping conversion")
        return str(output_path)

    print(f"Converting Megatron checkpoint to HuggingFace format...")
    print(f"  Source: {megatron_path}")
    print(f"  HF model ID: {hf_model_id}")
    print(f"  Output: {output_path}")

    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(hf_model_id, trust_remote_code=True)
    bridge.export_ckpt(
        megatron_path=str(megatron_path),
        hf_path=str(output_path),
    )

    print(f"Conversion complete: {output_path}")
    return str(output_path)


def setup_initial_checkpoint(initial_checkpoint_path: str, checkpoint_dir: str) -> None:
    """Set up a checkpoint structure from an initial Megatron checkpoint for finetuning."""
    import json
    from pathlib import Path

    checkpoint_dir = Path(checkpoint_dir)
    initial_path = Path(initial_checkpoint_path)

    existing_checkpoints = list(checkpoint_dir.glob("step_*"))
    if existing_checkpoints:
        print(f"Found existing checkpoints in {checkpoint_dir}, skipping initial checkpoint setup")
        return

    step_dir = checkpoint_dir / "step_0"
    policy_dir = step_dir / "policy"
    weights_dir = policy_dir / "weights"

    weights_dir.mkdir(parents=True, exist_ok=True)

    if not initial_path.exists():
        print(f"ERROR: Initial checkpoint path does not exist: {initial_path}")
        print(f"  Checking parent directory...")
        parent = initial_path.parent
        if parent.exists():
            print(f"  Parent exists: {parent}")
            print(f"  Contents: {list(parent.iterdir())[:10]}")
        else:
            print(f"  Parent also does not exist: {parent}")
        raise ValueError(
            f"Initial checkpoint path does not exist: {initial_path}. "
            f"Ensure the checkpoint is accessible from the compute node."
        )

    if not initial_path.is_dir():
        raise ValueError(
            f"Initial checkpoint path is not a directory: {initial_path} "
            f"(is_file={initial_path.is_file()}, is_symlink={initial_path.is_symlink()})"
        )

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
            print(f"Linked {item.name} -> {target}")

    training_info = {
        "step": 0,
        "epoch": 0,
        "global_step": 0,
        "initial_checkpoint": str(initial_path),
    }
    training_info_path = step_dir / "training_info.json"
    with open(training_info_path, "w") as f:
        json.dump(training_info, f, indent=2)

    print(f"Set up initial checkpoint at {step_dir}")


def setup_single_nemo_gym_dataset(jsonl_fpath: str, tokenizer, num_repeats: int | None = None):
    """Load and prepare a NeMo-Gym dataset from JSONL file."""
    from nemo_rl.data.datasets import AllTaskProcessedDataset
    from nemo_rl.data.interfaces import DatumSpec
    from nemo_rl.environments.nemo_gym import nemo_gym_example_to_nemo_rl_datum_spec

    with open(jsonl_fpath) as f:
        nemo_gym_examples = list(map(json.loads, f))

    print(f"Loaded data at {jsonl_fpath}. Found {len(nemo_gym_examples)} examples")

    if num_repeats:
        previous_length = len(nemo_gym_examples)
        nemo_gym_examples = list(
            chain.from_iterable(
                repeat(nemo_gym_example, num_repeats) for nemo_gym_example in nemo_gym_examples
            )
        )
        print(
            f"Repeating examples (in a pattern of abc to aabbcc) for {jsonl_fpath} "
            f"from {previous_length} to {len(nemo_gym_examples)}!"
        )

    nemo_rl_compatible_examples: list[DatumSpec] = [
        nemo_gym_example_to_nemo_rl_datum_spec(nemo_gym_example, idx)
        for idx, nemo_gym_example in enumerate(nemo_gym_examples)
    ]

    def passthrough_task_processor(datum_dict, *args, **kwargs):
        return datum_dict

    return AllTaskProcessedDataset(
        nemo_rl_compatible_examples,
        tokenizer,
        None,
        passthrough_task_processor,
    )


def main() -> None:
    """Main entry point for GRPO training."""
    from nemotron.kit.wandb import (
        patch_nemo_rl_checkpoint_logging,
        patch_wandb_http_handler_skip_digest_verification,
        patch_wandb_local_file_handler_skip_digest_verification,
        patch_wandb_runid_for_seeded_random,
    )

    patch_wandb_http_handler_skip_digest_verification()
    patch_wandb_local_file_handler_skip_digest_verification()
    patch_wandb_runid_for_seeded_random()
    patch_nemo_rl_checkpoint_logging()

    import wandb.util

    wandb.util.VALUE_BYTES_LIMIT = 10_000_000

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
    from nemo_rl.utils.config import load_config, parse_hydra_overrides
    from nemo_rl.utils.logger import get_next_experiment_dir
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("mul", lambda a, b: a * b)

    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "config",
            "default.yaml",
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    from nemo_runspec.config.resolvers import clear_artifact_cache, register_resolvers_from_config

    clear_artifact_cache()
    register_resolvers_from_config(
        config,
        artifacts_key="run",
        mode="pre_init",
        pre_init_patch_http_digest=False,
    )

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    initial_checkpoint = config.get("initial_checkpoint")
    if initial_checkpoint:
        if config.get("convert_initial_checkpoint_to_hf", False):
            hf_checkpoint_path = convert_megatron_to_hf(
                megatron_checkpoint_path=initial_checkpoint,
                hf_model_id=config["policy"]["model_name"],
                output_dir=config.get("converted_checkpoint_dir"),
            )
            config["policy"]["model_name"] = hf_checkpoint_path
            config["policy"]["tokenizer"]["name"] = hf_checkpoint_path
            print(f"Updated model_name to converted checkpoint: {hf_checkpoint_path}")

        elif config["checkpointing"]["enabled"]:
            setup_initial_checkpoint(initial_checkpoint, config["checkpointing"]["checkpoint_dir"])

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(f"Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}")

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, "A generation config is required for GRPO"
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    setup_nemo_gym_config(config, tokenizer)

    assert _should_use_nemo_gym(config)

    print("\nSetting up data...")
    train_dataset = setup_single_nemo_gym_dataset(
        jsonl_fpath=config["data"]["train_jsonl_fpath"],
        tokenizer=tokenizer,
    )
    val_dataset = setup_single_nemo_gym_dataset(
        jsonl_fpath=config["data"]["validation_jsonl_fpath"],
        tokenizer=tokenizer,
    )

    if config["grpo"]["max_val_samples"] is not None:
        raise ValueError(
            "A non-null `grpo.max_val_samples` parameter is not supported. "
            "The validation set you pass in will directly be used for validation "
            "with no additional preprocessing."
        )

    print(
        f"Setting `grpo.max_val_samples` and `grpo.val_batch_size` to the length "
        f"of the validation dataset, which is {len(val_dataset)}"
    )
    config["grpo"]["max_val_samples"] = len(val_dataset)
    config["grpo"]["val_batch_size"] = config["grpo"]["max_val_samples"]

    print("Final config:")
    pprint.pprint(config)

    init_ray()

    (
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
                data: list[list[str]] = value.data
                rows_to_log.extend(v[0] for v in data)

            logger.log_string_list_as_jsonl(rows_to_log, log_filename)

        policy_generation.finish_generation()
    else:
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


if __name__ == "__main__":
    main()
