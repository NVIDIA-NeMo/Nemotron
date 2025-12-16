#!/usr/bin/env python3

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

"""RL (Reinforcement Learning) script for Nemotron Nano3.

Uses NeMo-RL's GRPO algorithm for reinforcement learning training.
This script is designed to run inside a container with NeMo-RL installed.

Usage:
    # Direct execution inside container (nemo-run with Ray)
    python /path/to/train.py --config /path/to/grpo_config.yaml

    # With CLI overrides (Hydra syntax)
    python /path/to/train.py --config /path/to/grpo_config.yaml \
        grpo.num_iterations=100 \
        policy.generation.temperature=0.7
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
        description="Run GRPO training for Nemotron Nano3",
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
    # Apply wandb monkey patches early, before any wandb imports/init
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

    # Increase W&B single object size warning threshold
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

    # Parse arguments
    args, overrides = parse_args()

    # Use default config if not specified
    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "grpo_nanov3.yaml",
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    # Register nemotron artifact resolver for ${art:...} interpolations
    # This must happen before OmegaConf.to_container() resolves the config
    from nemotron.kit.resolvers import clear_artifact_cache, register_resolvers_from_config

    clear_artifact_cache()
    register_resolvers_from_config(
        config,
        artifacts_key="run",
        mode="pre_init",
        pre_init_patch_http_digest=False,
    )

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(f"Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}")

    # Setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, "A generation config is required for GRPO"
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # NeMo-Gym specific config setup
    setup_nemo_gym_config(config, tokenizer)

    # We assert here since this is right after the final config has been materialized
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

    # Validation dataset config setup
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

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Initialize Ray
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
    # Blocking wait for NeMo-Gym to spin up
    ray.get(nemo_gym.health_check.remote())
    task_to_env = {"nemo_gym": nemo_gym}
    val_task_to_env = task_to_env

    if is_trajectory_collection:
        from nemo_rl.algorithms.grpo import refit_policy_generation
        from nemo_rl.experience.rollouts import run_async_nemo_gym_rollout
        from wandb import Table

        # Run trajectory collection
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
