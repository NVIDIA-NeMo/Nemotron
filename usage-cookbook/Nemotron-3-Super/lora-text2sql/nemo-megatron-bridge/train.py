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

import argparse
import logging
import os
import sys
from typing import Tuple
import math
import shutil

import torch
from omegaconf import OmegaConf

from megatron.bridge.recipes.nemotronh.nemotron_3_super import (
    nemotron_3_super_finetune_config as finetune_config,
)
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)

from megatron.bridge.training.config import FinetuningDatasetConfig

# This script must be launched via torchrun (see 2-train.sh). Running it directly will misconfigure
# distributed training and can lead to confusing failures.
if "LOCAL_RANK" not in os.environ and "RANK" not in os.environ:
    raise RuntimeError(
        "This script must be launched with torchrun (e.g. `N_DEVICES=2 ./2-train.sh`). "
        "Do not run `python train.py` directly."
    )

# All variables below are mandatory and must be set by the caller (e.g. notebook %env cells).
# They have no defaults so that misconfiguration is caught early.
for _required_var in ("EXPERIMENT_NAME", "BASE_MODEL_PATH", "DATASET_DIR", "TRAINING_OUTPUT_DIR", "N_DEVICES", "MAX_SEQ_LEN"):
    if _required_var not in os.environ:
        raise RuntimeError(
            f"{_required_var} environment variable must be set explicitly. "
            f"See the notebook configuration cells or the README for details."
        )

# A human-readable name for the experiment (used in output folder names).
EXPERIMENT_NAME = os.environ["EXPERIMENT_NAME"]

BASE_MODEL_PATH = os.environ["BASE_MODEL_PATH"]
DATASET_DIR = os.environ["DATASET_DIR"]
TRAINING_OUTPUT_DIR = os.environ["TRAINING_OUTPUT_DIR"]  # Root directory for training artifacts
N_DEVICES = int(os.environ["N_DEVICES"])
MAX_SEQ_LEN = int(os.environ["MAX_SEQ_LEN"])
PER_DEVICE_BS = int(os.environ.get("PER_DEVICE_BS", "1"))
GLOBAL_BS = int(os.environ.get("GLOBAL_BS", "32"))  # Must be divisible by the number of GPUs
LORA_RANK = int(os.environ.get("LORA_RANK", "64"))
LR = float(os.environ.get("LR", "5e-5"))
MIN_LR = float(os.environ.get("MIN_LR", "0"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.001"))
CLIP_GRAD = float(os.environ.get("CLIP_GRAD", "1.0"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.03"))
LORA_TARGET_MODULES = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]

# Training length controls
N_EXAMPLES = os.environ.get("N_EXAMPLES")  # If unset, inferred from DATASET_DIR/training.jsonl
EPOCHS = int(os.environ.get("EPOCHS", "1"))
MAX_STEPS = os.environ.get("MAX_STEPS")  # If set, caps total optimizer steps (overrides EPOCHS).

SAVE_RATIO = 0.25  # Save every X% of entire training

# MoE dispatcher controls.
# The default "flex"+"deepep" path can fail if DeepEP kernels don't match the system/CUDA runtime.
# Use "alltoall" by default for portability; override if you have a working DeepEP build.
MOE_TOKEN_DISPATCHER_TYPE = os.environ.get("MOE_TOKEN_DISPATCHER_TYPE", "alltoall")  # e.g. alltoall|flex
MOE_FLEX_DISPATCHER_BACKEND = os.environ.get("MOE_FLEX_DISPATCHER_BACKEND")  # e.g. deepep (only used if type=flex)


# --------------------------------------------------------------------------------------------------
# You don't need to touch things below here
# --------------------------------------------------------------------------------------------------
def _training_jsonl_path(dataset_root: str) -> str:
    return dataset_root if dataset_root.endswith(".jsonl") else os.path.join(dataset_root, "training.jsonl")

def _count_jsonl_rows(fp: str) -> int:
    with open(fp, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

if N_EXAMPLES is None:
    jsonl_fp = _training_jsonl_path(DATASET_DIR)
    if not os.path.exists(jsonl_fp):
        raise FileNotFoundError(
            f"Couldn't find training JSONL at '{jsonl_fp}'. Set DATASET_DIR to the folder containing training.jsonl "
            f"(or set N_EXAMPLES explicitly)."
        )
    N_EXAMPLES = _count_jsonl_rows(jsonl_fp)
else:
    N_EXAMPLES = int(N_EXAMPLES)

# Determine how many steps of training is one epoch, based on the number of examples and batch size.
STEPS_PER_EPOCH = int(math.ceil(N_EXAMPLES / GLOBAL_BS))
N_STEPS = int(MAX_STEPS) if MAX_STEPS is not None else (EPOCHS * STEPS_PER_EPOCH)
CKPT_STEP = max(1, int(SAVE_RATIO * N_STEPS)) if N_STEPS > 0 else 1

run_name = f"{EXPERIMENT_NAME}-lora-{LORA_RANK}-gbs-{GLOBAL_BS}-lr-{LR}-warmup-{WARMUP_RATIO}-max_steps-{N_STEPS}-seq_len-{MAX_SEQ_LEN}"
output_dir = os.path.join(TRAINING_OUTPUT_DIR, run_name)

logger = logging.getLogger(__name__)

if os.environ["LOCAL_RANK"] == "0":
    os.makedirs(output_dir, exist_ok=True)
    # Copy this script to the output dir for reference
    shutil.copy(os.path.abspath(__file__), output_dir)
    # Update a "latest" symlink so downstream steps can find this run easily
    latest_link = os.path.join(TRAINING_OUTPUT_DIR, "latest")
    if os.path.islink(latest_link):
        os.remove(latest_link)
    os.symlink(output_dir, latest_link)

def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments, separating known script args from OmegaConf overrides."""
    parser = argparse.ArgumentParser(
        description="Finetune Llama3 8B model using Megatron-Bridge with YAML and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to the YAML OmegaConf override file. Default: conf/llama3_8b_pretrain_override_example.yaml", # TODO: update yaml file name
    )
    parser.add_argument("--packed-sequence", action="store_true", help="Whether to use sequence packing")
    parser.add_argument("--seq-length", type=int, default=MAX_SEQ_LEN,help="Sequence length")

    # Parse known args for the script, remaining will be treated as overrides
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def main() -> None:
    """
    Entry point for the Mamba 8B finetuning script.
    """
    args, cli_overrides = parse_cli_args()
    dataset = FinetuningDatasetConfig(
        dataset_root=DATASET_DIR,  # Path to your preprocessed dataset (JSONL, etc.)
        seq_length=MAX_SEQ_LEN,                    # Max sequence length for input tokens
        seed=1234,                          # Seed for reproducibility
        num_workers=8,                      # DataLoader worker threads
        pin_memory=True,                    # Optimize data transfer to GPU
        do_validation = False,                 # Whether to run validation
        do_test = False,
    )

    if not os.path.isdir(BASE_MODEL_PATH) or not os.listdir(BASE_MODEL_PATH):
        raise FileNotFoundError(
            f"BASE_MODEL_PATH is missing or empty: '{BASE_MODEL_PATH}'. "
            f"Run ./1-convert-hf-to-bridge.sh first (MEGATRON_PATH should match BASE_MODEL_PATH), "
            f"or set BASE_MODEL_PATH to the converted Megatron-Bridge checkpoint directory."
        )
    if N_DEVICES <= 0:
        raise ValueError(
            f"No GPUs detected/visible (torch.cuda.device_count()={torch.cuda.device_count()}). "
            f"Set N_DEVICES explicitly and/or check container GPU visibility."
        )
    if GLOBAL_BS % N_DEVICES != 0:
        raise ValueError(f"GLOBAL_BS must be divisible by N_DEVICES. Got GLOBAL_BS={GLOBAL_BS}, N_DEVICES={N_DEVICES}.")

    cfg: ConfigContainer = finetune_config(
        tensor_model_parallel_size=N_DEVICES,
        dir=output_dir,
        seq_length=args.seq_length,
        peft=LoRA(
            target_modules=LORA_TARGET_MODULES,
            dim=LORA_RANK,
            alpha=2 * LORA_RANK,
            dropout=0.0,
        ),
        packed_sequence=args.packed_sequence,
        expert_model_parallelism=N_DEVICES,
        global_batch_size=GLOBAL_BS,
        micro_batch_size=PER_DEVICE_BS,
        finetune_lr=LR,
        min_lr=MIN_LR,
        lr_warmup_iters=int(WARMUP_RATIO * N_STEPS),
        train_iters=N_STEPS,
    )
    cfg.model.seq_length = args.seq_length
    cfg.model.calculate_per_token_loss = True  # Why was this enabled in the config.yaml file?
    cfg.checkpoint.pretrained_checkpoint = BASE_MODEL_PATH
    cfg.checkpoint.save_interval = CKPT_STEP
    cfg.dataset = dataset
    cfg.optimizer.clip_grad = CLIP_GRAD
    cfg.optimizer.weight_decay = WEIGHT_DECAY
    cfg.logger.log_interval = 1

    # Prefer portable MoE dispatcher defaults to avoid DeepEP kernel issues.
    cfg.model.moe_token_dispatcher_type = MOE_TOKEN_DISPATCHER_TYPE
    if MOE_TOKEN_DISPATCHER_TYPE != "flex":
        # Ensure we don't route into DeepEP-backed flex dispatch.
        cfg.model.moe_enable_deepep = False
        cfg.model.moe_flex_dispatcher_backend = None
    elif MOE_FLEX_DISPATCHER_BACKEND:
        cfg.model.moe_flex_dispatcher_backend = MOE_FLEX_DISPATCHER_BACKEND

    # Convert the initial Python dataclass to an OmegaConf DictConfig for merging
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    # Load and merge YAML overrides if a config file is provided
    if args.config_file:
        logger.debug(f"Loading YAML overrides from: {args.config_file}")
        if not os.path.exists(args.config_file):
            logger.error(f"Override YAML file not found: {args.config_file}")
            sys.exit(1)
        yaml_overrides_omega = OmegaConf.load(args.config_file)
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides_omega)
        logger.debug("YAML overrides merged successfully.")

    # Apply command-line overrides using Hydra-style parsing
    if cli_overrides:
        logger.debug(f"Applying Hydra-style command-line overrides: {cli_overrides}")
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)
        logger.debug("Hydra-style command-line overrides applied successfully.")

    # Apply the final merged OmegaConf configuration back to the original ConfigContainer
    logger.debug("Applying final merged configuration back to Python ConfigContainer...")
    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    # Apply overrides while preserving excluded fields
    apply_overrides(cfg, final_overrides_as_dict, excluded_fields)

    # Start training
    logger.debug("Starting finetuning...")
    finetune(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
