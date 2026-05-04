#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/optimize/modelopt/distill"
# image = "nvcr.io/nvidia/nemo:26.02"
#
# [tool.runspec.run]
# launch = "python"
# workdir = "/opt/Model-Optimizer/examples/megatron_bridge"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
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

"""Generic ModelOpt distillation launcher through Megatron-Bridge."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from nemotron.steps._runners.modelopt import exec_torchrun_script

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"
UPSTREAM_SCRIPT = "/opt/Model-Optimizer/examples/megatron_bridge/distill.py"
MODELOPT_REPO = "https://github.com/NVIDIA/Model-Optimizer.git"
MODELOPT_INSTALL_ENV = "NEMOTRON_MODELOPT_INSTALL_CHECKOUT"
MODELOPT_SYNC_ENV = "NEMOTRON_MODELOPT_SYNC_EXAMPLES"

# Backward-compatible flat config keys. New configs should put upstream script
# arguments under `args:` so users can control ModelOpt without editing Python.
LEGACY_FORWARDED_FIELDS = (
    "tp_size",
    "pp_size",
    "teacher_hf_path",
    "student_hf_path",
    "student_hf_model",
    "data_paths",
    "data_path_to_cache",
    "seq_length",
    "mbs",
    "gbs",
    "train_iters",
    "lr",
    "min_lr",
    "lr_warmup_iters",
    "eval_interval",
    "eval_iters",
    "log_interval",
    "output_dir",
    "use_mock_data",
    "hf_export_path",
    "wandb_project",
    "wandb_entity",
    "wandb_exp_name",
    "trust_remote_code",
)


def ensure_model_optimizer_examples(script_path: str) -> None:
    script = Path(script_path)
    repo_root = script.parents[2]
    if script.exists() and not _env_flag(MODELOPT_SYNC_ENV):
        return
    if (repo_root / ".git").exists() and _env_flag(MODELOPT_SYNC_ENV):
        subprocess.run(
            ["git", "-C", str(repo_root), "fetch", "--depth", "1", "origin", "main"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_root), "checkout", "--force", "origin/main"],
            check=True,
        )
    elif not repo_root.exists():
        repo_root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", MODELOPT_REPO, str(repo_root)],
            check=True,
        )
    if not script.exists():
        raise FileNotFoundError(
            f"{script} was not found. Mount a ModelOpt checkout at {repo_root}, "
            f"or set {MODELOPT_SYNC_ENV}=1 to refresh a git checkout from {MODELOPT_REPO}."
        )


def install_model_optimizer_checkout(script_path: str) -> None:
    """Install the cloned ModelOpt checkout so examples and library match."""
    if not _env_flag(MODELOPT_INSTALL_ENV):
        repo_root = Path(script_path).parents[2]
        os.environ["PYTHONPATH"] = f"{repo_root}:{os.environ.get('PYTHONPATH', '')}"
        return
    repo_root = Path(script_path).parents[2]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "--break-system-packages",
            "-e",
            str(repo_root),
        ],
        check=True,
    )
    os.environ["PYTHONPATH"] = f"{repo_root}:{os.environ.get('PYTHONPATH', '')}"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def main() -> None:
    ensure_model_optimizer_examples(UPSTREAM_SCRIPT)
    install_model_optimizer_checkout(UPSTREAM_SCRIPT)
    exec_torchrun_script(
        default_config=DEFAULT_CONFIG,
        upstream_script=UPSTREAM_SCRIPT,
        forwarded_fields=LEGACY_FORWARDED_FIELDS,
        flag_style="underscore",
        default_nproc_per_node=8,
    )


if __name__ == "__main__":
    main()
