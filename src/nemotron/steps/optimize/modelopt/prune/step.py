#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/optimize/modelopt/prune"
# image = "nvcr.io/nvidia/nemo:26.04"
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

"""Generic ModelOpt structured-pruning launcher through Megatron-Bridge."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from nemotron.steps._runners.modelopt import (
    exec_torchrun_script,
    run_modelopt_setup_command,
    validate_model_optimizer_checkout,
)

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"
UPSTREAM_SCRIPT = "/opt/Model-Optimizer/examples/megatron_bridge/prune_minitron.py"
MODELOPT_REPO = "https://github.com/NVIDIA/Model-Optimizer.git"
MODELOPT_INSTALL_ENV = "NEMOTRON_MODELOPT_INSTALL_CHECKOUT"
MODELOPT_SYNC_ENV = "NEMOTRON_MODELOPT_SYNC_EXAMPLES"

# Backward-compatible flat config keys. New configs should put upstream script
# arguments under `args:` so users can control ModelOpt without editing Python.
LEGACY_FORWARDED_FIELDS = (
    "pp_size",
    "hf_model_name_or_path",
    "output_hf_path",
    "prune_target_params",
    "prune_export_config",
    "hparams_to_skip",
    "num_layers_in_first_pipeline_stage",
    "num_layers_in_last_pipeline_stage",
    "trust_remote_code",
)


def ensure_model_optimizer_examples(script_path: str) -> None:
    script = Path(script_path)
    repo_root = script.parents[2]
    if script.exists() and not _env_flag(MODELOPT_SYNC_ENV):
        return
    if (repo_root / ".git").exists() and _env_flag(MODELOPT_SYNC_ENV):
        run_modelopt_setup_command(
            ["git", "-C", str(repo_root), "fetch", "--depth", "1", "origin", "main"],
        )
        run_modelopt_setup_command(
            ["git", "-C", str(repo_root), "checkout", "--force", "origin/main"],
        )
    elif not repo_root.exists():
        repo_root.parent.mkdir(parents=True, exist_ok=True)
        run_modelopt_setup_command(
            ["git", "clone", "--depth", "1", MODELOPT_REPO, str(repo_root)],
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
    repo_root = validate_model_optimizer_checkout(script_path)
    run_modelopt_setup_command(
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
    )
    os.environ["PYTHONPATH"] = f"{repo_root}:{os.environ.get('PYTHONPATH', '')}"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def install_moe_grouped_gemm_compat_patch() -> None:
    """Drop the obsolete ``moe_grouped_gemm`` kwarg without mutating /opt files."""
    patch_dir = Path(tempfile.mkdtemp(prefix="nemotron-modelopt-prune-compat-"))
    (patch_dir / "sitecustomize.py").write_text(
        """
import inspect

try:
    from modelopt.torch.utils.plugins import mbridge

    original = mbridge.load_mbridge_model_from_hf
    if "moe_grouped_gemm" not in inspect.signature(original).parameters:
        def load_mbridge_model_from_hf_compat(*args, **kwargs):
            kwargs.pop("moe_grouped_gemm", None)
            return original(*args, **kwargs)

        mbridge.load_mbridge_model_from_hf = load_mbridge_model_from_hf_compat
except Exception as exc:
    print(f"[modelopt] prune compatibility patch disabled: {type(exc).__name__}: {exc}", flush=True)
""".lstrip(),
        encoding="utf-8",
    )
    os.environ["PYTHONPATH"] = f"{patch_dir}:{os.environ.get('PYTHONPATH', '')}"


def main() -> None:
    ensure_model_optimizer_examples(UPSTREAM_SCRIPT)
    install_model_optimizer_checkout(UPSTREAM_SCRIPT)
    install_moe_grouped_gemm_compat_patch()
    exec_torchrun_script(
        default_config=DEFAULT_CONFIG,
        upstream_script=UPSTREAM_SCRIPT,
        forwarded_fields=LEGACY_FORWARDED_FIELDS,
        flag_style="underscore",
        default_nproc_per_node=8,
    )


if __name__ == "__main__":
    main()
