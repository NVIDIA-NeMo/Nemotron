#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "omni3/sft"
# image = "oci-archive:///home/${oc.env:USER}/.cache/nemotron/containers/omni3-sft.tar"
# setup = "Build the Omni SFT container with `nemotron omni3 build sft` before training."
#
# [tool.runspec.run]
# launch = "torchrun"
# workdir = "/workspace/Megatron-Bridge"
# cmd = "python {script} --recipe {recipe} --step_func {step_func} --config {config}"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 2
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

"""Omni SFT entry point — thin forwarder to Megatron-Bridge's run_recipe.py.

This module exists primarily to carry the PEP 723 ``[tool.runspec]`` metadata
that the `nemotron omni3 sft` CLI reads (image, launch method, config
directory, resource footprint). The ``cmd`` template above declares the
contract: "torchrun-wrap ``python {script} --recipe … --step_func …
--config …``".

nemo-run's `Torchrun` launcher wraps this command with the correct
``--nproc-per-node`` / ``--nnodes`` / ``--node-rank`` / rendezvous flags
from Slurm env vars automatically — we do *not* hand-roll that logic. The
container's PATH is set so ``python`` resolves to the Megatron-Bridge
uv-synced venv (see stage0_sft/Dockerfile), which means no ``uv run``
prefix is needed anywhere in the runspec or CLI.

The module body is intentionally tiny: we accept the three ``--recipe`` /
``--step_func`` / ``--config`` flags the CLI passes through, then
``os.execvp`` into Megatron-Bridge's own ``run_recipe.py`` with the
same arguments plus any Hydra-style overrides. Using ``execvp`` means
there's no extra Python process — the child inherits torchrun's DDP
env vars directly.

If you want to run this recipe outside the Nemotron CLI, skip this file
and invoke Megatron-Bridge's script directly:

    cd /workspace/Megatron-Bridge
    torchrun --nproc-per-node=8 scripts/training/run_recipe.py \\
        --recipe nemotron_omni_valor32k_energon_sft_config \\
        --step_func nemotron_omni_step \\
        --config /path/to/default.yaml
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

RUN_RECIPE = Path("/workspace/Megatron-Bridge/scripts/training/run_recipe.py")


def main() -> None:
    """Forward all CLI args to run_recipe.py via execvp."""
    if not RUN_RECIPE.is_file():
        sys.exit(
            f"error: {RUN_RECIPE} not found. This wrapper expects to run inside the "
            f"omni3-sft container (`nemotron omni3 build sft`). Outside the container, "
            f"invoke Megatron-Bridge's scripts/training/run_recipe.py directly."
        )
    os.execvp("python", ["python", str(RUN_RECIPE), *sys.argv[1:]])


if __name__ == "__main__":
    main()
