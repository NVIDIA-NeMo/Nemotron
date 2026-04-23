#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "omni3/rl/vision"
# image = "oci-archive:///home/${oc.env:USER}/.cache/nemotron/containers/omni3-rl.tar"
# setup = "Build the Omni RL container with `nemotron omni3 build rl` before training."
#
# [tool.runspec.run]
# launch = "ray"
# workdir = "/opt/nemo-rl-omni"
# cmd = "uv run python {script} --config {config}"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "omegaconf"
#
# [tool.runspec.resources]
# # Stub allocation (nodes=1, gpus_per_node=0) — prevents an errant
# # `nemotron omni3 rl vision --batch` from tying up a real GPU allocation
# # while the launcher body is still pending upstream. When the real
# # launcher lands this should be bumped to the production footprint
# # (e.g. nodes=16, gpus_per_node=8) alongside the main() body.
# nodes = 1
# gpus_per_node = 0
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

"""Stub entrypoint for Omni vision RL until the upstream launcher lands."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "default.yaml"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse the placeholder CLI so `--help` works."""
    parser = argparse.ArgumentParser(
        description="Omni vision RL is not implemented yet.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML config file",
    )
    return parser.parse_known_args()


def main() -> None:
    """Raise until the upstream vision launcher is available."""
    parse_args()
    # TODO(wave3): replace this stub when the upstream vision RL launcher lands.
    raise NotImplementedError(
        "Vision RL launcher pending upstream — see omni-3 rl - 3-vision-rl"
    )


if __name__ == "__main__":
    main()
