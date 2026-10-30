# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""kit slurm Typer sub-group.

Slurm-only cluster container operations, explicit about the executor in the
command path (vs. a generic-looking verb that errors on non-Slurm executors).
Both commands are env.toml / nemo_runspec driven and run enroot on a compute
node via salloc:

    nemotron kit slurm build <profile> --recipe ultra3 --stage pretrain
    nemotron kit slurm squash <profile> <image>
"""

from __future__ import annotations

import typer

from nemotron.cli.kit.slurm.build import build
from nemotron.cli.kit.squash import squash

kit_slurm_app = typer.Typer(
    name="slurm",
    help="Slurm-only cluster container operations (build, squash).",
    no_args_is_help=True,
)

kit_slurm_app.command(name="build")(build)
kit_slurm_app.command(name="squash")(squash)
