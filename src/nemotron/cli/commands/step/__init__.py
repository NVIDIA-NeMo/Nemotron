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

"""Generic step CLI — list / show / run any discovered step.

Designed for agentic use: every step.py + step.toml in src/nemotron/steps/ is
auto-discovered. The agent's surface is uniform regardless of the underlying
framework (AutoModel, Megatron-Bridge, NeMo-RL, Data Designer).
"""
from __future__ import annotations

import typer

from nemotron.cli.commands.step.list_cmd import list_steps
from nemotron.cli.commands.step.run_cmd import run_step
from nemotron.cli.commands.step.show_cmd import show_step

step_app = typer.Typer(
    name="step",
    help="Discover, inspect, and run any registered step.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

step_app.command("list", help="List discovered steps. Use --json for machine-readable output.")(list_steps)
step_app.command("show", help="Show a step's manifest, runspec, and parameters.")(show_step)
step_app.command(
    "run",
    help="Run a step on the chosen executor profile.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)(run_step)

__all__ = ["step_app"]
