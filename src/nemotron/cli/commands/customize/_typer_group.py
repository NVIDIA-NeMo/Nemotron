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

"""Customize Typer group.

Contains the customize command group with subcommands for the full
Nemotron model customization pipeline (translate, data-prep, CPT, SFT,
SDG, RL, BYOB, eval, quantize).

When running inside the orchestrator container (``NEMOTRON_ORCHESTRATOR=1``),
commands are automatically dispatched to the correct sibling container via
``docker exec``.  When running directly inside a worker container (curator,
trainer, evaluator), commands execute locally as usual.

Design: LLM-Native Recipe Architecture
- Uses RecipeTyper for standardized command registration
- Each command module has visible execution logic
- Orchestrator auto-dispatch via nemotron_customize dispatcher
"""

from __future__ import annotations

import os
import sys

import typer

from nemotron.cli.commands.customize.byob import META as BYOB_META
from nemotron.cli.commands.customize.byob import byob
from nemotron.cli.commands.customize.cpt import META as CPT_META
from nemotron.cli.commands.customize.cpt import cpt
from nemotron.cli.commands.customize.data_prep import META as DATA_PREP_META
from nemotron.cli.commands.customize.data_prep import data_prep
from nemotron.cli.commands.customize.eval import META as EVAL_META
from nemotron.cli.commands.customize.eval import eval as eval_cmd
from nemotron.cli.commands.customize.quantize import META as QUANTIZE_META
from nemotron.cli.commands.customize.quantize import quantize
from nemotron.cli.commands.customize.rl import META as RL_META
from nemotron.cli.commands.customize.rl import rl
from nemotron.cli.commands.customize.sdg import META as SDG_META
from nemotron.cli.commands.customize.sdg import sdg
from nemotron.cli.commands.customize.sft import META as SFT_META
from nemotron.cli.commands.customize.sft import sft
from nemotron.cli.commands.customize.translate import META as TRANSLATE_META
from nemotron.cli.commands.customize.translate import translate
from nemo_runspec.recipe_typer import RecipeTyper

# Create customize app using RecipeTyper
customize_app = RecipeTyper(
    name="customize",
    help="Nemotron model customization recipes (translate, data-prep, CPT, SFT, SDG, RL, BYOB, eval, quantize)",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@customize_app.callback(invoke_without_command=True)
def _orchestrator_callback(ctx: typer.Context) -> None:
    """Intercept commands when running in the orchestrator container.

    If ``NEMOTRON_ORCHESTRATOR=1`` is set in the environment, this callback
    extracts the subcommand and remaining arguments from the Typer invocation
    context and delegates to the ``nemotron_customize`` dispatcher, which
    routes the command to the correct sibling container via ``docker exec``.

    When *not* in orchestrator mode (i.e. running directly inside a worker
    container or on the host without the env var), this callback is a no-op
    and the normal Typer command dispatch proceeds.
    """
    if os.environ.get("NEMOTRON_ORCHESTRATOR") != "1":
        return  # Normal execution — let Typer handle it

    # If no subcommand was given, let Typer show help as usual
    if ctx.invoked_subcommand is None:
        return

    # Build the argv to forward to the dispatcher.
    # ctx.invoked_subcommand is the subcommand name (e.g. "sft").
    # We need to reconstruct the remaining arguments. Typer stores
    # the *original* sys.argv, so we extract everything after "customize".
    original_args = sys.argv[1:]  # drop program name

    # Find "customize" in argv and take everything after it
    dispatch_args: list[str] = []
    found_customize = False
    for arg in original_args:
        if found_customize:
            dispatch_args.append(arg)
        elif arg == "customize":
            found_customize = True

    if not dispatch_args:
        return  # Safety: nothing to dispatch

    # Import and call the dispatcher
    from nemotron.cli.bin.nemotron_customize import dispatch

    dispatch(dispatch_args)
    # dispatch() calls sys.exit(), so we never reach here

# =============================================================================
# Register Customization Commands
#
# Each command exports a META object with config_dir, input/output_artifacts.
# Execution logic stays visible in each command module.
# =============================================================================

# Data Preparation
customize_app.add_recipe_command(data_prep, meta=DATA_PREP_META, rich_help_panel="Data Preparation")
customize_app.add_recipe_command(sdg, meta=SDG_META, rich_help_panel="Data Preparation")
customize_app.add_recipe_command(translate, meta=TRANSLATE_META, rich_help_panel="Data Preparation")

# Training Stages
customize_app.add_recipe_command(cpt, meta=CPT_META, rich_help_panel="Training Stages")
customize_app.add_recipe_command(sft, meta=SFT_META, rich_help_panel="Training Stages")
customize_app.add_recipe_command(rl, meta=RL_META, rich_help_panel="Training Stages")

# Benchmarking & Evaluation
customize_app.add_recipe_command(byob, meta=BYOB_META, rich_help_panel="Benchmarking & Evaluation")
customize_app.add_recipe_command(eval_cmd, meta=EVAL_META, rich_help_panel="Benchmarking & Evaluation")

# Export & Optimization
customize_app.add_recipe_command(quantize, meta=QUANTIZE_META, rich_help_panel="Export & Optimization")
