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

"""RL Typer subgroup.

Contains the RL command group with subcommands for each RL sub-stage:
- stage1_rlvr: RLVR (RL with Verifiable Rewards) — 3 sequential runs
- stage2_swe1: SWE-RL Stage 1 (SWE-Pivot)
- stage2_swe2: SWE-RL Stage 2 (SWE-bench with Apptainer)
- stage3_rlhf: RLHF (RL from Human Feedback) with GenRM

NOTE: RL training currently requires running inside the NeMo-RL repository
(nvcr.io/nvidia/nemo-rl container). The CLI commands are prepared but
commented out until Megatron checkpoint consumption is supported.
For now, run RL sub-stages directly via NeMo-RL:

    cd /opt/nemo-rl
    python train.py --config /path/to/config.yaml
"""

from __future__ import annotations

from nemo_runspec.recipe_typer import RecipeTyper

# Create rl app as a subgroup of super3
rl_app = RecipeTyper(
    name="rl",
    help="Reinforcement learning sub-stages (RLVR, SWE-RL, RLHF)",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# TODO(super3): Enable RL sub-stage CLI commands once Megatron checkpoint
# consumption is supported in the nemotron CLI. For now, run RL training
# directly inside the NeMo-RL container. Config files and train.py scripts
# are available in src/nemotron/recipes/super3/stage2_rl/{stage1_rlvr,stage2_swe1,stage2_swe2,stage3_rlhf}/.
#
# from nemotron.cli.commands.super3.rl.stage1_rlvr import META as RLVR_META, stage1_rlvr
# from nemotron.cli.commands.super3.rl.stage2_swe1 import META as SWE1_META, stage2_swe1
# from nemotron.cli.commands.super3.rl.stage2_swe2 import META as SWE2_META, stage2_swe2
# from nemotron.cli.commands.super3.rl.stage3_rlhf import META as RLHF_META, stage3_rlhf
#
# rl_app.add_recipe_command(stage1_rlvr, meta=RLVR_META, rich_help_panel="RL Sub-Stages")
# rl_app.add_recipe_command(stage2_swe1, meta=SWE1_META, rich_help_panel="RL Sub-Stages")
# rl_app.add_recipe_command(stage2_swe2, meta=SWE2_META, rich_help_panel="RL Sub-Stages")
# rl_app.add_recipe_command(stage3_rlhf, meta=RLHF_META, rich_help_panel="RL Sub-Stages")
