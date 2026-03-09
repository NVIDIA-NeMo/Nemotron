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

"""Data import Typer group for nano3."""

from __future__ import annotations

import typer

from nemotron.cli.commands.nano3.data.import_.pretrain import pretrain
from nemotron.cli.commands.nano3.data.import_.rl import rl
from nemotron.cli.commands.nano3.data.import_.sft import sft

# Create import app
import_app = typer.Typer(
    name="import",
    help="Import data as W&B artifacts",
    no_args_is_help=True,
)

# Register commands
import_app.command(name="pretrain")(pretrain)
import_app.command(name="sft")(sft)
import_app.command(name="rl")(rl)
