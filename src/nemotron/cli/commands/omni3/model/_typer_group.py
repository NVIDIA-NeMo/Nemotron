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

"""Model Typer group for omni3."""

from __future__ import annotations

import typer

from nemotron.cli.commands.omni3.model.adapter_export import adapter_export
from nemotron.cli.commands.omni3.model.eval import eval_cmd
from nemotron.cli.commands.omni3.model.export import export_app
from nemotron.cli.commands.omni3.model.import_ import import_app
from nemotron.cli.commands.omni3.model.lora_merge import lora_merge

model_app = typer.Typer(
    name="model",
    help="Model conversion and lifecycle commands",
    no_args_is_help=True,
)

model_app.add_typer(import_app, name="import")
model_app.add_typer(export_app, name="export")
model_app.command(name="eval")(eval_cmd)
model_app.command(name="lora-merge")(lora_merge)
model_app.command(name="adapter-export")(adapter_export)
