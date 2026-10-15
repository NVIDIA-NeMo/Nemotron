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

"""Data command group for super35."""

from __future__ import annotations

import typer

from nemotron.cli.commands.super35.data.import_ import import_app

# Create data app
data_app = typer.Typer(
    name="data",
    help="Data import commands (the Super 3.5 blends are not public; bring your own)",
    no_args_is_help=True,
)

# Register import subgroup
data_app.add_typer(import_app, name="import")
