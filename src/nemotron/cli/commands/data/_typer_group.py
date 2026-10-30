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

"""Typer group for ``nemotron data``."""

from __future__ import annotations

import typer

from nemotron.cli.commands.data.sdg import sdg_app

data_app = typer.Typer(
    name="data",
    help="Data preparation and curation pipelines.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

data_app.add_typer(sdg_app, name="sdg")
