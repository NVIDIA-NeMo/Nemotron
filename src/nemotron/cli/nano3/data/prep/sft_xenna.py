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

"""SFT data preparation command (Xenna execution)."""

from __future__ import annotations

import typer

from nemotron.kit.cli.recipe import recipe


@recipe(
    name="nano3/data/prep/sft-xenna",
    script_path="src/nemotron/recipes/nano3/stage1_sft/data_prep_xenna.py",
    config_dir="src/nemotron/recipes/nano3/stage1_sft/config/data_prep",
    default_config="default",
    torchrun=False,
    ray=True,
    packager="code",
    run_command="uv run --extra xenna python {script} --config {config}",
)
def sft_xenna(ctx: typer.Context) -> None:
    """Prepare data for SFT (packed .npy format) using Xenna."""
    ...
