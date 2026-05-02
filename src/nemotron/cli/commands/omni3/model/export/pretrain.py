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

"""Convert a Megatron Omni checkpoint back to HuggingFace format."""

from __future__ import annotations

from pathlib import Path

import typer

from nemotron.cli.commands.omni3.model._base import execute_model_command


def pretrain(
    ctx: typer.Context,
    hf_model: str = typer.Option(..., "--hf-model", help="Base HuggingFace model ID or local path."),
    megatron_path: Path = typer.Option(..., "--megatron-path", help="Input Megatron checkpoint directory."),
    hf_path: Path = typer.Option(..., "--hf-path", help="Output HuggingFace checkpoint directory."),
    not_strict: bool = typer.Option(
        True,
        "--not-strict/--strict",
        help="Forward --not-strict to the exporter.",
    ),
) -> None:
    """Run convert_checkpoints.py export."""
    command = [
        "uv",
        "run",
        "python",
        "examples/conversion/convert_checkpoints.py",
        "export",
        "--hf-model",
        hf_model,
        "--megatron-path",
        str(megatron_path),
        "--hf-path",
        str(hf_path),
    ]
    if not_strict:
        command.append("--not-strict")

    execute_model_command(
        ctx,
        job_name="omni3/model/export/pretrain",
        command=command,
        gpus_per_node=0,
        time_limit="02:00:00",
    )
