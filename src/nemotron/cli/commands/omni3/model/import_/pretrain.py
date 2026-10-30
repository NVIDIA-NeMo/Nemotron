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

"""Convert a HuggingFace Omni checkpoint to Megatron format."""

from __future__ import annotations

from pathlib import Path

import typer

from nemotron.cli.commands.omni3.model._base import execute_model_command


def pretrain(
    ctx: typer.Context,
    hf_model: str = typer.Option(..., "--hf-model", help="HuggingFace model ID or local path."),
    megatron_path: Path = typer.Option(..., "--megatron-path", help="Output Megatron checkpoint directory."),
    trust_remote_code: bool = typer.Option(
        True,
        "--trust-remote-code/--no-trust-remote-code",
        help="Forward --trust-remote-code to the converter.",
    ),
) -> None:
    """Run convert_checkpoints.py import."""
    command = [
        "uv",
        "run",
        "python",
        "examples/conversion/convert_checkpoints.py",
        "import",
        "--hf-model",
        hf_model,
        "--megatron-path",
        str(megatron_path),
    ]
    if trust_remote_code:
        command.append("--trust-remote-code")

    execute_model_command(
        ctx,
        job_name="omni3/model/import/pretrain",
        command=command,
        gpus_per_node=0,
        time_limit="02:00:00",
    )
