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

"""Export Omni LoRA adapters to HuggingFace PEFT format."""

from __future__ import annotations

from pathlib import Path

import typer

from nemotron.cli.commands.omni3.model._base import execute_model_command


def adapter_export(
    ctx: typer.Context,
    hf_model_path: str = typer.Option(..., "--hf-model-path", help="Base HuggingFace model ID or local path."),
    lora_checkpoint: Path = typer.Option(..., "--lora-checkpoint", help="LoRA iteration checkpoint directory."),
    output: Path = typer.Option(..., "--output", help="Output adapter directory."),
    trust_remote_code: bool = typer.Option(
        True,
        "--trust-remote-code/--no-trust-remote-code",
        help="Forward --trust-remote-code to the exporter.",
    ),
) -> None:
    """Run export_adapter.py."""
    command = [
        "uv",
        "run",
        "python",
        "examples/conversion/adapter/export_adapter.py",
        "--hf-model-path",
        hf_model_path,
        "--lora-checkpoint",
        str(lora_checkpoint),
        "--output",
        str(output),
    ]
    if trust_remote_code:
        command.append("--trust-remote-code")

    execute_model_command(
        ctx,
        job_name="omni3/model/adapter-export",
        command=command,
        gpus_per_node=0,
        time_limit="02:00:00",
    )
