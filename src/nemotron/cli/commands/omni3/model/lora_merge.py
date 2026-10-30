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

"""Merge Omni LoRA adapters into the base Megatron checkpoint."""

from __future__ import annotations

from pathlib import Path

import typer

from nemotron.cli.commands.omni3.model._base import execute_model_command


def lora_merge(
    ctx: typer.Context,
    lora_checkpoint: Path = typer.Option(..., "--lora-checkpoint", help="LoRA iteration checkpoint directory."),
    hf_model_path: str = typer.Option(..., "--hf-model-path", help="Base HuggingFace model ID or local path."),
    output: Path = typer.Option(..., "--output", help="Merged Megatron checkpoint directory."),
    tp: int = typer.Option(4, "--tp", help="Tensor parallel size."),
    nproc_per_node: int | None = typer.Option(
        None,
        "--nproc-per-node",
        help="torchrun local world size. Defaults to --tp when omitted.",
    ),
) -> None:
    """Run merge_lora.py."""
    local_world_size = nproc_per_node or tp
    command = [
        "uv",
        "run",
        "torchrun",
        f"--nproc-per-node={local_world_size}",
        "examples/peft/merge_lora.py",
        "--lora-checkpoint",
        str(lora_checkpoint),
        "--hf-model-path",
        hf_model_path,
        "--output",
        str(output),
        "--tp",
        str(tp),
    ]

    execute_model_command(
        ctx,
        job_name="omni3/model/lora-merge",
        command=command,
        gpus_per_node=local_world_size,
    )
