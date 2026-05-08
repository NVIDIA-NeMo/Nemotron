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

"""Run the multi-GPU HF↔Megatron Omni roundtrip check."""

from __future__ import annotations

import typer

from nemotron.cli.commands.omni3.model._base import execute_model_command


def roundtrip(
    ctx: typer.Context,
    hf_model_id: str = typer.Option(..., "--hf-model-id", help="HuggingFace model ID or local path."),
    tp: int = typer.Option(4, "--tp", help="Tensor parallel size."),
    nproc_per_node: int = typer.Option(4, "--nproc-per-node", help="torchrun local world size."),
    trust_remote_code: bool = typer.Option(
        True,
        "--trust-remote-code/--no-trust-remote-code",
        help="Forward --trust-remote-code to the roundtrip script.",
    ),
    not_strict: bool = typer.Option(
        True,
        "--not-strict/--strict",
        help="Forward --not-strict to the roundtrip script.",
    ),
) -> None:
    """Run hf_megatron_roundtrip_multi_gpu.py."""
    command = [
        "uv",
        "run",
        "torchrun",
        f"--nproc-per-node={nproc_per_node}",
        "examples/conversion/hf_megatron_roundtrip_multi_gpu.py",
        "--hf-model-id",
        hf_model_id,
        "--tp",
        str(tp),
    ]
    if trust_remote_code:
        command.append("--trust-remote-code")
    if not_strict:
        command.append("--not-strict")

    execute_model_command(
        ctx,
        job_name="omni3/model/import/roundtrip",
        command=command,
        gpus_per_node=nproc_per_node,
    )
