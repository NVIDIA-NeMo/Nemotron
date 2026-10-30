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

"""Run Omni multimodal sanity-check inference."""

from __future__ import annotations

from pathlib import Path

import typer

from nemotron.cli.commands.omni3.model._base import execute_model_command


def eval_cmd(
    ctx: typer.Context,
    hf_model_path: str = typer.Option(..., "--hf-model-path", help="HuggingFace model ID or local path."),
    megatron_model_path: Path | None = typer.Option(
        None,
        "--megatron-model-path",
        help="Megatron checkpoint directory. Omit to use the script's auto-conversion path.",
    ),
    prompt: str = typer.Option(..., "--prompt", help="Prompt text to evaluate."),
    image_path: Path | None = typer.Option(None, "--image-path", help="Optional image input."),
    video_path: Path | None = typer.Option(None, "--video-path", help="Optional video input."),
    audio_path: Path | None = typer.Option(None, "--audio-path", help="Optional audio input."),
    max_new_tokens: int = typer.Option(100, "--max-new-tokens", help="Generation length."),
    tp: int | None = typer.Option(None, "--tp", help="Tensor parallel size."),
    ep: int | None = typer.Option(None, "--ep", help="Expert parallel size."),
    nproc_per_node: int = typer.Option(1, "--nproc-per-node", help="torchrun local world size."),
) -> None:
    """Run hf_to_megatron_generate_nemotron_omni.py."""
    if not any((image_path, video_path, audio_path)):
        raise typer.BadParameter("Provide at least one of --image-path, --video-path, or --audio-path.")

    command = [
        "uv",
        "run",
        "torchrun",
        f"--nproc-per-node={nproc_per_node}",
        "examples/conversion/hf_to_megatron_generate_nemotron_omni.py",
        "--hf_model_path",
        hf_model_path,
        "--prompt",
        prompt,
        "--max_new_tokens",
        str(max_new_tokens),
    ]
    if megatron_model_path is not None:
        command.extend(["--megatron_model_path", str(megatron_model_path)])
    if image_path is not None:
        command.extend(["--image_path", str(image_path)])
    if video_path is not None:
        command.extend(["--video_path", str(video_path)])
    if audio_path is not None:
        command.extend(["--audio_path", str(audio_path)])
    if tp is not None:
        command.extend(["--tp", str(tp)])
    if ep is not None:
        command.extend(["--ep", str(ep)])

    execute_model_command(
        ctx,
        job_name="omni3/model/eval",
        command=command,
        gpus_per_node=nproc_per_node,
    )
