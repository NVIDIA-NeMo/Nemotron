# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select dependency projects without changing embedding recipe implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

_LOCAL_REVIEWED_RUNTIME_ERROR = (
    "The reviewed multimodal runtime uses commit-pinned local wheels that are ignored by Git and are not "
    "included by the container packager. Run this stage locally after preparing runtimes/wheels; "
    "container and Slurm execution are not supported for model_family=mistral3_vl yet."
)


def runtime_project(stage_dir: Path, model_family: Literal["text", "mistral3_vl"]) -> Path:
    """Choose compatible dependencies for the resolved model family.

    Args:
        stage_dir: Directory containing the embedding stage script.
        model_family: Validated recipe model family, after CLI overrides.

    Returns:
        The stage-local text project, legacy text finetuning project, or shared
        native VL project. Script location and configuration are unchanged.

    Raises:
        ValueError: If called for an unsupported family or embedding stage.
    """
    stages = {"stage1_data_prep", "stage2_finetune", "stage3_eval"}
    if stage_dir.name not in stages:
        raise ValueError(f"No embedding model runtime for stage: {stage_dir}")
    if model_family == "mistral3_vl":
        return Path(__file__).parent / "runtimes" / "native"
    if model_family != "text":
        raise ValueError(f"Unsupported embedding model family: {model_family}")
    if stage_dir.name == "stage2_finetune":
        return Path(__file__).parent / "runtimes" / "text_finetune"
    return stage_dir


def require_remote_runtime(stage: str, model_family: Literal["text", "mistral3_vl"] | None = None) -> None:
    """Reject remote lanes whose reviewed dependency artifacts are not packaged.

    Args:
        stage: Embedding stage name.
        model_family: Resolved model family for model-dependent stages.

    Raises:
        RuntimeError: If remote execution would omit required reviewed wheels.
    """
    if stage == "stage0_sdg":
        raise RuntimeError(
            "Stage 0 uses reviewed local Data Designer plugin and patched-core wheels that are ignored by Git "
            "and are not included by the container packager. Run Stage 0 locally after preparing runtimes/wheels."
        )
    if model_family == "mistral3_vl":
        raise RuntimeError(_LOCAL_REVIEWED_RUNTIME_ERROR)
