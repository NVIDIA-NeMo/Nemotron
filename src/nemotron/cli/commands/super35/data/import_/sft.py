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

"""Import an Energon SFT dataset as W&B artifact."""

from __future__ import annotations

from pathlib import Path

import typer

from nemo_runspec.env import get_wandb_config
from nemotron.kit.artifact import EnergonWebDatasetArtifact
from nemotron.kit.wandb_kit import WandbConfig, init_wandb_if_configured


def sft(
    data_dir: Path = typer.Argument(..., help="Energon WebDataset root (tar shards + .nv-meta/)"),
    name: str | None = typer.Option(
        None, "--name", "-n", help="Artifact name (default: super35-sft-data)"
    ),
    project: str | None = typer.Option(
        None, "--project", "-p", help="W&B project (overrides env.toml)"
    ),
    entity: str | None = typer.Option(
        None, "--entity", "-e", help="W&B entity (overrides env.toml)"
    ),
) -> None:
    """Import an Energon WebDataset (SFT blend) as a W&B artifact.

    The Super-VL SFT recipes read image/video-text samples through
    Megatron-Bridge's Energon loader (``dataset.path: ${art:data,path}``).
    Prepare the dataset with ``energon prepare`` first (no audio samples).
    The Super 3.5 SFT blends are not public; this registers your own.

    Examples:
        nemotron super35 data import sft /path/to/energon_dataset
        nemotron super35 data import sft /path/to/stage2_energon --name super35-sft-stage2-data
    """
    # Resolve data directory
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        typer.echo(f"Error: Data directory does not exist: {data_dir}", err=True)
        raise typer.Exit(1)

    # Energon datasets carry their metadata in .nv-meta/
    if not (data_dir / ".nv-meta").is_dir():
        typer.echo(
            f"Error: No .nv-meta/ directory in {data_dir}. "
            "Run `energon prepare` on the dataset first.",
            err=True,
        )
        raise typer.Exit(1)

    # Build W&B config from env.toml with CLI overrides
    env_wandb = get_wandb_config()
    wandb_project = project or (env_wandb.project if env_wandb else None)
    wandb_entity = entity or (env_wandb.entity if env_wandb else None)

    if not wandb_project:
        typer.echo("Error: W&B project required. Set in env.toml or use --project", err=True)
        raise typer.Exit(1)

    wandb_config = WandbConfig(project=wandb_project, entity=wandb_entity)

    # Initialize W&B
    init_wandb_if_configured(wandb_config, job_type="data-import", tags=["sft", "import"])

    # Create artifact (shard count only; per-split sample counts live in .nv-meta)
    artifact_name = name or "super35/sft/data"
    artifact = EnergonWebDatasetArtifact(
        path=data_dir,
        num_shards=len(list(data_dir.glob("*.tar"))),
        name=artifact_name,
    )

    # Save and register with W&B
    artifact.save()

    typer.echo(f"Imported SFT data from {data_dir}")
    typer.echo(f"Artifact: {artifact_name}")
