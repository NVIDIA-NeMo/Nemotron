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

"""Import SFT data as W&B artifact."""

from __future__ import annotations

from pathlib import Path

import typer

from nemotron.kit.artifact import DataBlendsArtifact
from nemo_runspec.env import get_wandb_config
from nemotron.kit.wandb_kit import WandbConfig, init_wandb_if_configured


def sft(
    data_dir: Path = typer.Argument(..., help="Path to data directory containing blend.json"),
    name: str | None = typer.Option(None, "--name", "-n", help="Custom artifact name"),
    project: str | None = typer.Option(
        None, "--project", "-p", help="W&B project (overrides env.toml)"
    ),
    entity: str | None = typer.Option(
        None, "--entity", "-e", help="W&B entity (overrides env.toml)"
    ),
) -> None:
    """Import SFT data directory as a W&B artifact.

    Examples:
        nemotron nano3 data import sft /path/to/data_dir
        nemotron nano3 data import sft /path/to/data_dir --project my-project
    """
    # Resolve data directory
    data_dir = data_dir.resolve()
    if not data_dir.exists():
        typer.echo(f"Error: Data directory does not exist: {data_dir}", err=True)
        raise typer.Exit(1)

    if not data_dir.is_dir():
        typer.echo(f"Error: Path is not a directory: {data_dir}", err=True)
        raise typer.Exit(1)

    # Look for blend.json in directory
    blend_path = data_dir / "blend.json"
    if not blend_path.exists():
        typer.echo(f"Error: No blend.json found in directory: {data_dir}", err=True)
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

    # Create artifact with minimal required fields
    artifact_name = name or "nano3/sft/data"
    artifact = DataBlendsArtifact(
        path=blend_path,
        total_tokens=0,
        total_sequences=0,
        name=artifact_name,
    )

    # Save and register with W&B
    artifact.save()

    typer.echo(f"Imported SFT data from {data_dir}")
    typer.echo(f"Artifact: {artifact_name}")
