"""Import pretrain data as W&B artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from nemotron.kit.artifact import DataBlendsArtifact
from nemotron.kit.cli.env import get_wandb_config
from nemotron.kit.wandb import WandbConfig, init_wandb_if_configured


def pretrain(
    data_path: Path = typer.Argument(..., help="Path to blend.json file"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Custom artifact name"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="W&B project (overrides env.toml)"),
    entity: Optional[str] = typer.Option(None, "--entity", "-e", help="W&B entity (overrides env.toml)"),
) -> None:
    """Import pretrain data (blend.json) as a W&B artifact.

    Examples:
        nemotron nano3 data import pretrain /path/to/blend.json
        nemotron nano3 data import pretrain /path/to/blend.json --project my-project
    """
    # Resolve data path
    data_path = data_path.resolve()
    if not data_path.exists():
        typer.echo(f"Error: Data path does not exist: {data_path}", err=True)
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
    init_wandb_if_configured(wandb_config, job_type="data-import", tags=["pretrain", "import"])

    # Create artifact with minimal required fields
    artifact_name = name or "nano3/pretrain/data"
    artifact = DataBlendsArtifact(
        path=data_path,
        total_tokens=0,
        total_sequences=0,
        name=artifact_name,
    )

    # Save and register with W&B
    artifact.save()

    typer.echo(f"Imported pretrain data from {data_path}")
    typer.echo(f"Artifact: {artifact_name}")
