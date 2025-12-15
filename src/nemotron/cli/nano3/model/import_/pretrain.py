"""Import pretrain model as W&B artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from nemotron.kit.artifact import ModelArtifact
from nemotron.kit.cli.env import get_wandb_config
from nemotron.kit.wandb import WandbConfig, init_wandb_if_configured


def pretrain(
    model_dir: Path = typer.Argument(..., help="Path to model checkpoint directory"),
    step: Optional[int] = typer.Option(None, "--step", "-s", help="Training step"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Custom artifact name"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="W&B project (overrides env.toml)"),
    entity: Optional[str] = typer.Option(None, "--entity", "-e", help="W&B entity (overrides env.toml)"),
) -> None:
    """Import a pretrain model checkpoint as a W&B artifact.

    Examples:
        nemotron nano3 model import pretrain /path/to/model --step 10000
        nemotron nano3 model import pretrain /path/to/model --step 10000 --project my-project
    """
    # Resolve model directory
    model_dir = model_dir.resolve()
    if not model_dir.exists():
        typer.echo(f"Error: Model directory does not exist: {model_dir}", err=True)
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
    init_wandb_if_configured(wandb_config, job_type="model-import", tags=["pretrain", "import"])

    # Create artifact
    artifact_name = name or "nano3/pretrain/model"
    artifact = ModelArtifact(
        path=model_dir,
        step=step or 0,
        name=artifact_name,
    )

    # Save and register with W&B
    artifact.save()

    typer.echo(f"Imported pretrain model from {model_dir}")
    typer.echo(f"Artifact: {artifact_name}")
