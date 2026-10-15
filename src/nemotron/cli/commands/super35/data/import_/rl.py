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

"""Import RL JSONL data (one RL stage) as W&B artifact."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from nemotron.kit.artifact import SplitJsonlDataArtifact
from nemo_runspec.env import get_wandb_config
from nemotron.kit.wandb_kit import WandbConfig, init_wandb_if_configured


def rl(
    train: Path = typer.Argument(..., help="Training JSONL file (NeMo Gym rows)"),
    val: Path | None = typer.Option(None, "--val", "-v", help="Validation JSONL file (default: the training file)"),
    name: str | None = typer.Option(
        None,
        "--name",
        "-n",
        help="Artifact name, one per RL stage (default: super35-rl-rlvr-data)",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Where to write manifest.json/metadata.json (default: <train dir>/<artifact name>)",
    ),
    project: str | None = typer.Option(
        None, "--project", "-p", help="W&B project (overrides env.toml)"
    ),
    entity: str | None = typer.Option(
        None, "--entity", "-e", help="W&B entity (overrides env.toml)"
    ),
) -> None:
    """Import one RL stage's train/val JSONL as a W&B artifact.

    Each RL stage config reads ``data.train.data_path: ${art:data,train}`` and
    ``data.validation.data_path: ${art:data,val}`` from its own data artifact,
    so register one artifact per stage with ``--name``. The JSONL files are
    referenced in place (keep them on storage the cluster can see); only a
    small manifest is written next to them.

    Examples:
        nemotron super35 data import rl /data/rlvr/train.jsonl --val /data/rlvr/val.jsonl
        nemotron super35 data import rl /data/video/train.jsonl --val /data/video/val.jsonl \\
            --name super35-rl-teacher-video-data
        nemotron super35 data import rl /data/mopd/p1.jsonl --name super35-rl-mopd-phase1-data
    """
    train = train.resolve()
    if not train.is_file():
        typer.echo(f"Error: Training file does not exist: {train}", err=True)
        raise typer.Exit(1)

    if val is not None:
        val = val.resolve()
        if not val.is_file():
            typer.echo(f"Error: Validation file does not exist: {val}", err=True)
            raise typer.Exit(1)
    else:
        # The reference runs validate on the training file (validation is off
        # in every Super 3.5 RL stage), so mirror that rather than leave val null.
        val = train
        typer.echo("No --val given; using the training file for validation (as the reference runs do).")

    # Build W&B config from env.toml with CLI overrides
    env_wandb = get_wandb_config()
    wandb_project = project or (env_wandb.project if env_wandb else None)
    wandb_entity = entity or (env_wandb.entity if env_wandb else None)

    if not wandb_project:
        typer.echo("Error: W&B project required. Set in env.toml or use --project", err=True)
        raise typer.Exit(1)

    artifact_name = name or "super35-rl-rlvr-data"

    # The artifact's path is a manifest.json; write it (plus metadata.json,
    # written by artifact.save()) into a per-artifact directory so several
    # stages can share one data directory without clobbering each other.
    out_dir = (output_dir or train.parent / artifact_name.replace("/", "-")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"train": str(train), "val": str(val)}
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    wandb_config = WandbConfig(project=wandb_project, entity=wandb_entity)

    # Initialize W&B
    init_wandb_if_configured(wandb_config, job_type="data-import", tags=["rl", "import"])

    artifact = SplitJsonlDataArtifact(
        path=manifest_path,
        total_sequences=0,
        train=str(train),
        val=str(val),
        name=artifact_name,
    )

    # Save and register with W&B
    artifact.save()

    typer.echo(f"Imported RL data: train={train} val={val}")
    typer.echo(f"Manifest: {manifest_path}")
    typer.echo(f"Artifact: {artifact_name}")
