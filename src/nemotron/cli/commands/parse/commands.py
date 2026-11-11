"""Explicit subprocess dispatch into the isolated Parse runtime."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import typer

from nemo_runspec import parse as parse_runspec
from nemo_runspec.recipe_config import parse_recipe_config
from nemo_runspec.recipe_typer import RecipeMeta
from nemotron.recipes.parse.config import ROOT, STAGES, load_config
from nemotron.recipes.parse.runtime.launch import run_module


def metadata(stage: str) -> RecipeMeta:
    model, directory, script = STAGES[stage]
    spec = parse_runspec(ROOT / directory / f"{script}.py")
    return RecipeMeta(
        name=spec.name,
        script_path=str(spec.script_path),
        config_dir=str(spec.config_dir),
        config_model=model,
        input_artifacts={} if stage == "data" else {"data_dir": "Parse dataset directory with artifact.json"},
        output_artifacts={"artifact": "Local artifact.json with data/model identity and paths"},
    )


def execute(stage: str, ctx: typer.Context) -> None:
    request = parse_recipe_config(ctx)
    if request.mode != "local" or request.stage:
        raise typer.BadParameter("Parse v1 runs commands directly; --run, --batch and --stage are not supported.")
    if request.passthrough:
        raise typer.BadParameter(f"Unrecognized arguments: {' '.join(request.passthrough)}")
    config = load_config(stage, request.config, request.dotlist)
    if request.dry_run:
        typer.echo(config.model_dump_json(indent=2))
        return
    _, directory, script = STAGES[stage]
    # Resolve the same profile once, then pass an explicit configuration to the worker.
    # No training imports or GPU allocation occur in this controller.
    with tempfile.TemporaryDirectory(prefix="nemotron-parse-config-") as temporary:
        path = Path(temporary) / "config.json"
        path.write_text(json.dumps(config.model_dump(mode="json")))
        try:
            run_module(f"nemotron.recipes.parse.{directory}.{script}", ["--config", path],
                       runtime_dir=config.runtime_dir)
        except subprocess.CalledProcessError as exc:
            raise typer.Exit(exc.returncode) from exc


def data(ctx: typer.Context) -> None:
    """Generate the reference dental images and complete Parse targets."""
    execute("data", ctx)


def sft(ctx: typer.Context) -> None:
    """Fine-tune the full Parse model with the reference SFT schedule."""
    execute("sft", ctx)


def rlvr(ctx: typer.Context) -> None:
    """Mine candidates, run decoder GRPO, and evaluate before/after."""
    execute("rlvr", ctx)


def evaluate(ctx: typer.Context) -> None:
    """Evaluate a checkpoint on the fixed held-out dental pages."""
    execute("eval", ctx)
