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

"""CLI command for the Curator-backed translation step."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
import typer

from nemo_runspec import parse as parse_runspec
from nemo_runspec.config import parse_config
from nemo_runspec.recipe_config import RecipeConfig, parse_recipe_config
from nemo_runspec.recipe_typer import RecipeMeta

SCRIPT_PATH = "src/nemotron/steps/translate/translation/step.py"
SPEC = parse_runspec(SCRIPT_PATH)

META = RecipeMeta(
    name=SPEC.name,
    script_path=SCRIPT_PATH,
    config_dir=str(SPEC.config_dir),
    default_config=SPEC.config.default,
    input_artifacts={"data": "JSONL or Parquet corpus to translate"},
    output_artifacts={"translated": "Translated JSONL or Parquet output shards"},
)


def _as_plain_dict(config: DictConfig) -> dict[str, Any]:
    data = OmegaConf.to_container(config, resolve=True)
    if not isinstance(data, dict):
        raise TypeError("Translation config must be a mapping")
    return data


def _load_translation_config(cfg: RecipeConfig) -> dict[str, Any]:
    """Load the translation step YAML and apply CLI dotlist overrides."""
    return _as_plain_dict(parse_config(cfg.ctx, SPEC.config_dir, SPEC.config.default))


def _run_translation_step(config: dict[str, Any]) -> Path:
    """Call the checked-in translation step runtime."""
    from nemotron.steps.translate.translation.step import run

    return run(config)


def translation(ctx: typer.Context) -> None:
    """Run corpus translation with NeMo Curator.

    Example:
        nemotron steps translation \\
          input_path=/data/source.jsonl \\
          output_dir=/data/translated \\
          source_language=en \\
          target_language=hi
    """
    cfg = parse_recipe_config(ctx)

    if cfg.mode != "local":
        typer.echo(
            "Error: nemotron steps translation currently supports local execution only.",
            err=True,
        )
        raise typer.Exit(1)

    if cfg.passthrough:
        typer.echo(
            "Error: nemotron steps translation accepts key=value config overrides only.",
            err=True,
        )
        raise typer.Exit(1)

    try:
        config = _load_translation_config(cfg)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if cfg.dry_run:
        typer.echo(OmegaConf.to_yaml(OmegaConf.create(config), resolve=True))
        return

    output_path = _run_translation_step(config)
    typer.echo(f"Translation complete. Output: {output_path}")
