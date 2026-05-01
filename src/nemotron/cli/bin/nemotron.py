#!/usr/bin/env python3

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

"""Nemotron CLI - Main entry point.

Usage:
    nemotron nano3 pretrain -c test                       # local execution
    nemotron nano3 pretrain --config test --run dlw       # nemo-run attached
    nemotron nano3 pretrain -c test -r dlw train.train_iters=5000
    nemotron nano3 pretrain -c test --dry-run             # preview config
"""

from __future__ import annotations

import importlib
import logging

import typer

from nemo_runspec.cli_context import global_callback

log = logging.getLogger(__name__)

# Create root app with global callback
app = typer.Typer(
    name="nemotron",
    help="Nemotron CLI - Reproducible training recipes",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="rich",
)


@app.callback()
def main_callback(
    ctx: typer.Context,
    config: str | None = typer.Option(
        None,
        "-c",
        "--config",
        help="Config name (looks in recipe's config/ dir) or path",
    ),
    run: str | None = typer.Option(
        None,
        "-r",
        "--run",
        help="Execute attached via nemo-run with specified env profile",
    ),
    batch: str | None = typer.Option(
        None,
        "-b",
        "--batch",
        help="Execute detached via nemo-run with specified env profile",
    ),
    dry_run: bool = typer.Option(
        False,
        "-d",
        "--dry-run",
        help="Print compiled config as rich table (no execution)",
    ),
    stage: bool = typer.Option(
        False,
        "--stage",
        help="Stage script + config to remote cluster for interactive debugging",
    ),
    force_squash: bool = typer.Option(
        False,
        "--force-squash",
        help="Force re-squash container image even if it already exists",
    ),
) -> None:
    """Nemotron CLI - Reproducible training recipes."""
    # Delegate to global_callback
    global_callback(ctx, config, run, batch, dry_run, stage, force_squash)


# Import and register recipe groups
def _safe_add_typer(module_path: str, attr_name: str, command_name: str) -> None:
    """Register a top-level CLI group without failing the whole CLI."""
    try:
        module = importlib.import_module(module_path)
        group = getattr(module, attr_name)
    except Exception as exc:  # pragma: no cover - defensive CLI bootstrap
        log.debug("Skipping CLI group '%s': %s", command_name, exc)
        return

    app.add_typer(group, name=command_name)


def _register_groups() -> None:
    """Register all recipe groups with the main app."""
    _safe_add_typer("nemotron.cli.commands.nano3", "nano3_app", "nano3")
    _safe_add_typer("nemotron.cli.commands.super3", "super3_app", "super3")
    _safe_add_typer("nemotron.cli.commands.steps", "steps_app", "steps")
    _safe_add_typer("nemotron.cli.kit", "kit_app", "kit")


# Register groups on import
_register_groups()


def main() -> None:
    """Entry point for the nemotron CLI."""
    app()


if __name__ == "__main__":
    main()
