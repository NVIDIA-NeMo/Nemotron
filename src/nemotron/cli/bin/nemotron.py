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

from importlib import metadata

import typer

from nemotron.kit.cli.globals import global_callback

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
def _register_groups() -> None:
    """Register all recipe groups with the main app."""
    from nemotron.cli.embed import embed_app
    from nemotron.cli.kit import kit_app
    from nemotron.cli.nano3 import nano3_app

    app.add_typer(nano3_app, name="nano3")
    app.add_typer(kit_app, name="kit")
    app.add_typer(embed_app, name="embed")


def _register_plugins() -> None:
    """Register external CLI plugins declared via entry points.

    Plugins should be registered under the `nemotron.cli.plugins` entry-point group,
    where each entry point provides either:
    - a `typer.Typer` instance, or
    - a zero-arg callable that returns a `typer.Typer` instance.

    Example (pyproject.toml):
        [project.entry-points."nemotron.cli.plugins"]
        embed = "nemotron_embed.cli:embed_app"
    """

    group = "nemotron.cli.plugins"

    try:
        eps = metadata.entry_points()
    except Exception:
        return

    # Python 3.10+: EntryPoints has .select(); older: dict-like mapping
    if hasattr(eps, "select"):
        plugins = list(eps.select(group=group))
    else:
        plugins = list(eps.get(group, []))  # type: ignore[call-arg]

    for ep in plugins:
        name = getattr(ep, "name", None)
        if not name:
            continue

        # Avoid collisions with built-in groups
        if name in {"nano3", "kit", "embed"}:
            continue

        try:
            loaded = ep.load()
        except Exception:
            # Best-effort: a bad plugin shouldn't break the CLI.
            continue

        plugin_app: typer.Typer | None = None
        if isinstance(loaded, typer.Typer):
            plugin_app = loaded
        elif callable(loaded):
            try:
                maybe_app = loaded()
            except Exception:
                continue
            if isinstance(maybe_app, typer.Typer):
                plugin_app = maybe_app

        if plugin_app is None:
            continue

        try:
            app.add_typer(plugin_app, name=name)
        except Exception:
            continue


# Register groups on import
_register_groups()
_register_plugins()


def main() -> None:
    """Entry point for the nemotron CLI."""
    app()


if __name__ == "__main__":
    main()
