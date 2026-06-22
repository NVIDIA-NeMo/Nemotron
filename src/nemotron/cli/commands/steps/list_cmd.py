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

"""`nemotron steps list` — discovery for humans and agents."""

from __future__ import annotations

import json as json_module
from collections import defaultdict
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from nemotron.steps.index import CATEGORY_TITLES, StepInfo, discover_steps

console = Console()


def _matches(
    step: StepInfo,
    *,
    category: str | None,
    consumes: str | None,
    produces: str | None,
    tag: str | None,
) -> bool:
    if category and step.category != category:
        return False
    if consumes and not any(a.type == consumes for a in step.consumes):
        return False
    if produces and not any(a.type == produces for a in step.produces):
        return False
    if tag and tag not in step.tags:
        return False
    return True


def _step_to_dict(step: StepInfo) -> dict:
    return {
        "id": step.id,
        "name": step.name,
        "category": step.category,
        "description": step.description,
        "tags": list(step.tags),
        "path": str(step.path),
        "consumes": [{"type": a.type, "required": a.required, "description": a.description} for a in step.consumes],
        "produces": [{"type": a.type, "required": a.required, "description": a.description} for a in step.produces],
        "parameters": [
            {"name": p.name, "default": p.default, "description": p.description, "choices": list(p.choices)}
            for p in step.parameters
        ],
    }


def _render_table(steps: list[StepInfo]) -> None:
    table = Table(title="Available Steps", show_lines=False)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Category")
    table.add_column("Consumes")
    table.add_column("Produces")
    table.add_column("Description", overflow="fold")

    for step in steps:
        consumes_str = ", ".join(a.type for a in step.consumes) or "-"
        produces_str = ", ".join(a.type for a in step.produces) or "-"
        table.add_row(step.id, step.category, consumes_str, produces_str, step.description.split("\n")[0])

    console.print(table)


def _render_tree(steps: list[StepInfo]) -> None:
    grouped: dict[str, list[StepInfo]] = defaultdict(list)
    for step in steps:
        grouped[step.category].append(step)

    root = Tree("[bold]Available Steps[/bold]")
    for category in sorted(grouped):
        title = CATEGORY_TITLES.get(category, category)
        branch = root.add(f"[cyan]{category}[/cyan] — {title}")
        for step in grouped[category]:
            summary = step.description.split("\n")[0] or "(no description)"
            branch.add(f"[bold]{step.id}[/bold]  [dim]{summary}[/dim]")

    console.print(root)


def list_steps(
    category: Annotated[
        str | None,
        typer.Option("--category", help="Filter by category (sft, peft, rl, …)."),
    ] = None,
    consumes: Annotated[
        str | None,
        typer.Option("--consumes", help="Only steps that consume this artifact type."),
    ] = None,
    produces: Annotated[
        str | None,
        typer.Option("--produces", help="Only steps that produce this artifact type."),
    ] = None,
    tag: Annotated[
        str | None,
        typer.Option("--tag", help="Only steps whose manifest tags include this value."),
    ] = None,
    tree: Annotated[
        bool,
        typer.Option("--tree", help="Group steps by category in a tree view."),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit JSON array (agent-friendly).")
    ] = False,
) -> None:
    """List discovered steps. Use ``--json`` for machine-readable output or ``--tree`` for a grouped view."""

    steps = [
        s
        for s in discover_steps()
        if _matches(s, category=category, consumes=consumes, produces=produces, tag=tag)
    ]

    if as_json:
        typer.echo(json_module.dumps([_step_to_dict(s) for s in steps], indent=2))
        return

    if not steps:
        console.print("[yellow]No steps matched.[/yellow]")
        return

    if tree:
        _render_tree(steps)
        return

    _render_table(steps)
