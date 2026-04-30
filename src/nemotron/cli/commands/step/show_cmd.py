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

"""`nemotron step show` — full manifest + runspec for one step."""
from __future__ import annotations

import json as json_module
from dataclasses import asdict
from typing import Annotated

import typer
from rich.console import Console

from nemo_runspec import parse as parse_runspec
from nemotron.cli.commands.step._resolve import resolve_step
from nemotron.cli.commands.step.list_cmd import _step_to_dict

console = Console()


def show_step(
    step_id: Annotated[str, typer.Argument(help="Step id, e.g. peft/automodel.")],
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON (agent-friendly).")] = False,
) -> None:
    step = resolve_step(step_id)
    script_path = step.path / "step.py"
    spec = parse_runspec(str(script_path)) if script_path.exists() else None

    if as_json:
        out = _step_to_dict(step)
        out["runspec"] = _runspec_to_dict(spec)
        typer.echo(json_module.dumps(out, indent=2, default=str))
        return

    console.rule(f"[bold cyan]{step.id}[/bold cyan] — {step.name}")
    console.print(f"[dim]{step.path}[/dim]\n")
    console.print(step.description + "\n")

    if step.consumes:
        console.print("[bold]Consumes[/bold]")
        for a in step.consumes:
            tag = "" if a.required else " (optional)"
            console.print(f"  • [green]{a.type}[/green]{tag} — {a.description}")
    if step.produces:
        console.print("\n[bold]Produces[/bold]")
        for a in step.produces:
            console.print(f"  • [green]{a.type}[/green] — {a.description}")
    if step.parameters:
        console.print("\n[bold]Parameters[/bold]")
        for p in step.parameters:
            default = "" if p.default is None else f" (default={p.default})"
            console.print(f"  • [yellow]{p.name}[/yellow]{default} — {p.description}")
    if spec is not None:
        console.print("\n[bold]Runspec[/bold]")
        console.print(f"  launcher: [magenta]{spec.run.launch}[/magenta]")
        console.print(f"  image: {spec.image or '-'}")
        console.print(f"  resources: nodes={spec.resources.nodes} gpus_per_node={spec.resources.gpus_per_node}")
        console.print(f"  config dir: {spec.config_dir}")
        console.print(f"  default config: {spec.config.default}")


def _runspec_to_dict(spec) -> dict | None:
    if spec is None:
        return None
    return {
        "name": spec.name,
        "image": spec.image,
        "launch": spec.run.launch,
        "config_dir": str(spec.config_dir),
        "default_config": spec.config.default,
        "resources": asdict(spec.resources),
    }
