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

"""Rich display utilities for CLI output.

Includes:
- Dry-run configuration display with YAML syntax highlighting
- Job submission summary with tree view
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree

from nemo_runspec.env import get_cli_config
from nemo_runspec.utils import rewrite_paths_for_remote, resolve_run_interpolations

# Global console instance
CONSOLE = Console()

# Default theme for syntax highlighting
DEFAULT_THEME = "monokai"


def _get_theme() -> str:
    """Get the syntax highlighting theme from env.toml or use default."""
    cli_config = get_cli_config()
    if cli_config and "theme" in cli_config:
        return str(cli_config.theme)
    return DEFAULT_THEME


def display_job_config(job_config: DictConfig, *, for_remote: bool = False) -> None:
    """Display the full job configuration as syntax-highlighted YAML.

    Config is flat - training config at root, run section has execution/provenance.

    Args:
        job_config: The compiled job configuration
        for_remote: If True, resolve all interpolations and rewrite paths for remote execution
    """
    CONSOLE.print()
    CONSOLE.print("[bold cyan]Compiled Configuration[/bold cyan]")
    CONSOLE.print()

    # Display run section (contains recipe, env, cli)
    _display_run_section(job_config)

    # Display training config (everything except run section)
    _display_config_section(job_config, for_remote=for_remote)

    CONSOLE.print()


def _display_run_section(job_config: DictConfig) -> None:
    """Display the run section as syntax-highlighted YAML."""
    run = job_config.get("run", {})
    if not run:
        return

    # Convert to YAML string
    yaml_str = OmegaConf.to_yaml(run, resolve=False)

    syntax = Syntax(yaml_str.rstrip(), "yaml", theme=_get_theme(), line_numbers=False)
    CONSOLE.print(
        Panel(
            syntax,
            title="[bold green]run[/bold green]",
            border_style="green",
            expand=False,
        )
    )
    CONSOLE.print()


def _display_config_section(job_config: DictConfig, *, for_remote: bool = False) -> None:
    """Display the training config as syntax-highlighted YAML."""
    # Create a copy without resolving interpolations
    config_dict = OmegaConf.to_container(job_config, resolve=False)
    run_section = config_dict.pop("run", {})

    if not config_dict:
        return

    if for_remote:
        # Rewrite paths for remote execution display
        import os

        repo_root_str = os.getcwd()
        config_dict = rewrite_paths_for_remote(config_dict, repo_root_str)
    else:
        # Resolve ${run.*} interpolations for display
        config_dict = resolve_run_interpolations(config_dict, run_section)

    # Convert back to OmegaConf for YAML serialization
    config_without_run = OmegaConf.create(config_dict)
    yaml_str = OmegaConf.to_yaml(config_without_run, resolve=False)

    syntax = Syntax(yaml_str.rstrip(), "yaml", theme=_get_theme(), line_numbers=False)
    CONSOLE.print(
        Panel(
            syntax,
            title="[bold green]config[/bold green]",
            border_style="green",
            expand=False,
        )
    )
    CONSOLE.print()


def display_job_submission(
    job_path: Path,
    train_path: Path,
    env_vars: dict[str, str],
    mode: str,
) -> None:
    """Display job submission summary as a Rich tree panel.

    Args:
        job_path: Path to job.yaml
        train_path: Path to train.yaml
        env_vars: Environment variables being set
        mode: Execution mode (run/batch/local)
    """
    # Build tree
    tree = Tree("[bold]Job Submission[/bold]")

    # Configs section
    configs = tree.add("[cyan]configs[/cyan]")
    configs.add(f"[dim]job:[/dim]   {job_path}")
    configs.add(f"[dim]train:[/dim] {train_path}")

    # Environment variables section (if any interesting ones)
    interesting_vars = {k: v for k, v in env_vars.items() if k not in ("NEMO_RUN_DIR",)}
    if interesting_vars:
        env_section = tree.add("[cyan]env[/cyan]")
        for key in sorted(interesting_vars.keys()):
            # Mask sensitive values
            if "KEY" in key or "TOKEN" in key or "SECRET" in key:
                env_section.add(f"[dim]{key}:[/dim] [green]✓ detected[/green]")
            else:
                env_section.add(f"[dim]{key}:[/dim] {interesting_vars[key]}")

    # Mode indicator
    mode_label = {
        "run": "[yellow]attached[/yellow]",
        "batch": "[blue]detached[/blue]",
        "local": "[green]local[/green]",
    }.get(mode, mode)
    tree.add(f"[cyan]mode:[/cyan] {mode_label}")

    CONSOLE.print()
    CONSOLE.print(Panel(tree, border_style="green", expand=False))
    CONSOLE.print()


def display_ray_job_submission(
    script_path: str,
    script_args: list[str],
    env_vars: dict[str, str],
    mode: str,
) -> None:
    """Display Ray job submission summary as a Rich tree panel.

    Args:
        script_path: Path to the script being executed
        script_args: Arguments passed to the script
        env_vars: Environment variables being set
        mode: Execution mode (attached/detached)
    """
    # Build tree
    tree = Tree("[bold]Job Submission[/bold]")

    # Script section
    script_section = tree.add("[cyan]script[/cyan]")
    script_section.add(f"[dim]path:[/dim] {script_path}")
    if script_args:
        script_section.add(f"[dim]args:[/dim] {' '.join(script_args)}")

    # Environment variables section (if any interesting ones)
    interesting_vars = {k: v for k, v in env_vars.items() if k not in ("NEMO_RUN_DIR",)}
    if interesting_vars:
        env_section = tree.add("[cyan]env[/cyan]")
        for key in sorted(interesting_vars.keys()):
            # Mask sensitive values
            if "KEY" in key or "TOKEN" in key or "SECRET" in key:
                env_section.add(f"[dim]{key}:[/dim] [green]✓ detected[/green]")
            else:
                env_section.add(f"[dim]{key}:[/dim] {interesting_vars[key]}")

    # Mode indicator
    mode_label = {
        "attached": "[yellow]attached[/yellow]",
        "detached": "[blue]detached[/blue]",
    }.get(mode, mode)
    tree.add(f"[cyan]mode:[/cyan] {mode_label}")

    CONSOLE.print()
    CONSOLE.print(Panel(tree, border_style="green", expand=False))
    CONSOLE.print()
