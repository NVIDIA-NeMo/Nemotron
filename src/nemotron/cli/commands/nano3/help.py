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

"""Custom help formatting for nano3 CLI commands.

Provides RecipeCommand class that extends TyperCommand with custom help panels
for global options, run overrides, artifact overrides, and examples.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from rich import box
from rich.panel import Panel
from rich.table import Table
from typer import rich_utils
from typer.core import TyperCommand

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def _get_env_profiles() -> list[str]:
    """Get list of profile names from env.toml, excluding special sections."""
    env_path = Path("env.toml")
    if not env_path.exists():
        return []

    try:
        with open(env_path, "rb") as f:
            config = tomllib.load(f)
        # Exclude special sections like 'wandb'
        special_sections = {"wandb"}
        return [k for k in config.keys() if k not in special_sections]
    except Exception:
        return []


def _get_available_configs(config_dir: str | None) -> list[str]:
    """Get list of available config names from a config directory.

    Args:
        config_dir: Path to config directory (relative to repo root).

    Returns:
        List of config names (without .yaml extension), excluding subdirectories.
    """
    if not config_dir:
        return []

    config_path = Path(config_dir)
    if not config_path.exists():
        return []

    try:
        configs = []
        for f in config_path.iterdir():
            if f.is_file() and f.suffix in (".yaml", ".yml"):
                configs.append(f.stem)
        return sorted(configs)
    except Exception:
        return []


class RecipeCommand(TyperCommand):
    """Custom TyperCommand that adds recipe-specific help panels.

    Class attributes:
        artifact_overrides: Dict mapping artifact names to descriptions.
            Example: {"data": "Data artifact", "model": "Model checkpoint"}
        config_dir: Path to config directory (relative to repo root).
    """

    artifact_overrides: ClassVar[dict[str, str]] = {}
    config_dir: ClassVar[str | None] = None

    def format_help(self, ctx, formatter):
        """Format help with custom recipe options section."""
        # First, render standard Typer help
        rich_utils.rich_format_help(
            obj=self,
            ctx=ctx,
            markup_mode=self.rich_markup_mode,
        )

        # Then add our custom panels
        console = rich_utils._get_rich_console()
        cmd_name = ctx.info_name

        # Global options table
        options_table = Table(
            show_header=False,
            box=box.SIMPLE,
            padding=(0, 2),
            pad_edge=False,
        )
        options_table.add_column("Option", style="green", no_wrap=True)
        options_table.add_column("Description")
        options_table.add_row("-c, --config NAME", "Config name or path")
        options_table.add_row("-r, --run PROFILE", "Submit to cluster (attached)")
        options_table.add_row("-b, --batch PROFILE", "Submit to cluster (detached)")
        options_table.add_row("-d, --dry-run", "Preview config without execution")
        options_table.add_row("--stage", "Stage files for interactive debugging")

        console.print(
            Panel(
                options_table,
                title="[bold]Global Options[/]",
                title_align="left",
                border_style=rich_utils.STYLE_OPTIONS_PANEL_BORDER,
            )
        )

        # Available configs (if config_dir is set)
        configs = _get_available_configs(self.config_dir)
        if configs:
            config_list = ", ".join(f"[cyan]{c}[/]" for c in configs)
            console.print(
                Panel(
                    f"Built-in: {config_list}\n"
                    "[dim]Custom:[/] -c /path/to/your/config.yaml",
                    title="[bold]Configs[/] (-c/--config)",
                    title_align="left",
                    border_style=rich_utils.STYLE_OPTIONS_PANEL_BORDER,
                )
            )

        # Artifact overrides (if any defined for this command)
        if self.artifact_overrides:
            artifact_table = Table(
                show_header=False,
                box=box.SIMPLE,
                padding=(0, 2),
                pad_edge=False,
            )
            artifact_table.add_column("Override", style="cyan", no_wrap=True)
            artifact_table.add_column("Description")
            for name, desc in self.artifact_overrides.items():
                artifact_table.add_row(f"run.{name}", desc)

            console.print(
                Panel(
                    artifact_table,
                    title="[bold]Artifact Overrides[/] (W&B artifact references)",
                    title_align="left",
                    border_style=rich_utils.STYLE_OPTIONS_PANEL_BORDER,
                )
            )

        # Run overrides table
        run_table = Table(
            show_header=False,
            box=box.SIMPLE,
            padding=(0, 2),
            pad_edge=False,
        )
        run_table.add_column("Override", style="yellow", no_wrap=True)
        run_table.add_column("Description")
        run_table.add_row("run.env.nodes", "Number of nodes")
        run_table.add_row("run.env.nproc_per_node", "GPUs per node")
        run_table.add_row("run.env.partition", "Slurm partition")
        run_table.add_row("run.env.account", "Slurm account")
        run_table.add_row("run.env.time", "Job time limit (e.g., 04:00:00)")
        run_table.add_row("run.env.container_image", "Override container image")

        console.print(
            Panel(
                run_table,
                title="[bold]Run Overrides[/] (override env.toml settings)",
                title_align="left",
                border_style=rich_utils.STYLE_OPTIONS_PANEL_BORDER,
            )
        )

        # env.toml profiles
        profiles = _get_env_profiles()
        if profiles:
            profile_list = ", ".join(f"[cyan]{p}[/]" for p in profiles)
            console.print(
                Panel(
                    f"Available profiles: {profile_list}\n"
                    "[dim]Usage:[/] --run PROFILE or --batch PROFILE",
                    title="[bold]env.toml Profiles[/]",
                    title_align="left",
                    border_style=rich_utils.STYLE_OPTIONS_PANEL_BORDER,
                )
            )

        console.print(
            Panel(
                "Override config values: [yellow]key.path=value[/]\n"
                f"[dim]Example:[/] ... {cmd_name} -c tiny [yellow]train.train_iters=5000[/]",
                title="[bold]Dotlist Overrides[/]",
                title_align="left",
                border_style=rich_utils.STYLE_OPTIONS_PANEL_BORDER,
            )
        )

        console.print(
            Panel(
                f"[green]$ ... {cmd_name} -c tiny[/]                    [dim]Local execution[/]\n"
                f"[green]$ ... {cmd_name} -c tiny --dry-run[/]          [dim]Preview config[/]\n"
                f"[green]$ ... {cmd_name} -c tiny --run my-cluster[/]   [dim]Submit to cluster[/]\n"
                f"[green]$ ... {cmd_name} -c tiny -r cluster run.env.nodes=4[/]",
                title="[bold]Examples[/]",
                title_align="left",
                border_style=rich_utils.STYLE_OPTIONS_PANEL_BORDER,
            )
        )


def make_recipe_command(
    artifact_overrides: dict[str, str] | None = None,
    config_dir: str | None = None,
):
    """Factory function to create a RecipeCommand subclass with custom options.

    Args:
        artifact_overrides: Dict mapping artifact names to descriptions.
            Example: {"data": "Data artifact", "model": "Model checkpoint"}
        config_dir: Path to config directory (relative to repo root).

    Returns:
        A RecipeCommand subclass with the specified options.
    """

    class CustomRecipeCommand(RecipeCommand):
        pass

    CustomRecipeCommand.artifact_overrides = artifact_overrides or {}
    CustomRecipeCommand.config_dir = config_dir
    return CustomRecipeCommand
