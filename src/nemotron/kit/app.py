# Copyright (c) Nemotron Contributors
# SPDX-License-Identifier: MIT

"""Minimal tyro CLI helper for nested subcommands.

This module provides an App class that simplifies nested subcommand definitions
with tyro, removing the need for manual wrapper dataclasses.

Example:
    >>> from nemotron.kit import App
    >>> from dataclasses import dataclass
    >>>
    >>> @dataclass
    ... class PreTrainConfig:
    ...     batch_size: int = 32
    ...
    >>> def pretrain_main(config: PreTrainConfig) -> None:
    ...     print(f"Training with batch_size={config.batch_size}")
    ...
    >>> app = App("myapp", description="My training app")
    >>> data = app.group("data", description="Data commands")
    >>> prep = data.group("prep", description="Prepare data")
    >>> prep.command("pretrain", PreTrainConfig, pretrain_main)
    >>>
    >>> if __name__ == "__main__":
    ...     app.run()
"""

from __future__ import annotations

from dataclasses import dataclass, field, make_dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, Union

import tyro
from tyro.conf import OmitArgPrefixes, OmitSubcommandPrefixes, subcommand

from nemotron.kit.wandb import WandbConfig


@dataclass
class GlobalOptions:
    """Global options (work at any CLI level)."""

    config_file: Annotated[
        Path | None,
        tyro.conf.arg(
            name="config-file",
            aliases=("-c",),
            help="Load config from YAML/TOML/JSON file",
        ),
    ] = None

    run: Annotated[
        str | None,
        tyro.conf.arg(
            name="run",
            aliases=("-r",),
            help="Execute via nemo-run with profile from run.toml. Use --run.<key> to override profile values.",
        ),
    ] = None

    # W&B options (flattened for single section display)
    wandb_project: Annotated[
        str | None,
        tyro.conf.arg(name="wandb.project", help="W&B project name (enables tracking)"),
    ] = None

    wandb_entity: Annotated[
        str | None,
        tyro.conf.arg(name="wandb.entity", help="W&B entity/team name"),
    ] = None

    wandb_run_name: Annotated[
        str | None,
        tyro.conf.arg(name="wandb.run-name", help="W&B run name (auto-generated if not set)"),
    ] = None

    wandb_tags: Annotated[
        tuple[str, ...],
        tyro.conf.arg(name="wandb.tags", help="W&B tags for filtering runs"),
    ] = ()

    wandb_notes: Annotated[
        str | None,
        tyro.conf.arg(name="wandb.notes", help="W&B notes/description for the run"),
    ] = None

    def to_wandb_config(self) -> WandbConfig:
        """Convert flattened fields to WandbConfig."""
        return WandbConfig(
            project=self.wandb_project,
            entity=self.wandb_entity,
            run_name=self.wandb_run_name,
            tags=self.wandb_tags,
            notes=self.wandb_notes,
        )


class App:
    """CLI application with nested subcommand support.

    Provides a declarative API for building CLI applications with nested
    subcommands, similar to typer but built on top of tyro.

    Features:
        - Clean API: Declarative `app.group()` and `app.command()` calls
        - No manual wrappers: Auto-generates wrapper dataclasses internally
        - No arg prefix: Uses `OmitArgPrefixes` → shows `--blend-path` not
          `--data.prep.blend-path`
        - Handler mapping: Automatically routes to correct handler based on
          config type
        - Composable: Groups can be nested arbitrarily deep
        - Global options: Supports `--config-file` and `--run` flags

    Example:
        >>> app = App("nano3", description="Nano3 training recipe")
        >>>
        >>> # Nested groups
        >>> data = app.group("data", description="Data commands")
        >>> prep = data.group("prep", description="Prepare data")
        >>>
        >>> # Register commands
        >>> prep.command("pretrain", PreTrainConfig, pretrain_main)
        >>> prep.command("sft", SFTConfig, sft_main)
        >>>
        >>> # Run CLI
        >>> app.run()
    """

    def __init__(self, name: str, description: str = ""):
        """Initialize a CLI application or group.

        Args:
            name: Name of the app or group (used in subcommand name)
            description: Description shown in help text
        """
        self.name = name
        self.description = description
        self._commands: list[tuple[str, type, Callable, str]] = []
        self._groups: dict[str, App] = {}

    def group(self, name: str, description: str = "") -> App:
        """Create a nested command group.

        Args:
            name: Name of the group (becomes the subcommand name)
            description: Description shown in help text

        Returns:
            A new App instance for registering nested commands
        """
        child = App(name, description)
        self._groups[name] = child
        return child

    def command(
        self,
        name: str,
        config: type,
        handler: Callable,
        description: str = "",
    ) -> None:
        """Register a command with its config and handler.

        Args:
            name: Name of the command (becomes the subcommand name)
            config: Dataclass type for command configuration
            handler: Function to call with the parsed config
            description: Description shown in help text (defaults to config's docstring)
        """
        self._commands.append((name, config, handler, description or config.__doc__ or ""))

    def _build_union(self, include_global_options: bool = False) -> tuple[type, dict[type, Callable]]:
        """Build Union type and handler mapping for tyro.

        Args:
            include_global_options: Whether to include GlobalOptions in leaf commands

        Returns:
            Tuple of (Union type for tyro, dict mapping config types to handlers)
        """
        handlers: dict[type, Callable] = {}
        union_members: list[type] = []

        # Add direct commands
        for name, config, handler, desc in self._commands:
            if include_global_options:
                # Create a wrapper that includes both config and global options
                # Command config first, global options last (appears at bottom of help)
                # Use Annotated with arg(name="global") to avoid trailing dash in help
                wrapper = make_dataclass(
                    f"_{config.__name__}WithGlobal",
                    [
                        (name, Annotated[config, OmitArgPrefixes]),
                        ("global_", Annotated[GlobalOptions, tyro.conf.arg(name="global")]),
                    ],
                    frozen=True,
                )
                wrapper.__doc__ = config.__doc__
                annotated = Annotated[wrapper, subcommand(name=name, description=desc, prefix_name=False)]
                union_members.append(annotated)
                # Store both wrapper and config -> handler mapping
                # (wrapper for lookup before unwrap, config for lookup after unwrap)
                handlers[wrapper] = handler
                handlers[config] = handler
            else:
                annotated = Annotated[config, subcommand(name=name, description=desc, prefix_name=False)]
                union_members.append(annotated)
                handlers[config] = handler

        # Add groups as wrapper dataclasses
        for group_name, group_app in self._groups.items():
            group_union, group_handlers = group_app._build_union(include_global_options)
            handlers.update(group_handlers)

            # Create wrapper dataclass with OmitArgPrefixes and OmitSubcommandPrefixes on the field
            wrapper = make_dataclass(
                f"_{group_name.title()}Wrapper",
                [(group_name, Annotated[group_union, OmitArgPrefixes, OmitSubcommandPrefixes])],
                frozen=True,
            )
            wrapper.__doc__ = group_app.description or ""

            annotated = Annotated[wrapper, subcommand(name=group_name, description=group_app.description, prefix_name=False)]
            union_members.append(annotated)
            # Mark wrapper for unwrapping (handler=None signals it's a wrapper)
            handlers[wrapper] = None  # type: ignore

        return Union[tuple(union_members)], handlers  # type: ignore

    def build(self, include_global_options: bool = False) -> tuple[type, dict[type, Callable]]:
        """Build the tyro-compatible Union type and handler mapping.

        Args:
            include_global_options: Whether to include GlobalOptions in leaf commands

        Returns:
            Tuple of (annotated Union type for tyro.cli, handler mapping)
        """
        union_type, handlers = self._build_union(include_global_options)
        # Create annotated type for tyro.cli - OmitArgPrefixes removes prefixes from args
        annotated_union = Annotated[union_type, OmitArgPrefixes]  # type: ignore
        return annotated_union, handlers

    def run(self) -> None:
        """Run the CLI application.

        This method:
        1. Checks for --run flag and dispatches to nemo-run if specified
        2. Loads config file if --config-file is specified
        3. Parses remaining args with tyro and invokes the handler
        """
        import sys

        args = sys.argv[1:]

        # Check for --run and execute via nemo-run if specified
        run_name, run_overrides, remaining_args = _extract_run_args(args)
        if run_name is not None:
            _execute_with_nemo_run(run_name, run_overrides, remaining_args)
            return

        # Pre-process: extract config file path (but don't load yet - we need the config class)
        # TODO: Config file loading requires knowing the config class, which we only know
        # after subcommand parsing. For now, just filter out the arg.
        filtered_args = _filter_config_file_args(remaining_args)

        # Build tyro Union type and handler mapping with global options
        union_type, handlers = self.build(include_global_options=True)

        # Run tyro directly on the Union type (not a function) for cleaner help output
        config = tyro.cli(union_type, args=filtered_args, description=self.description)

        # Unwrap nested wrappers - check for any single-field wrapper dataclass
        # Also handles the config+global_ wrapper
        global_options: GlobalOptions | None = None
        while True:
            fields = getattr(config, "__dataclass_fields__", {})
            if len(fields) == 1:
                field_name = next(iter(fields))
                config = getattr(config, field_name)
            elif "global_" in fields and len(fields) == 2:
                # This is a command wrapper with global options - extract both
                global_options = getattr(config, "global_")
                config_field = next(f for f in fields if f != "global_")
                config = getattr(config, config_field)
            else:
                break

        # Initialize wandb from global options or run.toml
        if global_options is not None and global_options.wandb_project is not None:
            # CLI args take precedence
            from nemotron.kit.wandb import init_wandb_if_configured
            init_wandb_if_configured(global_options.to_wandb_config(), job_type="cli")
        else:
            # Try loading from run.toml [wandb] section
            from nemotron.kit.run import load_wandb_config
            from nemotron.kit.wandb import init_wandb_if_configured
            wandb_config = load_wandb_config()
            if wandb_config is not None:
                init_wandb_if_configured(wandb_config, job_type="cli")

        # Dispatch to handler
        handler = handlers.get(type(config))
        if handler is None:
            print(f"Unknown command: {config}")
            sys.exit(1)

        result = handler(config)
        if isinstance(result, int) and result != 0:
            sys.exit(result)


def _extract_run_args(args: list[str]) -> tuple[str | None, dict[str, str], list[str]]:
    """Extract --run and --run.<key> arguments.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (run_name, run_overrides, remaining_args)
    """
    run_name: str | None = None
    run_overrides: dict[str, str] = {}
    remaining: list[str] = []

    i = 0
    while i < len(args):
        arg = args[i]

        if arg == "--run" or arg == "-r":
            if i + 1 < len(args):
                run_name = args[i + 1]
                i += 2
            else:
                remaining.append(arg)
                i += 1
        elif arg.startswith("--run."):
            key = arg[6:]  # Remove "--run."
            if i + 1 < len(args):
                run_overrides[key] = args[i + 1]
                i += 2
            else:
                remaining.append(arg)
                i += 1
        elif arg.startswith("--run="):
            run_name = arg[6:]
            i += 1
        elif arg.startswith("-r="):
            run_name = arg[3:]
            i += 1
        else:
            remaining.append(arg)
            i += 1

    return run_name, run_overrides, remaining


def _execute_with_nemo_run(run_name: str, overrides: dict[str, str], remaining_args: list[str]) -> None:
    """Execute command via nemo-run with the specified profile.

    Args:
        run_name: Name of the run profile from run.toml
        overrides: Key-value overrides for the run profile
        remaining_args: Additional CLI arguments
    """
    from nemotron.kit.run import load_run_profile, build_executor

    profile = load_run_profile(run_name, overrides)
    executor = build_executor(profile)

    # Import nemo_run and execute
    try:
        import nemo_run as run
    except ImportError:
        print("Error: nemo-run is required for --run support")
        print("Install with: pip install nemo-run")
        import sys
        sys.exit(1)

    # Execute with nemo-run
    # The remaining_args contain the subcommand and its arguments
    import sys
    script = sys.argv[0]

    with run.Experiment(run_name) as exp:
        exp.add(
            run.Script(
                inline=f"python {script} {' '.join(remaining_args)}",
            ),
            executor=executor,
        )
        exp.run()


def _maybe_load_config_file(args: list[str]) -> dict[str, Any] | None:
    """Load config file if --config-file or -c is specified.

    Args:
        args: Command line arguments

    Returns:
        Dict of config values or None if no config file specified
    """
    config_path: Path | None = None

    for i, arg in enumerate(args):
        if arg in ("--config-file", "-c", "--config") and i + 1 < len(args):
            config_path = Path(args[i + 1])
            break
        elif arg.startswith("--config-file="):
            config_path = Path(arg.split("=", 1)[1])
            break
        elif arg.startswith("--config="):
            config_path = Path(arg.split("=", 1)[1])
            break
        elif arg.startswith("-c="):
            config_path = Path(arg.split("=", 1)[1])
            break

    if config_path is None:
        return None

    from nemotron.kit.config import ConfigManager

    manager = ConfigManager()
    return manager.load_file(config_path)


def _filter_config_file_args(args: list[str]) -> list[str]:
    """Remove --config-file and -c arguments from args.

    Args:
        args: Command line arguments

    Returns:
        Filtered arguments without config file flags
    """
    filtered: list[str] = []
    i = 0

    while i < len(args):
        arg = args[i]

        if arg in ("--config-file", "-c", "--config") and i + 1 < len(args):
            i += 2  # Skip flag and value
        elif arg.startswith("--config-file=") or arg.startswith("--config=") or arg.startswith("-c="):
            i += 1  # Skip combined flag=value
        else:
            filtered.append(arg)
            i += 1

    return filtered
