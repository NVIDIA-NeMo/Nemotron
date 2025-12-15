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

from collections.abc import Callable
from dataclasses import dataclass, field, make_dataclass
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

import tyro
from tyro.conf import OmitArgPrefixes, OmitSubcommandPrefixes, subcommand

from nemotron.kit.artifact import ArtifactInput
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
            help="Execute via nemo-run with profile from run.toml. Use --run.<key> to override.",
        ),
    ] = None

    dry_run: Annotated[
        bool,
        tyro.conf.arg(name="dry-run", help="Print configuration and exit without executing"),
    ] = False

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


def _make_artifact_options_class(artifacts: dict[str, ArtifactInput]) -> type:
    """Create a dynamic dataclass for artifact options.

    Args:
        artifacts: Dict mapping artifact slot names to ArtifactInput definitions

    Returns:
        A dataclass with fields like art_data, art_checkpoint, etc.
    """
    fields = []
    for slot_name, artifact_input in artifacts.items():
        # Field name: art_<slot_name> (e.g., art_data)
        field_name = f"art_{slot_name}"

        # Help text shows the default artifact name prominently
        help_text = (
            f"W&B artifact reference. Default: {artifact_input.default_name}. "
            f"Use version only (v10, latest) or full path (entity/project/name:version)."
        )

        # Create annotated field with tyro.conf.arg for proper CLI naming
        annotated_type = Annotated[
            str | None,
            tyro.conf.arg(name=f"art.{slot_name}", help=help_text),
        ]

        fields.append((field_name, annotated_type, field(default=None)))

    # Create the dataclass
    artifact_options_class = make_dataclass(
        "_ArtifactOptions",
        fields,
        frozen=True,
    )
    artifact_options_class.__doc__ = "Artifact inputs (resolve W&B artifacts to local paths)."

    return artifact_options_class


def _typeddict_to_dataclass(td: type, prefix: str = "") -> type:
    """Convert a TypedDict to a dataclass for tyro CLI parsing.

    Args:
        td: A TypedDict class (with total=False for optional fields)
        prefix: Optional prefix for CLI arg names (e.g., "fn." -> "--fn.field-name")

    Returns:
        A dataclass with the same fields, all optional with None defaults.
        Each field is annotated with tyro.conf.arg for proper CLI naming.

    Raises:
        TypeError: If td is not a TypedDict

    Example:
        >>> class MyKwargs(TypedDict, total=False):
        ...     per_split_data_args_path: str | None
        ...     seq_length: int
        ...
        >>> dc = _typeddict_to_dataclass(MyKwargs, prefix="fn.")
        >>> # Creates a dataclass with --fn.per-split-data-args-path and --fn.seq-length CLI args
    """
    if not is_typeddict(td):
        raise TypeError(f"{td} is not a TypedDict")

    hints = get_type_hints(td)

    # Build field list - all fields get None default (total=False semantics)
    fields = []
    for name, type_hint in hints.items():
        # Convert underscore to hyphen for CLI arg name
        cli_name = f"{prefix}{name.replace('_', '-')}"

        # Wrap in Optional if not already, with None default
        origin = get_origin(type_hint)
        if origin is Union:
            args = get_args(type_hint)
            if type(None) not in args:
                type_hint = type_hint | None
        else:
            type_hint = type_hint | None

        # Create annotated type with proper CLI name
        annotated_type = Annotated[
            type_hint,
            tyro.conf.arg(name=cli_name),
        ]

        fields.append((name, annotated_type, field(default=None)))

    result = make_dataclass(
        f"_{td.__name__}Cli",
        fields,
        frozen=True,
    )
    result.__doc__ = td.__doc__ or f"CLI arguments from {td.__name__}"

    return result


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
        # Commands tuple: (name, config, handler, desc, artifacts, defaults_fn, kwargs, path)
        self._commands: list[
            tuple[
                str,
                type,
                Callable,
                str,
                dict[str, ArtifactInput] | None,
                Callable[..., Any] | None,
                type | None,
                str | None,
            ]
        ] = []
        self._groups: dict[str, App] = {}
        # Script paths for direct execution (subcommand -> script_path)
        self._script_paths: dict[str, str] = {}

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
        artifacts: dict[str, ArtifactInput] | None = None,
        defaults_fn: Callable[..., Any] | None = None,
        kwargs_schema: type | None = None,
        script_path: str | None = None,
    ) -> None:
        """Register a command with its config and handler.

        Args:
            name: Name of the command (becomes the subcommand name)
            config: Dataclass type for command configuration
            handler: Function to call with the parsed config
            description: Description shown in help text (defaults to config's docstring)
            artifacts: Named artifact inputs that can be provided via --art.<name>.
                      Each ArtifactInput defines a default artifact name and mappings
                      from artifact metadata fields to config field paths.
                      Use "fn." prefix in mapping values to pass to defaults_fn():
                      {"blend_path": "fn.per_split_data_args_path"}
            defaults_fn: Optional callable that returns a default config instance.
                        Called with kwargs extracted from artifact mappings with "fn."
                        prefix. Useful for recipe functions that need runtime arguments.
            kwargs_schema: Optional TypedDict class defining kwargs for defaults_fn.
                          Fields from this TypedDict become CLI arguments (--fn.<field-name>)
                          that are passed to defaults_fn(). Requires defaults_fn to be set.
            script_path: Optional path to Python script for direct execution via nemo-run.
                        When set, --run will execute this script directly without
                        installing nemotron. Path is relative to package root (e.g.,
                        "src/nemotron/recipes/nano3/stage0_pretrain/train.py").
                        This is useful for training scripts that only depend on
                        megatron-bridge, not nemotron.

        Example:
            >>> app.command(
            ...     "pretrain",
            ...     TrainingConfig,
            ...     training_main,
            ...     artifacts={
            ...         "data": ArtifactInput(
            ...             default_name="DataBlendsArtifact-pretrain",
            ...             mappings={"path": "dataset.data_path"},
            ...         ),
            ...     },
            ... )

            >>> # With defaults_fn, kwargs_schema, and fn. prefix
            >>> app.command(
            ...     "pretrain",
            ...     ConfigContainer,
            ...     training_main,
            ...     defaults_fn=nano_3_pretrain_config,
            ...     kwargs_schema=NemotronHCommonKwargs,  # TypedDict for CLI args
            ...     artifacts={
            ...         "data": ArtifactInput(
            ...             default_name="DataBlendsArtifact-pretrain",
            ...             mappings={"blend_path": "fn.per_split_data_args_path"},
            ...         ),
            ...     },
            ... )
        """
        if kwargs_schema is not None and defaults_fn is None:
            raise ValueError("kwargs_schema requires defaults_fn to be set")
        self._commands.append(
            (
                name,
                config,
                handler,
                description or config.__doc__ or "",
                artifacts,
                defaults_fn,
                kwargs_schema,
                script_path,
            )
        )
        if script_path is not None:
            self._script_paths[name] = script_path

    def _build_union(
        self, include_global_options: bool = False
    ) -> tuple[
        type,
        dict[type, Callable],
        dict[type, dict[str, ArtifactInput] | None],
        dict[type, Callable[..., Any] | None],
        dict[type, type | None],
    ]:
        """Build Union type and handler/artifacts/defaults_fn/kwargs_schema mappings for tyro.

        Args:
            include_global_options: Whether to include GlobalOptions in leaf commands

        Returns:
            Tuple of (Union type for tyro, dict mapping config types to handlers,
                     dict mapping config types to their artifact definitions,
                     dict mapping config types to their defaults_fn callables,
                     dict mapping config types to their kwargs_schema TypedDict classes)
        """
        handlers: dict[type, Callable] = {}
        artifacts_map: dict[type, dict[str, ArtifactInput] | None] = {}
        defaults_fn_map: dict[type, Callable[..., Any] | None] = {}
        kwargs_schema_map: dict[type, type | None] = {}
        union_members: list[type] = []

        # Add direct commands
        for (
            name,
            config,
            handler,
            desc,
            artifacts,
            defaults_fn,
            kwargs_schema,
            _script_path,
        ) in self._commands:
            if include_global_options:
                # Create a wrapper that includes both config and global options
                # Command config first, global options last (appears at bottom of help)
                # Use Annotated with arg(name="global") to avoid trailing dash in help
                wrapper_fields = [
                    (name, Annotated[config, OmitArgPrefixes]),
                ]

                # Add kwargs_schema options if defined (appears before artifacts)
                if kwargs_schema is not None:
                    kwargs_options_class = _typeddict_to_dataclass(kwargs_schema, prefix="fn.")
                    wrapper_fields.append(
                        ("fn_", Annotated[kwargs_options_class, tyro.conf.arg(name="fn")])
                    )

                # Add artifact options if defined for this command
                if artifacts:
                    artifact_options_class = _make_artifact_options_class(artifacts)
                    wrapper_fields.append(
                        ("art_", Annotated[artifact_options_class, tyro.conf.arg(name="art")])
                    )

                # Global options always at the end
                wrapper_fields.append(
                    ("global_", Annotated[GlobalOptions, tyro.conf.arg(name="global")])
                )

                wrapper = make_dataclass(
                    f"_{config.__name__}WithGlobal",
                    wrapper_fields,
                    frozen=True,
                )
                wrapper.__doc__ = config.__doc__
                annotated = Annotated[
                    wrapper, subcommand(name=name, description=desc, prefix_name=False)
                ]
                union_members.append(annotated)
                # Store both wrapper and config -> handler mapping
                # (wrapper for lookup before unwrap, config for lookup after unwrap)
                handlers[wrapper] = handler
                handlers[config] = handler
                # Store artifacts mapping for the unwrapped config type
                artifacts_map[config] = artifacts
                # Store defaults_fn for the unwrapped config type
                defaults_fn_map[config] = defaults_fn
                # Store kwargs_schema for the unwrapped config type
                kwargs_schema_map[config] = kwargs_schema
            else:
                annotated = Annotated[
                    config, subcommand(name=name, description=desc, prefix_name=False)
                ]
                union_members.append(annotated)
                handlers[config] = handler
                artifacts_map[config] = artifacts
                defaults_fn_map[config] = defaults_fn
                kwargs_schema_map[config] = kwargs_schema

        # Add groups as wrapper dataclasses
        for group_name, group_app in self._groups.items():
            group_union, group_handlers, group_artifacts, group_defaults_fn, group_kwargs_schema = (
                group_app._build_union(include_global_options)
            )
            handlers.update(group_handlers)
            artifacts_map.update(group_artifacts)
            defaults_fn_map.update(group_defaults_fn)
            kwargs_schema_map.update(group_kwargs_schema)

            # Create wrapper dataclass with OmitArgPrefixes and OmitSubcommandPrefixes on the field
            wrapper = make_dataclass(
                f"_{group_name.title()}Wrapper",
                [(group_name, Annotated[group_union, OmitArgPrefixes, OmitSubcommandPrefixes])],
                frozen=True,
            )
            wrapper.__doc__ = group_app.description or ""

            annotated = Annotated[
                wrapper,
                subcommand(name=group_name, description=group_app.description, prefix_name=False),
            ]
            union_members.append(annotated)
            # Mark wrapper for unwrapping (handler=None signals it's a wrapper)
            handlers[wrapper] = None  # type: ignore

        # Build union from tuple of members - must use Union[] for dynamic construction
        union_type: type = Union[tuple(union_members)]  # type: ignore[valid-type]  # noqa: UP007
        return (
            union_type,
            handlers,
            artifacts_map,
            defaults_fn_map,
            kwargs_schema_map,
        )

    def build(
        self, include_global_options: bool = False
    ) -> tuple[
        type,
        dict[type, Callable],
        dict[type, dict[str, ArtifactInput] | None],
        dict[type, Callable[..., Any] | None],
        dict[type, type | None],
    ]:
        """Build the tyro-compatible Union type and all mappings.

        Args:
            include_global_options: Whether to include GlobalOptions in leaf commands

        Returns:
            Tuple of (annotated Union type, handler map, artifacts map, defaults_fn map, kwargs map)
        """
        union_type, handlers, artifacts_map, defaults_fn_map, kwargs_schema_map = self._build_union(
            include_global_options
        )
        # Create annotated type for tyro.cli - OmitArgPrefixes removes prefixes from args
        annotated_union = Annotated[union_type, OmitArgPrefixes]  # type: ignore
        return annotated_union, handlers, artifacts_map, defaults_fn_map, kwargs_schema_map

    def get_script_path(self, subcommand_parts: list[str]) -> str | None:
        """Get the script path for a subcommand.

        Args:
            subcommand_parts: List of subcommand parts (e.g., ['pretrain'])

        Returns:
            Script path if the command has one, None otherwise.
        """
        if not subcommand_parts:
            return None

        first_part = subcommand_parts[0]

        # Check direct commands
        if first_part in self._script_paths and len(subcommand_parts) == 1:
            return self._script_paths[first_part]

        # Check nested groups
        if first_part in self._groups:
            return self._groups[first_part].get_script_path(subcommand_parts[1:])

        return None

    def run(self) -> None:
        """Run the CLI application.

        This method:
        1. Checks for --run flag and dispatches to nemo-run if specified
        2. Reads stdin artifacts if piped
        3. Extracts fn. kwargs from artifact mappings for defaults_fn
        4. Parses args with tyro (including --art.<name> options)
        5. Applies artifact references to config
        6. Invokes the handler
        """
        import sys

        args = sys.argv[1:]

        # Check for --run/--batch and execute via nemo-run if specified
        run_name, run_overrides, remaining_args, is_launch = _extract_run_args(args)
        if run_name is not None:
            _execute_with_nemo_run(run_name, run_overrides, remaining_args, is_launch, self)
            return

        # Read stdin artifacts if piped (for pipeline composition)
        from nemotron.kit.config import _read_stdin_artifacts

        stdin_artifacts = _read_stdin_artifacts()

        # Pre-process: extract config file path (but don't load yet - we need the config class)
        # TODO: Config file loading requires knowing the config class, which we only know
        # after subcommand parsing. For now, just filter out the arg.
        filtered_args = _filter_config_file_args(remaining_args)

        # Convert Hydra-style key=value arguments to tyro-style --key value
        # This allows users to use either: sample=1000 or --sample 1000
        filtered_args = _convert_hydra_to_tyro_args(filtered_args)

        # Build tyro Union type and all mappings with global options
        union_type, handlers, artifacts_map, defaults_fn_map, kwargs_schema_map = self.build(
            include_global_options=True
        )

        # Run tyro directly on the Union type (not a function) for cleaner help output
        # tyro now parses --art.<name>, --fn.<name> options directly
        config = tyro.cli(union_type, args=filtered_args, description=self.description)

        # Unwrap nested wrappers - check for any single-field wrapper dataclass
        # Also handles the config+fn_+art_+global_ wrapper
        global_options: GlobalOptions | None = None
        art_refs: dict[str, str] = {}
        cli_fn_kwargs: dict[str, Any] = {}
        while True:
            fields = getattr(config, "__dataclass_fields__", {})
            if len(fields) == 1:
                field_name = next(iter(fields))
                config = getattr(config, field_name)
            elif "global_" in fields:
                # This is a command wrapper with global options (and possibly fn_, art_)
                global_options = getattr(config, "global_")

                # Extract fn_ kwargs if present (from kwargs_schema CLI args)
                if "fn_" in fields:
                    fn_options = getattr(config, "fn_")
                    # Convert fn_ dataclass fields to cli_fn_kwargs dict
                    for field_name in getattr(fn_options, "__dataclass_fields__", {}):
                        value = getattr(fn_options, field_name)
                        if value is not None:
                            cli_fn_kwargs[field_name] = value

                # Extract artifact refs if present
                if "art_" in fields:
                    art_options = getattr(config, "art_")
                    # Convert art_ dataclass fields to art_refs dict
                    for field_name in getattr(art_options, "__dataclass_fields__", {}):
                        value = getattr(art_options, field_name)
                        if value is not None:
                            # field_name is like "art_data" -> slot_name is "data"
                            slot_name = field_name[4:]  # Remove "art_" prefix
                            art_refs[slot_name] = value

                # Find the config field (not global_, fn_, or art_)
                config_field = next(f for f in fields if f not in ("global_", "fn_", "art_"))
                config = getattr(config, config_field)
            else:
                break

        # Get artifacts definition and defaults_fn for this config type
        artifacts = artifacts_map.get(type(config))
        defaults_fn = defaults_fn_map.get(type(config))

        # Build fn_kwargs: CLI kwargs as base, artifact fn. kwargs override
        fn_kwargs: dict[str, Any] = dict(cli_fn_kwargs)  # Start with CLI kwargs

        # Extract fn. kwargs from artifact mappings (these override CLI kwargs)
        if stdin_artifacts and artifacts:
            artifact_fn_kwargs = _extract_fn_kwargs_from_artifacts(stdin_artifacts, artifacts)
            fn_kwargs.update(artifact_fn_kwargs)  # Artifact values take precedence

        # Call defaults_fn if provided with combined kwargs
        if defaults_fn is not None and fn_kwargs:
            # Call defaults_fn with extracted kwargs for its side effects and tracking
            # The actual config values come from tyro parsing and artifact application below
            defaults_fn(**fn_kwargs)

        # Apply artifact references to config by constructing art:// URIs
        # and letting the resolve_artifact_uri function handle resolution
        if (art_refs or stdin_artifacts) and artifacts:
            config = _apply_artifact_refs_to_config(config, art_refs, stdin_artifacts, artifacts)

        # Display compiled configuration (always, like pretrain)
        _display_config(config, global_options)

        # Handle dry-run mode
        if global_options is not None and global_options.dry_run:
            return

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


def _extract_fn_kwargs_from_artifacts(
    stdin_artifacts: dict[str, dict],
    artifacts: dict[str, ArtifactInput],
) -> dict[str, Any]:
    """Extract fn. kwargs from artifact mappings for defaults_fn.

    Iterates through all artifact inputs and their mappings. For mappings
    with "fn." prefix in the target, loads the artifact metadata and
    extracts the value to pass as a kwarg to defaults_fn().

    Args:
        stdin_artifacts: Artifacts piped from stdin (from previous step)
        artifacts: ArtifactInput definitions for this command

    Returns:
        Dict of kwargs to pass to defaults_fn()
    """
    from nemotron.kit.config import _load_artifact_metadata

    fn_kwargs: dict[str, Any] = {}

    for slot_name, artifact_input in artifacts.items():
        if slot_name not in stdin_artifacts:
            continue

        # Load artifact metadata
        try:
            metadata = _load_artifact_metadata(stdin_artifacts[slot_name])
        except FileNotFoundError:
            continue

        # Check each mapping for fn. prefix
        for artifact_field, target in artifact_input.mappings.items():
            if not target.startswith("fn."):
                continue

            # Extract kwarg name (remove "fn." prefix)
            kwarg_name = target[3:]

            # Get value from metadata
            if artifact_field in metadata:
                fn_kwargs[kwarg_name] = metadata[artifact_field]

    return fn_kwargs


def _extract_run_args(args: list[str]) -> tuple[str | None, dict[str, str], list[str], bool]:
    """Extract --run/--batch and --run.<key>/--batch.<key> arguments.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (run_name, run_overrides, remaining_args, is_launch).
        is_launch is True when --batch was used (implies detach=True).

    Raises:
        ValueError: If both --run and --batch are specified.
    """
    run_name: str | None = None
    launch_name: str | None = None
    run_overrides: dict[str, str] = {}
    remaining: list[str] = []

    i = 0
    while i < len(args):
        arg = args[i]

        # Handle --run / -r
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
        # Handle --batch / -b
        elif arg == "--batch" or arg == "-b":
            if i + 1 < len(args):
                launch_name = args[i + 1]
                i += 2
            else:
                remaining.append(arg)
                i += 1
        elif arg.startswith("--batch."):
            key = arg[8:]  # Remove "--batch."
            if i + 1 < len(args):
                run_overrides[key] = args[i + 1]
                i += 2
            else:
                remaining.append(arg)
                i += 1
        elif arg.startswith("--batch="):
            launch_name = arg[8:]
            i += 1
        elif arg.startswith("-b="):
            launch_name = arg[3:]
            i += 1
        else:
            remaining.append(arg)
            i += 1

    # Validate mutual exclusivity
    if run_name is not None and launch_name is not None:
        raise ValueError(
            "--run and --batch are mutually exclusive. "
            "Use --run for attached execution or --batch for detached execution."
        )

    # Determine final name and whether launch mode is active
    is_launch = launch_name is not None
    final_name = launch_name if is_launch else run_name

    return final_name, run_overrides, remaining, is_launch


def _load_artifact_metadata_from_path(artifact_path: str) -> dict[str, Any] | None:
    """Load artifact metadata from path.

    Args:
        artifact_path: Path to artifact directory or file.

    Returns:
        Artifact metadata dict, or None if not found.
    """
    import json
    from pathlib import Path

    path = Path(artifact_path)

    # If path is a file (e.g., manifest.json, blend.json), use parent directory
    if path.is_file():
        metadata_path = path.parent / "metadata.json"
    else:
        metadata_path = path / "metadata.json"

    if not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        data = json.load(f)
        # Metadata may be nested under "metadata" key or at top level
        return data.get("metadata", data)


def _apply_artifact_refs_to_config(
    config: Any,
    art_refs: dict[str, str],
    stdin_artifacts: dict[str, dict] | None,
    artifacts: dict[str, ArtifactInput],
) -> Any:
    """Apply artifact references to config after tyro parsing.

    This function:
    1. Constructs art:// URIs from artifact references
    2. Resolves them to local paths via resolve_artifact_uri
    3. Updates the config fields according to the mappings

    Args:
        config: The parsed config dataclass
        art_refs: Mapping from artifact slot names to references (from --art.<name>)
        stdin_artifacts: Artifacts piped from stdin (from previous step)
        artifacts: ArtifactInput definitions for this command

    Returns:
        Updated config with artifact paths resolved
    """
    from nemotron.kit.config import resolve_artifact_uri

    updates: dict[str, Any] = {}

    # Process --art.<name> references (highest priority)
    for slot_name, ref in art_refs.items():
        if slot_name not in artifacts:
            continue

        artifact_input = artifacts[slot_name]

        # Build the full art:// URI from the reference
        art_uri = _build_art_uri(ref, artifact_input.default_name)

        # Resolve base artifact path once (for metadata loading)
        base_artifact_path = resolve_artifact_uri(art_uri)

        # Load metadata once if any mapping uses metadata. prefix
        artifact_metadata = None
        if any(f.startswith("metadata.") for f in artifact_input.mappings.keys()):
            artifact_metadata = _load_artifact_metadata_from_path(base_artifact_path)

        # Apply each mapping (skip fn. prefixed targets - those are for defaults_fn)
        for artifact_field, config_field in artifact_input.mappings.items():
            # Skip fn. prefixed targets - those are handled by defaults_fn
            if config_field.startswith("fn."):
                continue

            if artifact_field.startswith("metadata."):
                # Extract value from artifact metadata
                metadata_key = artifact_field[9:]  # Remove "metadata." prefix
                if artifact_metadata and metadata_key in artifact_metadata:
                    value = artifact_metadata[metadata_key]
                    _set_nested_field(config, config_field, value, updates)
            elif artifact_field:
                # Build full URI with file path
                full_uri = f"{art_uri}/{artifact_field}"
                local_path = resolve_artifact_uri(full_uri)
                _set_nested_field(config, config_field, local_path, updates)
            else:
                # Use base artifact path
                _set_nested_field(config, config_field, base_artifact_path, updates)

    # Process stdin artifacts (lower priority than --art.<name>)
    if stdin_artifacts:
        for slot_name, artifact_info in stdin_artifacts.items():
            if slot_name not in artifacts:
                continue

            # Skip if already provided via --art.<name>
            if slot_name in art_refs:
                continue

            artifact_input = artifacts[slot_name]
            artifact_path = artifact_info.get("path")

            if artifact_path:
                # Load metadata once if any mapping uses metadata. prefix
                artifact_metadata = None
                if any(f.startswith("metadata.") for f in artifact_input.mappings.keys()):
                    artifact_metadata = _load_artifact_metadata_from_path(artifact_path)

                for artifact_field, config_field in artifact_input.mappings.items():
                    # Skip fn. prefixed targets - those are handled by defaults_fn
                    if config_field.startswith("fn."):
                        continue

                    if artifact_field.startswith("metadata."):
                        # Extract value from artifact metadata
                        metadata_key = artifact_field[9:]  # Remove "metadata." prefix
                        if artifact_metadata and metadata_key in artifact_metadata:
                            value = artifact_metadata[metadata_key]
                            _set_nested_field(config, config_field, value, updates)
                    elif artifact_field:
                        full_path = f"{artifact_path}/{artifact_field}"
                        _set_nested_field(config, config_field, full_path, updates)
                    else:
                        _set_nested_field(config, config_field, artifact_path, updates)

    # Apply updates to config using dataclass replace
    if updates:
        config = _apply_nested_updates(config, updates)

    return config


def _set_nested_field(config: Any, field_path: str, value: Any, updates: dict[str, Any]) -> None:
    """Set a nested field value in the updates dict.

    Args:
        config: The config dataclass (for type checking)
        field_path: Dot-separated field path like "dataset.data_path"
        value: Value to set
        updates: Dict to store updates (modified in place)
    """
    parts = field_path.split(".")
    if len(parts) == 1:
        updates[field_path] = value
    else:
        # Nested field - store as nested dict
        current = updates
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value


def _apply_nested_updates(config: Any, updates: dict[str, Any]) -> Any:
    """Apply nested updates to a dataclass config.

    Args:
        config: The config dataclass
        updates: Dict of updates (may be nested)

    Returns:
        Updated config
    """
    from dataclasses import is_dataclass, replace

    flat_updates = {}
    for key, value in updates.items():
        if isinstance(value, dict) and hasattr(config, key):
            # Nested update - recurse into the nested dataclass
            nested_config = getattr(config, key)
            if is_dataclass(nested_config):
                flat_updates[key] = _apply_nested_updates(nested_config, value)
            else:
                flat_updates[key] = value
        else:
            flat_updates[key] = value

    return replace(config, **flat_updates)


def _build_art_uri(ref: str, default_name: str) -> str:
    """Build a full art:// URI from an artifact reference.

    Args:
        ref: Artifact reference in various formats:
             - "v10" or "latest" -> use default_name
             - "Name:version" -> use as-is
             - "entity/project/name:version" -> full W&B path
        default_name: Default artifact name for version-only references

    Returns:
        Full art:// URI

    Examples:
        >>> _build_art_uri("v10", "DataBlendsArtifact-pretrain")
        "art://DataBlendsArtifact-pretrain:v10"
        >>> _build_art_uri("latest", "DataBlendsArtifact-pretrain")
        "art://DataBlendsArtifact-pretrain:latest"
        >>> _build_art_uri("DataBlendsArtifact-pretrain:v5", "default")
        "art://DataBlendsArtifact-pretrain:v5"
        >>> _build_art_uri("romeyn/nemotron/DataBlendsArtifact-pretrain:v10", "default")
        "art://romeyn/nemotron/DataBlendsArtifact-pretrain:v10"
    """
    # Already an art:// URI
    if ref.startswith("art://"):
        return ref

    # Check if it's a version-only reference (e.g., "v10", "latest", "10")
    is_version_only = ref == "latest" or ref.startswith("v") and ref[1:].isdigit() or ref.isdigit()

    if is_version_only:
        # Use default artifact name with the version
        version = ref if ref == "latest" or ref.startswith("v") else f"v{ref}"
        return f"art://{default_name}:{version}"

    # Has a colon - could be name:version or entity/project/name:version
    if ":" in ref:
        return f"art://{ref}"

    # No version specifier - assume it's a name and use :latest
    return f"art://{ref}:latest"


def _append_wandb_args(args: list[str], wandb_config: WandbConfig) -> list[str]:
    """Append wandb CLI arguments from WandbConfig.

    Args:
        args: Existing CLI arguments
        wandb_config: WandbConfig with project, entity, etc.

    Returns:
        New args list with wandb CLI flags appended.
    """
    result = list(args)

    if wandb_config.project:
        result.extend(["--wandb.project", wandb_config.project])
    if wandb_config.entity:
        result.extend(["--wandb.entity", wandb_config.entity])
    if wandb_config.run_name:
        result.extend(["--wandb.run-name", wandb_config.run_name])
    for tag in wandb_config.tags:
        result.extend(["--wandb.tags", tag])
    if wandb_config.notes:
        result.extend(["--wandb.notes", wandb_config.notes])

    return result


def _display_config(config: Any, global_options: GlobalOptions | None) -> None:
    """Display the compiled configuration as syntax-highlighted YAML.

    Mimics display_job_config() but works with dataclasses instead of OmegaConf.

    Args:
        config: The parsed config dataclass
        global_options: Global CLI options (for wandb config display)
    """
    from dataclasses import asdict, is_dataclass

    import yaml
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax

    from nemotron.kit.cli.env import get_cli_config
    from nemotron.kit.run import load_wandb_config

    console = Console()

    # Get theme from env.toml or use default
    cli_config = get_cli_config()
    theme = str(cli_config.theme) if cli_config and "theme" in cli_config else "monokai"

    console.print()
    console.print("[bold cyan]Compiled Configuration[/bold cyan]")
    console.print()

    # Build run section with wandb info
    run_info: dict[str, Any] = {}

    # Get wandb config from CLI args or env.toml
    if global_options is not None and global_options.wandb_project is not None:
        wandb = global_options.to_wandb_config()
        run_info["wandb"] = {
            "project": wandb.project,
            "entity": wandb.entity,
        }
    else:
        wandb_config = load_wandb_config()
        if wandb_config is not None and wandb_config.project:
            run_info["wandb"] = {
                "project": wandb_config.project,
                "entity": wandb_config.entity,
            }

    if run_info:
        yaml_str = yaml.dump(run_info, default_flow_style=False, sort_keys=False)
        syntax = Syntax(yaml_str.rstrip(), "yaml", theme=theme, line_numbers=False)
        console.print(
            Panel(
                syntax,
                title="[bold green]run[/bold green]",
                border_style="green",
                expand=False,
            )
        )
        console.print()

    # Display config
    if is_dataclass(config):
        config_dict = asdict(config)
        # Convert Path objects to strings for YAML serialization
        config_dict = _convert_paths_to_strings(config_dict)
        yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
        syntax = Syntax(yaml_str.rstrip(), "yaml", theme=theme, line_numbers=False)
        console.print(
            Panel(
                syntax,
                title="[bold green]config[/bold green]",
                border_style="green",
                expand=False,
            )
        )
        console.print()


def _convert_paths_to_strings(obj: Any) -> Any:
    """Recursively convert Path objects to strings for YAML serialization."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _convert_paths_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_paths_to_strings(item) for item in obj]
    return obj


def _get_script_config_path(subcommand_parts: list[str]) -> Path | None:
    """Get the config file path for a script based on subcommand.

    Args:
        subcommand_parts: CLI subcommand parts (e.g., ['data', 'prep', 'pretrain'])

    Returns:
        Path to the config file or None if not a configurable script.
    """
    # Normalize: skip 'nano3' prefix if present
    parts = subcommand_parts
    if parts and parts[0] == "nano3":
        parts = parts[1:]

    # Handle data prep: ['data', 'prep', 'pretrain'] -> stage0_pretrain/config/data_prep.yaml
    if len(parts) >= 3 and parts[0] == "data" and parts[1] == "prep":
        stage = parts[2]  # e.g., "pretrain", "sft", "rl"

        stage_map = {
            "pretrain": "stage0_pretrain",
            "sft": "stage1_sft",
            "rl": "stage2_rl",
        }
        stage_dir = stage_map.get(stage)
        if stage_dir is None:
            return None

        # Build config path relative to cwd
        return Path(f"src/nemotron/recipes/nano3/{stage_dir}/config/data_prep.yaml")

    return None


def _display_script_config(
    subcommand_parts: list[str],
    script_args: list[str],
    console: Any,
    theme: str,
) -> None:
    """Display the script configuration as syntax-highlighted YAML.

    Loads the script's config file and applies CLI overrides to show
    the effective configuration that will be used.

    Args:
        subcommand_parts: CLI subcommand parts (e.g., ['data', 'prep', 'pretrain'])
        script_args: Script arguments (e.g., ['--sample', '10000', '--force'])
        console: Rich console instance
        theme: Syntax highlighting theme
    """
    from rich.panel import Panel
    from rich.syntax import Syntax

    # Get config file path for this script
    config_path = _get_script_config_path(subcommand_parts)
    if config_path is None or not config_path.exists():
        return

    try:
        from omegaconf import OmegaConf

        from nemotron.kit.train_script import (
            apply_hydra_overrides,
            load_omegaconf_yaml,
        )

        # Load base config
        config = load_omegaconf_yaml(config_path)

        # CLI flags that are NOT config overrides (handled by CLI framework)
        cli_only_flags = {"dry-run", "run", "batch", "config"}

        # Parse script_args into overrides
        # Handle both --key value and key=value formats
        overrides = []
        i = 0
        while i < len(script_args):
            arg = script_args[i]
            if arg.startswith("--"):
                # --key value format: convert to key=value
                key = arg[2:]  # Remove --

                # Skip CLI-only flags that aren't config fields
                if key in cli_only_flags:
                    i += 1
                    continue

                if i + 1 < len(script_args) and not script_args[i + 1].startswith("-"):
                    value = script_args[i + 1]
                    # Convert --sample 10000 to sample=10000
                    overrides.append(f"{key}={value}")
                    i += 2
                else:
                    # Boolean flag like --force
                    overrides.append(f"{key}=true")
                    i += 1
            elif "=" in arg:
                # key=value format: use directly
                overrides.append(arg)
                i += 1
            else:
                i += 1

        # Apply overrides
        config = apply_hydra_overrides(config, overrides)

        # Convert to YAML for display (resolve interpolations)
        yaml_str = OmegaConf.to_yaml(config, resolve=True)

        syntax = Syntax(yaml_str.rstrip(), "yaml", theme=theme, line_numbers=False)
        console.print(
            Panel(
                syntax,
                title="[bold green]config[/bold green]",
                border_style="green",
                expand=False,
            )
        )
        console.print()

    except Exception:
        # Silently skip config display on any error
        pass


def _display_run_config(
    profile: Any,
    run_name: str,
    wandb_config: WandbConfig | None,
    remaining_args: list[str],
    is_launch: bool,
) -> None:
    """Display the run configuration as syntax-highlighted YAML.

    Args:
        profile: The RunConfig profile
        run_name: Name of the run profile
        wandb_config: Optional wandb config from env.toml
        remaining_args: CLI arguments (for showing command)
        is_launch: Whether running in batch/launch mode
    """
    import sys

    import yaml
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax

    from nemotron.kit.cli.env import get_cli_config

    console = Console()

    # Get theme from env.toml or use default
    cli_config = get_cli_config()
    theme = str(cli_config.theme) if cli_config and "theme" in cli_config else "monokai"

    console.print()
    console.print("[bold cyan]Compiled Configuration[/bold cyan]")
    console.print()

    # Build run section
    run_info: dict[str, Any] = {
        "mode": "batch" if is_launch else "run",
        "profile": run_name,
        "env": {
            "executor": profile.executor,
        },
    }

    # Add all non-None env fields from the profile
    # These are all the Slurm/executor-related fields
    env_fields = [
        "account",
        "partition",
        "run_partition",
        "batch_partition",
        "time",
        "job_name",
        "nodes",
        "nproc_per_node",
        "ntasks_per_node",
        "gpus_per_node",
        "mem",
        "exclusive",
        "cpus_per_task",
        "cpus_per_gpu",
        "gpus_per_task",
        "mem_per_gpu",
        "mem_per_cpu",
        "qos",
        "constraint",
        "exclude",
        "gres",
        "array",
        "tunnel",
        "host",
        "user",
        "identity",
        "remote_job_dir",
        "container_image",
        "mounts",
    ]
    for field_name in env_fields:
        value = getattr(profile, field_name, None)
        # Skip None values and default "local" tunnel
        if value is None:
            continue
        if field_name == "tunnel" and value == "local":
            continue
        # Skip empty lists
        if isinstance(value, list) and not value:
            continue
        run_info["env"][field_name] = value

    # Add CLI info - extract script args (everything after subcommand)
    # Stop at first flag OR key=value override (Hydra-style)
    subcommand_parts = []
    script_args = []
    for i, arg in enumerate(remaining_args):
        if arg.startswith("-") or "=" in arg:
            script_args = remaining_args[i:]
            break
        subcommand_parts.append(arg)

    run_info["cli"] = {
        "argv": sys.argv,
        "script_args": script_args if script_args else None,
    }

    # Add wandb config if available
    if wandb_config is not None and wandb_config.project:
        run_info["wandb"] = {
            "project": wandb_config.project,
            "entity": wandb_config.entity,
        }

    yaml_str = yaml.dump(run_info, default_flow_style=False, sort_keys=False)
    syntax = Syntax(yaml_str.rstrip(), "yaml", theme=theme, line_numbers=False)
    console.print(
        Panel(
            syntax,
            title="[bold green]run[/bold green]",
            border_style="green",
            expand=False,
        )
    )
    console.print()

    # Display data prep config panel if this is a data prep command
    _display_script_config(subcommand_parts, script_args, console, theme)


def _execute_with_nemo_run(
    run_name: str,
    overrides: dict[str, str],
    remaining_args: list[str],
    is_launch: bool = False,
    app: App | None = None,
) -> None:
    """Execute command via nemo-run with the specified profile.

    Args:
        run_name: Name of the run profile from run.toml
        overrides: Key-value overrides for the run profile
        remaining_args: Additional CLI arguments
        is_launch: If True, force detach=True for detached execution
        app: Optional App instance for looking up script paths
    """
    from nemotron.kit.run import (
        build_executor,
        load_run_profile,
        load_wandb_config,
        resolve_partition,
        run_with_nemo_run,
    )

    # Load profile from run.toml
    profile = load_run_profile(run_name)

    # Force detach=True and ray_mode="job" when using --batch
    # ray_mode="job" ensures the cluster terminates after the job completes
    if is_launch:
        profile.detach = True
        profile.ray_mode = "job"

    # Resolve partition based on execution mode (run vs launch)
    # This allows different partitions for attached vs detached execution
    profile.partition = resolve_partition(profile, is_launch)

    # Load wandb config for potential use later
    wandb_config = load_wandb_config()

    # Apply CLI overrides
    for key, value in overrides.items():
        if hasattr(profile, key):
            # Handle type conversion for common types
            field_type = type(getattr(profile, key))
            if field_type is bool:
                value = value.lower() in ("true", "1", "yes")
            elif field_type is int:
                value = int(value)
            elif field_type is list:
                value = value.split(",") if value else []
            setattr(profile, key, value)
        else:
            raise ValueError(f"Unknown run config field: {key}")

    # Check for --dry-run in remaining args
    dry_run = "--dry-run" in remaining_args

    # Display compiled configuration (like pretrain does)
    _display_run_config(profile, run_name, wandb_config, remaining_args, is_launch)

    # Handle dry-run mode
    if dry_run:
        return

    # Build descriptive experiment name from subcommand path
    # Extract subcommand parts (stop at first flag OR key=value override)
    subcommand_parts = []
    for arg in remaining_args:
        if arg.startswith("-") or "=" in arg:
            break
        subcommand_parts.append(arg)
    experiment_name = "-".join(subcommand_parts) if subcommand_parts else run_name

    # Check if command has a direct script path (for commands that don't need pip install)
    direct_script_path = app.get_script_path(subcommand_parts) if app else None

    # Check if target module has RAY = True
    use_ray = _check_module_ray_flag(subcommand_parts)

    # For non-Ray jobs, append wandb CLI args if configured.
    # Ray jobs get wandb config from env vars (WANDB_PROJECT, WANDB_ENTITY)
    # set in the Ray runtime_env, so we don't append CLI args.
    if not use_ray and wandb_config is not None and wandb_config.enabled:
        remaining_args = _append_wandb_args(remaining_args, wandb_config)

    if use_ray:
        # Use Ray execution path via run_with_nemo_run
        if direct_script_path:
            # Direct script execution - use relative path from workdir
            # nemo-run's SlurmRayJob rsyncs to {cluster_dir}/code and sets workdir there
            # so we can use relative paths directly (no /nemo_run mount for Ray jobs)
            script_args = remaining_args[len(subcommand_parts) :]
            run_with_nemo_run(
                script_path=direct_script_path,
                script_args=script_args,
                run_config=profile,
                ray=True,
                pre_ray_start_commands=None,
            )
        else:
            # Fallback to module execution (requires pip install)
            # Build the command - need to include 'nano3' since it was stripped by __main__.py
            script_path = f"python -m nemotron nano3 {' '.join(remaining_args)}"
            run_with_nemo_run(
                script_path=script_path,
                script_args=[],  # Args already included in script_path
                run_config=profile,
                ray=True,
                pre_ray_start_commands=None,
            )
    else:
        # Standard execution via nemo-run Experiment
        executor = build_executor(profile)

        try:
            import nemo_run as run
        except ImportError:
            print("Error: nemo-run is required for --run support")
            print("Install with: pip install nemo-run")
            import sys

            sys.exit(1)

        with run.Experiment(experiment_name) as exp:
            # Inject experiment_id for artifact aliasing across tasks
            executor.env_vars["NEMO_EXPERIMENT_ID"] = exp._id

            if not direct_script_path:
                subcommand_str = " ".join(subcommand_parts)
                raise ValueError(
                    f"No script_path configured for command '{subcommand_str}'.\n"
                    f"Add script_path parameter to app.command() in __main__.py."
                )

            # Direct script execution - no pip install needed
            # Script only depends on megatron-bridge which is in the container
            # Extract script args (everything after the subcommand)
            script_args = remaining_args[len(subcommand_parts) :]
            # Script path is relative to /nemo_run/code
            container_script_path = f"/nemo_run/code/{direct_script_path}"

            exp.add(
                run.Script(path=container_script_path, args=script_args, entrypoint="python"),
                executor=executor,
                name=experiment_name,
            )
            exp.run(detach=profile.detach)


def _check_module_ray_flag(subcommand_parts: list[str]) -> bool:
    """Check if the target module has RAY = True.

    Args:
        subcommand_parts: CLI subcommand parts (e.g., ['nano3', 'data', 'prep', 'pretrain'])

    Returns:
        True if the module has RAY = True, False otherwise.
    """
    # Map CLI subcommands to module paths (e.g., ['nano3', 'data', 'prep', 'pretrain']
    # -> nemotron.recipes.nano3.stage0_pretrain.data_prep)
    module_path = _subcommand_to_module(subcommand_parts)
    if module_path is None:
        return False

    try:
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, "RAY", False)
    except (ImportError, AttributeError):
        return False


def _subcommand_to_module(subcommand_parts: list[str]) -> str | None:
    """Convert CLI subcommand parts to a module path.

    Args:
        subcommand_parts: CLI subcommand parts

    Returns:
        Module path string or None if not mappable.
    """
    # Handle nano3 recipe subcommands
    # Note: The 'nano3' prefix may already be consumed by __main__.py before we get here
    # Format without nano3: data prep <stage> -> nemotron.recipes.nano3.stage<N>_<stage>.data_prep
    # Format with nano3: nano3 data prep <stage> -> same

    # Normalize: skip 'nano3' prefix if present
    parts = subcommand_parts
    if parts and parts[0] == "nano3":
        parts = parts[1:]

    # Now expecting: ['data', 'prep', 'pretrain'] or ['train', 'pretrain']
    if len(parts) >= 3 and parts[0] == "data" and parts[1] == "prep":
        stage = parts[2]  # e.g., "pretrain", "sft", "rl"

        # Map stage names to directory names
        stage_map = {
            "pretrain": "stage0_pretrain",
            "sft": "stage1_sft",
            "rl": "stage2_rl",
        }
        stage_dir = stage_map.get(stage)
        if stage_dir is None:
            return None

        return f"nemotron.recipes.nano3.{stage_dir}.data_prep"

    elif len(parts) >= 2 and parts[0] == "train":
        stage = parts[1]  # e.g., "pretrain", "sft", "rl"

        stage_map = {
            "pretrain": "stage0_pretrain",
            "sft": "stage1_sft",
            "rl": "stage2_rl",
        }
        stage_dir = stage_map.get(stage)
        if stage_dir is None:
            return None

        return f"nemotron.recipes.nano3.{stage_dir}.train"

    elif len(parts) >= 1:
        # Direct training commands: ['pretrain'], ['sft'], ['rl']
        stage = parts[0]

        stage_map = {
            "pretrain": "stage0_pretrain",
            "sft": "stage1_sft",
            "rl": "stage2_rl",
        }
        stage_dir = stage_map.get(stage)
        if stage_dir is None:
            return None

        return f"nemotron.recipes.nano3.{stage_dir}.train"

    return None


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
        elif (
            arg.startswith("--config-file=") or arg.startswith("--config=") or arg.startswith("-c=")
        ):
            i += 1  # Skip combined flag=value
        else:
            filtered.append(arg)
            i += 1

    return filtered


def _convert_hydra_to_tyro_args(args: list[str]) -> list[str]:
    """Convert Hydra-style key=value arguments to tyro-style --key value.

    This allows users to use either syntax:
    - Hydra-style: sample=1000 force=true
    - Tyro-style: --sample 1000 --force

    Args:
        args: Command line arguments

    Returns:
        Arguments with key=value converted to --key value format.

    Examples:
        >>> _convert_hydra_to_tyro_args(["data", "prep", "sft", "sample=1000", "--force"])
        ["data", "prep", "sft", "--sample", "1000", "--force"]
        >>> _convert_hydra_to_tyro_args(["pretrain", "train.batch_size=32"])
        ["pretrain", "--train.batch-size", "32"]
    """
    result: list[str] = []

    for arg in args:
        # Skip if it's already a flag (starts with -)
        if arg.startswith("-"):
            result.append(arg)
            continue

        # Check for key=value pattern (no leading dash, contains =)
        if "=" in arg:
            key, value = arg.split("=", 1)
            # Convert underscores to hyphens for CLI compatibility
            key = key.replace("_", "-")
            # Handle boolean values: force=true -> --force, force=false -> --no-force
            if value.lower() in ("true", "1", "yes"):
                result.append(f"--{key}")
            elif value.lower() in ("false", "0", "no"):
                result.append(f"--no-{key}")
            else:
                result.append(f"--{key}")
                result.append(value)
        else:
            # Not a key=value, pass through as-is (likely a subcommand)
            result.append(arg)

    return result
