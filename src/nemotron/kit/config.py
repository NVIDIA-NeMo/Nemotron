# Copyright (c) Nemotron Contributors
# SPDX-License-Identifier: MIT

"""
ConfigManager for loading configs from YAML, TOML, and JSON files.

Provides seamless integration with tyro for CLI argument parsing.
Adapted from torchtitan's ConfigManager with extended format support.
"""

import importlib
import json
import sys
from dataclasses import field, fields, is_dataclass, make_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generic, Type, TypeVar, overload

if TYPE_CHECKING:
    from nemotron.kit.wandb import WandbConfig

import tyro

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

T = TypeVar("T")

# Valid CLI argument names for config file path
CONFIG_FILE_KEYS = {"--config-file", "--config_file", "--config"}


class ConfigManager(Generic[T]):
    """
    Parses, merges, and validates configuration from files and CLI.

    Supports YAML, TOML, and JSON configuration files. The format is
    detected automatically based on file extension.

    Configuration precedence:
        CLI args > config file > defaults() > dataclass defaults

    Global sections (wandb, run) in config files are extracted separately
    and can be accessed via get_wandb_config() after parsing.

    Example:
        >>> from dataclasses import dataclass
        >>> from nemotron.config import ConfigManager
        >>>
        >>> @dataclass
        ... class TrainingConfig:
        ...     batch_size: int = 32
        ...     learning_rate: float = 1e-4
        >>>
        >>> manager = ConfigManager(TrainingConfig)
        >>> config = manager.parse_args(["--config-file", "config.yaml"])
        >>> wandb_config = manager.get_wandb_config()  # From [wandb] section
    """

    # Sections that are extracted from config files before validation
    # These are global settings, not recipe-specific
    GLOBAL_SECTIONS = {"wandb", "run", "custom_config_module"}

    def __init__(
        self,
        config_cls: Type[T],
        defaults: Callable[[], T] | None = None,
    ):
        """
        Initialize ConfigManager with a dataclass type.

        Args:
            config_cls: A dataclass type to use as the configuration schema.
            defaults: Optional callable that returns a default config instance.
                     Used for model-specific defaults from external recipe functions.
        """
        if not is_dataclass(config_cls):
            raise TypeError(f"{config_cls.__name__} must be a dataclass")
        self.config_cls = config_cls
        self.defaults = defaults
        self.config: T | None = None
        self._wandb_config: dict[str, Any] | None = None
        self._setup_tyro_registry()

    def parse_args(self, args: list[str] | None = None) -> T:
        """
        Parse configuration from file and CLI arguments.

        Args:
            args: CLI arguments. Defaults to sys.argv[1:].

        Returns:
            Populated configuration dataclass instance.
        """
        if args is None:
            args = sys.argv[1:]

        # Load config file if specified
        file_values = self._maybe_load_config_file(args)

        # Extract global sections (wandb, run) before validation
        if file_values:
            file_values, global_sections = self._extract_global_sections(file_values)
            self._wandb_config = global_sections.get("wandb")
        else:
            global_sections = {}

        # Filter out config file args before passing to tyro
        filtered_args = self._filter_config_file_args(args)

        # Optionally merge with custom config module
        config_cls = self._maybe_add_custom_config(filtered_args, file_values)

        # Build base config with precedence: defaults() < config file
        # Start with defaults() if provided, otherwise dataclass defaults
        if self.defaults is not None:
            base_config = self.defaults()
            # Overlay file values on top of defaults
            if file_values:
                base_config = self._merge_dict_into_dataclass(base_config, file_values)
        elif file_values:
            base_config = self._dict_to_dataclass(config_cls, file_values)
        else:
            base_config = config_cls()

        # Parse CLI with base config as defaults (CLI has highest precedence)
        self.config = tyro.cli(
            config_cls,
            args=filtered_args,
            default=base_config,
            registry=self._registry,
        )

        return self.config

    def _extract_global_sections(
        self, file_values: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Extract global sections (wandb, run) from config file values.

        These sections are not part of the recipe config and should be
        handled separately.

        Args:
            file_values: Raw config file contents.

        Returns:
            Tuple of (recipe_config, global_sections).
        """
        recipe_config = {}
        global_sections = {}

        for key, value in file_values.items():
            if key in self.GLOBAL_SECTIONS:
                global_sections[key] = value
            else:
                recipe_config[key] = value

        return recipe_config, global_sections

    def get_wandb_config(self) -> "WandbConfig | None":
        """Get WandbConfig from the [wandb] section of the config file.

        Returns:
            WandbConfig instance if [wandb] section exists, None otherwise.
        """
        if self._wandb_config is None:
            return None

        from nemotron.kit.wandb import WandbConfig

        wandb_dict = dict(self._wandb_config)

        # Convert tags from list to tuple if present
        if "tags" in wandb_dict and isinstance(wandb_dict["tags"], list):
            wandb_dict["tags"] = tuple(wandb_dict["tags"])

        # Map run_name from TOML (allow shorthand "name")
        if "name" in wandb_dict and "run_name" not in wandb_dict:
            wandb_dict["run_name"] = wandb_dict.pop("name")

        return WandbConfig(**wandb_dict)

    def _filter_config_file_args(self, args: list[str]) -> list[str]:
        """Remove --config-file related arguments from args list."""
        filtered = []
        skip_next = False
        for i, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue

            # Handle --config-file=path format
            if "=" in arg:
                key = arg.split("=", 1)[0]
                if key in CONFIG_FILE_KEYS:
                    continue
            # Handle --config-file path format
            elif arg in CONFIG_FILE_KEYS:
                skip_next = True
                continue

            filtered.append(arg)
        return filtered

    def _maybe_load_config_file(self, args: list[str]) -> dict[str, Any] | None:
        """
        Load config file if --config-file is specified in CLI args.

        Args:
            args: CLI arguments to search.

        Returns:
            Parsed config dict, or None if no config file specified.
        """
        file_path = self._find_config_file_path(args)
        if file_path is None:
            return None

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        return self._load_config_file(path)

    def _find_config_file_path(self, args: list[str]) -> str | None:
        """Extract config file path from CLI arguments."""
        for i, arg in enumerate(args):
            # Handle --config-file=path format
            if "=" in arg:
                key, value = arg.split("=", 1)
                if key in CONFIG_FILE_KEYS:
                    return value
            # Handle --config-file path format
            elif arg in CONFIG_FILE_KEYS and i < len(args) - 1:
                return args[i + 1]
        return None

    def _load_config_file(self, path: Path) -> dict[str, Any]:
        """
        Load config file based on extension.

        Args:
            path: Path to config file.

        Returns:
            Parsed config as dictionary.

        Raises:
            ValueError: If file format is not supported.
        """
        suffix = path.suffix.lower()

        if suffix in (".yaml", ".yml"):
            if not YAML_AVAILABLE:
                raise ImportError(
                    "PyYAML is required for YAML config files. "
                    "Install with: pip install pyyaml"
                )
            with open(path) as f:
                return yaml.safe_load(f) or {}

        elif suffix == ".toml":
            with open(path, "rb") as f:
                return tomllib.load(f)

        elif suffix == ".json":
            with open(path) as f:
                return json.load(f)

        else:
            raise ValueError(
                f"Unsupported config format: {suffix}. "
                "Supported formats: .yaml, .yml, .toml, .json"
            )

    def _maybe_add_custom_config(
        self, args: list[str], file_values: dict[str, Any] | None
    ) -> Type[T]:
        """
        Find and merge custom config module if specified.

        Searches CLI args first, then config file for custom_config_module.
        """
        module_path = None

        # Check CLI for --custom-config-module
        custom_keys = {"--custom-config-module", "--custom_config_module"}
        for i, arg in enumerate(args):
            key = arg.split("=")[0]
            if key in custom_keys:
                module_path = arg.split("=", 1)[1] if "=" in arg else args[i + 1]
                break

        # Check config file
        if not module_path and file_values:
            module_path = file_values.get("custom_config_module")

        if not module_path:
            return self.config_cls

        # Import and merge
        custom_config = importlib.import_module(module_path)
        if not hasattr(custom_config, "Config"):
            raise AttributeError(
                f"Custom config module {module_path} must define a 'Config' class"
            )
        return self._merge_configs(self.config_cls, custom_config.Config)

    @staticmethod
    def _merge_configs(base: Type[T], custom: Type) -> Type[T]:
        """
        Merge base config with custom extensions.

        Merge behavior:
        - Fields in both: recursively merge if both are dataclasses,
          otherwise custom overrides base
        - Fields only in base: preserved
        - Fields only in custom: added
        """
        if not is_dataclass(base) or not is_dataclass(custom):
            return base

        result = []
        base_fields = {f.name: f for f in fields(base)}
        custom_fields = {f.name: f for f in fields(custom)}

        # Process base fields
        for name, f in base_fields.items():
            if name in custom_fields:
                cf = custom_fields[name]
                # Recursively merge nested dataclasses
                if is_dataclass(f.type) and is_dataclass(cf.type):
                    merged = ConfigManager._merge_configs(f.type, cf.type)
                    result.append((name, merged, field(default_factory=merged)))
                else:
                    # Custom overrides base
                    result.append((name, cf.type, cf))
            else:
                result.append((name, f.type, f))

        # Add custom-only fields
        for name, f in custom_fields.items():
            if name not in base_fields:
                result.append((name, f.type, f))

        return make_dataclass(f"Merged{base.__name__}", result, bases=(base,))

    def _dict_to_dataclass(self, cls: Type[T], data: dict[str, Any]) -> T:
        """
        Convert dictionary to dataclass, handling nested structures.

        Args:
            cls: Target dataclass type.
            data: Dictionary with config values.

        Returns:
            Dataclass instance.

        Raises:
            ValueError: If data contains invalid field names.
        """
        if not is_dataclass(cls):
            return data  # type: ignore

        valid_fields = {f.name for f in fields(cls)}
        if invalid := set(data) - valid_fields - {"custom_config_module"}:
            raise ValueError(
                f"Invalid fields in config for {cls.__name__}: {invalid}. "
                f"Valid fields: {valid_fields}"
            )

        result = {}
        for f in fields(cls):
            if f.name in data:
                value = data[f.name]
                if is_dataclass(f.type) and isinstance(value, dict):
                    result[f.name] = self._dict_to_dataclass(f.type, value)
                else:
                    result[f.name] = value
        return cls(**result)

    def _merge_dict_into_dataclass(self, instance: T, data: dict[str, Any]) -> T:
        """
        Merge dictionary values into an existing dataclass instance.

        Creates a new instance with values from data overlaid on instance.
        Handles nested dataclasses recursively.

        Args:
            instance: Existing dataclass instance (e.g., from defaults()).
            data: Dictionary with config values to overlay.

        Returns:
            New dataclass instance with merged values.

        Raises:
            ValueError: If data contains invalid field names.
        """
        cls = type(instance)
        if not is_dataclass(cls):
            return data  # type: ignore

        valid_fields = {f.name for f in fields(cls)}
        if invalid := set(data) - valid_fields - {"custom_config_module"}:
            raise ValueError(
                f"Invalid fields in config for {cls.__name__}: {invalid}. "
                f"Valid fields: {valid_fields}"
            )

        # Start with all values from the existing instance
        result = {}
        for f in fields(cls):
            existing_value = getattr(instance, f.name)
            if f.name in data:
                new_value = data[f.name]
                # Recursively merge nested dataclasses
                if is_dataclass(f.type) and isinstance(new_value, dict):
                    if is_dataclass(existing_value):
                        result[f.name] = self._merge_dict_into_dataclass(
                            existing_value, new_value
                        )
                    else:
                        result[f.name] = self._dict_to_dataclass(f.type, new_value)
                else:
                    result[f.name] = new_value
            else:
                result[f.name] = existing_value

        return cls(**result)

    def _setup_tyro_registry(self) -> None:
        """Set up custom tyro parsing rules."""
        self._registry = tyro.constructors.ConstructorRegistry()

        @self._registry.primitive_rule
        def list_str_comma_rule(type_info: tyro.constructors.PrimitiveTypeInfo):
            """Support comma-separated string lists."""
            if type_info.type != list[str]:
                return None
            return tyro.constructors.PrimitiveConstructorSpec(
                nargs=1,
                metavar="A,B,C,...",
                instance_from_str=lambda args: args[0].split(","),
                is_instance=lambda instance: all(isinstance(i, str) for i in instance),
                str_from_instance=lambda instance: [",".join(instance)],
            )

        @self._registry.primitive_rule
        def art_uri_rule(type_info: tyro.constructors.PrimitiveTypeInfo):
            """Resolve art:// URIs to local paths during parsing.

            When a string field receives an art:// URI, this rule:
            1. Resolves the artifact to a local path via ArtifactFileSystem
            2. Registers W&B lineage via wandb.use_artifact() if a run is active
            3. Returns the local path string

            This ensures config fields receive actual local paths that work
            with code expecting filesystem paths (like Megatron-Bridge).
            """
            # Only apply to str fields (not Path - let those be handled normally)
            if type_info.type != str:
                return None

            return tyro.constructors.PrimitiveConstructorSpec(
                nargs=1,
                metavar="PATH_OR_ART_URI",
                instance_from_str=lambda args: resolve_artifact_uri(args[0]),
                is_instance=lambda x: isinstance(x, str),
                str_from_instance=lambda x: [str(x)],
            )


def resolve_artifact_uri(uri: str) -> str:
    """Resolve art:// URI to local path, registering W&B lineage.

    This function is called during tyro CLI parsing when a string field
    receives an art:// URI. It:
    1. Resolves the artifact to a local filesystem path
    2. Registers W&B lineage via wandb.use_artifact() if a run is active
    3. Returns the local path as a string

    Args:
        uri: Either an art:// URI or a local path. If not an art:// URI,
             returns the input unchanged.

    Returns:
        Local filesystem path (as string).

    Examples:
        >>> resolve_artifact_uri("art://DataBlendsArtifact-pretrain:v10/blend.json")
        '/path/to/wandb/artifacts/DataBlendsArtifact-pretrain-v10/blend.json'

        >>> resolve_artifact_uri("art://romeyn/nemotron/DataBlendsArtifact-pretrain:v10")
        '/path/to/wandb/artifacts/...'

        >>> resolve_artifact_uri("/local/path/to/file.json")
        '/local/path/to/file.json'  # Unchanged
    """
    if not uri.startswith("art://"):
        return uri  # Already a local path, return unchanged

    from nemotron.kit.filesystem import ArtifactFileSystem

    fs = ArtifactFileSystem()

    # Resolve the URI to local path
    artifact_path, file_path = fs._resolve(uri)

    # Register W&B lineage if in an active run
    try:
        import wandb

        if wandb.run is not None:
            # Extract artifact name and version from URI for use_artifact()
            name, version, _ = fs._parse_uri(uri)

            # Build the artifact reference for W&B
            if version is not None:
                artifact_ref = f"{name}:v{version}"
            else:
                artifact_ref = f"{name}:latest"

            # Register usage for lineage tracking
            wandb.use_artifact(artifact_ref)
    except ImportError:
        pass  # wandb not installed, skip lineage tracking

    # Build the full path including any file path within the artifact
    full_path = artifact_path / file_path if file_path else artifact_path
    return str(full_path)


def _read_stdin_artifacts() -> dict[str, dict] | None:
    """Read artifacts from stdin if piped.

    Returns:
        Dictionary mapping artifact names to their info (path, type),
        or None if stdin is not piped or invalid JSON.
    """
    if sys.stdin.isatty():
        return None
    try:
        data = json.loads(sys.stdin.read())
        # Validate structure: should be dict of {name: {path, type}}
        if isinstance(data, dict) and all(
            isinstance(v, dict) and "path" in v for v in data.values()
        ):
            return data
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def _load_artifact_metadata(artifact_info: dict) -> dict:
    """Load full metadata from artifact path.

    Args:
        artifact_info: Dict with at least "path" key pointing to artifact directory.

    Returns:
        Full artifact metadata from metadata.json.

    Raises:
        FileNotFoundError: If metadata.json doesn't exist.
    """
    path = Path(artifact_info["path"])
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Artifact metadata not found: {metadata_path}")
    with open(metadata_path) as f:
        return json.load(f)


def _apply_parse_inputs(
    args: list[str],
    parse_inputs: dict[str, str],
    stdin_artifacts: dict[str, dict],
) -> tuple[list[str], dict[str, Any]]:
    """Inject artifact fields as CLI args or collect as fn kwargs.

    Args:
        args: Original CLI arguments.
        parse_inputs: Mapping of "artifact.field" -> "config.field" or "fn.kwarg_name".
                     Entries with "fn." prefix are collected as kwargs for defaults_fn().
        stdin_artifacts: Artifacts read from stdin.

    Returns:
        Tuple of (modified_args, fn_kwargs):
        - modified_args: Args list with artifact values prepended (for non-fn. entries)
        - fn_kwargs: Dict of kwargs to pass to defaults_fn() (for fn. entries)

    Raises:
        ValueError: If artifact field doesn't exist.
    """
    new_args = list(args)
    fn_kwargs: dict[str, Any] = {}

    for artifact_field, target in parse_inputs.items():
        # Parse "data.blend_path" -> artifact="data", field="blend_path"
        if "." not in artifact_field:
            raise ValueError(
                f"Invalid parse_inputs key: '{artifact_field}'. "
                "Expected format: 'artifact_name.field_name'"
            )
        artifact_name, field_name = artifact_field.split(".", 1)

        if artifact_name not in stdin_artifacts:
            continue  # Not provided via stdin, maybe via CLI

        # Load artifact metadata
        metadata = _load_artifact_metadata(stdin_artifacts[artifact_name])

        # Extract field value
        if field_name not in metadata:
            available = ", ".join(sorted(metadata.keys()))
            raise ValueError(
                f"Artifact '{artifact_name}' has no field '{field_name}'. "
                f"Available fields: {available}"
            )

        value = metadata[field_name]
        if value is not None:
            # Check if this is a fn.* target (kwarg for defaults_fn)
            if target.startswith("fn."):
                kwarg_name = target[3:]  # Remove "fn." prefix
                fn_kwargs[kwarg_name] = value
            else:
                # Prepend as CLI arg (lowest priority, CLI overrides)
                new_args = [f"--{target}", str(value)] + new_args

    return new_args, fn_kwargs


@overload
def cli(
    config_cls: Type[T],
    /,
    *,
    args: list[str] | None = None,
    parse_inputs: dict[str, str] | None = None,
    defaults: Callable[[], T] | None = None,
    defaults_fn: Callable[..., T] | None = None,
    kwargs_schema: type | None = None,
    ray: bool = False,
    pre_ray_start_commands: list[str] | None = None,
) -> T: ...


@overload
def cli(
    main: Callable[..., T],
    /,
    *,
    args: list[str] | None = None,
    parse_inputs: dict[str, str] | None = None,
    defaults: Callable[[], Any] | None = None,
    defaults_fn: Callable[..., Any] | None = None,
    kwargs_schema: type | None = None,
    ray: bool = False,
    pre_ray_start_commands: list[str] | None = None,
) -> T: ...


def cli(
    config_or_main: Type[T] | Callable[..., T],
    /,
    *,
    args: list[str] | None = None,
    parse_inputs: dict[str, str] | None = None,
    defaults: Callable[[], Any] | None = None,
    defaults_fn: Callable[..., Any] | None = None,
    kwargs_schema: type | None = None,
    ray: bool = False,
    pre_ray_start_commands: list[str] | None = None,
) -> T:
    """
    Drop-in replacement for tyro.cli with config file, artifact piping, and nemo-run support.

    Supports loading configuration from YAML, TOML, or JSON files via
    --config-file. CLI arguments override config file values.

    When parse_inputs is provided, enables Unix-style piping between steps:
    - Reads artifact JSON from stdin (output of previous step's print_step_complete())
    - Maps artifact fields to config fields per the parse_inputs mapping
    - Entries with "fn." prefix are passed as kwargs to defaults_fn()
    - Other entries are injected as CLI args (lowest priority, can be overridden)

    When --run <profile> is provided, executes via nemo-run with the named profile
    from run.toml/yaml/json. Supports all nemo-run executors: local, docker, slurm,
    skypilot, dgxcloud, lepton.

    Configuration precedence (highest to lowest):
        CLI args > config file > defaults() > dataclass defaults

    Args:
        config_or_main: Either a dataclass type or a function with typed parameters.
        args: CLI arguments. Defaults to sys.argv[1:].
        parse_inputs: Mapping of artifact fields to config fields for stdin piping.
                     Format: {"artifact_name.field": "config.nested.field"}
                     Use "fn." prefix to pass values to defaults_fn():
                     {"artifact_name.field": "fn.kwarg_name"}
        defaults: Optional callable that returns a default config instance.
                 Used for model-specific defaults from external recipe functions.
                 Example: defaults=nemotron_nano_v2
        defaults_fn: Optional callable that returns a default config instance.
                    Unlike defaults, this is called with kwargs extracted from
                    parse_inputs entries with "fn." prefix. The result becomes
                    the defaults value. Useful for recipe functions that need
                    runtime arguments (e.g., per_split_data_args_path).
                    Example: defaults_fn=nano_3_pretrain_config
        kwargs_schema: Optional TypedDict class defining kwargs for defaults_fn.
                      Fields from this TypedDict become CLI arguments (--fn.<field-name>)
                      that are passed to defaults_fn(). Requires defaults_fn to be set.
                      Artifact fn. values take precedence over CLI kwargs.
                      Example: kwargs_schema=NemotronNext3Bv2CommonKwargs
        ray: Whether this recipe requires Ray for execution (e.g., RL training).
        pre_ray_start_commands: Commands to run before Ray starts (only used if ray=True).

    Returns:
        For dataclass: populated instance.
        For function: return value of calling the function.

    Examples:
        # With a dataclass
        >>> @dataclass
        ... class Config:
        ...     batch_size: int = 32
        >>> config = cli(Config)

        # With a function (like tyro.cli)
        >>> def main(batch_size: int = 32) -> None:
        ...     print(batch_size)
        >>> cli(main)

        # With config file
        >>> cli(Config, args=["--config-file", "config.yaml"])

        # With artifact piping
        >>> cli(main, parse_inputs={"data.blend_path": "data.data_path"})
        # Enables: python data_prep.py | python train.py

        # With model-specific defaults
        >>> from megatron.bridge.recipes import nemotron_nano_v2
        >>> cli(main, defaults=nemotron_nano_v2)

        # With defaults_fn and fn. prefix (passes artifact value to recipe function)
        >>> cli(main, defaults_fn=nano_3_pretrain_config,
        ...     parse_inputs={"data.blend_path": "fn.per_split_data_args_path"})

        # With kwargs_schema (exposes TypedDict fields as CLI args for defaults_fn)
        >>> cli(main, defaults_fn=nano_3_pretrain_config,
        ...     kwargs_schema=NemotronNext3Bv2CommonKwargs,
        ...     parse_inputs={"data.blend_path": "fn.per_split_data_args_path"})
        # Now supports: python train.py --fn.seq-length 4096 --fn.mock

        # Execute via nemo-run
        >>> # python train.py --run draco
        >>> cli(main, defaults=nemotron_nano_v2)

        # With Ray support (for RL training)
        >>> cli(main, ray=True, pre_ray_start_commands=["pip install nemo-rl"])
    """
    if args is None:
        args = sys.argv[1:]

    # Check for --run/--batch <profile> and handle nemo-run execution
    run_name, run_overrides, remaining_args, is_launch = _extract_run_args(args)
    if run_name is not None:
        return _execute_with_nemo_run(
            run_name=run_name,
            run_overrides=run_overrides,
            script_args=remaining_args,
            ray=ray,
            pre_ray_start_commands=pre_ray_start_commands,
            is_launch=is_launch,
        )

    # Validate kwargs_schema requires defaults_fn
    if kwargs_schema is not None and defaults_fn is None:
        raise ValueError("kwargs_schema requires defaults_fn to be set")

    # Parse kwargs_schema CLI args (--fn.* args) if provided
    cli_fn_kwargs: dict[str, Any] = {}
    if kwargs_schema is not None:
        from nemotron.kit.app import _typeddict_to_dataclass
        kwargs_dataclass = _typeddict_to_dataclass(kwargs_schema, prefix="fn.")
        # Extract --fn.* args from args list
        fn_args = [a for a in args if a.startswith("--fn.")]
        fn_args_with_values = []
        i = 0
        while i < len(args):
            if args[i].startswith("--fn."):
                fn_args_with_values.append(args[i])
                # Check if next arg is the value (not another flag)
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    fn_args_with_values.append(args[i + 1])
                    i += 2
                else:
                    i += 1
            else:
                i += 1

        if fn_args_with_values:
            # Parse the fn.* args using tyro
            parsed_kwargs = tyro.cli(kwargs_dataclass, args=fn_args_with_values)
            # Extract non-None values
            for field_name in getattr(parsed_kwargs, "__dataclass_fields__", {}):
                value = getattr(parsed_kwargs, field_name)
                if value is not None:
                    cli_fn_kwargs[field_name] = value

        # Remove --fn.* args from main args list
        filtered_args = []
        i = 0
        while i < len(args):
            if args[i].startswith("--fn."):
                # Skip flag and its value if present
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    i += 2
                else:
                    i += 1
            else:
                filtered_args.append(args[i])
                i += 1
        args = filtered_args

    # Apply parse_inputs from stdin artifacts if provided
    artifact_fn_kwargs: dict[str, Any] = {}
    if parse_inputs:
        stdin_artifacts = _read_stdin_artifacts()
        if stdin_artifacts:
            args, artifact_fn_kwargs = _apply_parse_inputs(args, parse_inputs, stdin_artifacts)

    # Validate fn.* usage requires defaults_fn
    if artifact_fn_kwargs and defaults_fn is None:
        fn_keys = [k for k, v in parse_inputs.items() if v.startswith("fn.")]
        raise ValueError(
            f"parse_inputs contains 'fn.' targets {fn_keys} but defaults_fn is not provided. "
            "Either remove the 'fn.' prefix or provide defaults_fn."
        )

    # Merge kwargs: CLI kwargs as base, artifact fn. kwargs override
    fn_kwargs: dict[str, Any] = dict(cli_fn_kwargs)
    fn_kwargs.update(artifact_fn_kwargs)  # Artifact values take precedence

    # If defaults_fn is provided, call it with fn_kwargs and use result as defaults
    if defaults_fn is not None:
        defaults = lambda: defaults_fn(**fn_kwargs)

    # Helper to initialize wandb from config file if present
    def _maybe_init_wandb_from_config(manager: ConfigManager) -> None:
        """Initialize wandb from [wandb] section in config file if present."""
        wandb_config = manager.get_wandb_config()
        if wandb_config is not None:
            from nemotron.kit.wandb import init_wandb_if_configured
            init_wandb_if_configured(wandb_config, job_type="cli")

    # Check if it's a dataclass or a callable
    if is_dataclass(config_or_main):
        manager = ConfigManager(config_or_main, defaults=defaults)
        config = manager.parse_args(args)
        _maybe_init_wandb_from_config(manager)
        return config

    # It's a callable - extract config type from signature if possible
    func = config_or_main
    import inspect

    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    # Check if function takes a single dataclass argument
    if len(params) == 1:
        param = params[0]
        if param.annotation != inspect.Parameter.empty and is_dataclass(param.annotation):
            # Function takes a single dataclass - use ConfigManager
            manager = ConfigManager(param.annotation, defaults=defaults)
            config = manager.parse_args(args)
            _maybe_init_wandb_from_config(manager)
            return func(config)

    # Fall back to building a dataclass from function parameters
    # and using ConfigManager with that
    config_cls = _func_to_dataclass(func)
    if config_cls is not None:
        # Note: defaults is not supported for auto-generated dataclasses from functions
        manager = ConfigManager(config_cls)
        config = manager.parse_args(args)
        _maybe_init_wandb_from_config(manager)
        # Convert dataclass back to kwargs
        kwargs = {f.name: getattr(config, f.name) for f in fields(config_cls)}
        return func(**kwargs)

    # If we can't extract a config, fall back to plain tyro.cli
    return tyro.cli(func, args=args)


def _func_to_dataclass(func: Callable) -> Type | None:
    """
    Convert function signature to a dataclass for config file support.

    Returns None if the function signature can't be converted.
    """
    import inspect

    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if not params:
        return None

    # Build dataclass fields from parameters
    dc_fields = []
    for param in params:
        if param.annotation == inspect.Parameter.empty:
            # Can't build dataclass without type annotations
            return None

        if param.default != inspect.Parameter.empty:
            dc_fields.append((param.name, param.annotation, field(default=param.default)))
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            # Required keyword-only param without default - not supported
            return None
        else:
            # Required positional param - use MISSING-like pattern
            dc_fields.append((param.name, param.annotation))

    return make_dataclass(f"{func.__name__}_Config", dc_fields)


def _extract_run_args(args: list[str]) -> tuple[str | None, dict[str, str], list[str], bool]:
    """Extract --run/--batch arguments from CLI args.

    Parses --run <profile> and --run.<key> <value> overrides from args,
    or --batch <profile> and --batch.<key> <value> overrides.
    Returns the profile name, overrides dict, remaining args, and whether
    batch mode (detached execution) was used.

    Args:
        args: Original CLI arguments.

    Returns:
        Tuple of (profile_name, overrides_dict, remaining_args, is_launch).
        profile_name is None if neither --run nor --batch specified.
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

        # Handle --run <profile>
        if arg == "--run":
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                run_name = args[i + 1]
                i += 2
                continue
            else:
                raise ValueError("--run requires a profile name")

        # Handle --run=<profile>
        if arg.startswith("--run="):
            run_name = arg.split("=", 1)[1]
            i += 1
            continue

        # Handle --run.<key> <value> or --run.<key>=<value>
        if arg.startswith("--run."):
            key = arg[6:]  # Remove "--run."
            if "=" in key:
                key, value = key.split("=", 1)
                run_overrides[key] = value
            elif i + 1 < len(args):
                run_overrides[key] = args[i + 1]
                i += 2
                continue
            else:
                raise ValueError(f"--run.{key} requires a value")
            i += 1
            continue

        # Handle --batch / -b <profile>
        if arg == "--batch" or arg == "-b":
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                launch_name = args[i + 1]
                i += 2
                continue
            else:
                raise ValueError("--batch requires a profile name")

        # Handle --batch=<profile> or -b=<profile>
        if arg.startswith("--batch="):
            launch_name = arg.split("=", 1)[1]
            i += 1
            continue

        if arg.startswith("-b="):
            launch_name = arg.split("=", 1)[1]
            i += 1
            continue

        # Handle --batch.<key> <value> or --batch.<key>=<value>
        if arg.startswith("--batch."):
            key = arg[8:]  # Remove "--batch."
            if "=" in key:
                key, value = key.split("=", 1)
                run_overrides[key] = value
            elif i + 1 < len(args):
                run_overrides[key] = args[i + 1]
                i += 2
                continue
            else:
                raise ValueError(f"--batch.{key} requires a value")
            i += 1
            continue

        remaining.append(arg)
        i += 1

    # Validate mutual exclusivity
    if run_name is not None and launch_name is not None:
        raise ValueError("--run and --batch are mutually exclusive. Use --run for attached execution or --batch for detached execution.")

    # Determine final name and whether batch mode is active
    is_launch = launch_name is not None
    final_name = launch_name if is_launch else run_name

    return final_name, run_overrides, remaining, is_launch


def _execute_with_nemo_run(
    run_name: str,
    run_overrides: dict[str, str],
    script_args: list[str],
    ray: bool,
    pre_ray_start_commands: list[str] | None,
    is_launch: bool = False,
) -> None:
    """Execute the current script via nemo-run.

    Args:
        run_name: Profile name from run.toml/yaml/json.
        run_overrides: CLI overrides for the run config.
        script_args: Arguments to pass to the script.
        ray: Whether to use Ray execution.
        pre_ray_start_commands: Commands to run before Ray starts.
        is_launch: If True, force detach=True for detached execution.

    Raises:
        SystemExit: After nemo-run execution completes.
    """
    from nemotron.kit.run import load_run_profile, run_with_nemo_run

    # Load and apply overrides to run config
    run_config = load_run_profile(run_name)

    # Force detach=True and ray_mode="job" when using --batch
    # ray_mode="job" ensures the cluster terminates after the job completes
    if is_launch:
        run_config.detach = True
        run_config.ray_mode = "job"

    # Apply CLI overrides
    for key, value in run_overrides.items():
        if hasattr(run_config, key):
            # Handle type conversion for common types
            field_type = type(getattr(run_config, key))
            if field_type == bool:
                value = value.lower() in ("true", "1", "yes")
            elif field_type == int:
                value = int(value)
            elif field_type == list:
                value = value.split(",") if value else []
            setattr(run_config, key, value)
        else:
            raise ValueError(f"Unknown run config field: {key}")

    # Get the script path from sys.argv[0]
    script_path = sys.argv[0]

    # Execute via nemo-run
    exit_code = run_with_nemo_run(
        script_path=script_path,
        script_args=script_args,
        run_config=run_config,
        ray=ray,
        pre_ray_start_commands=pre_ray_start_commands,
    )

    # Exit with the same code
    sys.exit(exit_code)
