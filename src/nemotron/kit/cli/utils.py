"""Shared utilities for CLI components."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def resolve_run_interpolations(obj: Any, run_data: dict) -> Any:
    """Recursively resolve ${run.*} interpolations in a dict/list.

    Only resolves ${run.X.Y} style interpolations, preserves other
    interpolations like ${art:data,path}.

    Args:
        obj: Object to process (dict, list, or scalar)
        run_data: The run section data to resolve from

    Returns:
        Object with ${run.*} interpolations resolved
    """
    if isinstance(obj, dict):
        return {k: resolve_run_interpolations(v, run_data) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_run_interpolations(item, run_data) for item in obj]
    elif isinstance(obj, str) and obj.startswith("${run.") and obj.endswith("}"):
        # Extract the path: ${run.wandb.project} -> wandb.project
        path = obj[6:-1]  # Remove "${run." and "}"
        # Navigate run_data to get the value
        parts = path.split(".")
        value = run_data
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return obj  # Can't resolve, keep original
        return value
    else:
        return obj


def rewrite_paths_for_remote(obj: Any, repo_root: Path | str) -> Any:
    """Recursively rewrite paths for remote execution.

    Rewrites:
    - ${oc.env:PWD}/... -> /nemo_run/code/...
    - ${oc.env:NEMO_RUN_DIR,...}/... -> /nemo_run/...
    - Absolute paths under repo_root -> /nemo_run/code/...

    Args:
        obj: Object to process (dict, list, or scalar)
        repo_root: Local repository root path

    Returns:
        Object with paths rewritten for remote execution
    """
    repo_root_str = str(repo_root)

    if isinstance(obj, dict):
        return {k: rewrite_paths_for_remote(v, repo_root_str) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [rewrite_paths_for_remote(item, repo_root_str) for item in obj]
    elif isinstance(obj, str):
        # Rewrite ${oc.env:PWD}/... to /nemo_run/code/...
        if "${oc.env:PWD}" in obj:
            return obj.replace("${oc.env:PWD}", "/nemo_run/code")

        # Rewrite ${oc.env:NEMO_RUN_DIR,...}/... to /nemo_run/...
        # Handles both ${oc.env:NEMO_RUN_DIR} and ${oc.env:NEMO_RUN_DIR,.}
        match = re.match(r"\$\{oc\.env:NEMO_RUN_DIR[^}]*\}(.*)", obj)
        if match:
            suffix = match.group(1)
            return f"/nemo_run{suffix}"

        # Rewrite absolute paths under repo_root to /nemo_run/code/...
        if obj.startswith(repo_root_str):
            rel_path = obj[len(repo_root_str) :].lstrip("/")
            return f"/nemo_run/code/{rel_path}"

    return obj


def extract_run_args(args: list[str]) -> tuple[str | None, dict[str, str], list[str], bool]:
    """Extract --run/--batch arguments from CLI args.

    Parses --run <profile> and --run.<key> <value> overrides from args,
    or --batch <profile> and --batch.<key> <value> overrides.
    Returns the profile name, overrides dict, remaining args, and whether
    batch mode (detached execution) was used.

    Supports both long and short forms:
    - --run <profile> or -r <profile>
    - --run=<profile> or -r=<profile>
    - --run.<key> <value> or --run.<key>=<value>
    - --batch <profile> or -b <profile>
    - --batch=<profile> or -b=<profile>
    - --batch.<key> <value> or --batch.<key>=<value>

    Args:
        args: Original CLI arguments.

    Returns:
        Tuple of (profile_name, overrides_dict, remaining_args, is_launch).
        profile_name is None if neither --run nor --batch specified.
        is_launch is True when --batch was used (implies detach=True).

    Raises:
        ValueError: If both --run and --batch are specified, or if a required value is missing.
    """
    run_name: str | None = None
    launch_name: str | None = None
    run_overrides: dict[str, str] = {}
    remaining: list[str] = []

    i = 0
    while i < len(args):
        arg = args[i]

        # Handle --run <profile> or -r <profile>
        if arg == "--run" or arg == "-r":
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                run_name = args[i + 1]
                i += 2
                continue
            else:
                raise ValueError("--run requires a profile name")

        # Handle --run=<profile> or -r=<profile>
        if arg.startswith("--run="):
            run_name = arg.split("=", 1)[1]
            i += 1
            continue
        if arg.startswith("-r="):
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
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
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
        raise ValueError(
            "--run and --batch are mutually exclusive. "
            "Use --run for attached execution or --batch for detached execution."
        )

    # Determine final name and whether batch mode is active
    is_launch = launch_name is not None
    final_name = launch_name if is_launch else run_name

    return final_name, run_overrides, remaining, is_launch


# Valid CLI argument names for config file path
CONFIG_FILE_KEYS = {"--config-file", "--config_file", "--config", "-c"}


def filter_config_file_args(args: list[str]) -> list[str]:
    """Remove --config-file and related arguments from args list.

    Handles both formats:
    - --config-file path
    - --config-file=path
    - -c path
    - -c=path

    Args:
        args: Command line arguments

    Returns:
        Filtered arguments without config file flags
    """
    filtered: list[str] = []
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
