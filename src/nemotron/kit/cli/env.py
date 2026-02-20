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

"""Environment profile loading from env.toml.

Handles loading executor configurations and profile inheritance.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from nemotron.kit.cli.globals import GlobalContext

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def find_env_file(start_dir: Path | None = None) -> Path | None:
    """Find env.toml walking up from start_dir.

    Args:
        start_dir: Directory to start searching from. Defaults to cwd.

    Returns:
        Path to env.toml if found, None otherwise.
    """
    if start_dir is None:
        start_dir = Path.cwd()

    for path in [start_dir, *start_dir.parents]:
        env_file = path / "env.toml"
        if env_file.exists():
            return env_file

        # Stop at project root (has pyproject.toml)
        if (path / "pyproject.toml").exists():
            break

    return None


def load_env_file(config_path: Path | None = None) -> dict[str, Any]:
    """Load env.toml file contents.

    Args:
        config_path: Path to env.toml. If None, searches for it.

    Returns:
        Dictionary with all profiles from the file.
    """
    if config_path is None:
        config_path = find_env_file()

    if config_path is None:
        return {}

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def parse_env(ctx: GlobalContext, *, config_path: Path | None = None) -> DictConfig | None:
    """Parse environment profile from env.toml based on CLI context.

    This is the main entry point for loading env configs. It:
    1. Checks if --run or --batch was specified (ctx.profile)
    2. Loads and resolves the profile from env.toml
    3. Returns OmegaConf DictConfig for attribute access

    Args:
        ctx: Global CLI context with profile name from --run/--batch
        config_path: Optional explicit path to env.toml

    Returns:
        OmegaConf DictConfig with resolved profile, or None if no profile specified
    """
    if not ctx.profile:
        return None

    profile = load_env_profile(ctx.profile, config_path=config_path)
    # Return resolved config for clean attribute access
    return OmegaConf.create(OmegaConf.to_container(profile, resolve=True))


def load_env_profile(name: str, config_path: Path | None = None) -> DictConfig:
    """Load a named profile from env.toml with inheritance support.

    Profiles can extend other profiles using `extends = "parent"`.

    Args:
        name: Profile name (e.g., "dlw", "local")
        config_path: Optional path to env.toml

    Returns:
        OmegaConf DictConfig with the resolved profile

    Raises:
        ValueError: If profile not found
    """
    all_profiles = load_env_file(config_path)

    if name not in all_profiles:
        available = [k for k in all_profiles.keys() if k != "wandb"]
        raise ValueError(f"Profile '{name}' not found in env.toml. Available: {available}")

    return _resolve_profile(name, all_profiles, set())


def _resolve_profile(
    name: str,
    all_profiles: dict[str, Any],
    visited: set[str],
) -> DictConfig:
    """Recursively resolve profile with inheritance.

    Args:
        name: Profile name to resolve
        all_profiles: All profiles from env.toml
        visited: Set of already visited profiles (for cycle detection)

    Returns:
        Resolved profile as DictConfig
    """
    if name in visited:
        raise ValueError(f"Circular profile inheritance detected: {name}")

    visited.add(name)
    profile = all_profiles[name].copy()

    # Handle 'extends' inheritance
    if "extends" in profile:
        parent_name = profile.pop("extends")
        if parent_name not in all_profiles:
            raise ValueError(f"Profile '{name}' extends unknown profile '{parent_name}'")
        parent = _resolve_profile(parent_name, all_profiles, visited)
        # Merge: child overrides parent
        profile = OmegaConf.merge(parent, OmegaConf.create(profile))
    else:
        profile = OmegaConf.create(profile)

    return profile


def get_wandb_config(config_path: Path | None = None) -> DictConfig | None:
    """Get wandb configuration from env.toml if present.

    Args:
        config_path: Optional path to env.toml

    Returns:
        WandB config as DictConfig, or None if not present
    """
    all_profiles = load_env_file(config_path)

    if "wandb" in all_profiles:
        return OmegaConf.create(all_profiles["wandb"])

    return None


def get_cli_config(config_path: Path | None = None) -> DictConfig | None:
    """Get CLI display configuration from env.toml if present.

    Args:
        config_path: Optional path to env.toml

    Returns:
        CLI config as DictConfig, or None if not present
    """
    all_profiles = load_env_file(config_path)

    if "cli" in all_profiles:
        return OmegaConf.create(all_profiles["cli"])

    return None


def get_cache_config(config_path: Path | None = None) -> DictConfig | None:
    """Get cache configuration from env.toml if present.

    The [cache] section can contain:
    - git_dir: Directory for caching git repos (default: ~/.nemotron/git-cache)

    Example env.toml:
        [cache]
        git_dir = "/path/to/git-cache"

    Args:
        config_path: Optional path to env.toml

    Returns:
        Cache config as DictConfig, or None if not present
    """
    all_profiles = load_env_file(config_path)

    if "cache" in all_profiles:
        return OmegaConf.create(all_profiles["cache"])

    return None


def get_git_cache_dir(config_path: Path | None = None) -> Path:
    """Get the git cache directory from env.toml or use default.

    Looks for [cache] git_dir in env.toml. Falls back to ~/.nemotron/git-cache
    if the configured path is not accessible (e.g., remote cluster path on local machine).

    Args:
        config_path: Optional path to env.toml

    Returns:
        Path to git cache directory (created if it doesn't exist)
    """
    cache_config = get_cache_config(config_path)
    default_cache_dir = Path.home() / ".nemotron" / "git-cache"

    if cache_config and cache_config.get("git_dir"):
        cache_dir = Path(cache_config.git_dir).expanduser()
        # Try to create/access the configured directory
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            return cache_dir
        except OSError:
            # Fall back to default if the configured path is not accessible
            # (e.g., remote cluster path like /lustre/... on local machine)
            pass

    default_cache_dir.mkdir(parents=True, exist_ok=True)
    return default_cache_dir
