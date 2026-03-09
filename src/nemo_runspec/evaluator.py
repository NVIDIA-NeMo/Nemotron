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

"""Evaluator helpers for nemo-evaluator-launcher integration.

Provides shared utilities for evaluator CLI commands:
- Task flag parsing (-t/--task)
- W&B credential auto-propagation
- Container auto-squash for deployment images
- Eval config save (strips 'run' section before passing to launcher)

These are generic nemo_runspec utilities with no nemotron imports.
The CLI commands import and call them explicitly — visible execution logic,
no decorator magic.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


# =============================================================================
# Task Flag Parsing
# =============================================================================


def parse_task_flags(passthrough: list[str]) -> list[str] | None:
    """Parse -t/--task flags from passthrough args.

    Args:
        passthrough: List of passthrough arguments

    Returns:
        List of task names, or None if no tasks specified
    """
    tasks = []
    i = 0
    while i < len(passthrough):
        if passthrough[i] in ("-t", "--task") and i + 1 < len(passthrough):
            tasks.append(passthrough[i + 1])
            i += 2
        else:
            i += 1
    return tasks if tasks else None


def get_non_task_args(passthrough: list[str]) -> list[str]:
    """Get passthrough args that are not -t/--task flags.

    Args:
        passthrough: List of passthrough arguments

    Returns:
        List of non-task arguments
    """
    extra = []
    i = 0
    while i < len(passthrough):
        if passthrough[i] in ("-t", "--task") and i + 1 < len(passthrough):
            i += 2  # Skip -t and its value
        else:
            extra.append(passthrough[i])
            i += 1
    return extra


# =============================================================================
# W&B Token Auto-Propagation
# =============================================================================
# Similar to how nemo-run automatically passes WANDB_API_KEY when logged in,
# these helpers ensure the evaluator launcher receives the W&B credentials.


def needs_wandb(cfg: Any) -> bool:
    """Check if config requires W&B credentials.

    Returns True if:
    - execution.auto_export.destinations contains "wandb", OR
    - export.wandb section exists

    Args:
        cfg: Job configuration (OmegaConf DictConfig or dict)

    Returns:
        True if W&B credentials are needed
    """
    from omegaconf import OmegaConf

    # Convert to dict for easier access
    if hasattr(cfg, "_content"):
        cfg_dict = OmegaConf.to_container(cfg, resolve=False)
    else:
        cfg_dict = cfg

    # Check execution.auto_export.destinations
    try:
        destinations = cfg_dict.get("execution", {}).get("auto_export", {}).get("destinations", [])
        if "wandb" in destinations:
            return True
    except (AttributeError, TypeError):
        pass

    # Check export.wandb section
    try:
        if cfg_dict.get("export", {}).get("wandb") is not None:
            return True
    except (AttributeError, TypeError):
        pass

    return False


def ensure_wandb_host_env() -> None:
    """Ensure W&B environment variables are set on the host.

    Auto-detects WANDB_API_KEY from local wandb login (same as nemo-run).
    Also sets WANDB_PROJECT/WANDB_ENTITY from env.toml [wandb] section.

    This is required because nemo-evaluator-launcher checks os.getenv()
    for env_vars mappings at submission time.
    """
    from nemo_runspec.env import get_wandb_config

    # Auto-detect WANDB_API_KEY from wandb login
    if "WANDB_API_KEY" not in os.environ:
        try:
            import wandb

            api_key = wandb.api.api_key
            if api_key:
                os.environ["WANDB_API_KEY"] = api_key
                sys.stderr.write("[info] Detected W&B login, setting WANDB_API_KEY\n")
        except Exception:
            pass  # wandb not installed or not logged in

    # Load WANDB_PROJECT/WANDB_ENTITY from env.toml [wandb] section
    wandb_config = get_wandb_config()
    if wandb_config is not None:
        if wandb_config.get("project") and "WANDB_PROJECT" not in os.environ:
            os.environ["WANDB_PROJECT"] = wandb_config.project
        if wandb_config.get("entity") and "WANDB_ENTITY" not in os.environ:
            os.environ["WANDB_ENTITY"] = wandb_config.entity


def inject_wandb_env_mappings(cfg: Any) -> None:
    """Inject W&B env var mappings into evaluator config.

    The nemo-evaluator-launcher expects:
    - evaluation.env_vars: mapping of container env var -> host env var name
    - execution.env_vars.export: env vars for the W&B export container

    This function adds the WANDB_API_KEY (and optionally PROJECT/ENTITY)
    mappings so the launcher knows to forward these from the host environment.

    Note: This only adds string mappings (e.g., "WANDB_API_KEY": "WANDB_API_KEY"),
    not actual secrets. The launcher resolves these via os.getenv() at runtime.

    Args:
        cfg: Job configuration (OmegaConf DictConfig) - modified in place
    """
    from omegaconf import open_dict

    # Helper to safely set nested dict value
    def _ensure_nested(cfg_node: Any, *keys: str) -> Any:
        """Ensure nested dict path exists, creating dicts as needed."""
        current = cfg_node
        for key in keys:
            if key not in current or current[key] is None:
                with open_dict(current):
                    current[key] = {}
            current = current[key]
        return current

    # Inject into evaluation.env_vars (for evaluation containers)
    try:
        eval_env = _ensure_nested(cfg, "evaluation", "env_vars")
        with open_dict(eval_env):
            if "WANDB_API_KEY" not in eval_env:
                eval_env["WANDB_API_KEY"] = "WANDB_API_KEY"
            if "WANDB_PROJECT" not in eval_env:
                eval_env["WANDB_PROJECT"] = "WANDB_PROJECT"
            if "WANDB_ENTITY" not in eval_env:
                eval_env["WANDB_ENTITY"] = "WANDB_ENTITY"
    except Exception:
        pass  # Config structure doesn't support this

    # Inject into execution.env_vars.export (for W&B export container)
    try:
        export_env = _ensure_nested(cfg, "execution", "env_vars", "export")
        with open_dict(export_env):
            if "WANDB_API_KEY" not in export_env:
                export_env["WANDB_API_KEY"] = "WANDB_API_KEY"
            if "WANDB_PROJECT" not in export_env:
                export_env["WANDB_PROJECT"] = "WANDB_PROJECT"
            if "WANDB_ENTITY" not in export_env:
                export_env["WANDB_ENTITY"] = "WANDB_ENTITY"
    except Exception:
        pass  # Config structure doesn't support this


# =============================================================================
# Container Auto-Squash for Slurm
# =============================================================================
# Similar to how training recipes auto-squash Docker images for Slurm,
# these helpers ensure evaluator container images are squashed before execution.


def collect_evaluator_images(cfg: Any) -> list[tuple[str, str]]:
    """Collect (dotpath, image) for all container images in eval config.

    Args:
        cfg: Evaluator configuration (OmegaConf DictConfig)

    Returns:
        List of (dotpath, image_value) tuples for images that need squashing
    """
    from omegaconf import OmegaConf

    images = []

    # Deployment image
    dep_image = OmegaConf.select(cfg, "deployment.image")
    if dep_image and isinstance(dep_image, str):
        images.append(("deployment.image", dep_image))

    # Proxy image (if present)
    proxy_image = OmegaConf.select(cfg, "execution.proxy.image")
    if proxy_image and isinstance(proxy_image, str):
        images.append(("execution.proxy.image", proxy_image))

    return images


def maybe_auto_squash_evaluator(
    job_config: Any,
    *,
    mode: str,
    dry_run: bool,
    force_squash: bool,
) -> None:
    """Auto-squash container images for Slurm execution.

    Checks if the executor is Slurm with SSH tunnel, and if so, squashes
    any Docker images to .sqsh files on the remote cluster. Modifies
    job_config in-place with the squashed paths.

    Args:
        job_config: Full job configuration (OmegaConf DictConfig) - modified in place
        mode: Execution mode ("local", "run", "batch")
        dry_run: Whether this is a dry-run (skip squash to avoid remote side effects)
        force_squash: Whether to force re-squash
    """
    from omegaconf import OmegaConf, open_dict

    from nemo_runspec.squash import ensure_squashed_image

    # Only for remote slurm execution
    if mode not in ("run", "batch"):
        return

    # Skip on dry-run to avoid remote side effects
    if dry_run:
        return

    # Get env config
    env_config = OmegaConf.to_container(job_config.run.env, resolve=True)

    # Only for Slurm executor
    if env_config.get("executor") != "slurm":
        return

    # Need SSH tunnel support
    if env_config.get("tunnel") != "ssh":
        return

    # Need SSH connection info
    host = env_config.get("host")
    remote_job_dir = env_config.get("remote_job_dir")

    if not all([host, remote_job_dir]):
        return

    # Check for nemo-run (optional dependency for SSH tunnel)
    try:
        import nemo_run as run
    except ImportError:
        console.print(
            "[yellow]Warning:[/yellow] nemo-run not installed, skipping auto-squash. "
            "Install with: pip install nemo-run"
        )
        return

    # Collect images to squash
    images = collect_evaluator_images(job_config)
    if not images:
        return

    # Filter out already-squashed images
    images_to_squash = [(dp, img) for dp, img in images if not img.endswith(".sqsh")]
    if not images_to_squash:
        return

    # Create SSH tunnel
    user = env_config.get("user") or ""
    tunnel = run.SSHTunnel(
        host=host,
        user=user,
        job_dir=remote_job_dir,
    )

    try:
        tunnel.connect()

        # Squash each image and update config
        for dotpath, image in images_to_squash:
            console.print(f"[blue]Auto-squashing:[/blue] {image}")
            sqsh_path = ensure_squashed_image(
                tunnel=tunnel,
                container_image=image,
                remote_job_dir=remote_job_dir,
                env_config=env_config,
                force=force_squash,
            )

            # Update config with squashed path
            with open_dict(job_config):
                OmegaConf.update(job_config, dotpath, sqsh_path, merge=False)

    finally:
        # Cleanup tunnel if it has a disconnect method
        if hasattr(tunnel, "disconnect"):
            try:
                tunnel.disconnect()
            except Exception:
                pass


# =============================================================================
# Eval Config Save
# =============================================================================


def save_eval_configs(
    job_config: Any,
    recipe_name: str,
    *,
    for_remote: bool = False,
) -> tuple[Path, Path]:
    """Save job and eval configs to disk.

    The eval config has the 'run' section stripped and ${run.*} interpolations
    resolved. This is what gets passed to nemo-evaluator-launcher.

    Args:
        job_config: Full job configuration (OmegaConf DictConfig)
        recipe_name: Recipe name for job directory
        for_remote: If True, rewrite paths for remote execution

    Returns:
        Tuple of (job_yaml_path, eval_yaml_path)
    """
    from omegaconf import OmegaConf

    from nemo_runspec.config import generate_job_dir
    from nemo_runspec.utils import rewrite_paths_for_remote, resolve_run_interpolations

    job_dir = generate_job_dir(recipe_name)

    # Extract eval config (everything except 'run' section, with ${run.*} resolved)
    config_dict = OmegaConf.to_container(job_config, resolve=False)
    run_section = config_dict.pop("run", {})

    # Rewrite paths for remote execution if needed
    if for_remote:
        repo_root = Path.cwd()
        config_dict = rewrite_paths_for_remote(config_dict, repo_root)

    # Resolve ${run.*} interpolations (${run.env.host}, ${run.wandb.entity}, etc.)
    config_dict = resolve_run_interpolations(config_dict, run_section)

    eval_config = OmegaConf.create(config_dict)

    # Save configs
    job_dir.mkdir(parents=True, exist_ok=True)

    job_path = job_dir / "job.yaml"
    eval_path = job_dir / "eval.yaml"

    OmegaConf.save(job_config, job_path)
    OmegaConf.save(eval_config, eval_path)

    return job_path, eval_path
