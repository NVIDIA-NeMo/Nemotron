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

"""@evaluator decorator for evaluation commands.

Reuses ConfigBuilder from recipe infrastructure for consistent config handling,
but executes via nemo-evaluator-launcher instead of nemo-run.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from nemotron.kit.cli.config import ConfigBuilder, generate_job_dir
from nemotron.kit.cli.display import display_job_config, display_job_submission
from nemotron.kit.cli.env import get_wandb_config
from nemotron.kit.cli.globals import GlobalContext, split_unknown_args
from nemotron.kit.cli.utils import resolve_run_interpolations

console = Console()


@dataclass
class EvaluatorMetadata:
    """Metadata attached to an evaluator command function.

    Attributes:
        name: Evaluator identifier (e.g., "nano3/eval")
        config_dir: Path to config directory relative to repo root
        default_config: Default config name (default: "default")
        require_explicit_config: If True, requires -c/--config to be provided
    """

    name: str
    config_dir: str
    default_config: str = "default"
    require_explicit_config: bool = False


def evaluator(
    name: str,
    config_dir: str,
    default_config: str = "default",
    *,
    require_explicit_config: bool = False,
) -> Callable:
    """Decorator marking a function as an evaluator command.

    Similar to @recipe but executes via nemo-evaluator-launcher.
    Supports --run/--batch for cluster execution, local execution when no profile.

    Args:
        name: Evaluator identifier (e.g., "nano3/eval")
        config_dir: Path to config directory
                   (e.g., "src/nemotron/recipes/nano3/stage3_eval/config")
        default_config: Default config name (stem) or path used when -c/--config
            is not provided (default: "default").
        require_explicit_config: If True, requires -c/--config to be provided.
            Used for top-level `nemotron evaluate` command.

    Example:
        @evaluator(
            name="nano3/eval",
            config_dir="src/nemotron/recipes/nano3/stage3_eval/config",
        )
        def eval(ctx: typer.Context):
            '''Run evaluation with NeMo-Evaluator (stage3).'''
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(ctx: typer.Context) -> None:
            # Get global context
            global_ctx: GlobalContext = ctx.obj
            if global_ctx is None:
                global_ctx = GlobalContext()

            # Split unknown args into dotlist and passthrough
            # Also extract any global options that appear after the subcommand
            dotlist, passthrough, global_ctx = split_unknown_args(ctx.args or [], global_ctx)
            global_ctx.dotlist = dotlist
            global_ctx.passthrough = passthrough

            # Validate options after split_unknown_args has extracted all global options
            if global_ctx.run and global_ctx.batch:
                typer.echo("Error: --run and --batch cannot both be set", err=True)
                raise typer.Exit(1)

            # --stage is not supported for evaluator
            if global_ctx.stage:
                typer.echo("Error: --stage is not supported for evaluator commands", err=True)
                raise typer.Exit(1)

            # Check if explicit config is required
            if require_explicit_config and not global_ctx.config:
                typer.echo(
                    "Error: -c/--config is required for this command.\n"
                    "Example: nemotron evaluate -c /path/to/eval.yaml --run CLUSTER",
                    err=True,
                )
                raise typer.Exit(1)

            # Build configuration (reuses ConfigBuilder)
            builder = ConfigBuilder(
                recipe_name=name,
                script_path="",  # Not used for evaluator
                config_dir=config_dir,
                default_config=default_config,
                ctx=global_ctx,
                argv=sys.argv,
            )

            # Load and merge config
            builder.load_and_merge()

            # Build full job config
            builder.build_job_config()

            # Auto-inject W&B env mappings if W&B export is configured
            # This mirrors nemo-run's behavior of auto-passing WANDB_API_KEY
            if _needs_wandb(builder.job_config):
                _inject_wandb_env_mappings(builder.job_config)

            # Auto-squash container images for Slurm execution
            # This mirrors nemo-run's behavior of auto-squashing Docker images
            _maybe_auto_squash_evaluator(builder.job_config, global_ctx)

            # Display compiled configuration
            # Show resolved paths for remote execution (--run/--batch)
            for_remote = global_ctx.mode in ("run", "batch")
            display_job_config(builder.job_config, for_remote=for_remote)

            # Handle dry-run mode
            if global_ctx.dry_run:
                return

            # Save configs (job.yaml for provenance, eval.yaml for launcher)
            job_path, eval_path = _save_eval_configs(builder, for_remote=for_remote)

            # Display job submission summary
            display_job_submission(job_path, eval_path, {}, global_ctx.mode)

            # Execute via evaluator launcher
            _execute_evaluator(
                job_config=builder.job_config,
                passthrough=passthrough,
            )

        # Attach metadata to function for introspection
        wrapper._evaluator_metadata = EvaluatorMetadata(
            name=name,
            config_dir=config_dir,
            default_config=default_config,
            require_explicit_config=require_explicit_config,
        )

        return wrapper

    return decorator


def _save_eval_configs(
    builder: ConfigBuilder,
    *,
    for_remote: bool = False,
) -> tuple[Path, Path]:
    """Save job and eval configs to disk.

    Args:
        builder: ConfigBuilder with loaded configuration
        for_remote: If True, rewrite paths for remote execution

    Returns:
        Tuple of (job_yaml_path, eval_yaml_path)
    """
    from omegaconf import OmegaConf

    from nemotron.kit.cli.utils import rewrite_paths_for_remote

    job_config = builder.job_config
    job_dir = generate_job_dir(builder.recipe_name)

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


def _execute_evaluator(
    job_config: Any,
    passthrough: list[str],
) -> None:
    """Execute evaluation via nemo-evaluator-launcher.

    1. Ensure W&B env vars are set (needed for artifact resolution)
    2. Resolve artifacts (${art:model,path})
    3. Extract evaluator config (everything except 'run' section)
    4. Call run_eval() with fully resolved config

    Args:
        job_config: Full job configuration
        passthrough: Passthrough arguments (for -t/--task flags)
    """
    from omegaconf import OmegaConf

    from nemotron.kit.resolvers import (
        clear_artifact_cache,
        register_resolvers_from_config,
    )

    # Ensure W&B host env vars BEFORE artifact resolution
    # The resolver uses WANDB_ENTITY/WANDB_PROJECT from environment to locate artifacts
    # This loads entity/project from env.toml [wandb] section if not already set
    _ensure_wandb_host_env()

    # Resolve artifacts (${art:model,path} etc.)
    clear_artifact_cache()
    register_resolvers_from_config(
        job_config,
        artifacts_key="run",
        mode="pre_init",
    )

    # Resolve all interpolations
    # This resolves: ${run.env.host}, ${run.wandb.entity}, ${art:model,path}, etc.
    resolved_config = OmegaConf.to_container(job_config, resolve=True)

    # Extract evaluator-specific config (everything except 'run' section)
    # The 'run' section was only needed for interpolation, not for the launcher
    eval_config = {k: v for k, v in resolved_config.items() if k != "run"}
    eval_config = OmegaConf.create(eval_config)

    # Parse -t/--task flags from passthrough
    task_list = _parse_task_flags(passthrough)

    # Validate that no extra passthrough args exist (only -t/--task allowed)
    extra_args = _get_non_task_args(passthrough)
    if extra_args:
        typer.echo(
            f"Error: Unknown arguments: {' '.join(extra_args)}\n"
            "Only -t/--task flags are supported for passthrough.",
            err=True,
        )
        raise typer.Exit(1)

    # Import and call evaluator launcher
    try:
        from nemo_evaluator_launcher.api.functional import run_eval
    except ImportError:
        typer.echo("Error: nemo-evaluator-launcher is required for evaluation", err=True)
        typer.echo('Install with: pip install "nemotron[evaluator]"', err=True)
        raise typer.Exit(1)

    # Inject W&B env var mappings into eval_config if needed
    # (env vars were already set earlier for artifact resolution)
    if _needs_wandb(eval_config):
        _inject_wandb_env_mappings(eval_config)

    # Call the launcher
    console.print("\n[bold blue]Starting evaluation...[/bold blue]")
    invocation_id = run_eval(eval_config, dry_run=False, tasks=task_list)

    if invocation_id:
        console.print(f"\n[green]✓[/green] Evaluation submitted: [cyan]{invocation_id}[/cyan]")
        console.print(
            f"[dim]Check status: nemo-evaluator-launcher status {invocation_id}[/dim]"
        )
        console.print(f"[dim]Stream logs: nemo-evaluator-launcher logs {invocation_id}[/dim]")


def _parse_task_flags(passthrough: list[str]) -> list[str] | None:
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


def _get_non_task_args(passthrough: list[str]) -> list[str]:
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


def _needs_wandb(cfg: Any) -> bool:
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


def _ensure_wandb_host_env() -> None:
    """Ensure W&B environment variables are set on the host.

    Auto-detects WANDB_API_KEY from local wandb login (same as nemo-run).
    Also sets WANDB_PROJECT/WANDB_ENTITY from env.toml [wandb] section.

    This is required because nemo-evaluator-launcher checks os.getenv()
    for env_vars mappings at submission time.
    """
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


def _inject_wandb_env_mappings(cfg: Any) -> None:
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


def _collect_evaluator_images(cfg: Any) -> list[tuple[str, str]]:
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


def _maybe_auto_squash_evaluator(
    job_config: Any,
    global_ctx: GlobalContext,
) -> None:
    """Auto-squash container images for Slurm execution.

    Checks if the executor is Slurm with SSH tunnel, and if so, squashes
    any Docker images to .sqsh files on the remote cluster. Modifies
    job_config in-place with the squashed paths.

    Args:
        job_config: Full job configuration (OmegaConf DictConfig) - modified in place
        global_ctx: Global CLI context with mode and force_squash flag
    """
    from omegaconf import OmegaConf, open_dict

    from nemotron.kit.cli.squash import ensure_squashed_image, is_sqsh_image

    # Only for remote slurm execution
    if global_ctx.mode not in ("run", "batch"):
        return

    # Skip on dry-run to avoid remote side effects
    if global_ctx.dry_run:
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
    user = env_config.get("user")
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
    images = _collect_evaluator_images(job_config)
    if not images:
        return

    # Filter out already-squashed images
    images_to_squash = [(dp, img) for dp, img in images if not is_sqsh_image(img)]
    if not images_to_squash:
        return

    # Create SSH tunnel
    tunnel = run.SSHTunnel(
        host=host,
        user=user or "",
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
                force=global_ctx.force_squash,
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
