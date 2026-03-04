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

"""Top-level evaluate command.

Provides a generic `nemotron evaluate` command with pre-built configs for
common evaluation scenarios. Unlike recipe-specific commands (nano3/eval),
this command has no default config and requires explicit config selection.

Design: LLM-Native Recipe Architecture
- Execution logic visible and modifiable
- Same evaluator pipeline as nano3/eval, but without recipe-specific defaults
"""

from __future__ import annotations

import typer
from rich.console import Console

from nemo_runspec.config import (
    build_job_config,
    clear_artifact_cache,
    generate_job_dir,
    parse_config,
    register_resolvers_from_config,
)
from nemo_runspec.display import display_job_config, display_job_submission
from nemo_runspec.env import parse_env
from nemo_runspec.evaluator import (
    ensure_wandb_host_env,
    get_non_task_args,
    inject_wandb_env_mappings,
    maybe_auto_squash_evaluator,
    needs_wandb,
    parse_task_flags,
    save_eval_configs,
)
from nemo_runspec.recipe_config import RecipeConfig, parse_recipe_config
from nemo_runspec.recipe_typer import RecipeMeta

console = Console()

# =============================================================================
# Recipe Metadata
# =============================================================================

EVAL_CONFIG_DIR = "src/nemotron/recipes/evaluator/config"

META = RecipeMeta(
    name="evaluate",
    script_path="",  # No recipe script
    config_dir=EVAL_CONFIG_DIR,
    default_config="default",
    input_artifacts={},
    output_artifacts={},
)


# =============================================================================
# Execution Logic
# =============================================================================


def _execute_evaluate(cfg: RecipeConfig):
    """Execute evaluation with nemo-evaluator-launcher.

    Same pipeline as nano3/eval but without recipe-specific defaults.

    Args:
        cfg: Parsed recipe configuration
    """
    from pathlib import Path

    from omegaconf import OmegaConf

    # --stage is not supported for evaluator
    if cfg.stage:
        typer.echo("Error: --stage is not supported for evaluator commands", err=True)
        raise typer.Exit(1)

    # Require explicit config
    if not cfg.config:
        typer.echo(
            "Error: -c/--config is required for this command.\n"
            "Example: nemotron evaluate -c /path/to/eval.yaml --run CLUSTER",
            err=True,
        )
        raise typer.Exit(1)

    # =========================================================================
    # 1. Parse configuration
    # =========================================================================
    config_dir = Path(EVAL_CONFIG_DIR)
    train_config = parse_config(cfg.ctx, config_dir, "default")
    env = parse_env(cfg.ctx)

    # Build full job config with provenance
    job_config = build_job_config(
        train_config,
        cfg.ctx,
        "evaluate",
        "",  # No script path
        cfg.argv,
        env_profile=env,
    )

    # =========================================================================
    # 2. Auto-inject W&B env mappings if W&B export is configured
    # =========================================================================
    if needs_wandb(job_config):
        inject_wandb_env_mappings(job_config)

    # =========================================================================
    # 3. Auto-squash container images for Slurm execution
    # =========================================================================
    maybe_auto_squash_evaluator(
        job_config,
        mode=cfg.mode,
        dry_run=cfg.dry_run,
        force_squash=cfg.force_squash,
    )

    # =========================================================================
    # 4. Display compiled configuration
    # =========================================================================
    for_remote = cfg.mode in ("run", "batch")
    display_job_config(job_config, for_remote=for_remote)

    # Handle dry-run mode
    if cfg.dry_run:
        return

    # =========================================================================
    # 5. Save configs (job.yaml for provenance, eval.yaml for launcher)
    # =========================================================================
    job_path, eval_path = save_eval_configs(
        job_config, "evaluate", for_remote=for_remote
    )

    # Display job submission summary
    display_job_submission(job_path, eval_path, {}, cfg.mode)

    # =========================================================================
    # 6. Execute via evaluator launcher
    # =========================================================================

    # Ensure W&B host env vars BEFORE artifact resolution
    ensure_wandb_host_env()

    # Resolve artifacts (${art:model,path} etc.)
    clear_artifact_cache()
    register_resolvers_from_config(
        job_config,
        artifacts_key="run",
        mode="pre_init",
    )

    # Resolve all interpolations
    resolved_config = OmegaConf.to_container(job_config, resolve=True)

    # Extract evaluator-specific config (everything except 'run' section)
    eval_config = {k: v for k, v in resolved_config.items() if k != "run"}
    eval_config = OmegaConf.create(eval_config)

    # Parse -t/--task flags from passthrough
    task_list = parse_task_flags(cfg.passthrough)

    # Validate that no extra passthrough args exist (only -t/--task allowed)
    extra_args = get_non_task_args(cfg.passthrough)
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
        typer.echo(
            "Error: nemo-evaluator-launcher is required for evaluation", err=True
        )
        typer.echo('Install with: pip install "nemotron[evaluator]"', err=True)
        raise typer.Exit(1)

    # Inject W&B env var mappings into eval_config if needed
    if needs_wandb(eval_config):
        inject_wandb_env_mappings(eval_config)

    # Call the launcher
    console.print("\n[bold blue]Starting evaluation...[/bold blue]")
    invocation_id = run_eval(eval_config, dry_run=False, tasks=task_list)

    if invocation_id:
        console.print(
            f"\n[green]\u2713[/green] Evaluation submitted: [cyan]{invocation_id}[/cyan]"
        )
        console.print(
            f"[dim]Check status: nemo-evaluator-launcher status {invocation_id}[/dim]"
        )
        console.print(
            f"[dim]Stream logs: nemo-evaluator-launcher logs {invocation_id}[/dim]"
        )


# =============================================================================
# CLI Entry Point
# =============================================================================


def evaluate(ctx: typer.Context) -> None:
    """Run model evaluation with nemo-evaluator.

    Generic evaluation command with pre-built configs for common models.
    For recipe-specific evaluation with artifact resolution, use `nemotron nano3 eval`.

    Available configs:
        nemotron-3-nano-nemo-ray  NeMo Framework Ray deployment for Nemotron-3-Nano

    Examples:
        # Evaluate Nemotron-3-Nano with NeMo Ray deployment
        nemotron evaluate -c nemotron-3-nano-nemo-ray --run MY-CLUSTER

        # Override checkpoint path
        nemotron evaluate -c nemotron-3-nano-nemo-ray --run MY-CLUSTER \\
            deployment.checkpoint_path=/path/to/checkpoint

        # Filter specific tasks
        nemotron evaluate -c nemotron-3-nano-nemo-ray --run MY-CLUSTER -t adlr_mmlu

        # Dry run (preview config)
        nemotron evaluate -c nemotron-3-nano-nemo-ray --run MY-CLUSTER --dry-run

        # Use custom config file
        nemotron evaluate -c /path/to/custom.yaml --run MY-CLUSTER
    """
    cfg = parse_recipe_config(ctx)
    _execute_evaluate(cfg)
