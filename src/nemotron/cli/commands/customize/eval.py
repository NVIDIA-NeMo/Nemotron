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

"""Eval command implementation for customization recipes (stage4).

Supports two evaluation modes:
- ``--mode model`` (default): Model benchmark evaluation using nemo-evaluator-launcher.
  Follows the SAME execution pattern as nano3/super3 eval -- calls run_eval() directly,
  no recipe script, no nemo-run submission.
- ``--mode data``: Data quality assessment using NeMo Curator filters/scorers.
  Runs locally via the AssessmentTool pipeline.

Design: LLM-Native Recipe Architecture
- Execution logic visible and modifiable
- Fork this file to change how evaluations are submitted
"""

from __future__ import annotations

import typer
from rich.console import Console

from nemo_runspec.config import (
    build_job_config,
    clear_artifact_cache,
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
# Recipe Metadata (no SPEC -- evaluator has no recipe script)
# =============================================================================

CONFIG_DIR = "src/nemotron/customization_recipes/nemotron/stage4_eval/config"

META = RecipeMeta(
    name="customize/eval",
    script_path="",  # No recipe script -- evaluator calls run_eval() directly
    config_dir=CONFIG_DIR,
    default_config="default",
    input_artifacts={"model": "Trained model checkpoint to evaluate"},
    output_artifacts={},
)


# =============================================================================
# Model Evaluation (nemo-evaluator-launcher)
# =============================================================================


def _execute_model_eval(cfg: RecipeConfig):
    """Execute model benchmark evaluation with nemo-evaluator-launcher.

    This follows the EXACT same pattern as nano3/super3 eval:
    parse_config -> build_job_config -> resolve artifacts -> save configs -> run_eval()

    Args:
        cfg: Parsed recipe configuration
    """
    from pathlib import Path

    from omegaconf import OmegaConf

    # --stage is not supported for evaluator
    if cfg.stage:
        typer.echo("Error: --stage is not supported for evaluator commands", err=True)
        raise typer.Exit(1)

    # =========================================================================
    # 1. Parse configuration
    # =========================================================================
    config_dir = Path(CONFIG_DIR)
    train_config = parse_config(cfg.ctx, config_dir, "default")
    env = parse_env(cfg.ctx)

    # Build full job config with provenance
    job_config = build_job_config(
        train_config,
        cfg.ctx,
        "customize/eval",
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
        job_config, "customize/eval", for_remote=for_remote
    )

    # Display job submission summary
    display_job_submission(job_path, eval_path, {}, cfg.mode, artifacts=job_config.get("artifacts"))

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
    console.print("\n[bold blue]Starting model evaluation...[/bold blue]")
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
# Data Quality Evaluation (NeMo Curator)
# =============================================================================


def _execute_data_eval(cfg: RecipeConfig):
    """Execute data quality evaluation using NeMo Curator filters/scorers.

    This mode runs the AssessmentTool pipeline from data_prep/quality.py
    to score training data quality using configurable filters and scorers.

    Args:
        cfg: Parsed recipe configuration
    """
    from pathlib import Path

    from omegaconf import OmegaConf

    from nemotron.customization_recipes.data_prep import evaluate_data_quality

    # =========================================================================
    # 1. Parse configuration
    # =========================================================================
    config_dir = Path(CONFIG_DIR)
    train_config = parse_config(cfg.ctx, config_dir, "default")
    env = parse_env(cfg.ctx)

    job_config = build_job_config(
        train_config,
        cfg.ctx,
        "customize/eval-data",
        "",
        cfg.argv,
        env_profile=env,
    )

    display_job_config(job_config)

    if cfg.dry_run:
        return

    # =========================================================================
    # 2. Extract data_eval section and run quality assessment
    # =========================================================================
    resolved = OmegaConf.to_container(job_config, resolve=True)
    data_eval_cfg = resolved.get("data_eval", {})

    if not data_eval_cfg:
        typer.echo(
            "Error: No 'data_eval' section found in config. "
            "Add a data_eval section or use --mode model for benchmark evaluation.",
            err=True,
        )
        raise typer.Exit(1)

    console.print("\n[bold blue]Starting data quality evaluation...[/bold blue]")
    result = evaluate_data_quality(OmegaConf.create(data_eval_cfg))

    console.print(f"\n[green]\u2713[/green] Data evaluation complete")
    if result.get("details"):
        console.print(f"  Details:    [cyan]{result['details']}[/cyan]")
    if result.get("aggregates"):
        console.print(f"  Aggregates: [cyan]{result['aggregates']}[/cyan]")


# =============================================================================
# CLI Entry Point
# =============================================================================


def eval(ctx: typer.Context) -> None:
    """Run evaluation (stage4).

    By default runs model benchmark evaluation using nemo-evaluator-launcher
    (same infrastructure as nano3/super3 eval).

    Pass ``--mode data`` to run data quality evaluation using NeMo Curator
    filters and scorers.

    Examples:
        # Model eval on cluster (loads env.toml profile)
        nemotron customize eval --run MY-CLUSTER

        # Override model artifact
        nemotron customize eval --run MY-CLUSTER run.model=sft:v2

        # Filter specific benchmark tasks
        nemotron customize eval --run MY-CLUSTER -t adlr_mmlu -t hellaswag

        # Dry run (show resolved config without executing)
        nemotron customize eval --run MY-CLUSTER --dry-run

        # Local execution
        nemotron customize eval execution.type=local

        # Data quality evaluation mode
        nemotron customize eval --mode data data_eval.input_file=/data/train.jsonl
    """
    # Check for --mode flag in passthrough args
    data_mode = False
    args = list(ctx.args) if ctx.args else []
    if "--mode" in args:
        idx = args.index("--mode")
        if idx + 1 < len(args):
            mode_val = args[idx + 1].lower()
            if mode_val == "data":
                data_mode = True
            elif mode_val != "model":
                typer.echo(
                    f"Error: Unknown eval mode '{mode_val}'. Use 'model' (default) or 'data'.",
                    err=True,
                )
                raise typer.Exit(1)
            args.pop(idx + 1)
            args.pop(idx)
            ctx.args = args

    cfg = parse_recipe_config(ctx)

    if data_mode:
        _execute_data_eval(cfg)
    else:
        _execute_model_eval(cfg)
