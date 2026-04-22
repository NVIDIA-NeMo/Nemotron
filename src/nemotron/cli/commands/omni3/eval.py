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

"""Eval command implementation for omni3 recipe (stage2)."""

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

CONFIG_DIR = "src/nemotron/recipes/omni3/stage2_eval/config"

META = RecipeMeta(
    name="omni3/eval",
    script_path="",
    config_dir=CONFIG_DIR,
    default_config="default",
    input_artifacts={"model": "Model checkpoint to evaluate"},
    output_artifacts={},
)


def _execute_eval(cfg: RecipeConfig):
    """Execute evaluation with nemo-evaluator-launcher."""
    from pathlib import Path

    from omegaconf import OmegaConf

    if cfg.stage:
        typer.echo("Error: --stage is not supported for evaluator commands", err=True)
        raise typer.Exit(1)

    config_dir = Path(CONFIG_DIR)
    train_config = parse_config(cfg.ctx, config_dir, "default")
    env = parse_env(cfg.ctx)

    job_config = build_job_config(
        train_config,
        cfg.ctx,
        "omni3/eval",
        "",
        cfg.argv,
        env_profile=env,
    )

    if needs_wandb(job_config):
        inject_wandb_env_mappings(job_config)

    maybe_auto_squash_evaluator(
        job_config,
        mode=cfg.mode,
        dry_run=cfg.dry_run,
        force_squash=cfg.force_squash,
    )

    for_remote = cfg.mode in ("run", "batch")
    display_job_config(job_config, for_remote=for_remote)

    if cfg.dry_run:
        return

    job_path, eval_path = save_eval_configs(
        job_config, "omni3/eval", for_remote=for_remote
    )

    display_job_submission(
        job_path, eval_path, {}, cfg.mode, artifacts=job_config.get("artifacts")
    )

    ensure_wandb_host_env()

    clear_artifact_cache()
    register_resolvers_from_config(
        job_config,
        artifacts_key="run",
        mode="pre_init",
    )

    resolved_config = OmegaConf.to_container(job_config, resolve=True)
    eval_config = {k: v for k, v in resolved_config.items() if k != "run"}
    eval_config = OmegaConf.create(eval_config)

    task_list = parse_task_flags(cfg.passthrough)
    extra_args = get_non_task_args(cfg.passthrough)
    if extra_args:
        typer.echo(
            f"Error: Unknown arguments: {' '.join(extra_args)}\n"
            "Only -t/--task flags are supported for passthrough.",
            err=True,
        )
        raise typer.Exit(1)

    try:
        from nemo_evaluator_launcher.api.functional import run_eval
    except ImportError:
        typer.echo(
            "Error: nemo-evaluator-launcher is required for evaluation", err=True
        )
        typer.echo('Install with: pip install "nemotron[evaluator]"', err=True)
        raise typer.Exit(1)

    if needs_wandb(eval_config):
        inject_wandb_env_mappings(eval_config)

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


def eval(ctx: typer.Context) -> None:
    """Run evaluation with NeMo-Evaluator (stage2)."""
    cfg = parse_recipe_config(ctx)
    _execute_eval(cfg)
