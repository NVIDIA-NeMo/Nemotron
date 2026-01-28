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

"""Pipeline orchestration for the embed recipe."""

from __future__ import annotations

import shutil
import subprocess

import typer

from nemotron.kit.cli.recipe_config import RecipeConfig, parse_recipe_config

# Stage order defines the DAG
STAGE_ORDER = ["sdg", "prep", "finetune", "eval", "export", "deploy"]
DEFAULT_TO = "eval"  # export/deploy are opt-in

# Map stage names to their modules (lazy import to avoid circular deps)
STAGE_MODULES = {
    "sdg": "nemotron.cli.embed.sdg",
    "prep": "nemotron.cli.embed.prep",
    "finetune": "nemotron.cli.embed.finetune",
    "eval": "nemotron.cli.embed.eval",
    "export": "nemotron.cli.embed.export",
    "deploy": "nemotron.cli.embed.deploy",
}

# Map stage names to their _*_nemo_run function names
STAGE_RUN_FNS = {
    "sdg": "_sdg_nemo_run",
    "prep": "_prep_nemo_run",
    "finetune": "_finetune_nemo_run",
    "eval": "_eval_nemo_run",
    "export": "_export_nemo_run",
    # deploy has no _nemo_run function
}


def _resolve_stages(from_stage: str, to_stage: str) -> list[str]:
    """Return the ordered list of stages to run."""
    if from_stage not in STAGE_ORDER:
        raise typer.BadParameter(f"Unknown stage: {from_stage}")
    if to_stage not in STAGE_ORDER:
        raise typer.BadParameter(f"Unknown stage: {to_stage}")
    start = STAGE_ORDER.index(from_stage)
    end = STAGE_ORDER.index(to_stage)
    if start > end:
        raise typer.BadParameter(f"--from {from_stage} is after --to {to_stage}")
    return STAGE_ORDER[start : end + 1]


def _split_stage_overrides(dotlist: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Split dotlist into per-stage overrides and global overrides.

    'sdg.num_pairs=20' -> stage='sdg', override='num_pairs=20'
    'num_epochs=5'      -> global override (applied to all stages)
    """
    stage_overrides: dict[str, list[str]] = {}
    global_overrides: list[str] = []

    for item in dotlist:
        key = item.split("=", 1)[0]
        parts = key.split(".", 1)
        if len(parts) == 2 and parts[0] in STAGE_ORDER:
            stage_name = parts[0]
            rest = parts[1] + "=" + item.split("=", 1)[1] if "=" in item else parts[1]
            stage_overrides.setdefault(stage_name, []).append(rest)
        else:
            global_overrides.append(item)

    return stage_overrides, global_overrides


def _run_pipeline_local(
    stages: list[str],
    base_options: RecipeConfig,
    stage_overrides: dict[str, list[str]],
    global_overrides: list[str],
) -> None:
    """Run stages sequentially as local subprocesses."""
    for stage_name in stages:
        print(f"\n{'=' * 60}")
        print(f"  Stage: {stage_name}")
        print(f"{'=' * 60}\n")

        stage_dotlist = global_overrides + stage_overrides.get(stage_name, [])

        uv_cmd = shutil.which("uv")
        cmd = [uv_cmd or "uv", "run", "nemotron", "embed", stage_name]
        if base_options.config:
            cmd.extend(["-c", base_options.config])
        cmd.extend(stage_dotlist)
        cmd.extend(base_options.passthrough)

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nError: Stage '{stage_name}' failed with exit code {result.returncode}")
            raise typer.Exit(result.returncode)


def _run_pipeline_remote(
    stages: list[str],
    base_options: RecipeConfig,
    stage_overrides: dict[str, list[str]],
    global_overrides: list[str],
) -> None:
    """Run stages as tasks in a single nemo-run Experiment."""
    import nemo_run as run
    from importlib import import_module

    experiment_name = "embed-" + "-".join(stages)

    with run.Experiment(experiment_name) as exp:
        for stage_name in stages:
            if stage_name not in STAGE_RUN_FNS:
                print(f"Warning: Stage '{stage_name}' does not support remote execution, skipping")
                continue

            stage_dotlist = global_overrides + stage_overrides.get(stage_name, [])
            options = RecipeConfig(
                config=base_options.config,
                run=base_options.run,
                batch=base_options.batch,
                dry_run=base_options.dry_run,
                stage=base_options.stage,
                force_squash=base_options.force_squash,
                dotlist=stage_dotlist,
                passthrough=base_options.passthrough,
            )

            mod = import_module(STAGE_MODULES[stage_name])
            run_fn = getattr(mod, STAGE_RUN_FNS[stage_name])
            run_fn(options, experiment=exp)

        exp.run(detach=not base_options.attached, tail_logs=base_options.attached)


def run(
    ctx: typer.Context,
    from_stage: str = typer.Option("sdg", "--from", help="Stage to start from"),
    to_stage: str = typer.Option(DEFAULT_TO, "--to", help="Stage to stop at (inclusive)"),
) -> None:
    """Run the embed pipeline end-to-end.

    Executes stages sequentially from --from to --to (default: sdg through eval).
    Supports local, Docker, and Slurm execution via --run/--batch.

    Per-stage overrides use stage prefix: sdg.num_pairs=20, finetune.num_epochs=5
    """
    options = parse_recipe_config(ctx)
    stages = _resolve_stages(from_stage, to_stage)
    stage_overrides, global_overrides = _split_stage_overrides(options.dotlist)

    print(f"Embed pipeline: {' → '.join(stages)}")
    if options.config:
        print(f"Config: {options.config}")
    print()

    if options.dry_run:
        print("Dry run — would execute these stages:")
        for s in stages:
            overrides = global_overrides + stage_overrides.get(s, [])
            print(f"  {s}: {overrides or '(no overrides)'}")
        return

    if options.mode == "local":
        _run_pipeline_local(stages, options, stage_overrides, global_overrides)
    else:
        _run_pipeline_remote(stages, options, stage_overrides, global_overrides)
