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

"""Pipeline command: compose pretrain + sft into a single nemo-run Experiment.

Requires --run or --batch (remote execution only). Each stage resolves
its own config from its own [tool.runspec] metadata, so the pipe command
only needs pipeline-level options (profile, mode, dry-run).

RL uses Ray and cannot compose with nemo-run Experiments -- run it separately.
"""

from __future__ import annotations

import typer

from nemo_runspec.recipe_config import parse_recipe_config
from nemo_runspec.recipe_typer import RecipeMeta

META = RecipeMeta(
    name="lightning35/pipe",
    script_path="",
    config_dir="",
    default_config="",
    input_artifacts={"data": "Pretrain data artifact (bin/idx blends)"},
    output_artifacts={"model": "Fine-tuned model checkpoint (after SFT)"},
)


def _execute_pipe(cfg):
    """Compose pretrain → sft into a single nemo-run Experiment."""
    if cfg.mode == "local":
        typer.echo("Error: pipe requires --run or --batch (remote execution)", err=True)
        raise typer.Exit(1)

    try:
        import nemo_run as run
    except ImportError:
        typer.echo("Error: nemo-run is required for pipe execution", err=True)
        typer.echo("Install with: pip install nemo-run", err=True)
        raise typer.Exit(1)

    from nemo_runspec.env import parse_env
    from nemo_runspec.execution import get_executor_type
    from nemotron.cli.commands.lightning35.pretrain import _execute_pretrain
    from nemotron.cli.commands.lightning35.sft import _execute_sft

    # Dry runs display each stage's compiled config and stop — don't create
    # (and run) an empty experiment afterwards.
    if cfg.dry_run:
        _execute_pretrain(cfg)
        _execute_sft(cfg)
        return

    # Cloud executors submit immediately inside the stage helpers instead of
    # composing into the experiment, so sequencing cannot be guaranteed there.
    executor_type = get_executor_type(parse_env(cfg.ctx))
    if executor_type in ("dgxcloud", "lepton"):
        typer.echo(
            f"Error: pipe composes stages via a Slurm nemo-run Experiment and does "
            f"not support the '{executor_type}' executor. Run the stages "
            "individually: `nemotron lightning35 pretrain` then `... sft`.",
            err=True,
        )
        raise typer.Exit(1)

    with run.Experiment("lightning35-pipe") as exp:
        _execute_pretrain(cfg, experiment=exp)
        _execute_sft(cfg, experiment=exp)
        # sequential=True is essential: nemo-run defaults to launching all
        # experiment tasks in PARALLEL, which would let SFT start before
        # pretrain finishes and resolve a stale (or missing) model artifact.
        # With Slurm this schedules both jobs at once chained via afterok.
        exp.run(sequential=True, detach=not cfg.attached)


def pipe(ctx: typer.Context) -> None:
    """Run full training pipeline: pretrain → sft.

    Composes pretrain and SFT stages into a single nemo-run Experiment
    for coordinated remote execution. Each stage uses its own default
    config. RL must be run separately (uses Ray).

    Requires --run or --batch.
    """
    cfg = parse_recipe_config(ctx)
    _execute_pipe(cfg)
