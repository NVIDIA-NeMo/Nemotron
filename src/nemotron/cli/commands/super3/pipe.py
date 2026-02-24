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
    name="super3/pipe",
    script_path="",
    config_dir="",
    default_config="",
    input_artifacts={"data": "Pretrain data artifact (bin/idx blends)"},
    output_artifacts={"model": "Fine-tuned model checkpoint (after SFT)"},
)


def _execute_pipe(cfg):
    """Compose pretrain -> sft into a single nemo-run Experiment."""
    if cfg.mode == "local":
        typer.echo("Error: pipe requires --run or --batch (remote execution)", err=True)
        raise typer.Exit(1)

    try:
        import nemo_run as run
    except ImportError:
        typer.echo("Error: nemo-run is required for pipe execution", err=True)
        typer.echo("Install with: pip install nemo-run", err=True)
        raise typer.Exit(1)

    from nemotron.cli.commands.super3.pretrain import _execute_pretrain
    from nemotron.cli.commands.super3.sft import _execute_sft

    with run.Experiment("super3-pipe") as exp:
        _execute_pretrain(cfg, experiment=exp)
        _execute_sft(cfg, experiment=exp)
        exp.run(detach=not cfg.attached)


def pipe(ctx: typer.Context) -> None:
    """Run full training pipeline: pretrain -> sft.

    Composes pretrain and SFT stages into a single nemo-run Experiment
    for coordinated remote execution. Each stage uses its own default
    config. RL must be run separately (uses Ray).

    Requires --run or --batch.
    """
    cfg = parse_recipe_config(ctx)
    _execute_pipe(cfg)
