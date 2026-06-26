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

"""SFT command implementation for omni3.

Mirrors ``commands/nano3/sft.py``: the dispatcher is a thin shim that parses
the recipe config + env profile, builds a nemo-run job, and submits it. The
training entry point lives in ``recipes/omni3/stage0_sft/train.py`` — it owns
the recipe / step-function / dataset-selector resolution by reading them out
of the YAML's ``recipe:`` block.
"""

from __future__ import annotations

from pathlib import Path

import typer

from nemo_runspec import parse as parse_runspec
from nemo_runspec.config import (
    build_job_config,
    extract_train_config,
    generate_job_dir,
    parse_config,
    save_configs,
)
from nemo_runspec.display import display_job_config, display_job_submission
from nemo_runspec.env import parse_env
from nemo_runspec.execution import (
    build_env_vars,
    create_executor,
    execute_local,
    get_startup_commands,
    prepend_startup_to_cmd,
)
from nemo_runspec.packaging import REMOTE_CONFIG, REMOTE_SCRIPT
from nemo_runspec.recipe_config import RecipeConfig, parse_recipe_config
from nemo_runspec.recipe_typer import RecipeMeta

SCRIPT_PATH = "src/nemotron/recipes/omni3/stage0_sft/train.py"
SPEC = parse_runspec(SCRIPT_PATH)

META = RecipeMeta(
    name=SPEC.name,
    script_path=SCRIPT_PATH,
    config_dir=str(SPEC.config_dir),
    default_config=SPEC.config.default,
    input_artifacts={
        "model": "Megatron-format base checkpoint path",
        "data": "Valor32k Energon dataset path",
    },
    output_artifacts={"model": "Fine-tuned Omni SFT checkpoint"},
)


def _execute_sft(cfg: RecipeConfig, *, experiment=None):
    """Execute omni3 SFT with nemo-run."""
    train_config = parse_config(cfg.ctx, SPEC.config_dir, SPEC.config.default)
    env = parse_env(cfg.ctx)

    # Allow config to override the train script (e.g. test.yaml → test_train.py),
    # matching the nano3/super3 training command pattern.
    script_path = SCRIPT_PATH
    if "run" in train_config and "train_script" in train_config.run:
        script_path = train_config.run.train_script

    # Parse runspec from the effective script for resource defaults.
    script_spec = parse_runspec(script_path) if script_path != SCRIPT_PATH else SPEC

    job_config = build_job_config(
        train_config,
        cfg.ctx,
        SPEC.name,
        script_path,
        cfg.argv,
        env_profile=env,
    )

    for_remote = cfg.mode in ("run", "batch")
    display_job_config(job_config, for_remote=for_remote)

    if cfg.dry_run:
        return

    job_dir = generate_job_dir(SPEC.name)
    train_config_for_script = extract_train_config(job_config, for_remote=for_remote)
    job_path, train_path = save_configs(job_config, train_config_for_script, job_dir)

    env_for_executor = job_config.run.env if hasattr(job_config.run, "env") else None
    env_vars = build_env_vars(job_config, env_for_executor)

    display_job_submission(job_path, train_path, env_vars, cfg.mode, artifacts=job_config.get("artifacts"))
    startup_commands = get_startup_commands(env_for_executor)

    if cfg.mode == "local":
        execute_local(
            script_path,
            train_path,
            cfg.passthrough,
            torchrun=(script_spec.run.launch == "torchrun"),
            env_vars=env_vars,
            startup_commands=startup_commands,
        )
    else:
        _execute_remote(
            script_path=script_path,
            script_resources=script_spec.resources,
            train_path=train_path,
            env=env_for_executor,
            passthrough=cfg.passthrough,
            attached=cfg.attached,
            env_vars=env_vars,
            startup_commands=startup_commands,
            force_squash=cfg.force_squash,
            experiment=experiment,
        )


def _execute_remote(
    *,
    script_path: str,
    script_resources=None,
    train_path: Path,
    env,
    passthrough: list[str],
    attached: bool,
    env_vars: dict[str, str],
    startup_commands: list[str] | None,
    force_squash: bool,
    experiment=None,
):
    """Execute via nemo-run with Slurm backend.

    Recipe / step-function / dataset selection live in the YAML's ``recipe:``
    block and are resolved inside ``train.py``. The ``launch = "torchrun"``
    declaration in the train script's PEP 723 frontmatter drives nemo-run to
    wrap the command with ``torchrun --nproc-per-node=N`` and populate the
    multi-node rendezvous flags from Slurm env vars automatically.
    """
    try:
        import nemo_run as run
    except ImportError:
        typer.echo("Error: nemo-run is required for --run/--batch execution", err=True)
        typer.echo("Install with: pip install nemo-run", err=True)
        raise typer.Exit(1)

    from nemo_runspec.packaging import SelfContainedPackager
    from nemo_runspec.run import (
        patch_nemo_run_ray_template_for_cpu,
        patch_nemo_run_rsync_accept_new_host_keys,
    )

    patch_nemo_run_rsync_accept_new_host_keys()
    patch_nemo_run_ray_template_for_cpu()

    packager = SelfContainedPackager(
        script_path=script_path,
        train_path=str(train_path),
    )

    executor = create_executor(
        env=env,
        env_vars=env_vars,
        packager=packager,
        attached=attached,
        force_squash=force_squash,
        default_image=SPEC.image,
        script_resources=script_resources,
    )

    recipe_name = SPEC.name.replace("/", "-")
    script_args = ["--config", REMOTE_CONFIG, *passthrough]

    if startup_commands:
        import shlex

        train_cmd = shlex.join(["python", REMOTE_SCRIPT, *script_args])
        full_cmd = prepend_startup_to_cmd(startup_commands, train_cmd)
        script_task = run.Script(path="bash", args=["-lc", full_cmd])
    else:
        script_task = run.Script(
            path=REMOTE_SCRIPT,
            args=script_args,
            entrypoint="python",
        )

    if experiment is not None:
        return experiment.add(script_task, executor=executor, name=recipe_name)

    with run.Experiment(recipe_name) as exp:
        exp.add(script_task, executor=executor, name=recipe_name)
        exp.run(detach=not attached)


def sft(ctx: typer.Context) -> None:
    """Run omni3 supervised fine-tuning."""
    cfg = parse_recipe_config(ctx)
    _execute_sft(cfg)
