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

"""Export command implementation.

Exports embedding models to ONNX and TensorRT for optimized inference.
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
from nemo_runspec.execution import build_env_vars
from nemo_runspec.recipe_config import RecipeConfig, parse_recipe_config
from nemo_runspec.recipe_typer import RecipeMeta
from nemotron.recipes.embed.stage4_export.export import ExportConfig

SCRIPT_PATH = "src/nemotron/recipes/embed/stage4_export/export.py"
SCRIPT_REMOTE = "src/nemotron/recipes/embed/stage4_export/run_uv.py"
SPEC = parse_runspec(SCRIPT_PATH)

META = RecipeMeta(
    name=SPEC.name,
    script_path=SCRIPT_PATH,
    config_dir=str(SPEC.config_dir),
    config_model=ExportConfig,
    default_config=SPEC.config.default,
    input_artifacts={"model": "Fine-tuned model checkpoint to export"},
    output_artifacts={"model": "Exported model (ONNX / TensorRT)"},
)


def _execute_export(cfg: RecipeConfig, *, experiment=None):
    """Execute export with visible execution logic."""
    train_config = parse_config(cfg.ctx, SPEC.config_dir, SPEC.config.default)
    env = parse_env(cfg.ctx)

    script_path = SCRIPT_PATH if cfg.mode == "local" else SCRIPT_REMOTE

    job_config = build_job_config(
        train_config,
        cfg.ctx,
        SPEC.name,
        script_path,
        cfg.argv,
        env_profile=env,
    )

    display_job_config(job_config, for_remote=False)

    if cfg.dry_run:
        return

    job_dir = generate_job_dir(SPEC.name)
    train_config_for_script = extract_train_config(job_config, for_remote=False)
    job_path, train_path = save_configs(job_config, train_config_for_script, job_dir)

    env_for_executor = job_config.run.env if hasattr(job_config.run, "env") else None
    env_vars = build_env_vars(job_config, env_for_executor)

    display_job_submission(job_path, train_path, env_vars, cfg.mode)

    if cfg.mode == "local":
        _execute_uv_local(train_path, cfg.passthrough, job_config)
    else:
        _execute_remote(
            train_path=train_path,
            env=env_for_executor,
            passthrough=cfg.passthrough,
            attached=cfg.attached,
            env_vars=env_vars,
            force_squash=cfg.force_squash,
            experiment=experiment,
        )


def _execute_uv_local(train_path: Path, passthrough: list[str], job_config) -> None:
    """Execute export locally via UV isolated environment.

    Conditionally includes TensorRT dependency based on config.
    """
    import os
    import shutil
    import subprocess

    uv_cmd = shutil.which("uv")
    if not uv_cmd:
        typer.echo("Error: 'uv' command not found. Please install uv.", err=True)
        raise typer.Exit(1)

    script_abs = SPEC.script_path
    stage_dir = script_abs.parent
    repo_root = SPEC.script_path.parents[len(Path(SCRIPT_PATH).parts) - 1]
    cmd = [
        uv_cmd, "run",
        "--with", str(repo_root),
        "--project", str(stage_dir),
    ]

    # Conditionally include TensorRT dependency
    export_to_trt = job_config.get("export_to_trt", False)
    if export_to_trt:
        cmd += ["--extra", "tensorrt"]

    cmd += [
        "python", str(script_abs),
        "--config", str(train_path),
        *passthrough,
    ]

    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)

    typer.echo(f"Executing with uv isolated environment: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    raise typer.Exit(result.returncode)


def _execute_remote(
    train_path: Path,
    env,
    passthrough: list[str],
    attached: bool,
    env_vars: dict[str, str],
    force_squash: bool,
    experiment=None,
):
    """Execute export via nemo-run with remote backend."""
    try:
        import nemo_run as run
    except ImportError:
        typer.echo("Error: nemo-run is required for --run/--batch execution", err=True)
        raise typer.Exit(1)

    from nemo_runspec.packaging import CodePackager
    from nemo_runspec.execution import create_executor
    from nemo_runspec.run import (
        patch_nemo_run_ray_template_for_cpu,
        patch_nemo_run_rsync_accept_new_host_keys,
    )

    patch_nemo_run_rsync_accept_new_host_keys()
    patch_nemo_run_ray_template_for_cpu()

    packager = CodePackager(
        script_path=str(SCRIPT_REMOTE),
        train_path=str(train_path),
    )

    executor = create_executor(
        env=env,
        env_vars=env_vars,
        packager=packager,
        attached=attached,
        force_squash=force_squash,
        default_image=SPEC.image,
    )

    recipe_name = SPEC.name.replace("/", "-")
    script_args = [*passthrough]

    if experiment is not None:
        return experiment.add(
            run.Script(path="main.py", args=script_args, entrypoint="python"),
            executor=executor,
            name=recipe_name,
        )

    with run.Experiment(recipe_name) as exp:
        exp.add(
            run.Script(path="main.py", args=script_args, entrypoint="python"),
            executor=executor,
            name=recipe_name,
        )
        exp.run(detach=not attached, tail_logs=attached)


def export(ctx: typer.Context) -> None:
    """Export embedding models to ONNX and TensorRT for optimized inference."""
    cfg = parse_recipe_config(ctx)
    _execute_export(cfg)
