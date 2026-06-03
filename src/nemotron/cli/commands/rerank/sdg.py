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

"""SDG command for rerank recipe.

Runs the same SDG pipeline as embed but with rerank-specific config
that writes output to output/rerank/ instead of output/embed/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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
from nemotron.recipes.embed.stage0_sdg.data_prep import SDGConfig

# Use embed's script but rerank's config directory
SCRIPT_PATH = "src/nemotron/recipes/embed/stage0_sdg/data_prep.py"
SCRIPT_REMOTE = "src/nemotron/recipes/embed/stage0_sdg/run_uv.py"
SPEC = parse_runspec(SCRIPT_PATH)
RERANK_CONFIG_DIR = Path("src/nemotron/recipes/rerank/stage0_sdg/config").resolve()


_SECRET_PLACEHOLDER = "<redacted>"


def _config_get(container: Any, key: str) -> Any:
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(key)
    get = getattr(container, "get", None)
    if callable(get):
        try:
            return get(key)
        except Exception:  # noqa: BLE001
            pass
    return getattr(container, key, None)


def _config_set(container: Any, key: str, value: Any) -> None:
    if container is None:
        return
    if isinstance(container, dict):
        container[key] = value
        return
    try:
        setattr(container, key, value)
    except Exception:  # noqa: BLE001
        try:
            container[key] = value
        except Exception:  # noqa: BLE001
            pass


def _redact_api_key_args(args: Any) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for raw_arg in list(args or []):
        arg = str(raw_arg)
        if redact_next:
            redacted.append(_SECRET_PLACEHOLDER)
            redact_next = False
            continue
        if arg in {"nvidia_api_key", "--nvidia_api_key", "--nvidia-api-key"}:
            redacted.append(arg)
            redact_next = True
            continue
        for prefix in ("nvidia_api_key=", "--nvidia_api_key=", "--nvidia-api-key="):
            if arg.startswith(prefix):
                redacted.append(f"{prefix}{_SECRET_PLACEHOLDER}")
                break
        else:
            redacted.append(arg)
    return redacted


def _extract_and_redact_nvidia_api_key(job_config: Any, train_config_for_script: Any | None = None) -> str | None:
    """Move SDG API keys out of persisted config and into executor env vars."""
    secret = _config_get(job_config, "nvidia_api_key")
    if secret:
        _config_set(job_config, "nvidia_api_key", None)
    if train_config_for_script is not None:
        train_secret = _config_get(train_config_for_script, "nvidia_api_key")
        if train_secret:
            _config_set(train_config_for_script, "nvidia_api_key", None)
            secret = secret or train_secret

    run_config = _config_get(job_config, "run")
    cli_config = _config_get(run_config, "cli")
    if cli_config is not None:
        for field in ("argv", "dotlist"):
            values = _config_get(cli_config, field)
            if values is not None:
                _config_set(cli_config, field, _redact_api_key_args(values))

    return str(secret) if secret else None


META = RecipeMeta(
    name="rerank/sdg",
    script_path=SCRIPT_PATH,
    config_dir=str(RERANK_CONFIG_DIR),
    config_model=SDGConfig,
    default_config="default",
    input_artifacts={"corpus": "Document corpus directory"},
    output_artifacts={"data": "Synthetic Q&A pairs (JSON)"},
)


def _execute_sdg(cfg: RecipeConfig, *, experiment=None):
    """Execute SDG with rerank output paths."""
    train_config = parse_config(cfg.ctx, RERANK_CONFIG_DIR, "default")
    env = parse_env(cfg.ctx)

    script_path = SCRIPT_PATH if cfg.mode == "local" else SCRIPT_REMOTE

    job_config = build_job_config(
        train_config,
        cfg.ctx,
        "rerank/sdg",
        script_path,
        cfg.argv,
        env_profile=env,
    )

    secret_api_key = _extract_and_redact_nvidia_api_key(job_config)

    for_remote = cfg.mode != "local"
    display_job_config(job_config, for_remote=for_remote)

    if cfg.dry_run:
        return

    if cfg.stage:
        typer.echo("Error: --stage is not supported for rerank stage commands yet.", err=True)
        raise typer.Exit(1)

    job_dir = generate_job_dir("rerank/sdg")
    # Preserve ${oc.env:NEMO_RUN_DIR,.}; recipe scripts resolve it from the executor's
    # environment so remote pipeline stages can share a configured remote_job_dir.
    train_config_for_script = extract_train_config(job_config, for_remote=False)
    train_secret_api_key = _extract_and_redact_nvidia_api_key(job_config, train_config_for_script)
    secret_api_key = secret_api_key or train_secret_api_key
    job_path, train_path = save_configs(job_config, train_config_for_script, job_dir)

    env_for_executor = job_config.run.env if hasattr(job_config.run, "env") else None
    env_vars = build_env_vars(job_config, env_for_executor)
    if secret_api_key:
        env_vars["NVIDIA_API_KEY"] = secret_api_key

    display_job_submission(job_path, train_path, env_vars, cfg.mode)

    if cfg.mode == "local":
        if secret_api_key:
            os.environ["NVIDIA_API_KEY"] = secret_api_key
        _execute_uv_local(train_path, cfg.passthrough)
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


def _execute_uv_local(train_path: Path, passthrough: list[str]) -> None:
    """Execute SDG locally via UV isolated environment."""
    from nemo_runspec.execution import execute_uv_local_from_spec

    execute_uv_local_from_spec(
        spec=SPEC,
        train_path=train_path,
        passthrough=passthrough,
    )


def _execute_remote(
    train_path: Path,
    env,
    passthrough: list[str],
    attached: bool,
    env_vars: dict[str, str],
    force_squash: bool,
    experiment=None,
):
    """Execute SDG via nemo-run with remote backend."""
    try:
        import nemo_run as run
    except ImportError:
        typer.echo("Error: nemo-run is required for --run/--batch execution", err=True)
        raise typer.Exit(1)

    from nemo_runspec.execution import create_executor
    from nemo_runspec.packaging import CodePackager
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
        script_resources=SPEC.resources,
    )

    recipe_name = "rerank-sdg"
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


def sdg(ctx: typer.Context) -> None:
    """Generate synthetic Q&A pairs from document corpus."""
    cfg = parse_recipe_config(ctx)
    _execute_sdg(cfg)
