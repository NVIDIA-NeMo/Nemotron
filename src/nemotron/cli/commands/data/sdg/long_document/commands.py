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

"""Command handlers for ``nemotron data sdg long-document <stage>``.

Each handler dispatches its corresponding ``recipes/data/sdg/long-document/0X-*.py``
recipe via nemo-run when a profile is selected (``--run dlw`` / ``--batch dlw``)
or via ``uv run`` locally.  Heavy script-level deps (PyMuPDF, datasets, vLLM
clients) are resolved at runtime from each script's PEP 723 inline
``dependencies`` list.

Pydantic config classes are loaded from the scripts via importlib so the
rich ``--help`` panel can introspect every config field — the scripts cannot
be imported by their normal dotted path because their parent directory uses
a dash and filenames begin with digits.

Producer stages (``ocr``, ``text-qa``, ...) optionally accept ``--serve``,
which composes a multi-task ``nemo_run.Experiment``: a serve task brings vLLM
up on a GPU partition and publishes its endpoint to a sentinel file on
shared storage; the recipe (client) task waits on that sentinel and uses the
endpoint as its ``vllm_endpoint``.  See ``_deployment.py`` for the schema
and bash-template generators.

Design: LLM-Native Recipe Architecture
- Execution logic visible and modifiable
- Fork this file to change how long-document SDG jobs are submitted
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import typer

from nemo_runspec import parse as parse_runspec
from nemo_runspec.config import build_job_config, generate_job_dir, parse_config, save_configs
from nemo_runspec.display import display_job_config, display_job_submission
from nemo_runspec.env import parse_env
from nemo_runspec.execution import build_env_vars
from nemo_runspec.recipe_config import RecipeConfig, parse_recipe_config
from nemo_runspec.recipe_typer import RecipeMeta

from nemotron.cli.commands.data.sdg.long_document._config_loader import load_config_class
from nemotron.cli.commands.data.sdg.long_document._deployment import (
    STAGE_DEFAULT_DEPLOYMENT,
    DeploymentConfig,
    build_serve_bash,
    load_deployment_config,
)
from nemotron.cli.commands.data.sdg.long_document._packager import LongDocPackager

# --------------------------------------------------------------------------- #
# Per-stage SPEC + RecipeMeta tables
#
# Each stage is described by:
#   - SCRIPT_*: repo-relative path to the recipe script (used by CodePackager).
#   - SPEC_*  : runspec parsed from the script's PEP 723 [tool.runspec] block.
#   - *_CFG   : Pydantic config class (loaded via importlib for rich --help).
#   - META_*  : RecipeMeta wired into Typer for help rendering.
# --------------------------------------------------------------------------- #

_RECIPES_ROOT = "src/nemotron/recipes/data/sdg/long-document"

# Stage 01: seed -----------------------------------------------------------------
SCRIPT_SEED = f"{_RECIPES_ROOT}/01-seed-dataset-preparation.py"
SPEC_SEED = parse_runspec(SCRIPT_SEED)
SEED_CFG = load_config_class(SPEC_SEED.script_path, "SeedConfig", "_long_doc_seed_module")
META_SEED = RecipeMeta(
    name=SPEC_SEED.name,
    script_path=SCRIPT_SEED,
    config_dir=str(SPEC_SEED.config_dir),
    config_model=SEED_CFG,
    default_config=SPEC_SEED.config.default,
    input_artifacts={},
    output_artifacts={"seed": "Per-page / windowed / whole-document seed parquet files"},
)

# Stage 02: ocr -------------------------------------------------------------------
SCRIPT_OCR = f"{_RECIPES_ROOT}/02-nemotron-parse-ocr-sdg.py"
SPEC_OCR = parse_runspec(SCRIPT_OCR)
OCR_CFG = load_config_class(SPEC_OCR.script_path, "OcrConfig", "_long_doc_ocr_module")
META_OCR = RecipeMeta(
    name=SPEC_OCR.name,
    script_path=SCRIPT_OCR,
    config_dir=str(SPEC_OCR.config_dir),
    config_model=OCR_CFG,
    default_config=SPEC_OCR.config.default,
    input_artifacts={"seed": "Per-page seed parquet (output of 01)"},
    output_artifacts={"ocr": "Parquet with `transcribed_texts` and bbox metadata"},
)

# Stage 03: text-qa ---------------------------------------------------------------
SCRIPT_TEXT_QA = f"{_RECIPES_ROOT}/03-text-qa-sdg.py"
SPEC_TEXT_QA = parse_runspec(SCRIPT_TEXT_QA)
TEXT_QA_CFG = load_config_class(SPEC_TEXT_QA.script_path, "TextQAConfig", "_long_doc_text_qa_module")
META_TEXT_QA = RecipeMeta(
    name=SPEC_TEXT_QA.name,
    script_path=SCRIPT_TEXT_QA,
    config_dir=str(SPEC_TEXT_QA.config_dir),
    config_model=TEXT_QA_CFG,
    default_config=SPEC_TEXT_QA.config.default,
    input_artifacts={"ocr": "OCR parquet (output of 02)"},
    output_artifacts={"text_qa": "Parquet with text-grounded QA pairs"},
)

# Stage 04: page-classification ---------------------------------------------------
SCRIPT_PAGE_CLASSIFICATION = f"{_RECIPES_ROOT}/04-page-classification-sdg.py"
SPEC_PAGE_CLASSIFICATION = parse_runspec(SCRIPT_PAGE_CLASSIFICATION)
PAGE_CLASSIFICATION_CFG = load_config_class(
    SPEC_PAGE_CLASSIFICATION.script_path,
    "PageClassificationConfig",
    "_long_doc_page_classification_module",
)
META_PAGE_CLASSIFICATION = RecipeMeta(
    name=SPEC_PAGE_CLASSIFICATION.name,
    script_path=SCRIPT_PAGE_CLASSIFICATION,
    config_dir=str(SPEC_PAGE_CLASSIFICATION.config_dir),
    config_model=PAGE_CLASSIFICATION_CFG,
    default_config=SPEC_PAGE_CLASSIFICATION.config.default,
    input_artifacts={"seed": "Per-page seed parquet (output of 01)"},
    output_artifacts={"page_classification": "Parquet with `page_classification` column"},
)

# Stage 05: visual-qa -------------------------------------------------------------
SCRIPT_VISUAL_QA = f"{_RECIPES_ROOT}/05-visual-qa-sdg.py"
SPEC_VISUAL_QA = parse_runspec(SCRIPT_VISUAL_QA)
VISUAL_QA_CFG = load_config_class(SPEC_VISUAL_QA.script_path, "VisualQAConfig", "_long_doc_visual_qa_module")
META_VISUAL_QA = RecipeMeta(
    name=SPEC_VISUAL_QA.name,
    script_path=SCRIPT_VISUAL_QA,
    config_dir=str(SPEC_VISUAL_QA.config_dir),
    config_model=VISUAL_QA_CFG,
    default_config=SPEC_VISUAL_QA.config.default,
    input_artifacts={"page_classification": "Page-classified parquet (output of 04)"},
    output_artifacts={"visual_qa": "Parquet with visual QA pairs"},
)

# Stage 06: single-page-qa --------------------------------------------------------
SCRIPT_SINGLE_PAGE_QA = f"{_RECIPES_ROOT}/06-single-page-qa-sdg.py"
SPEC_SINGLE_PAGE_QA = parse_runspec(SCRIPT_SINGLE_PAGE_QA)
SINGLE_PAGE_QA_CFG = load_config_class(
    SPEC_SINGLE_PAGE_QA.script_path,
    "SinglePageQAConfig",
    "_long_doc_single_page_qa_module",
)
META_SINGLE_PAGE_QA = RecipeMeta(
    name=SPEC_SINGLE_PAGE_QA.name,
    script_path=SCRIPT_SINGLE_PAGE_QA,
    config_dir=str(SPEC_SINGLE_PAGE_QA.config_dir),
    config_model=SINGLE_PAGE_QA_CFG,
    default_config=SPEC_SINGLE_PAGE_QA.config.default,
    input_artifacts={"seed": "Per-page seed parquet (output of 01)"},
    output_artifacts={"single_page_qa": "Parquet with anchored single-page QA pairs"},
)

# Stage 07: windowed-qa -----------------------------------------------------------
SCRIPT_WINDOWED_QA = f"{_RECIPES_ROOT}/07-multi-page-windowed-qa-sdg.py"
SPEC_WINDOWED_QA = parse_runspec(SCRIPT_WINDOWED_QA)
WINDOWED_QA_CFG = load_config_class(
    SPEC_WINDOWED_QA.script_path,
    "WindowedQAConfig",
    "_long_doc_windowed_qa_module",
)
META_WINDOWED_QA = RecipeMeta(
    name=SPEC_WINDOWED_QA.name,
    script_path=SCRIPT_WINDOWED_QA,
    config_dir=str(SPEC_WINDOWED_QA.config_dir),
    config_model=WINDOWED_QA_CFG,
    default_config=SPEC_WINDOWED_QA.config.default,
    input_artifacts={"windowed_seed": "Windowed seed parquet (output of 01)"},
    output_artifacts={"windowed_qa": "Parquet with multi-page windowed QA pairs"},
)

# Stage 08: whole-document-qa -----------------------------------------------------
SCRIPT_WHOLE_DOCUMENT_QA = f"{_RECIPES_ROOT}/08-whole-document-qa-sdg.py"
SPEC_WHOLE_DOCUMENT_QA = parse_runspec(SCRIPT_WHOLE_DOCUMENT_QA)
WHOLE_DOCUMENT_QA_CFG = load_config_class(
    SPEC_WHOLE_DOCUMENT_QA.script_path,
    "WholeDocumentQAConfig",
    "_long_doc_whole_document_qa_module",
)
META_WHOLE_DOCUMENT_QA = RecipeMeta(
    name=SPEC_WHOLE_DOCUMENT_QA.name,
    script_path=SCRIPT_WHOLE_DOCUMENT_QA,
    config_dir=str(SPEC_WHOLE_DOCUMENT_QA.config_dir),
    config_model=WHOLE_DOCUMENT_QA_CFG,
    default_config=SPEC_WHOLE_DOCUMENT_QA.config.default,
    input_artifacts={"whole_document_seed": "Whole-document seed parquet (output of 01)"},
    output_artifacts={"whole_document_qa": "Parquet with whole-document QA pairs"},
)

# Stage 09: judge -----------------------------------------------------------------
SCRIPT_JUDGE = f"{_RECIPES_ROOT}/09-frontier-judge-sdg.py"
SPEC_JUDGE = parse_runspec(SCRIPT_JUDGE)
JUDGE_CFG = load_config_class(SPEC_JUDGE.script_path, "JudgeConfig", "_long_doc_judge_module")
META_JUDGE = RecipeMeta(
    name=SPEC_JUDGE.name,
    script_path=SCRIPT_JUDGE,
    config_dir=str(SPEC_JUDGE.config_dir),
    config_model=JUDGE_CFG,
    default_config=SPEC_JUDGE.config.default,
    input_artifacts={"qa": "Any QA-output parquet (05/06/07/08)"},
    output_artifacts={"judged_qa": "Parquet with rubric scores + weighted composite"},
)


# --------------------------------------------------------------------------- #
# Shared execution helper
# --------------------------------------------------------------------------- #


def _execute_long_doc_stage(
    cfg: RecipeConfig,
    *,
    script_path: str,
    spec,
    serve_with: DeploymentConfig | None = None,
) -> None:
    """Execute a long-document SDG stage either locally (via ``uv``) or remotely
    (via nemo-run).

    When ``serve_with`` is provided and the run mode is ``run`` or ``batch``,
    the dispatch composes a multi-task experiment (serve + client) so the
    required vLLM model is brought up automatically on a GPU partition.
    """

    train_config = parse_config(cfg.ctx, spec.config_dir, spec.config.default)
    env = parse_env(cfg.ctx)

    job_config = build_job_config(
        train_config,
        cfg.ctx,
        spec.name,
        script_path,
        cfg.argv,
        env_profile=env,
    )
    display_job_config(job_config, for_remote=cfg.mode != "local")

    if serve_with is not None:
        typer.echo(
            f"[serve] auto-deploy enabled: deployment={serve_with.name} "
            f"model={serve_with.hf_model_handle} tp={serve_with.tensor_parallel_size}"
        )

    if cfg.dry_run:
        return

    if serve_with is not None and cfg.mode == "local":
        typer.echo(
            "Error: --serve is only meaningful with --run/--batch (auto-deploy "
            "spins up vLLM via nemo-run on the cluster).  Drop --serve for local runs.",
            err=True,
        )
        raise typer.Exit(1)

    job_dir = generate_job_dir(spec.name)
    job_path, train_path = save_configs(job_config, train_config, job_dir)

    env_for_executor = job_config.run.env if hasattr(job_config.run, "env") else None
    env_vars = build_env_vars(job_config, env_for_executor)
    display_job_submission(
        job_path,
        train_path,
        env_vars,
        cfg.mode,
        artifacts=job_config.get("artifacts"),
    )

    if cfg.mode == "local":
        _execute_uv_local(spec.script_path, train_path, cfg.passthrough)
    else:
        _execute_remote(
            script_path=script_path,
            train_path=train_path,
            spec=spec,
            env=env_for_executor,
            passthrough=cfg.passthrough,
            attached=cfg.attached,
            env_vars=env_vars,
            force_squash=cfg.force_squash,
            serve_with=serve_with,
        )


def _execute_uv_local(script_abs: Path, train_path: Path, passthrough: list[str]) -> None:
    """Execute a long-document recipe locally via ``uv run --no-project``.

    PEP 723 inline deps are resolved by uv at runtime; we don't pin a project
    so the parent Nemotron env doesn't get pulled in.
    """
    import os

    uv_cmd = shutil.which("uv")
    if not uv_cmd:
        typer.echo(
            "Error: 'uv' command not found. Install uv (https://docs.astral.sh/uv/) "
            "or run via --run/--batch to use a remote container.",
            err=True,
        )
        raise typer.Exit(1)

    cmd = [
        uv_cmd, "run", "--no-project",
        str(script_abs),
        "--config", str(train_path),
        *passthrough,
    ]

    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)

    typer.echo(f"Executing locally via uv: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    raise typer.Exit(result.returncode)


def _build_serve_executor(
    *,
    serve_with: DeploymentConfig,
    env,
    env_vars: dict[str, str],
    attached: bool,
    force_squash: bool,
):
    """Build a ``nemo_run.SlurmExecutor`` for the auto-deploy serve task.

    Resolves the GPU partition in this priority order:
      1. ``serve_with.partition`` (deployment YAML)
      2. ``env.sdg_serve_partition`` (env.toml profile field)
      3. ``env.run_partition`` (existing field, typically ``interactive``)
      4. ``env.partition`` (final fallback)
    """
    import nemo_run as run

    def _get(key: str, default=None):
        if env is None:
            return default
        return env.get(key, default) if hasattr(env, "get") else getattr(env, key, default)

    partition = (
        serve_with.partition
        or _get("sdg_serve_partition")
        or _get("run_partition")
        or _get("partition")
    )

    tunnel = None
    remote_job_dir = _get("remote_job_dir")
    if _get("tunnel") == "ssh":
        tunnel = run.SSHTunnel(
            host=_get("host", "localhost"),
            user=_get("user"),
            job_dir=remote_job_dir,
        )

    raw_mounts = list(_get("mounts") or [])
    mounts = [m for m in raw_mounts if not m.startswith("__auto_mount__:")]
    if "/lustre:/lustre" not in mounts:
        mounts.append("/lustre:/lustre")

    return run.SlurmExecutor(
        account=_get("account"),
        partition=partition,
        nodes=serve_with.nodes,
        ntasks_per_node=1,
        gpus_per_node=serve_with.gpus_per_node,
        cpus_per_task=_get("cpus_per_task"),
        time=serve_with.walltime,
        container_image=serve_with.image,
        container_mounts=mounts,
        tunnel=tunnel,
        mem=_get("mem"),
        env_vars=env_vars,
        launcher=None,
    )


def _build_sentinel_path(remote_job_dir: str | None, recipe_name: str) -> str:
    """Generate a unique sentinel path on shared storage for one --serve run."""
    base = remote_job_dir or "/tmp"
    return f"{base}/sdg-deploy/{recipe_name}/{int(time.time())}/endpoint"


def _execute_remote(
    *,
    script_path: str,
    train_path: Path,
    spec,
    env,
    passthrough: list[str],
    attached: bool,
    env_vars: dict[str, str],
    force_squash: bool,
    serve_with: DeploymentConfig | None = None,
    experiment=None,
):
    """Dispatch a long-document recipe remotely via nemo-run + Slurm.

    When ``serve_with`` is provided, the experiment composes a serve task
    (vLLM on a GPU partition) and a client task (the recipe) coordinated by
    a sentinel file on shared storage.
    """
    try:
        import nemo_run as run
    except ImportError:
        typer.echo("Error: nemo-run is required for --run/--batch execution", err=True)
        raise typer.Exit(1)

    from nemo_runspec.execution import create_executor
    from nemo_runspec.run import (
        patch_nemo_run_ray_template_for_cpu,
        patch_nemo_run_rsync_accept_new_host_keys,
    )

    patch_nemo_run_rsync_accept_new_host_keys()
    patch_nemo_run_ray_template_for_cpu()

    recipe_name = spec.name.replace("/", "-")

    # Sentinel + serve-task plumbing (only when --serve was passed).
    sentinel_path: str | None = None
    serve_executor = None
    serve_bash: str | None = None
    if serve_with is not None:
        remote_job_dir = (
            env.get("remote_job_dir")
            if env is not None and hasattr(env, "get")
            else getattr(env, "remote_job_dir", None) if env is not None else None
        )
        sentinel_path = _build_sentinel_path(remote_job_dir, recipe_name)
        serve_executor = _build_serve_executor(
            serve_with=serve_with,
            env=env,
            env_vars=env_vars,
            attached=attached,
            force_squash=force_squash,
        )
        serve_bash = build_serve_bash(serve_with, sentinel_path)
        typer.echo(f"[serve] sentinel: {sentinel_path}")
        typer.echo(f"[serve] partition: {serve_executor.partition}")

    packager = LongDocPackager(
        script_path=script_path,
        train_path=str(train_path),
        sentinel_path=sentinel_path,
    )

    client_executor = create_executor(
        env=env,
        env_vars=env_vars,
        packager=packager,
        attached=attached,
        force_squash=force_squash,
        default_image=spec.image,
    )

    script_args = list(passthrough)

    if experiment is not None:
        if serve_with is not None and serve_bash is not None and serve_executor is not None:
            experiment.add(
                run.Script(inline=serve_bash, entrypoint="bash"),
                executor=serve_executor,
                name=f"{recipe_name}-serve",
            )
        return experiment.add(
            run.Script(path="main.py", args=script_args, entrypoint="python"),
            executor=client_executor,
            name=f"{recipe_name}-client" if serve_with is not None else recipe_name,
        )

    with run.Experiment(recipe_name) as exp:
        if serve_with is not None and serve_bash is not None and serve_executor is not None:
            exp.add(
                run.Script(inline=serve_bash, entrypoint="bash"),
                executor=serve_executor,
                name=f"{recipe_name}-serve",
            )
        exp.add(
            run.Script(path="main.py", args=script_args, entrypoint="python"),
            executor=client_executor,
            name=f"{recipe_name}-client" if serve_with is not None else recipe_name,
        )
        exp.run(detach=not attached, tail_logs=attached)


# --------------------------------------------------------------------------- #
# Per-stage command functions — each one is a thin wrapper that pins SCRIPT
# and SPEC.  Add more stages here as scripts 02–09 are refactored.
# --------------------------------------------------------------------------- #


def seed(ctx: typer.Context) -> None:
    """Build per-page / windowed / whole-document seed parquet files from FinePDFs.

    CPU-only.  Override defaults with Hydra-style ``key=value`` pairs, e.g.:
    ``num_docs=50 subset=fra_Latn``.
    """
    cfg = parse_recipe_config(ctx)
    _execute_long_doc_stage(cfg, script_path=SCRIPT_SEED, spec=SPEC_SEED)


def _resolve_serve(stage_key: str, *, serve: bool, serve_config: str | None) -> DeploymentConfig | None:
    """Resolve a ``DeploymentConfig`` for ``--serve`` flag handling.

    Returns ``None`` if ``--serve`` was not passed.  Otherwise loads the
    deployment YAML named by ``--serve-config`` (or the stage's default).
    """
    if not serve:
        return None
    deploy_name = serve_config or STAGE_DEFAULT_DEPLOYMENT.get(stage_key)
    if deploy_name is None:
        typer.echo(
            f"Error: stage '{stage_key}' has no default deployment registered. "
            f"Pass --serve-config <name> explicitly.",
            err=True,
        )
        raise typer.Exit(1)
    try:
        return load_deployment_config(deploy_name)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


def ocr(
    ctx: typer.Context,
    serve: bool = typer.Option(
        False,
        "--serve",
        help="Auto-deploy the required vLLM model on a GPU partition before running the recipe.",
    ),
    serve_config: str = typer.Option(
        None,
        "--serve-config",
        help="Override the default deployment-config name (looked up in deployment/<name>.yaml).",
    ),
) -> None:
    """Run Nemotron-Parse OCR over per-page seed images via a vLLM endpoint."""
    cfg = parse_recipe_config(ctx)
    serve_with = _resolve_serve("ocr", serve=serve, serve_config=serve_config)
    _execute_long_doc_stage(cfg, script_path=SCRIPT_OCR, spec=SPEC_OCR, serve_with=serve_with)


def text_qa(
    ctx: typer.Context,
    serve: bool = typer.Option(
        False, "--serve",
        help="Auto-deploy the required vLLM model on a GPU partition before running the recipe.",
    ),
    serve_config: str = typer.Option(
        None, "--serve-config",
        help="Override the default deployment-config name (looked up in deployment/<name>.yaml).",
    ),
) -> None:
    """Generate text-QA pairs from OCR-transcribed document text."""
    cfg = parse_recipe_config(ctx)
    serve_with = _resolve_serve("text-qa", serve=serve, serve_config=serve_config)
    _execute_long_doc_stage(cfg, script_path=SCRIPT_TEXT_QA, spec=SPEC_TEXT_QA, serve_with=serve_with)


def page_classification(
    ctx: typer.Context,
    serve: bool = typer.Option(
        False, "--serve",
        help="Auto-deploy the required vLLM model on a GPU partition before running the recipe.",
    ),
    serve_config: str = typer.Option(
        None, "--serve-config",
        help="Override the default deployment-config name (looked up in deployment/<name>.yaml).",
    ),
) -> None:
    """Classify pages by visual element type and reasoning complexity."""
    cfg = parse_recipe_config(ctx)
    serve_with = _resolve_serve("page-classification", serve=serve, serve_config=serve_config)
    _execute_long_doc_stage(
        cfg, script_path=SCRIPT_PAGE_CLASSIFICATION, spec=SPEC_PAGE_CLASSIFICATION,
        serve_with=serve_with,
    )


def visual_qa(
    ctx: typer.Context,
    serve: bool = typer.Option(
        False, "--serve",
        help="Auto-deploy the required vLLM model on a GPU partition before running the recipe.",
    ),
    serve_config: str = typer.Option(
        None, "--serve-config",
        help="Override the default deployment-config name (looked up in deployment/<name>.yaml).",
    ),
) -> None:
    """Generate visual QA pairs grounded in page images and their classification."""
    cfg = parse_recipe_config(ctx)
    serve_with = _resolve_serve("visual-qa", serve=serve, serve_config=serve_config)
    _execute_long_doc_stage(cfg, script_path=SCRIPT_VISUAL_QA, spec=SPEC_VISUAL_QA, serve_with=serve_with)


def single_page_qa(
    ctx: typer.Context,
    serve: bool = typer.Option(
        False, "--serve",
        help="Auto-deploy the required vLLM model on a GPU partition before running the recipe.",
    ),
    serve_config: str = typer.Option(
        None, "--serve-config",
        help="Override the default deployment-config name (looked up in deployment/<name>.yaml).",
    ),
) -> None:
    """Generate anchored single-page QA across Text/Table/Chart/Image/Layout."""
    cfg = parse_recipe_config(ctx)
    serve_with = _resolve_serve("single-page-qa", serve=serve, serve_config=serve_config)
    _execute_long_doc_stage(
        cfg, script_path=SCRIPT_SINGLE_PAGE_QA, spec=SPEC_SINGLE_PAGE_QA,
        serve_with=serve_with,
    )


def windowed_qa(
    ctx: typer.Context,
    serve: bool = typer.Option(
        False, "--serve",
        help="Auto-deploy the required vLLM model on a GPU partition before running the recipe.",
    ),
    serve_config: str = typer.Option(
        None, "--serve-config",
        help="Override the default deployment-config name (looked up in deployment/<name>.yaml).",
    ),
) -> None:
    """Generate multi-page QA from sliding windows of consecutive pages."""
    cfg = parse_recipe_config(ctx)
    serve_with = _resolve_serve("windowed-qa", serve=serve, serve_config=serve_config)
    _execute_long_doc_stage(
        cfg, script_path=SCRIPT_WINDOWED_QA, spec=SPEC_WINDOWED_QA,
        serve_with=serve_with,
    )


def whole_document_qa(
    ctx: typer.Context,
    serve: bool = typer.Option(
        False, "--serve",
        help="Auto-deploy the required vLLM model on a GPU partition before running the recipe.",
    ),
    serve_config: str = typer.Option(
        None, "--serve-config",
        help="Override the default deployment-config name (looked up in deployment/<name>.yaml).",
    ),
) -> None:
    """Generate whole-document QA requiring cross-page reasoning."""
    cfg = parse_recipe_config(ctx)
    serve_with = _resolve_serve("whole-document-qa", serve=serve, serve_config=serve_config)
    _execute_long_doc_stage(
        cfg, script_path=SCRIPT_WHOLE_DOCUMENT_QA, spec=SPEC_WHOLE_DOCUMENT_QA,
        serve_with=serve_with,
    )


def judge(ctx: typer.Context) -> None:
    """Score QA pairs using a frontier LLM-as-a-judge."""
    cfg = parse_recipe_config(ctx)
    _execute_long_doc_stage(cfg, script_path=SCRIPT_JUDGE, spec=SPEC_JUDGE)
