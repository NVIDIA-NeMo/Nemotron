"""Stage uv environments and standard NeMo-Run container execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from nemo_runspec import parse as parse_runspec
from nemo_runspec.recipe_config import parse_recipe_config
from nemo_runspec.recipe_typer import RecipeMeta
from nemotron.recipes.parse.config import ROOT, STAGES

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from nemo_runspec import Runspec
    from nemo_runspec.recipe_config import RecipeConfig


def metadata(stage: str) -> RecipeMeta:
    model, directory, script = STAGES[stage]
    spec = parse_runspec(ROOT / directory / f"{script}.py")
    return RecipeMeta(
        name=spec.name,
        script_path=str(spec.script_path),
        config_dir=str(spec.config_dir),
        config_model=model,
        input_artifacts={} if stage == "data" else {"data_dir": "Parse dataset directory with artifact.json"},
        output_artifacts={"artifact": "Local artifact.json with data/model identity and paths"},
    )


def execute(stage: str, ctx: typer.Context) -> None:
    """Use the same uv-local and NeMo-Run executor path as retrieval recipes."""
    from omegaconf import OmegaConf

    from nemo_runspec.config import build_job_config, generate_job_dir, parse_config, save_configs
    from nemo_runspec.env import parse_env
    from nemo_runspec.execution import build_env_vars, execute_uv_local_from_spec

    request = parse_recipe_config(ctx)
    if request.stage:
        raise typer.BadParameter("--stage is not supported for Parse commands")
    if request.passthrough:
        raise typer.BadParameter(f"Unrecognized arguments: {' '.join(request.passthrough)}")
    if not request.dry_run and not (ROOT.parents[3] / "src/nemotron").is_dir():
        raise typer.BadParameter("Parse execution requires a Nemotron source checkout or editable install.")
    model, directory, script = STAGES[stage]
    spec = parse_runspec(ROOT / directory / f"{script}.py")
    raw = parse_config(request.ctx, spec.config_dir, spec.config.default)
    values = OmegaConf.to_container(raw, resolve=True)
    values.pop("run", None)
    config = model.model_validate(values)
    repo = ROOT.parents[3]
    payload = config.model_dump(mode="json")
    job = build_job_config(
        raw, request.ctx, spec.name, "src/nemotron/kit/run_uv.py", request.argv, env_profile=parse_env(request.ctx)
    )
    env = job.run.env
    executor = env.get("executor", "local")
    local = request.mode == "local" or executor == "local"
    _validate_resources(env, stage)
    if not local:
        if executor not in {"docker", "slurm"}:
            raise typer.BadParameter("Parse supports local uv, Docker, and single-task Slurm execution.")
        _validate_remote_paths(payload, env, stage)
        if "training_config" in payload:
            path = Path(payload["training_config"])
            if path.is_relative_to(repo):
                payload["training_config"] = str(path.relative_to(repo))
        if executor == "slurm":
            env["nodes"] = env["ntasks_per_node"] = 1
        if executor == "docker" and stage in {"rlvr", "eval"}:
            # vLLM images serve by default; these stages execute batch scripts.
            env["additional_kwargs"] = {"entrypoint": "", **(env.get("additional_kwargs") or {})}
    if request.dry_run:
        typer.echo(
            json.dumps(
                payload if local else dict(payload, run=OmegaConf.to_container(job.run, resolve=True)),
                indent=2,
            )
        )
        return
    job_dir = generate_job_dir(spec.name)
    _, train_path = save_configs(job, OmegaConf.create(payload), job_dir)
    env_vars = build_env_vars(job, env)
    env_vars["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    # Ray actors already receive the stage interpreter explicitly. Its automatic
    # uv hook would otherwise resync workers and upload the submitter's cwd.
    if stage in ("rlvr", "eval"):
        env_vars["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    if local:
        execute_uv_local_from_spec(
            spec=spec,
            train_path=train_path,
            passthrough=[],
            env_vars=env_vars,
            locked=True,
            isolate_environment=True,
        )
    else:
        _execute_remote(spec, train_path, env, env_vars, request, directory)


def _validate_resources(env: DictConfig, stage: str) -> None:
    """Reject profiles outside the stage's single-controller support boundary."""
    for name in ("nodes", "ntasks_per_node", "nproc_per_node", "nprocs_per_node", "gpus_per_node"):
        value = env.get(name)
        allowed = {0, 1} if stage == "data" and name == "gpus_per_node" else {1}
        if value is not None and value not in allowed:
            raise typer.BadParameter(f"Parse requires one node, one task, and one GPU; {name}={value} is unsupported.")


def _validate_remote_paths(payload: dict[str, Any], env: DictConfig, stage: str) -> None:
    """Require persistent remote artifacts rather than job-private relative paths."""
    mounts = list(env.get("mounts") or [])
    if env.get("executor") == "docker":
        from nemo_runspec.execution import resolve_docker_mounts

        mounts = resolve_docker_mounts(mounts)
    if env.get("executor") == "slurm" and env.get("lustre_mount", "/lustre:/lustre"):
        mounts.append(env.get("lustre_mount", "/lustre:/lustre"))
    destinations = []
    for mount in mounts:
        parts = str(mount).split(":")
        destinations.append(
            (
                Path(os.path.normpath(parts[1] if len(parts) > 1 else parts[0])),
                any("ro" in option.split(",") for option in parts[2:]),
            )
        )
    fields = ["data_dir"]
    if "output_dir" in payload:
        fields.append("output_dir")
    if stage == "rlvr":
        fields.append("sft_output")
    if stage == "eval":
        fields.append("model_artifact" if payload.get("model") is None else "model")
    training = payload.get("training_config")
    if training is not None:
        path = Path(training)
        source = path if path.is_absolute() else Path.cwd() / path
        source = Path(os.path.normpath(source))
        if source.is_relative_to(ROOT.parents[3]) and source.is_file():
            payload["training_config"] = str(source.relative_to(ROOT.parents[3]))
        else:
            fields.append("training_config")
    model = payload.get("model")
    if stage != "eval" and model is not None:
        path = Path(model)
        # SFT also accepts Hub repository IDs, which are resolved in the worker.
        if stage == "rlvr" or path.is_absolute() or path.exists() or str(model).startswith((".", "~")):
            fields.append("model")
    if payload.get("processor_source") is not None:
        fields.append("processor_source")
    for name in fields:
        path = Path(os.path.normpath(payload[name]))
        writes = name == "output_dir" or (stage == "data" and name == "data_dir")
        matching = [
            (len(root.parts), index, readonly)
            for index, (root, readonly) in enumerate(destinations)
            if path.is_relative_to(root)
        ]
        if not path.is_absolute() or not matching or (writes and max(matching)[2]):
            raise typer.BadParameter(
                f"Remote {name} must be absolute and beneath a profile mount: {path}. "
                "See docs/nemotron/parse/runtime.md."
            )


def _execute_remote(
    spec: Runspec,
    train_path: Path,
    env: DictConfig,
    env_vars: dict[str, str],
    request: RecipeConfig,
    directory: str,
) -> None:
    """Submit one stage controller and propagate attached execution failures."""
    import nemo_run as run

    from nemo_runspec.execution import create_executor
    from nemo_runspec.packaging import CodePackager

    if env.get("executor") == "docker" and not request.attached:
        raise typer.BadParameter("Docker --batch is unavailable with this NeMo-Run version; use --run.")
    packager = CodePackager(
        script_path="src/nemotron/kit/run_uv.py", train_path=str(train_path), exit_code_file=".parse-stage-exit-code"
    )
    executor = create_executor(
        env=env,
        env_vars=env_vars,
        packager=packager,
        attached=request.attached,
        force_squash=request.force_squash,
        default_image=spec.image,
        script_resources=spec.resources,
    )
    # Automodel manages its own single-rank process group; RLVR manages Ray.
    # An outer torchrun would launch duplicate controllers and stage outputs.
    executor.launcher = None
    name = spec.name.replace("/", "-")
    with run.Experiment(name) as experiment:
        experiment.add(
            run.Script(
                path="main.py", entrypoint="python3", args=["--stage-dir", f"src/nemotron/recipes/parse/{directory}"]
            ),
            executor=executor,
            name=name,
        )
        experiment.run(detach=not request.attached, tail_logs=request.attached)
    if request.attached and isinstance(executor, run.DockerExecutor):
        # Some NeMo-Run versions report tee's exit status. Read the stage's
        # result from the host-mounted package rather than accepting that.
        receipt = Path(experiment.jobs[0].executor.job_dir) / "code" / ".parse-stage-exit-code"
        if not receipt.is_file():
            typer.echo("Container did not report a stage exit code; inspect its logs.", err=True)
            raise typer.Exit(1)
        code = int(receipt.read_text())
        if code != 0:
            typer.echo(f"Parse stage failed (exit code {code}); see the container log above.", err=True)
            raise typer.Exit(code)
    if request.attached:
        statuses = experiment.status(return_dict=True) or {}
        if any(str(value["status"]) in {"FAILED", "CANCELLED"} for value in statuses.values()):
            raise typer.Exit(1)


def data(ctx: typer.Context) -> None:
    """Generate the dental example images and complete Parse targets."""
    execute("data", ctx)


def sft(ctx: typer.Context) -> None:
    """Fine-tune the full Parse model with the configured SFT schedule."""
    execute("sft", ctx)


def rlvr(ctx: typer.Context) -> None:
    """Mine candidates, run decoder GRPO, and evaluate before/after."""
    execute("rlvr", ctx)


def evaluate(ctx: typer.Context) -> None:
    """Compare base, SFT and RLVR checkpoints on the held-out dental pages."""
    execute("eval", ctx)
