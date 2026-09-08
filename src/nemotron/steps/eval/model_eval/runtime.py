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

"""Runtime helpers for `eval/model_eval`.

Two execution modes, selected by ``mode``:

``launcher`` (default)
    Hand the config to NeMo Evaluator Launcher, which owns scheduling. Its
    executors cover local / slurm / lepton only — there is no Run:ai backend —
    and each cloud executor re-implements job submission, which is where the
    Lepton-specific defects documented in this step's README live.

``direct``
    Run the harness in-process with ``nemo-evaluator run_eval`` and let
    **Nemotron's own runspec executors** place the job (``--run/--batch``
    profile: local, docker, slurm, lepton, dgxcloud/Run:ai). The step container
    must ship the harness; the model is reached over HTTP, so the endpoint is
    either hosted separately (Lepton, Run:ai) or brought up in the same
    allocation (Slurm).

    Direct mode is backend-agnostic by construction: it never touches the
    launcher's executor layer, so Lepton/Run:ai support comes from the same
    runspec code every other Nemotron step uses.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from nemo_runspec.config import clear_artifact_cache, register_resolvers_from_config
from nemo_runspec.evaluator import (
    ensure_wandb_host_env,
    get_non_task_args,
    inject_wandb_env_mappings,
    maybe_auto_squash_evaluator,
    needs_wandb,
    parse_task_flags,
    save_eval_configs,
)
from nemotron.kit.train_script import (
    apply_hydra_overrides,
    load_omegaconf_yaml,
    parse_config_and_overrides,
)

_STEP_ONLY_KEYS = {
    "dry_run",
    "harness_image",
    "mode",
    "output_dir",
    "overwrite",
    "task_filters",
}


class ModelEvalDependencyError(RuntimeError):
    """The launcher this step delegates to is not importable.

    Raised instead of exiting the process, so a caller that embeds this launch
    boundary can map the failure into its own error contract; `run_model_eval`
    turns it back into a process exit for direct CLI use.
    """


@dataclass(frozen=True)
class ModelEvalLaunchResult:
    launcher_config_path: Path
    invocation_id: str | None


def run_model_eval(*, default_config: Path) -> None:
    config_path, cfg, overrides = _load_config(default_config)
    passthrough = _passthrough_args(overrides)
    _validate_passthrough(passthrough)

    if str(cfg.get("mode", "launcher")).lower() == "direct":
        run_direct(cfg, task_filters=parse_task_flags(passthrough))
        return

    task_filters = parse_task_flags(passthrough) or None
    try:
        result = launch_model_eval_config(
            config_path=config_path,
            cfg=cfg,
            task_filters=task_filters,
        )
    except ModelEvalDependencyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Install with: uv sync --extra evaluator", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"launcher_config: {result.launcher_config_path}")
    if result.invocation_id:
        print(f"launcher_invocation_id: {result.invocation_id}")
        print(f"status_command: nemo-evaluator-launcher status {result.invocation_id}")
        print(f"logs_command: nemo-evaluator-launcher logs {result.invocation_id}")


def launch_model_eval_config(
    *,
    config_path: Path,
    cfg: DictConfig,
    task_filters: list[str] | None = None,
) -> ModelEvalLaunchResult:
    """Submit a prepared model-eval config without reparsing process arguments."""
    # Resolve the launcher before writing anything, so a missing dependency does
    # not leave a materialized config behind that was never submitted.
    try:
        from nemo_evaluator_launcher.api.functional import (  # type: ignore[import-untyped]
            run_eval,
        )
    except ImportError as exc:
        raise ModelEvalDependencyError(
            "nemo-evaluator-launcher is required for evaluation"
        ) from exc

    launcher_cfg, dry_run, configured_tasks = _build_launcher_config(cfg)
    selected_tasks = task_filters if task_filters is not None else configured_tasks
    eval_path = _save_launcher_config(config_path, cfg, launcher_cfg)

    invocation_id = run_eval(
        launcher_cfg,
        dry_run=dry_run,
        tasks=selected_tasks,
    )
    return ModelEvalLaunchResult(
        launcher_config_path=eval_path,
        invocation_id=str(invocation_id) if invocation_id else None,
    )


def _load_config(default_config: Path) -> tuple[Path, DictConfig, list[str]]:
    config_path, overrides = parse_config_and_overrides(default_config=default_config)
    cfg = apply_hydra_overrides(load_omegaconf_yaml(config_path), overrides)
    return Path(config_path), cfg, overrides


def _build_launcher_config(cfg: DictConfig) -> tuple[DictConfig, bool, list[str] | None]:
    dry_run = bool(cfg.get("dry_run", False))
    output_dir = cfg.get("output_dir")
    task_filters = cfg.get("task_filters")

    _maybe_auto_squash(cfg, dry_run=dry_run)

    if needs_wandb(cfg):
        ensure_wandb_host_env()

    clear_artifact_cache()
    register_resolvers_from_config(cfg, artifacts_key="run", mode="pre_init")
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError("model evaluation config must resolve to one object")
    launcher_dict = dict(resolved)
    launcher_dict.pop("run", None)
    for key in _STEP_ONLY_KEYS:
        launcher_dict.pop(key, None)

    if output_dir:
        launcher_dict.setdefault("execution", {})
        launcher_dict["execution"].setdefault("output_dir", output_dir)

    launcher_cfg = OmegaConf.create(launcher_dict)
    if needs_wandb(launcher_cfg):
        ensure_wandb_host_env()
        inject_wandb_env_mappings(launcher_cfg)

    # AFTER injection, not before. `inject_wandb_env_mappings` mutates the
    # config, so a config that validates clean on its own can still be invalid
    # by the time the launcher sees it -- which is precisely how a rejected key
    # survived both a hand-audit of the YAML and an inspection of the compiled
    # config.
    _validate_launcher_schema(launcher_cfg)

    return launcher_cfg, dry_run, list(task_filters) if task_filters else None


def _validate_launcher_schema(launcher_cfg: DictConfig) -> None:
    """Run the launcher's own schema checks on the composed config.

    The launcher validates at submission time, so a config error surfaces only
    once a job has been scheduled. Running the same checks here means a
    ``dry_run=true`` run fails identically to a real one, for free and with no
    schema of our own to keep in sync.

    Note this cannot cover ``nemotron steps run -d``: the generic runner prints
    the compiled config and returns before the step executes, so nothing in
    this file runs at all on that path.
    """
    try:
        from nemo_evaluator_launcher.api.functional import (
            _validate_config_sections,
            _validate_no_missing_values,
        )
    except ImportError:
        # Private helpers; if upstream renames them, submission-time validation
        # still applies. Losing the early check must not break the run.
        return

    for check in (_validate_no_missing_values, _validate_config_sections):
        try:
            check(launcher_cfg)
        except ValueError as exc:
            raise SystemExit(f"launcher config is invalid:\n{exc}") from exc
        except Exception:  # noqa: BLE001 - an unexpected internal error must not block
            continue  # skip only this check, not the ones after it


def _passthrough_args(overrides: list[str]) -> list[str]:
    """Return non-Hydra passthrough args from direct step.py invocation."""
    return [arg for arg in overrides if arg != "--" and "=" not in arg]


def _validate_passthrough(passthrough: list[str]) -> None:
    extra_args = get_non_task_args(passthrough)
    if extra_args:
        print(
            f"Error: Unknown arguments: {' '.join(extra_args)}\nOnly -t/--task flags are supported for passthrough.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _maybe_auto_squash(cfg: DictConfig, *, dry_run: bool) -> None:
    run = cfg.get("run")
    if not isinstance(run, DictConfig):
        return

    mode = str(run.get("mode", "local"))
    force_squash = bool(run.get("force_squash", False))
    maybe_auto_squash_evaluator(
        cfg,
        mode=mode,
        dry_run=dry_run,
        force_squash=force_squash,
    )


def _save_launcher_config(
    config_path: Path,
    cfg: DictConfig,
    launcher_cfg: DictConfig,
) -> Path:
    if config_path.name == "train.yaml":
        eval_path = config_path.with_name("eval.yaml")
    else:
        _, eval_path = save_eval_configs(cfg, "eval/model_eval")

    OmegaConf.save(launcher_cfg, eval_path)
    return eval_path


# =============================================================================
# Direct mode — harness in-process, Nemotron's executors do the scheduling
# =============================================================================


def run_direct(cfg: DictConfig, *, task_filters: list[str] | None = None) -> None:
    """Run each task with ``nemo-evaluator run_eval`` inside the current job.

    The launcher is bypassed entirely: this process *is* the eval worker, so
    the backend is whatever runspec profile submitted the step. That is what
    makes Lepton and Run:ai work without a launcher executor for either.
    """
    clear_artifact_cache()
    register_resolvers_from_config(cfg, artifacts_key="run", mode="pre_init")
    plain = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(plain, dict):
        raise SystemExit("config did not resolve to a mapping")

    endpoint = (plain.get("target") or {}).get("api_endpoint") or {}
    url = endpoint.get("url") if _is_set(endpoint.get("url")) else None
    model_id = endpoint.get("model_id") if _is_set(endpoint.get("model_id")) else None
    if not url or not model_id:
        raise SystemExit(
            "direct mode needs target.api_endpoint.url and .model_id "
            "(serve the model first, or use mode=launcher with deployment.type=vllm)"
        )
    model_type = str(endpoint.get("type", "completions"))
    if model_type not in {"chat", "completions"}:
        raise SystemExit(f"target.api_endpoint.type must be 'chat' or 'completions', got {model_type!r}")
    api_key_name = endpoint.get("api_key_name") if _is_set(endpoint.get("api_key_name")) else None

    # Keep the whole task entry, not just its name: a task may carry its own
    # nemo_evaluator_config, which must be merged over the global params.
    # Reducing tasks to names silently dropped per-task settings such as
    # `top_p` on adlr_mmlu.
    configured: dict[str, dict] = {}
    for entry in (plain.get("evaluation") or {}).get("tasks") or []:
        if isinstance(entry, dict):
            configured[str(entry.get("name"))] = entry
        else:
            configured[str(entry)] = {"name": str(entry)}

    wanted = task_filters or plain.get("task_filters") or None
    if wanted:
        # The requested list is authoritative and is used EXACTLY as supplied.
        # The previous intersection silently dropped any requested task that was
        # not already in evaluation.tasks, so `-t configured -t new` ran only the
        # configured one -- a silent partial run, the worst possible failure mode.
        tasks = list(dict.fromkeys(str(t) for t in wanted))
    else:
        tasks = list(configured)
    if not tasks:
        raise SystemExit("no tasks configured (evaluation.tasks is empty)")
    _validate_task_names(tasks)

    out_root = Path(str(plain.get("output_dir") or "./results"))
    _warn_if_tokenizer_missing(plain, model_id=str(model_id), model_type=model_type)
    # NB: not created here. A failed preflight must not leave a new output root
    # behind; creation happens in the execute phase below.
    dry_run = bool(plain.get("dry_run", False))
    global_evaluator_config = (plain.get("evaluation") or {}).get("nemo_evaluator_config") or {}
    global_params = (global_evaluator_config.get("config") or {}).get("params") or {}
    global_adapter = _adapter_config(global_evaluator_config)

    # Harnesses that ignore --api_key_name read the bearer token from
    # OPENAI_API_KEY (lm-eval's `local-completions`, which the sovereign MILU
    # tasks use). Without this every request 401s and summary.json reports only
    # `failed(1)`. Forward the SAME credential the config already names.
    child_env = {**os.environ}
    if api_key_name and os.environ.get(str(api_key_name)) and not child_env.get("OPENAI_API_KEY"):
        child_env["OPENAI_API_KEY"] = os.environ[str(api_key_name)]

    # `overwrite` REPLACES a task directory; it does not merge into it. Merging
    # is what produces the contaminated eval_factory_metrics.json this guard
    # exists to prevent, so bypassing the guard without clearing would preserve
    # the exact bug.
    overwrite = bool(plain.get("overwrite", False))

    # ---- PREFLIGHT -----------------------------------------------------------
    # Resolve every task's path, output state and serialised overrides BEFORE
    # touching the filesystem or launching anything. Doing this inside the
    # execution loop meant a bad task N aborted after task 1 had already run,
    # leaving spent compute and no summary.json.
    plan: list[tuple[str, Path, list[str]]] = []
    to_clear: list[Path] = []
    claimed: dict[Path, str] = {}
    planned_params: dict[str, dict] = {}
    planned_adapter: dict[str, dict] = {}
    secrets_by_task: dict[str, set[str]] = {}
    for task in tasks:
        task_out = out_root / task
        # A symlinked task directory can point anywhere, and two names can
        # resolve to the SAME directory. Either way `overwrite` would delete
        # real results, or two tasks would write into one directory while the
        # summary reported both as separate successes.
        if task_out.is_symlink():
            raise SystemExit(f"refusing to use {task_out}: task directory is a symlink")
        # Both sides must be normalised the SAME way. Mixing resolve() with
        # absolute() made a legitimate path under a symlinked parent, or one
        # containing '..', compare unequal and get rejected as an escape.
        # resolve() is non-strict, so it works for a directory not yet created.
        resolved = task_out.resolve()
        root = out_root.resolve()
        if resolved == root or root not in resolved.parents:
            raise SystemExit(f"refusing to use {task_out}: not a strict child of {out_root}")
        if resolved in claimed:
            raise SystemExit(
                f"tasks {claimed[resolved]!r} and {task!r} both resolve to {resolved}; "
                f"results would be written into one directory"
            )
        claimed[resolved] = task
        if task_out.exists() and not task_out.is_dir():
            # iterdir() would raise NotADirectoryError and bypass the clean
            # SystemExit every other preflight failure uses.
            raise SystemExit(f"refusing to use {task_out}: exists and is not a directory")
        occupied = task_out.is_dir() and any(task_out.iterdir())
        if occupied:
            if overwrite:
                # A dry run reports the plan but never deletes.
                if not dry_run:
                    to_clear.append(task_out)
            else:
                # Raised on dry runs too: a dry run is a plan check, and this is
                # exactly the kind of problem it should surface before you spend
                # GPU time on the real run.
                # eval_factory_metrics.json accumulates across runs in a reused
                # directory, so a stale failure's status codes silently blend
                # into a fresh run's. Refuse rather than produce a mixture.
                raise SystemExit(
                    f"{task_out} is not empty. Results would be mixed with a previous run "
                    f"(eval_factory_metrics.json accumulates across runs). Use a fresh "
                    f"output_dir, or set overwrite=true to REPLACE this directory."
                )
        cmd = [
            "nemo-evaluator",
            "run_eval",
            "--eval_type",
            task,
            "--model_id",
            str(model_id),
            "--model_type",
            model_type,
            "--model_url",
            str(url),
            "--output_dir",
            str(task_out),
        ]
        if api_key_name:
            cmd += ["--api_key_name", str(api_key_name)]
        # _direct_overrides raises on unserialisable values; surface that here,
        # before anything has been deleted or run.
        params = _merged_params(global_params, configured.get(task))
        adapter = _merged_adapter(global_adapter, configured.get(task))
        # The adapter writes its cache and logs here; per task, so two tasks in
        # one run cannot collide.
        adapter = {
            k: (str(task_out / "adapter") if k == "output_dir" and not _is_set(v) else v) for k, v in adapter.items()
        }
        # The adapter carries its own endpoint_type, which defaults to `chat`.
        # Leaving it disagreeing with target.api_endpoint.type on a completions
        # run is inert today, but it is a latent trap the moment an interceptor
        # branches on it. Only filled in when not set explicitly.
        if adapter and not _is_set(adapter.get("endpoint_type")):
            adapter["endpoint_type"] = model_type
        overrides = _direct_overrides(params, adapter)
        # Adapter settings can carry credentials as readily as params can.
        secrets_by_task[task] = _secret_values(params) | _secret_values(adapter)
        if overrides:
            cmd += ["--overrides", overrides]
        plan.append((task, task_out, cmd))
        planned_params[task] = params
        planned_adapter[task] = adapter

    # ---- EXECUTE -------------------------------------------------------------
    out_root.mkdir(parents=True, exist_ok=True)

    # Claim the directory BEFORE deleting anything. Two submissions racing on
    # the SAME fresh output_dir both clear the not-empty check above -- it only
    # catches a directory that already has results in it. With the claim taken
    # after the rmtree, a resubmit with overwrite=true would delete the first
    # run's IN-PROGRESS results and only then discover that run holds the lock:
    # the destruction happened before the check meant to prevent it.
    lock = _claim_output_dir(out_root, dry_run=dry_run, overwrite=overwrite)
    try:
        _execute(
            plan=plan,
            plan_context=(plain, out_root, dry_run, overwrite),
            to_clear=to_clear,
            planned_params=planned_params,
            planned_adapter=planned_adapter,
            secrets_by_task=secrets_by_task,
            child_env=child_env,
            url=str(url),
            model_id=str(model_id),
            model_type=model_type,
            api_key_name=str(api_key_name) if api_key_name else None,
            summary_name="summary.dry-run.json" if dry_run else "summary.json",
        )
    finally:
        # Released on ANY exit. Removing it only on the normal path meant a
        # Popen failure, an unwritable log, or any other exception left the
        # claim behind and blocked every subsequent run until someone deleted
        # it by hand.
        if lock is not None:
            lock.unlink(missing_ok=True)


def _execute(
    *,
    plan: list[tuple[str, Path, list[str]]],
    plan_context: tuple[dict, Path, bool, bool],
    to_clear: list[Path],
    planned_params: dict[str, dict],
    planned_adapter: dict[str, dict],
    secrets_by_task: dict[str, set[str]],
    child_env: dict[str, str],
    url: str,
    model_id: str,
    model_type: str,
    api_key_name: str | None,
    summary_name: str,
) -> None:
    """Run the planned tasks. Split out so the caller can release the claim."""
    plain, out_root, dry_run, overwrite = plan_context
    for stale in to_clear:
        shutil.rmtree(stale)

    # Written BEFORE the tasks run. A job that is preempted or OOM-killed
    # mid-suite still leaves a record of what it was running and against what;
    # per-task outcomes are summary.json's job, not this file's.
    manifest_path = out_root / ("run_manifest.dry-run.json" if dry_run else "run_manifest.json")
    manifest_path.write_text(
        json.dumps(
            _run_manifest(
                plain=plain,
                out_root=out_root,
                summary_name=summary_name,
                plan=plan,
                planned_params=planned_params,
                planned_adapter=planned_adapter,
                model_id=str(model_id),
                url=str(url),
                model_type=model_type,
                api_key_name=str(api_key_name) if api_key_name else None,
                dry_run=dry_run,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    print(f"eval_manifest: {manifest_path}")

    summary_path = out_root / summary_name
    summary: dict[str, str] = {}
    failures: dict[str, str] = {}
    for task, task_out, cmd in plan:
        task_out.mkdir(parents=True, exist_ok=True)
        print(f"[{task}] {shlex.join(_safe_cmd(cmd, str(url), secrets_by_task.get(task, set())))}", flush=True)
        if dry_run:
            summary[task] = "dry-run"
            continue
        # Persist the harness's own output. In a reaped pod the container log is
        # gone, and summary.json records only `failed(N)` -- which is not enough
        # to tell a 401 from a timeout from a bad tokenizer.
        log_path = task_out / "harness.log"
        # Tee incrementally rather than buffering: an eval can emit hundreds of
        # MB, and a run that dies mid-way must still leave a usable log.
        with log_path.open("wb") as log_file:
            proc = subprocess.Popen(  # noqa: S603
                cmd, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            assert proc.stdout is not None
            # read1() returns whatever is available rather than waiting for a
            # full buffer, so a child that prints and flushes is visible at once
            # and nothing is lost if the pod dies mid-run.
            for chunk in iter(lambda: proc.stdout.read1(8192), b""):
                log_file.write(chunk)
                log_file.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()
            returncode = proc.wait()
        summary[task] = "ok" if returncode == 0 else f"failed({returncode})"
        # Rewritten after every task. Written once at the end, a job preempted
        # during task 3 of 4 left a manifest and three logs but no summary --
        # the one file the tooling reads. Preemption is an expected path, not
        # an exotic one: it is what a preemptible queue does.
        summary_path.write_text(json.dumps(summary, indent=2))
        if returncode != 0:
            print(f"[{task}] FAILED with exit {returncode}; log: {log_path}", file=sys.stderr)
            failures[task] = _log_tail(log_path)

    # Final write: dry runs never entered the loop, and a completed run gets a
    # last consistent flush. `summary_name` keeps a dry run from clobbering a
    # real summary.json from a previous run.
    summary_path.write_text(json.dumps(summary, indent=2))
    failures_path = out_root / "failures.txt"
    if failures:
        # One small file with every failure's tail. Otherwise the cause is
        # buried in a per-task harness.log that, on a cluster whose job logs
        # are unreadable after termination, costs a whole extra job to reach.
        failures_path.write_text("\n\n".join(f"===== {task} =====\n{tail}" for task, tail in failures.items()))
        print(f"eval_failures: {failures_path}", file=sys.stderr)
    else:
        # A previous run's failures.txt beside a clean summary.json reads as a
        # failure that did not happen.
        failures_path.unlink(missing_ok=True)
    print(f"eval_results: {out_root}")
    print(f"eval_summary: {summary_path}")
    for task, state in summary.items():
        print(f"  {task}: {state}")
    if any(v.startswith("failed") for v in summary.values()):
        raise SystemExit(1)


# The manifest is a stable, provider-neutral record of WHAT was run. Bump this
# when a field changes meaning or is removed; adding a field does not require it.
_MANIFEST_SCHEMA_VERSION = 1


def _redact_url(url: str) -> str:
    """Endpoint URL with credentials and query string removed.

    Tokens turn up in URLs more often than anyone admits -- as userinfo
    (``https://user:token@host/...``) or as a query parameter. The manifest is
    meant to be attached to a report, so it must not be a way to leak one.
    """
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return "<unparseable>"
    if not parts.scheme and not parts.netloc:
        return url
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    redacted = parts._replace(netloc=host, query="", fragment="")
    if parts.username or parts.password:
        # Say so rather than silently returning a different URL than was used.
        return urllib.parse.urlunsplit(redacted) + "  (credentials redacted)"
    return urllib.parse.urlunsplit(redacted)


# A digest is `sha256:` plus exactly 64 hex characters. Matching only on the
# substring `@sha256:` let `image@sha256:abc` be certified as pinned, which is
# worse than no check at all: the manifest would vouch for an unreproducible run.
_DIGEST_RE = re.compile(r"@sha256:[0-9a-f]{64}$")


def _is_digest_pinned(image: str | None) -> bool:
    return bool(image and _DIGEST_RE.search(image.strip()))


# Keys whose VALUE is a credential. Matched on the whole key, so `tokenizer`
# (which contains "token") is not caught, and `*_name` is excluded because it
# holds the NAME of a credential variable, not the credential.
_SENSITIVE_KEY_RE = re.compile(
    r"^(?!.*_name$).*(?:^|_)(token|api_?key|secret|password|passwd|credential|bearer)s?$",
    re.IGNORECASE,
)

_REDACTED = "***"


def _walk_params(params: object, *, redact: bool) -> tuple[object, set[str]]:
    """Recursively redact credential-valued params, collecting what was removed.

    Returns the (optionally redacted) structure and the set of secret values, so
    the same values can be scrubbed from the command line, where they appear
    inside a single `--overrides key=value` argv element rather than on their own.
    """
    secrets: set[str] = set()
    if isinstance(params, dict):
        out: dict = {}
        for key, value in params.items():
            if _SENSITIVE_KEY_RE.match(str(key)) and isinstance(value, (str, int, float)):
                text = str(value)
                if text:
                    secrets.add(text)
                out[key] = _REDACTED if redact else value
                continue
            child, child_secrets = _walk_params(value, redact=redact)
            out[key] = child
            secrets |= child_secrets
        return out, secrets
    if isinstance(params, (list, tuple)):
        items = []
        for value in params:
            child, child_secrets = _walk_params(value, redact=redact)
            items.append(child)
            secrets |= child_secrets
        return items, secrets
    return params, secrets


def _redact_params(params: dict) -> dict:
    redacted, _ = _walk_params(params, redact=True)
    return redacted  # type: ignore[return-value]


def _secret_values(params: object) -> set[str]:
    _, secrets = _walk_params(params, redact=False)
    return secrets


def _safe_cmd(cmd: list[str], url: str, secrets: set[str] = frozenset()) -> list[str]:
    """The command with the raw endpoint URL replaced by its redacted form.

    The URL is passed as ``--model_url``, so both the echoed command line and
    the manifest's recorded command would otherwise carry any token embedded in
    the URL's userinfo or query string straight into a log file. `secrets` adds
    any credential-valued parameter, which lands in `--overrides` the same way.
    """
    safe = _redact_url(url)
    out = [safe if part == url else part for part in cmd]
    # Substring replacement: a credential passed as `--overrides
    # config.params.extra.api_key=xyz` is embedded in one argv element.
    for secret in sorted(secrets, key=len, reverse=True):
        out = [part.replace(secret, _REDACTED) for part in out]
    return out


def _harness_versions() -> dict[str, str]:
    """Versions of the harness packages present in THIS container.

    A task name means whatever the image says it means, so the image reference
    alone is not enough when the tag is mutable.
    """
    from importlib.metadata import PackageNotFoundError, version

    found: dict[str, str] = {}
    for pkg in ("nemo-evaluator", "nemo-evaluator-launcher", "lm-eval"):
        try:
            found[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
        except Exception:  # noqa: BLE001 - a broken dist must not fail the run
            continue
    return found


def _code_provenance() -> dict[str, object]:
    """Which Nemotron code produced this run.

    The harness versions identify what computed the score; this identifies the
    orchestration around it -- the config plumbing, parameter merging and
    overrides that decide what the harness is actually asked to do. Fields are
    always present, null when unavailable, so a reader can tell "not a git
    checkout" from "provenance silently omitted".
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        pkg_version: str | None = version("nemotron")
    except PackageNotFoundError:
        pkg_version = None
    except Exception:  # noqa: BLE001 - a broken dist must not fail the run
        pkg_version = None

    git_sha: str | None = None
    git_dirty: bool | None = None
    try:
        repo = Path(__file__).resolve().parent
        common: dict = {"cwd": repo, "capture_output": True, "text": True, "timeout": 10}
        rev = subprocess.run(["git", "rev-parse", "HEAD"], check=False, **common)  # noqa: S603, S607
        if rev.returncode == 0 and rev.stdout.strip():
            git_sha = rev.stdout.strip()
            status = subprocess.run(["git", "status", "--porcelain"], check=False, **common)  # noqa: S603, S607
            # Uncommitted changes mean the SHA does not describe what ran.
            git_dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    except Exception:  # noqa: BLE001 - no git, not a checkout, or a timeout
        pass

    return {"nemotron_version": pkg_version, "git_sha": git_sha, "git_dirty": git_dirty}


def _run_manifest(
    *,
    plain: dict,
    out_root: Path,
    summary_name: str,
    plan: list[tuple[str, Path, list[str]]],
    planned_params: dict[str, dict],
    planned_adapter: dict[str, dict],
    model_id: str,
    url: str,
    model_type: str,
    api_key_name: str | None,
    dry_run: bool,
) -> dict:
    """Provider-neutral record of what this run was, written before it starts.

    Everything needed to say "this number came from this model, over this
    endpoint, with these parameters, using this harness" -- without which a
    published score is not reproducible and not auditable.
    """
    # `harness_image` first: the generic runner strips `run.env` before invoking
    # the step, so `run.env.container_image` is populated in a local run and
    # ABSENT in every submitted one -- which silently recorded image: null and
    # image_pinned_by_digest: false for exactly the runs worth citing.
    image = str(plain.get("harness_image") or ((plain.get("run") or {}).get("env") or {}).get("container_image") or "")
    image = image or None
    return {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "mode": "direct",
        "dry_run": dry_run,
        "model": {
            "id": model_id,
            # Redacted: this file is meant to be publishable alongside results.
            "endpoint_url": _redact_url(url),
            "endpoint_type": model_type,
            # The NAME of the credential variable, never its value.
            "api_key_name": api_key_name,
        },
        "harness": {
            "image": image,
            # A mutable tag makes the run unreproducible; record whether the
            # reference actually pins anything.
            "image_pinned_by_digest": _is_digest_pinned(image),
            "packages": _harness_versions(),
        },
        # The harness computed the score; this code decided what it was asked.
        "code": _code_provenance(),
        "tasks": {
            task: {
                "output_dir": str(task_out),
                # Fully merged (global + per-task), i.e. what the task actually ran with.
                "params": _redact_params(planned_params.get(task, {})),
                "adapter_config": _redact_params(planned_adapter.get(task, {})),
                "command": _safe_cmd(
                    cmd,
                    url,
                    # Both namespaces: a credential is as likely in adapter_config.
                    _secret_values(planned_params.get(task, {})) | _secret_values(planned_adapter.get(task, {})),
                ),
            }
            for task, task_out, cmd in plan
        },
        "output_dir": str(out_root),
        "summary_path": str(out_root / summary_name),
    }


def _describe_claim(lock: Path) -> str:
    """Describe the holder of a claim, including its age.

    A preempted pod leaves its claim behind, and every retry then fails until
    someone deletes it by hand. The payload already records when the run
    started, so say so: a claim hours old on a preemptible queue is almost
    certainly stale, and the reader can decide without guessing.
    """
    try:
        raw = lock.read_text().strip()
    except OSError:
        return "<unreadable>"
    try:
        held = json.loads(raw)
        started = _dt.datetime.fromisoformat(str(held["started_at"]))
        age = _dt.datetime.now(_dt.timezone.utc) - started
        hours = age.total_seconds() / 3600
        note = " -- likely stale, no eval runs this long" if hours >= 24 else ""
        return f"pid {held.get('pid')} on {held.get('host')}, started {started.isoformat()} ({hours:.1f}h ago){note}"
    except Exception:  # noqa: BLE001 - a malformed claim must still be reported
        return raw


def _claim_output_dir(out_root: Path, *, dry_run: bool, overwrite: bool) -> Path | None:
    """Take an exclusive claim on ``out_root`` for the duration of the run.

    The not-empty check refuses a directory that already HAS results. It cannot
    catch two jobs starting against the same empty directory -- which is what
    happens when a submission is retried because the first was believed to have
    failed. Both pass preflight, both write into one directory, and the summary
    reports whichever finished last.

    ``O_CREAT | O_EXCL`` is atomic on POSIX and on the shared filesystems these
    results land on. A dry run takes no claim: it writes nothing that could
    collide, and it must never block a real run.
    """
    if dry_run:
        return None
    lock = out_root / ".nemotron-eval.lock"
    payload = json.dumps(
        {
            "pid": os.getpid(),
            "host": os.uname().nodename,
            "started_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        }
    )
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        held = _describe_claim(lock)
        if overwrite:
            # `overwrite` is about replacing stale RESULTS, not about racing a
            # live job. Say so rather than silently proceeding.
            raise SystemExit(
                f"{out_root} is claimed by another run ({held}). overwrite=true replaces stale "
                f"results; it does not override a live claim. Use a different output_dir, or "
                f"delete {lock} if that run is definitely gone."
            ) from None
        raise SystemExit(
            f"{out_root} is claimed by another run ({held}). Two runs writing here would "
            f"interleave and the summary would report only the last one to finish. Use a "
            f"different output_dir, or delete {lock} if that run is definitely gone."
        ) from None
    with os.fdopen(fd, "w") as handle:
        handle.write(payload)
    return lock


def _warn_if_tokenizer_missing(plain: dict, *, model_id: str, model_type: str) -> None:
    """Warn when no client-side tokenizer is configured, for any endpoint type.

    lm-evaluation-harness constructs a tokenizer with
    ``AutoTokenizer.from_pretrained`` and, when none is given, falls back to the
    ``model=`` value -- which here is the endpoint's served-model-name, not an
    HF repo. The result is a ``RepositoryNotFoundError`` naming your
    served-model-name as a missing repo, tens of lines into a Hub traceback,
    with nothing to connect it to the tokenizer setting you left unset.

    This is NOT limited to completions. `local-chat-completions` loads a
    tokenizer too: `ifeval`, `mmlu_instruct` and `gsm8k_cot_instruct` all ship
    `extra.tokenizer` and thread it through their command templates. An earlier
    version of this check returned early for chat endpoints and so could never
    fire on the runs that needed it most.

    A warning rather than an error: a harness that needs no client-side
    tokenizer is possible, and refusing to run would be worse than saying so.
    """
    params = (((plain.get("evaluation") or {}).get("nemo_evaluator_config") or {}).get("config") or {}).get(
        "params"
    ) or {}
    if _is_set((params.get("extra") or {}).get("tokenizer")):
        return
    print(
        f"Warning: no tokenizer configured (extra.tokenizer / EVAL_TOKENIZER is unset)\n"
        f"  for a {model_type} endpoint. lm-evaluation-harness loads a tokenizer\n"
        f"  client-side for chat AND completions tasks, and with none given it falls\n"
        f"  back to loading '{model_id}' as a HuggingFace repo -- which fails with\n"
        f"  RepositoryNotFoundError unless that happens to be a real repo id.\n"
        f"  Set EVAL_TOKENIZER to the model's HF repo id or a local snapshot path.\n"
        f"  A tokenizer-extended checkpoint MUST use its own tokenizer, not the base one.",
        file=sys.stderr,
        flush=True,
    )


def _log_tail(log_path: Path, *, lines: int = 40) -> str:
    """Last lines of a harness log, for the run-level failures file."""
    try:
        text = log_path.read_text(errors="replace")
    except OSError as exc:
        return f"<could not read {log_path}: {exc}>"
    return "\n".join(text.splitlines()[-lines:])


def _validate_task_names(tasks: list[str]) -> None:
    """Task names become directory names under ``output_dir``.

    An empty or whitespace name resolves to ``output_dir`` itself, so with
    ``overwrite=true`` the subsequent rmtree would delete the entire output
    root. ``Path("").name == ""`` means a naive `name == Path(name).name` check
    passes, so test for emptiness explicitly.
    """
    bad = [t for t in tasks if not t or not t.strip() or t != Path(t).name or t in (".", "..")]
    if bad:
        raise SystemExit(f"task names are used as directory names; refusing: {bad!r}")


def _adapter_config(nemo_evaluator_config: dict | None) -> dict:
    return (((nemo_evaluator_config or {}).get("target") or {}).get("api_endpoint") or {}).get("adapter_config") or {}


def _merged_adapter(global_adapter: dict, task_entry: dict | None) -> dict:
    """Suite-wide adapter settings with the task's own layered on top."""
    merged = dict(global_adapter or {})
    merged.update(_adapter_config((task_entry or {}).get("nemo_evaluator_config")))
    return merged


def _merged_params(global_params: dict, task_entry: dict | None) -> dict:
    """Global params with the task's own nemo_evaluator_config layered on top.

    A per-task setting must win over the suite-wide default; otherwise a mixed
    suite runs some tasks with the wrong generation parameters and says nothing.
    """
    merged = dict(global_params or {})
    task_params = (((task_entry or {}).get("nemo_evaluator_config") or {}).get("config") or {}).get("params") or {}
    for key, value in task_params.items():
        if key == "extra" and isinstance(value, dict):
            merged["extra"] = {**(merged.get("extra") or {}), **value}
        else:
            merged[key] = value
    return merged


_KNOWN_PARAM_KEYS = {
    "task",
    "limit_samples",
    "parallelism",
    "request_timeout",
    "max_retries",
    "temperature",
    "top_p",
    "max_new_tokens",
    "extra",
}


def _direct_overrides(params: dict, adapter: dict | None = None) -> str:
    """Translate the step's config into nemo-evaluator ``--overrides``.

    Two namespaces are forwarded:

    ``config.params.*``
        generation and run settings.
    ``target.api_endpoint.adapter_config.*``
        the adapter proxy that sits between the harness and the endpoint.
        Forwarding this is not optional: without ``use_caching`` the harness
        holds every request and response for a task in memory, and a task past
        roughly 30k requests dies client-side on a large CPU shape while the
        endpoint is still answering 200 to everything. Dropping this namespace
        put a hard ceiling on which benchmarks direct mode could run at all.

    Everything else in ``nemo_evaluator_config`` is launcher-shaped and has no
    direct analogue.
    """
    unknown = sorted(set(params or {}) - _KNOWN_PARAM_KEYS)
    if unknown:
        # Silently dropping these meant a task ran with different generation
        # settings than the config asked for, and said nothing.
        raise SystemExit(
            f"unsupported nemo_evaluator_config params: {unknown}. "
            f"Supported: {sorted(_KNOWN_PARAM_KEYS - {'extra'})}. "
            f"Pass harness-specific arguments under `extra:`."
        )
    pairs: list[str] = []
    simple = {
        # `task` selects the language/subset for multi-subset harnesses. Dropping
        # it turns `mmlu_prox_completions` from the intended single language into
        # the full 29-language run (~350k requests) -- which does not error, it
        # just never finishes.
        "task": "config.params.task",
        "limit_samples": "config.params.limit_samples",
        "parallelism": "config.params.parallelism",
        "request_timeout": "config.params.request_timeout",
        "max_retries": "config.params.max_retries",
        "temperature": "config.params.temperature",
        "top_p": "config.params.top_p",
        "max_new_tokens": "config.params.max_new_tokens",
    }
    for key, dotted in simple.items():
        val = params.get(key)
        if _is_set(val):
            pairs.append(f"{dotted}={_override_value(val)}")
    for key, val in (params.get("extra") or {}).items():
        if _is_set(val):
            pairs.append(f"config.params.extra.{key}={_override_value(val)}")
    # Not allowlisted: the adapter's interceptor set is open-ended and version
    # dependent, and an unknown key here is inert rather than silently changing
    # how the model is scored.
    for key, val in (adapter or {}).items():
        if _is_set(val):
            pairs.append(f"target.api_endpoint.adapter_config.{key}={_override_value(val)}")
    return ",".join(pairs)


def _override_value(val: object) -> str:
    """Render one override value for the comma-joined ``--overrides`` string.

    The override expression is comma-separated, so a value containing a comma
    (or a list/dict) would be parsed as extra key=value pairs and corrupt every
    following override. Reject those rather than emit something that silently
    means the wrong thing.
    """
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (list, tuple, dict, set)):
        raise SystemExit(
            f"cannot pass {type(val).__name__} through --overrides (comma-joined): {val!r}. "
            "Put collection-valued params in the harness config instead."
        )
    text = str(val)
    if "," in text:
        raise SystemExit(f"override value contains a comma and would corrupt the override list: {text!r}")
    return text


def _is_set(val: object) -> bool:
    """True unless the value is unset or an unresolved null sentinel.

    ``${oc.env:VAR,null}`` yields the *string* ``"null"`` when VAR is unset, and
    forwarding that verbatim makes the harness try to load a tokenizer named
    ``None`` ("Unrecognized model in None"). Treat the usual spellings as unset.
    """
    if val is None:
        return False
    return str(val).strip().lower() not in {"null", "none", ""}
