# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Nemotron BYOB CLI orchestration for BFCL evaluation."""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
from typing import Any, Final, Literal

import yaml  # type: ignore[import-untyped]
from omegaconf import OmegaConf
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    ValidationError,
    model_validator,
)

from nemotron.steps.byob.runtime.benchmark_families.base import BenchmarkRunResult
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.config import (
    BfclEvalConfig,
    load_eval_config,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.contamination import (
    evaluate_contamination,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.error_taxonomy import (
    FATAL_EVAL_ERROR_CODES,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.eval_runner import (
    BfclEvalRunResult,
    BfclTraceEvalRunResult,
    run_declared_eval_sync,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.nemo_native_adapter import (
    NEMO_NATIVE_ADAPTER_VERSION,
    NemoNativeAdapterConfig,
    install_native_framework,
    launcher_task_entry,
    native_bundle_tree_hash,
    native_framework_distribution,
    write_native_adapter_config,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_verification import (
    verify_eval_source,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.nemo_evaluator_export import (
    NEMO_EVALUATOR_ROOT,
)
from nemotron.steps.eval.model_eval.runtime import (
    ModelEvalDependencyError,
    launch_model_eval_config,
)

BFCL_EVAL_CLI_VERSION: Final = "1.0"


class BfclEvalCliError(RuntimeError):
    """A stable CLI-boundary failure with a process exit status."""

    code = "eval_cli_invalid"
    cli_exit_code = 2

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        cli_exit_code: int | None = None,
    ) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        if cli_exit_code is not None:
            self.cli_exit_code = cli_exit_code


class BfclEvalCliRuntimeError(BfclEvalCliError):
    code = "eval_cli_runtime_failed"
    cli_exit_code = 3


class BfclEvalCliArtifactConflictError(BfclEvalCliError):
    code = "eval_cli_artifact_conflict"
    cli_exit_code = 7


class BfclEvalCliFrameworkNotInstalledError(BfclEvalCliError):
    code = "eval_cli_framework_not_installed"
    cli_exit_code = 3


class BfclEvalCliFrameworkVersionMismatchError(BfclEvalCliError):
    code = "eval_cli_framework_version_mismatch"
    cli_exit_code = 3


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BfclLauncherCliConfig(_Frozen):
    bundle_root: Path | None = None
    native_output_dir: Path
    adapter_config_path: Path
    framework_build_dir: Path
    task_config_path: Path
    launcher_base_config_path: Path
    launcher_config_path: Path
    launcher_output_dir: Path
    dataset_mount_path: StrictStr = "/datasets/bfcl"
    container: StrictStr | None = None
    evaluation_mounts: dict[StrictStr, StrictStr] = Field(default_factory=dict)
    submit: StrictBool = False

    @model_validator(mode="after")
    def _mounts(self) -> BfclLauncherCliConfig:
        for host, container in self.evaluation_mounts.items():
            host_path = Path(host)
            container_path = Path(container)
            if not host_path.is_absolute() or not container_path.is_absolute():
                raise ValueError("Launcher evaluation mounts must map absolute paths")
            if host_path != container_path:
                raise ValueError(
                    "Launcher evaluation_mounts must be identity mounts because the "
                    "adapter and resolved eval config contain absolute paths"
                )
        if self.container is not None and not self.evaluation_mounts:
            raise ValueError(
                "a Launcher evaluation container requires explicit evaluation_mounts "
                "for the adapter, eval source, oracle resources, and output trees"
            )
        return self


class BfclEvalCliConfig(_Frozen):
    """Operational envelope kept outside the hash-bearing eval config."""

    schema_version: Literal["1.0"] = BFCL_EVAL_CLI_VERSION
    family: Literal["bfcl"] = "bfcl"
    stage: Literal["eval"] = "eval"
    eval_config_path: Path
    execution_backend: Literal["direct", "nemo_launcher"] = "direct"
    output_format: Literal["human", "json"] = "human"
    probe_oracle: StrictBool = True
    dry_run: StrictBool = False
    launcher: BfclLauncherCliConfig | None = None

    @model_validator(mode="after")
    def _backend_options(self) -> BfclEvalCliConfig:
        if self.execution_backend == "nemo_launcher" and self.launcher is None:
            raise ValueError("nemo_launcher backend requires launcher options")
        if self.execution_backend == "direct" and self.launcher is not None:
            raise ValueError("direct backend may not carry launcher options")
        return self


def _resolved_path(value: Any, *, base: Path, field: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise BfclEvalCliError(f"{field} must be a filesystem path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_bfcl_eval_cli_config(path: str | os.PathLike[str]) -> BfclEvalCliConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise BfclEvalCliError(f"CLI orchestration config does not exist: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise BfclEvalCliError(
            "CLI orchestration config is not valid UTF-8 YAML"
        ) from exc
    if not isinstance(raw, dict):
        raise BfclEvalCliError("CLI orchestration config must contain one object")
    raw["eval_config_path"] = _resolved_path(
        raw.get("eval_config_path"),
        base=source.parent,
        field="eval_config_path",
    )
    launcher = raw.get("launcher")
    if launcher is not None:
        if not isinstance(launcher, dict):
            raise BfclEvalCliError("launcher must contain one object")
        for field in (
            "native_output_dir",
            "adapter_config_path",
            "framework_build_dir",
            "task_config_path",
            "launcher_base_config_path",
            "launcher_config_path",
            "launcher_output_dir",
        ):
            launcher[field] = _resolved_path(
                launcher.get(field),
                base=source.parent,
                field=f"launcher.{field}",
            )
        if launcher.get("bundle_root") is not None:
            launcher["bundle_root"] = _resolved_path(
                launcher["bundle_root"],
                base=source.parent,
                field="launcher.bundle_root",
            )
    try:
        return BfclEvalCliConfig.model_validate(raw)
    except ValidationError as exc:
        raise BfclEvalCliError(
            f"CLI orchestration config violates schema {BFCL_EVAL_CLI_VERSION}: "
            f"{_schema_violations(exc)}"
        ) from exc


def _schema_violations(exc: ValidationError) -> str:
    """Name the offending fields, because that is what an operator has to edit."""
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
        for error in exc.errors()
    )


# Exit statuses are a published contract, so they are assigned per registered
# error code rather than inferred from substrings of the code name. Inference put
# sibling codes in contradictory buckets: a candidate cache conflict is an artifact
# conflict, not a candidate failure, and a drifted oracle pack is a stale
# publication rather than live oracle infrastructure. The import-time totality
# check below makes a newly registered code a build failure instead of a silent
# reclassification.
_EXIT_CODE_GROUPS: Final[tuple[tuple[int, tuple[str, ...]], ...]] = (
    (
        2,  # The operator declared something impossible; nothing was executed.
        (
            "eval_config_invalid",
            "eval_config_schema_invalid",
            "eval_config_path_invalid",
            "candidate_identity_invalid",
            "candidate_revision_mutable",
            "secret_in_eval_config",
            "unsupported_eval_mode",
            "eval_runner_mode_unsupported",
            "eval_executable_scoring_policy_unsupported",
            "eval_trace_scoring_policy_unsupported",
            "eval_cli_invalid",
        ),
    ),
    (
        4,  # The eval would have measured a contaminated or leaked task set.
        (
            "eval_contamination_invalid",
            "eval_contamination_candidate_exposed",
            "eval_contamination_unresolved",
            "eval_contamination_empty_task_set",
            "eval_contamination_task_set_inconsistent",
            "eval_contamination_plan_drift",
            "eval_source_model_exposure_invalid",
            "eval_conversation_answer_key_leak",
        ),
    ),
    (
        5,  # The candidate endpoint, not the harness, made the run impossible.
        (
            "eval_candidate_client_invalid",
            "eval_candidate_credentials_missing",
            "eval_candidate_authentication_failed",
            "eval_candidate_request_invalid",
            "eval_candidate_provider_extension_invalid",
            "eval_candidate_response_invalid",
        ),
    ),
    (
        6,  # Live oracle or assertion infrastructure failed.
        (
            "eval_oracle_session_failed",
            "eval_oracle_reset_failed",
            "eval_oracle_call_failed",
            "eval_oracle_state_failed",
            "eval_assertion_infrastructure_failed",
        ),
    ),
    (
        7,  # An immutable artifact already exists with different evidence.
        (
            "eval_artifact_invalid",
            "eval_publication_policy_violation",
            "eval_candidate_cache_invalid",
            "eval_candidate_cache_conflict",
            "eval_tool_trace_cache_invalid",
            "eval_tool_trace_cache_conflict",
            "eval_cli_artifact_conflict",
        ),
    ),
    (
        3,  # Setup, source verification, scoring, or aggregation refused the run.
        (
            "eval_source_invalid",
            "eval_source_manifest_invalid",
            "eval_source_manifest_drift",
            "eval_source_benchmark_hash_mismatch",
            "eval_source_benchmark_schema_mismatch",
            "eval_source_publication_invalid",
            "eval_source_task_index_invalid",
            "eval_source_oracle_pack_drift",
            "eval_source_oracle_resource_mismatch",
            "eval_source_translation_lineage_invalid",
            "eval_source_changed_during_eval",
            "eval_conversation_invalid",
            "eval_conversation_script_invalid",
            "eval_conversation_unauthorized",
            "eval_conversation_transition_invalid",
            "eval_executable_invalid",
            "eval_executable_projection_invalid",
            "eval_executable_unauthorized",
            "eval_executable_scoring_invalid",
            "eval_executable_evidence_mismatch",
            "eval_executable_aggregation_invalid",
            "eval_trace_scoring_invalid",
            "eval_trace_evidence_mismatch",
            "eval_trace_aggregation_invalid",
            "eval_runner_invalid",
            "eval_nemo_adapter_invalid",
            "eval_cli_runtime_failed",
            "eval_cli_framework_not_installed",
            "eval_cli_framework_version_mismatch",
        ),
    ),
)
EVAL_CLI_EXIT_CODES: Final[dict[str, int]] = {
    code: status for status, codes in _EXIT_CODE_GROUPS for code in codes
}

if set(EVAL_CLI_EXIT_CODES) != set(FATAL_EVAL_ERROR_CODES):
    _unmapped = sorted(FATAL_EVAL_ERROR_CODES - set(EVAL_CLI_EXIT_CODES))
    _unknown = sorted(set(EVAL_CLI_EXIT_CODES) - FATAL_EVAL_ERROR_CODES)
    raise RuntimeError(
        "the CLI exit-status contract must cover the eval error taxonomy exactly: "
        f"unmapped={_unmapped}, unregistered={_unknown}"
    )


def _wrap_runtime_failure(exc: Exception) -> BfclEvalCliError:
    exception_code = str(getattr(exc, "code", ""))
    code = (
        exception_code
        if exception_code in EVAL_CLI_EXIT_CODES
        else BfclEvalCliRuntimeError.code
    )
    return BfclEvalCliError(
        str(exc),
        code=code,
        cli_exit_code=EVAL_CLI_EXIT_CODES[code],
    )


def _write_immutable_text(path: Path, text: str) -> None:
    payload = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise BfclEvalCliArtifactConflictError(
                f"{path.name} already exists with different orchestration evidence"
            )
        return
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        if not path.is_file() or path.read_bytes() != payload:
            raise BfclEvalCliArtifactConflictError(
                f"{path.name} appeared concurrently with different evidence"
            ) from exc


def _launcher_document(
    base_path: Path,
    *,
    task: dict[str, Any],
    model_id: str,
    target_url: str,
    api_key_name: str,
    output_dir: Path,
    dry_run: bool,
    evaluation_mounts: dict[str, str],
) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise BfclEvalCliError("Launcher base config is not valid UTF-8 YAML") from exc
    if not isinstance(raw, dict):
        raise BfclEvalCliError("Launcher base config must contain one object")
    raw["dry_run"] = dry_run
    raw["output_dir"] = str(output_dir)
    raw["task_filters"] = [task["name"]]
    execution = raw.setdefault("execution", {})
    execution["output_dir"] = str(output_dir)
    mounts = execution.setdefault("mounts", {})
    mounts.setdefault("evaluation", {}).update(evaluation_mounts)
    # A base config that deploys a checkpoint owns the endpoint: the adapter's
    # `launcher` target binding exists precisely so the URL can arrive at runtime.
    # Forcing `deployment.type: none` here would silently discard that base config
    # and pin the run to the eval config's already-served candidate instead.
    deployment = raw.setdefault("deployment", {})
    deployment.setdefault("type", "none")
    endpoint = raw.setdefault("target", {}).setdefault("api_endpoint", {})
    endpoint.update({"api_key_name": api_key_name, "type": "chat"})
    if deployment["type"] == "none":
        endpoint.update({"model_id": model_id, "url": target_url})
    else:
        # Launcher 0.2.6 rejects both fields for a managed deployment and fills
        # them from the endpoint it creates. Remove stale values inherited from a
        # hosted-endpoint base config as well as values this orchestrator controls.
        endpoint.pop("model_id", None)
        endpoint.pop("url", None)
    raw.setdefault("evaluation", {})["tasks"] = [task]
    if _needs_evaluation_mounts(raw, task) and not mounts.get("evaluation"):
        raise BfclEvalCliError(
            "a containerized Launcher evaluation requires explicit evaluation_mounts "
            "for the adapter config, verified eval source, oracle resources, and "
            "output trees; the bundle keeps its own dataset_mount_path contract"
        )
    return raw


def _needs_evaluation_mounts(document: dict[str, Any], task: dict[str, Any]) -> bool:
    """Detect a containerized evaluation wherever the merged config declares it.

    Keying this on the CLI's own `container` override would miss a base config that
    names the evaluation image globally or per task, and those runs are exactly the
    ones whose absolute host paths are invisible without a mount.
    """
    if task.get("container") is not None:
        return True
    evaluation = document.get("evaluation")
    return isinstance(evaluation, dict) and evaluation.get("container") is not None


def _assert_bundle_untouched(bundle_root: Path, launcher: BfclLauncherCliConfig) -> None:
    """Keep every orchestration artifact outside the immutable exported bundle.

    The bundle is verified by exact file set, so writing one extra YAML into it
    would make the very next verification of that publication fail.
    """
    written = {
        "adapter_config_path": launcher.adapter_config_path,
        "framework_build_dir": launcher.framework_build_dir,
        "task_config_path": launcher.task_config_path,
        "launcher_config_path": launcher.launcher_config_path,
        "launcher_output_dir": launcher.launcher_output_dir,
        "native_output_dir": launcher.native_output_dir,
    }
    for field, path in written.items():
        if _paths_overlap(path, bundle_root):
            raise BfclEvalCliError(
                f"launcher.{field} may not contain or be contained by the "
                "immutable native bundle"
            )


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def _required_evaluation_paths(
    cli: BfclEvalCliConfig,
    config: BfclEvalConfig,
) -> dict[str, Path]:
    """Return every host path the native framework opens or writes at runtime."""
    launcher = cli.launcher
    if launcher is None:
        raise BfclEvalCliError("nemo_launcher backend requires launcher options")
    required = {
        "adapter_config": launcher.adapter_config_path,
        "eval_config": cli.eval_config_path,
        "source_publication": config.source.publication_dir,
        "bfcl_output": config.outputs.output_dir,
        "native_output": launcher.native_output_dir,
        "launcher_output": launcher.launcher_output_dir,
    }
    oracle = config.source.oracle
    if oracle is not None:
        required.update(
            {
                "oracle_pack_manifest": oracle.pack_manifest.path,
                "oracle_execution_resource": oracle.execution_resource.path,
            }
        )
    return required


def _validate_evaluation_mount_coverage(
    mounts: dict[str, str],
    required: dict[str, Path],
) -> None:
    roots = tuple(Path(host).resolve() for host in mounts)
    uncovered = {
        label: str(path)
        for label, path in required.items()
        if not any(
            path.resolve() == root or root in path.resolve().parents for root in roots
        )
    }
    if uncovered:
        detail = ", ".join(
            f"{label}={path}" for label, path in sorted(uncovered.items())
        )
        raise BfclEvalCliError(
            "Launcher evaluation_mounts do not expose every absolute runtime path: "
            f"{detail}"
        )


def _run_nemo_launcher(
    cli: BfclEvalCliConfig,
    *,
    config: BfclEvalConfig,
) -> BenchmarkRunResult:
    launcher = cli.launcher
    if launcher is None:
        raise BfclEvalCliError("nemo_launcher backend requires launcher options")
    if len(config.candidates) != 1:
        raise BfclEvalCliError(
            "one NeMo Launcher orchestration config requires exactly one candidate"
        )
    candidate = config.candidates[0]
    bundle_root = (
        launcher.bundle_root
        if launcher.bundle_root is not None
        else (config.source.publication_dir / NEMO_EVALUATOR_ROOT).resolve()
    )
    _assert_bundle_untouched(bundle_root, launcher)
    adapter = NemoNativeAdapterConfig(
        bundle_root=bundle_root,
        bundle_content_hash=native_bundle_tree_hash(bundle_root),
        eval_config_path=cli.eval_config_path,
        candidate_alias=candidate.alias,
        native_output_dir=launcher.native_output_dir,
        target_binding="launcher",
        # A Launcher task is submitted once, so the operator's probe choice has to
        # be pinned into the adapter config rather than dropped at this boundary.
        probe_oracle=cli.probe_oracle,
    )
    write_native_adapter_config(adapter, launcher.adapter_config_path)
    package = install_native_framework(
        adapter,
        adapter_config_path=launcher.adapter_config_path,
        install_dir=launcher.framework_build_dir,
    )
    task = launcher_task_entry(
        adapter,
        adapter_config_path=launcher.adapter_config_path,
        container=launcher.container,
        dataset_mount_path=launcher.dataset_mount_path,
    )
    _write_immutable_text(
        launcher.task_config_path,
        yaml.safe_dump(task, sort_keys=False, allow_unicode=True),
    )
    launcher_document = _launcher_document(
        launcher.launcher_base_config_path,
        task=task,
        model_id=candidate.model,
        target_url=candidate.api.base_url,
        api_key_name=candidate.api.api_key_env,
        output_dir=launcher.launcher_output_dir,
        dry_run=cli.dry_run,
        evaluation_mounts=dict(launcher.evaluation_mounts),
    )
    if _needs_evaluation_mounts(launcher_document, task):
        _validate_evaluation_mount_coverage(
            dict(launcher.evaluation_mounts),
            _required_evaluation_paths(cli, config),
        )
    _write_immutable_text(
        launcher.launcher_config_path,
        yaml.safe_dump(launcher_document, sort_keys=False, allow_unicode=True),
    )
    payload: dict[str, Any] = {
        "status": "launcher_materialized",
        "candidate": candidate.alias,
        "framework_package": str(package),
        "adapter_config_path": str(launcher.adapter_config_path),
        "task_config_path": str(launcher.task_config_path),
        "launcher_config_path": str(launcher.launcher_config_path),
    }
    if launcher.submit:
        distribution = native_framework_distribution(adapter)
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise BfclEvalCliFrameworkNotInstalledError(
                f"{distribution} is not installed; install the generated package {package}"
            ) from exc
        if installed != NEMO_NATIVE_ADAPTER_VERSION:
            raise BfclEvalCliFrameworkVersionMismatchError(
                f"{distribution} version {installed} differs from "
                f"{NEMO_NATIVE_ADAPTER_VERSION}"
            )
        try:
            launch = launch_model_eval_config(
                config_path=launcher.launcher_config_path,
                cfg=OmegaConf.create(launcher_document),
                task_filters=[task["name"]],
            )
        except ModelEvalDependencyError as exc:
            raise BfclEvalCliFrameworkNotInstalledError(str(exc)) from exc
        payload.update(
            {
                "status": "launcher_submitted",
                "launcher_invocation_id": launch.invocation_id,
                "submitted_launcher_config_path": str(
                    launch.launcher_config_path
                ),
            }
        )
    return BenchmarkRunResult(
        output_path=launcher.launcher_config_path,
        output_format=cli.output_format,
        payload=payload,
    )


def _completed_result(
    result: BfclEvalRunResult | BfclTraceEvalRunResult,
    *,
    output_format: Literal["human", "json"],
) -> BenchmarkRunResult:
    # Scope is a property of the run, so it is read from the aggregates and
    # required to agree rather than sampled from whichever candidate came first.
    scopes = {score.scope for score in result.candidate_scores}
    if len(scopes) != 1:
        raise BfclEvalCliError(
            f"one eval run reports one measurement scope, not {sorted(scopes)}",
            code="eval_artifact_invalid",
            cli_exit_code=EVAL_CLI_EXIT_CODES["eval_artifact_invalid"],
        )
    artifacts = result.artifacts
    return BenchmarkRunResult(
        output_path=artifacts.report_path,
        output_format=output_format,
        payload={
            "status": "completed",
            "eval_run_id": result.eval_run_id,
            "scope": scopes.pop(),
            "candidates": list(result.plan.candidate_aliases),
            "report_path": str(artifacts.report_path),
            "task_results_path": (
                str(artifacts.task_results_path)
                if artifacts.task_results_path is not None
                else None
            ),
            "manifest_path": (
                str(artifacts.manifest_path)
                if artifacts.manifest_path is not None
                else None
            ),
        },
    )


def run_bfcl_eval_cli(config_path: str | os.PathLike[str]) -> BenchmarkRunResult:
    """Preflight or execute one declared BFCL evaluation from the Nemotron CLI."""
    cli = load_bfcl_eval_cli_config(config_path)
    # Every failure below, including projecting the finished run into a payload,
    # has to leave through the published exit-status contract; an unmapped
    # exception would reach the operator as a bare traceback with status 1.
    try:
        if cli.execution_backend == "nemo_launcher":
            return _run_nemo_launcher(cli, config=load_eval_config(cli.eval_config_path))
        if cli.dry_run:
            config = load_eval_config(cli.eval_config_path)
            source = verify_eval_source(config, probe_oracle=cli.probe_oracle)
            plan = evaluate_contamination(config, source)
            return BenchmarkRunResult(
                output_path=config.outputs.output_dir,
                output_format=cli.output_format,
                payload={
                    "status": "preflight_passed",
                    "scope": config.publication_scope,
                    "eval_config_hash": config.eval_config_hash,
                    "candidates": list(plan.candidate_aliases),
                    "task_counts": {
                        alias: len(plan.evaluation_task_ids(alias))
                        for alias in plan.candidate_aliases
                    },
                    "candidate_network_used": False,
                    "oracle_probed": cli.probe_oracle,
                },
            )
        # The runner owns the single authoritative load of the eval config: loading
        # it here too would widen the window in which the pinned file can change
        # between the CLI's view of it and the run that is actually authorized.
        return _completed_result(
            run_declared_eval_sync(
                cli.eval_config_path,
                probe_oracle=cli.probe_oracle,
            ),
            output_format=cli.output_format,
        )
    except BfclEvalCliError:
        raise
    except Exception as exc:
        raise _wrap_runtime_failure(exc) from exc


__all__ = [
    "BFCL_EVAL_CLI_VERSION",
    "EVAL_CLI_EXIT_CODES",
    "BfclEvalCliConfig",
    "BfclEvalCliArtifactConflictError",
    "BfclEvalCliError",
    "BfclEvalCliFrameworkNotInstalledError",
    "BfclEvalCliFrameworkVersionMismatchError",
    "BfclEvalCliRuntimeError",
    "BfclLauncherCliConfig",
    "load_bfcl_eval_cli_config",
    "run_bfcl_eval_cli",
]
