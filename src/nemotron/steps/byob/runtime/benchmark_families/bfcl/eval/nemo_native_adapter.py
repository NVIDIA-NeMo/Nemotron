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

"""NeMo Evaluator native framework bridge for BFCL evaluation.

The generic NeMo Evaluator BYOB runner is intentionally not used for execution:
it renders one text prompt and averages scalar sample scores, while BFCL needs
native tool-call messages, incremental replay, live oracle sessions, explicit N/A
denominators, and authorization-bound aggregation.  This adapter registers one
NeMo framework command, then delegates those semantics to the BFCL runner and
only translates the completed aggregate into ``EvaluationResult``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictStr,
    field_validator,
    model_validator,
)

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.config import (
    load_eval_config,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.eval_artifacts import (
    CandidateAggregate,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.eval_runner import (
    AuthorizedEvalRun,
    BfclEvalRunResult,
    BfclTraceEvalRunResult,
    authorize_bfcl_eval,
    run_bfcl_eval,
    run_bfcl_trace_eval,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.schemas import (
    BfclEvalConfig,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    NEMO_EVALUATOR_SCHEMA_VERSION,
    ContentHash,
    NemoEvaluatorBundle,
    NemoEvaluatorRecord,
    export_content_hash,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.nemo_evaluator_export import (
    NEMO_BUNDLE_FILE,
    NEMO_BUNDLE_FILES,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

NEMO_NATIVE_ADAPTER_VERSION: Final = "1.2"
SUPPORTED_NEMO_EVALUATOR_VERSION: Final = "0.2.8"
SUPPORTED_NEMO_LAUNCHER_VERSION: Final = "0.2.6"
NEMO_NATIVE_RESULT_FILE: Final = "nemo_evaluator_results.json"
NEMO_NATIVE_MANIFEST_FILE: Final = "nemo_native_adapter_manifest.json"
NEMO_NATIVE_FAILURE_FILE: Final = "nemo_native_adapter_failure.json"


class NemoNativeAdapterError(RuntimeError):
    """The bundle, launcher boundary, or translated result is not publishable."""

    code = "eval_nemo_adapter_invalid"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NemoNativeAdapterConfig(_Frozen):
    """Pinned input to one native NeMo Evaluator task invocation."""

    schema_version: Literal["1.2"] = NEMO_NATIVE_ADAPTER_VERSION
    bundle_root: Path
    bundle_content_hash: ContentHash
    eval_config_path: Path
    candidate_alias: StrictStr
    native_output_dir: Path
    target_binding: Literal["launcher", "exact"] = "launcher"
    # Whether the run verifies the oracle pack by executing it. This is part of the
    # pinned setup rather than a runtime flag, because a Launcher task is submitted
    # once and an orchestrator that could not state it here would silently probe
    # against an operator who declared otherwise.
    probe_oracle: StrictBool = True
    nemo_evaluator_version: Literal["0.2.8"] = SUPPORTED_NEMO_EVALUATOR_VERSION
    nemo_launcher_version: Literal["0.2.6"] = SUPPORTED_NEMO_LAUNCHER_VERSION

    @field_validator("bundle_root", "eval_config_path", "native_output_dir")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        expanded = value.expanduser()
        if not expanded.is_absolute():
            raise ValueError("native adapter paths must be absolute")
        return expanded.resolve()

    @field_validator("candidate_alias")
    @classmethod
    def _alias(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("candidate_alias must be non-empty")
        return value.strip()

    @model_validator(mode="after")
    def _separate_output(self) -> NemoNativeAdapterConfig:
        if _paths_overlap(self.native_output_dir, self.bundle_root):
            raise ValueError(
                "native output may not contain or be contained by the immutable input bundle"
            )
        return self

    def semantic_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @property
    def config_hash(self) -> str:
        return _sha256_json(self.semantic_payload())


@dataclass(frozen=True)
class VerifiedNemoBundle:
    root: Path
    descriptor: NemoEvaluatorBundle
    records: tuple[NemoEvaluatorRecord, ...]
    evaluator_config: dict[str, Any]
    metadata: dict[str, Any]
    system_prompts: dict[str, str]
    content_hash: str

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(record.task_id for record in self.records)


@dataclass(frozen=True)
class NemoNativeRunResult:
    config: NemoNativeAdapterConfig
    bundle: VerifiedNemoBundle
    bfcl_result: BfclEvalRunResult | BfclTraceEvalRunResult
    result_path: Path
    result_hash: str
    manifest_path: Path
    manifest_hash: str


def _sha256_json(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left in right.parents or right in left.parents


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_immutable(path: Path, payload: bytes) -> str:
    expected = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or _file_hash(path) != expected:
            raise NemoNativeAdapterError(
                f"{path.name} already exists with different immutable evidence"
            )
        return expected
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or _file_hash(path) != expected:
                raise NemoNativeAdapterError(
                    f"{path.name} appeared concurrently with different evidence"
                )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return expected


def _check_immutable_destination(path: Path, payload: bytes) -> None:
    if not path.exists():
        return
    expected = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    if not path.is_file() or _file_hash(path) != expected:
        raise NemoNativeAdapterError(
            f"{path.name} already exists with different immutable evidence"
        )


def load_native_adapter_config(path: str | Path) -> NemoNativeAdapterConfig:
    source = Path(path)
    if not source.is_file():
        raise NemoNativeAdapterError(f"native adapter config does not exist: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        return NemoNativeAdapterConfig.model_validate(raw)
    except Exception as exc:
        if isinstance(exc, NemoNativeAdapterError):
            raise
        raise NemoNativeAdapterError(
            f"native adapter config is invalid: {type(exc).__name__}"
        ) from exc


def write_native_adapter_config(
    config: NemoNativeAdapterConfig,
    path: str | Path,
) -> str:
    payload = yaml.safe_dump(
        config.semantic_payload(),
        sort_keys=True,
        allow_unicode=True,
        width=1_000_000,
    ).encode("utf-8")
    return _write_immutable(Path(path), payload)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NemoNativeAdapterError(f"{path.name} is not valid UTF-8 JSON") from exc


def _read_dataset(path: Path) -> tuple[NemoEvaluatorRecord, ...]:
    records: list[NemoEvaluatorRecord] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise NemoNativeAdapterError(
                        f"{path.name} line {number} is blank"
                    )
                records.append(NemoEvaluatorRecord.model_validate_json(line))
    except NemoNativeAdapterError:
        raise
    except Exception as exc:
        raise NemoNativeAdapterError(
            f"{path.name} does not satisfy the evaluator record contract"
        ) from exc
    if not records:
        raise NemoNativeAdapterError("the native evaluator dataset is empty")
    task_ids = [record.task_id for record in records]
    if len(set(task_ids)) != len(task_ids):
        raise NemoNativeAdapterError("the native evaluator dataset repeats a task_id")
    return tuple(records)


def read_native_bundle_tree(root: str | Path) -> tuple[dict[str, bytes], str]:
    """Read the exact bundle file set and digest it.

    An orchestrator has to digest a bundle before it can name that digest in an
    adapter config, so this is the one definition of what the tree is and how it is
    hashed; a second copy could accept a bundle this verifier would reject.
    """
    bundle_root = Path(root).expanduser().resolve()
    if not bundle_root.is_dir():
        raise NemoNativeAdapterError(f"bundle root is not a directory: {bundle_root}")
    actual_files = tuple(
        sorted(
            path.relative_to(bundle_root).as_posix()
            for path in bundle_root.rglob("*")
            if path.is_file()
        )
    )
    if actual_files != tuple(sorted(NEMO_BUNDLE_FILES)):
        raise NemoNativeAdapterError(
            f"bundle files differ from the contract: {list(actual_files)}"
        )
    try:
        contents = {name: (bundle_root / name).read_bytes() for name in NEMO_BUNDLE_FILES}
    except OSError as exc:
        raise NemoNativeAdapterError("the native bundle cannot be read") from exc
    return contents, export_content_hash(contents)


def native_bundle_tree_hash(root: str | Path) -> str:
    """Digest one exported bundle tree without an adapter config to compare to."""
    return read_native_bundle_tree(root)[1]


def verify_native_bundle(
    config: NemoNativeAdapterConfig,
    *,
    bundle_root: str | Path | None = None,
) -> VerifiedNemoBundle:
    """Read and content-verify every file the native framework will consume."""
    root = (
        Path(bundle_root).expanduser().resolve()
        if bundle_root is not None
        else config.bundle_root
    )
    contents, content_hash = read_native_bundle_tree(root)
    if content_hash != config.bundle_content_hash:
        raise NemoNativeAdapterError("bundle tree hash differs from the adapter config")

    try:
        descriptor = NemoEvaluatorBundle.model_validate(
            _read_json(root / NEMO_BUNDLE_FILE)
        )
    except Exception as exc:
        raise NemoNativeAdapterError("bundle.json violates its descriptor contract") from exc
    records = _read_dataset(root / descriptor.dataset_file)
    if descriptor.record_count != len(records):
        raise NemoNativeAdapterError("bundle record_count differs from dataset.jsonl")
    dataset_hash = export_content_hash(
        {descriptor.dataset_file: contents[descriptor.dataset_file]}
    )
    if dataset_hash != descriptor.dataset_content_hash:
        raise NemoNativeAdapterError("dataset.jsonl differs from its descriptor hash")
    if _read_json(root / descriptor.dataset_schema_file) != NemoEvaluatorRecord.model_json_schema():
        raise NemoNativeAdapterError("dataset.schema.json differs from the record model")

    metadata = _read_json(root / descriptor.metadata_file)
    prompts = _read_json(root / descriptor.system_prompt_file)
    if not isinstance(metadata, dict) or not isinstance(prompts, dict):
        raise NemoNativeAdapterError("bundle metadata and prompt catalog must be objects")
    if (
        metadata.get("schema_version") != NEMO_EVALUATOR_SCHEMA_VERSION
        or metadata.get("task_name") != descriptor.task_name
        or metadata.get("records") != len(records)
    ):
        raise NemoNativeAdapterError("metadata.json disagrees with bundle.json")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in prompts.items()):
        raise NemoNativeAdapterError("system_prompts.json must map strings to strings")

    try:
        evaluator_config = yaml.safe_load(
            (root / descriptor.evaluator_config_file).read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise NemoNativeAdapterError("evaluator.yaml is not valid UTF-8 YAML") from exc
    if not isinstance(evaluator_config, dict):
        raise NemoNativeAdapterError("evaluator.yaml must contain one object")
    task = evaluator_config.get("task", {})
    scoring = evaluator_config.get("scoring", {})
    if (
        evaluator_config.get("schema_version") != NEMO_EVALUATOR_SCHEMA_VERSION
        or task.get("name") != descriptor.task_name
        or task.get("dataset") != descriptor.dataset_file
        or tuple(scoring.get("metrics", ())) != descriptor.scoring.metrics
    ):
        raise NemoNativeAdapterError("evaluator.yaml disagrees with bundle.json")
    return VerifiedNemoBundle(
        root=root,
        descriptor=descriptor,
        records=records,
        evaluator_config=evaluator_config,
        metadata=metadata,
        system_prompts=prompts,
        content_hash=content_hash,
    )


def _installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise NemoNativeAdapterError(
            f"{distribution} is required; install the evaluator extra"
        ) from exc


def verify_nemo_runtime(
    config: NemoNativeAdapterConfig,
    *,
    require_launcher: bool = True,
) -> dict[str, str]:
    versions: dict[str, str] = {
        "nemo-evaluator": _installed_version("nemo-evaluator")
    }
    expected: dict[str, str] = {
        "nemo-evaluator": config.nemo_evaluator_version
    }
    if require_launcher:
        versions["nemo-evaluator-launcher"] = _installed_version(
            "nemo-evaluator-launcher"
        )
        expected["nemo-evaluator-launcher"] = config.nemo_launcher_version
    if versions != expected:
        raise NemoNativeAdapterError(
            f"NeMo runtime versions differ: actual={versions}, expected={expected}"
        )
    return versions


def _score_document(numerator: int, denominator: int) -> dict[str, Any]:
    mean = numerator / denominator
    variance = mean * (1.0 - mean)
    stddev = math.sqrt(variance)
    return {
        "value": mean,
        "stats": {
            "count": denominator,
            "sum": float(numerator),
            "sum_squared": float(numerator),
            "min": 1.0 if numerator == denominator else 0.0,
            "max": 1.0 if numerator else 0.0,
            "mean": mean,
            "variance": variance,
            "stddev": stddev,
            "stderr": stddev / math.sqrt(denominator),
        },
    }


def native_evaluation_result_document(
    aggregate: CandidateAggregate,
    *,
    task_name: str,
    metric_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Translate BFCL counts without using NeMo BYOB's mean-of-ratios reducer."""
    metric_aliases = {
        "tool_selection_pass_rate": "tool_selection",
        "arguments_pass_rate": "arguments",
        "call_ordering_pass_rate": "call_ordering",
    }
    def native_name(metric: str) -> str:
        if metric_names is None:
            return metric
        return metric_aliases.get(metric, metric)

    scores = {
        native_name(metric.metric): _score_document(
            metric.numerator,
            metric.denominator,
        )
        for metric in aggregate.metrics
        if metric.denominator
        and (
            metric_names is None
            or native_name(metric.metric) in metric_names
        )
    }
    document = {
        "tasks": {task_name: {"metrics": {"pass@1": {"scores": scores}}}},
        "groups": {},
    }
    try:
        from nemo_evaluator.api.api_dataclasses import (  # type: ignore[import-untyped]
            EvaluationResult,
        )

        EvaluationResult.model_validate(document)
    except ImportError as exc:
        raise NemoNativeAdapterError(
            "nemo-evaluator is required to validate the native result"
        ) from exc
    except Exception as exc:
        raise NemoNativeAdapterError(
            "translated metrics do not satisfy NeMo EvaluationResult"
        ) from exc
    return document


def _validate_eval_boundary(
    adapter: NemoNativeAdapterConfig,
    bundle: VerifiedNemoBundle,
    eval_config: BfclEvalConfig,
    *,
    target_url: str | None,
    target_model_id: str | None,
) -> None:
    aliases = tuple(candidate.alias for candidate in eval_config.candidates)
    if aliases != (adapter.candidate_alias,):
        raise NemoNativeAdapterError(
            "one native Launcher task requires exactly its configured candidate"
        )
    candidate = eval_config.candidate(adapter.candidate_alias)
    if target_model_id is not None and target_model_id != candidate.model:
        raise NemoNativeAdapterError("Launcher model id differs from the eval config")
    if (
        adapter.target_binding == "exact"
        and target_url is not None
        and target_url.rstrip("/") != candidate.api.base_url
    ):
        raise NemoNativeAdapterError("Launcher target URL differs from the eval config")
    if _paths_overlap(eval_config.outputs.output_dir, bundle.root):
        raise NemoNativeAdapterError(
            "BFCL eval output may not contain or be contained by its input bundle"
        )
    if _paths_overlap(adapter.native_output_dir, bundle.root):
        raise NemoNativeAdapterError(
            "NeMo native output may not contain or be contained by the runtime bundle"
        )
    if _paths_overlap(eval_config.outputs.output_dir, adapter.native_output_dir):
        raise NemoNativeAdapterError(
            "BFCL and NeMo native outputs require separate directories"
        )


def _runtime_eval_config(
    adapter: NemoNativeAdapterConfig,
    config: BfclEvalConfig,
    *,
    target_url: str | None,
    target_model_id: str | None,
) -> BfclEvalConfig:
    if adapter.target_binding != "launcher" or (
        target_url is None and target_model_id is None
    ):
        return config
    payload = config.model_dump(mode="python")
    candidate_payload = next(
        candidate
        for candidate in payload["candidates"]
        if candidate["alias"] == adapter.candidate_alias
    )
    if (
        target_model_id is not None
        and target_model_id != candidate_payload["model"]
    ):
        raise NemoNativeAdapterError(
            "Launcher model id differs from the eval config; refusing to score "
            "a served target under another candidate identity"
        )
    if target_url is not None:
        candidate_payload["api"]["base_url"] = target_url.rstrip("/")
    try:
        return BfclEvalConfig.model_validate(payload)
    except Exception as exc:
        raise NemoNativeAdapterError(
            "Launcher target endpoint cannot form a valid BFCL runtime config"
        ) from exc


def _validate_authorization(
    bundle: VerifiedNemoBundle,
    authorized: AuthorizedEvalRun,
    *,
    candidate_alias: str,
) -> None:
    if (
        authorized.source.evaluation_benchmark.content_hash
        != bundle.descriptor.source.benchmark_content_hash
    ):
        raise NemoNativeAdapterError(
            "BFCL verified source differs from the bundle source identity"
        )
    if bundle.task_ids != authorized.projection.task_ids:
        raise NemoNativeAdapterError(
            "native bundle task order differs from the verified BFCL source"
        )
    projected = tuple(
        NemoEvaluatorRecord.from_canonical(row) for row in authorized.projection.rows
    )
    if projected != bundle.records:
        raise NemoNativeAdapterError(
            "native bundle records differ from the verified BFCL source projection"
        )
    eligible = authorized.plan.evaluation_task_ids(candidate_alias)
    if not eligible or any(task_id not in bundle.task_ids for task_id in eligible):
        raise NemoNativeAdapterError(
            "BFCL authorization is not a non-empty subset of the native bundle"
        )


def _run_authorized_eval(
    config: BfclEvalConfig,
    *,
    authorized: AuthorizedEvalRun,
    probe_oracle: bool,
) -> BfclEvalRunResult | BfclTraceEvalRunResult:
    if config.settings.executable:
        return asyncio.run(
            run_bfcl_eval(
                config,
                eval_run_id=authorized.eval_run_id,
                probe_oracle=probe_oracle,
                authorized=authorized,
            )
        )
    return asyncio.run(
        run_bfcl_trace_eval(
            config,
            eval_run_id=authorized.eval_run_id,
            probe_oracle=probe_oracle,
            authorized=authorized,
        )
    )


def _run_nemo_native_adapter(
    adapter: NemoNativeAdapterConfig,
    *,
    target_url: str | None = None,
    target_model_id: str | None = None,
    probe_oracle: bool | None = None,
    bundle_root: str | Path | None = None,
) -> NemoNativeRunResult:
    """Verify, run through BFCL, and publish a NeMo-native result sidecar."""
    if probe_oracle is not None and probe_oracle != adapter.probe_oracle:
        raise NemoNativeAdapterError(
            "runtime probe_oracle differs from the immutable adapter config"
        )
    effective_probe_oracle = adapter.probe_oracle
    versions = verify_nemo_runtime(adapter, require_launcher=False)
    bundle = verify_native_bundle(adapter, bundle_root=bundle_root)
    eval_config = load_eval_config(adapter.eval_config_path)
    _validate_eval_boundary(
        adapter,
        bundle,
        eval_config,
        target_url=target_url,
        target_model_id=target_model_id,
    )
    runtime_config = _runtime_eval_config(
        adapter,
        eval_config,
        target_url=target_url,
        target_model_id=target_model_id,
    )
    authorized = authorize_bfcl_eval(
        runtime_config,
        eval_run_id=None,
        probe_oracle=effective_probe_oracle,
    )
    _validate_authorization(
        bundle,
        authorized,
        candidate_alias=adapter.candidate_alias,
    )
    result = _run_authorized_eval(
        runtime_config,
        authorized=authorized,
        probe_oracle=effective_probe_oracle,
    )
    aggregates = {
        aggregate.candidate_alias: aggregate for aggregate in result.candidate_scores
    }
    if set(aggregates) != {adapter.candidate_alias}:
        raise NemoNativeAdapterError("BFCL result covers a different candidate set")
    aggregate = aggregates[adapter.candidate_alias]
    document = native_evaluation_result_document(
        aggregate,
        task_name=_launcher_task_name(bundle.descriptor.task_name),
        metric_names=bundle.descriptor.scoring.metrics,
    )
    output_dir = adapter.native_output_dir
    result_path = output_dir / NEMO_NATIVE_RESULT_FILE
    result_payload = _json_bytes(document)
    result_hash = f"sha256:{hashlib.sha256(result_payload).hexdigest()}"
    manifest = {
        "schema_version": NEMO_NATIVE_ADAPTER_VERSION,
        "adapter_config_hash": adapter.config_hash,
        "bundle_content_hash": bundle.content_hash,
        "task_name": bundle.descriptor.task_name,
        "launcher_task_name": _launcher_task_name(bundle.descriptor.task_name),
        "candidate_alias": adapter.candidate_alias,
        "served_identity": {
            "model": target_model_id
            or runtime_config.candidate(adapter.candidate_alias).model,
            "base_url": (
                target_url.rstrip("/")
                if target_url is not None
                else runtime_config.candidate(adapter.candidate_alias).api.base_url
            ),
            "canonical_model_identity": aggregate.canonical_model_identity,
        },
        "eval_run_id": result.eval_run_id,
        "eval_scope": aggregate.scope,
        "aggregate_hash": aggregate.aggregate_hash,
        "omitted_not_applicable_metrics": {
            metric.metric: metric.not_applicable_reason
            for metric in aggregate.metrics
            if metric.denominator == 0
        },
        "nemo_versions": versions,
        "nemo_launcher_version_pin": adapter.nemo_launcher_version,
        "eval_config_hash": aggregate.eval_config_hash,
        "plan_identity": aggregate.plan_identity,
        "source_verification_identity": aggregate.source_verification_identity,
        "authorized_task_ids": list(
            result.plan.evaluation_task_ids(adapter.candidate_alias)
        ),
        "artifacts": {
            "nemo_evaluator_result": {
                "file": NEMO_NATIVE_RESULT_FILE,
                "content_hash": result_hash,
            },
            "bfcl_eval_report": {
                "file": str(result.artifacts.report_path),
                "content_hash": result.artifacts.report_hash,
            },
        },
    }
    manifest_path = output_dir / NEMO_NATIVE_MANIFEST_FILE
    manifest_payload = _json_bytes(manifest)
    _check_immutable_destination(result_path, result_payload)
    _check_immutable_destination(manifest_path, manifest_payload)
    _write_immutable(result_path, result_payload)
    manifest_hash = _write_immutable(manifest_path, manifest_payload)
    return NemoNativeRunResult(
        config=adapter,
        bundle=bundle,
        bfcl_result=result,
        result_path=result_path,
        result_hash=result_hash,
        manifest_path=manifest_path,
        manifest_hash=manifest_hash,
    )


def run_nemo_native_adapter(
    adapter: NemoNativeAdapterConfig,
    *,
    target_url: str | None = None,
    target_model_id: str | None = None,
    probe_oracle: bool | None = None,
    bundle_root: str | Path | None = None,
) -> NemoNativeRunResult:
    """Run the native bridge and leave machine-readable evidence on failure."""
    try:
        return _run_nemo_native_adapter(
            adapter,
            target_url=target_url,
            target_model_id=target_model_id,
            probe_oracle=probe_oracle,
            bundle_root=bundle_root,
        )
    except Exception as exc:
        runtime_bundle = (
            Path(bundle_root).expanduser().resolve()
            if bundle_root is not None
            else adapter.bundle_root
        )
        if not _paths_overlap(adapter.native_output_dir, runtime_bundle):
            failure = {
                "schema_version": NEMO_NATIVE_ADAPTER_VERSION,
                "adapter_config_hash": adapter.config_hash,
                "error_code": getattr(exc, "code", NemoNativeAdapterError.code),
                "error_type": type(exc).__name__,
            }
            try:
                _write_immutable(
                    adapter.native_output_dir / NEMO_NATIVE_FAILURE_FILE,
                    _json_bytes(failure),
                )
            except Exception:
                pass
        raise


def _framework_name(task_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", task_name.lower()).strip("_")[:32]
    digest = hashlib.sha256(task_name.encode("utf-8")).hexdigest()[:12]
    return f"byob_bfcl_{normalized or 'task'}_{digest}"


def _launcher_task_name(task_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9_-]+", "_", task_name.lower()).strip("_")[:48]
    if normalized == task_name and normalized:
        return normalized
    digest = hashlib.sha256(task_name.encode("utf-8")).hexdigest()[:12]
    return f"{normalized or 'task'}_{digest}"


def native_framework_distribution(adapter: NemoNativeAdapterConfig) -> str:
    """Name the distribution `install_native_framework` builds for this bundle.

    An orchestrator has to verify that exact distribution is installed before it
    submits, so the name is derived here rather than reconstructed from the built
    directory by every caller.
    """
    bundle = verify_native_bundle(adapter)
    return f"nemo-evaluator-{_framework_name(bundle.descriptor.task_name)}"


def native_framework_definition(
    adapter: NemoNativeAdapterConfig,
    *,
    adapter_config_path: str,
) -> dict[str, Any]:
    """Build the FDF registered with NeMo Evaluator for this immutable bundle."""
    bundle = verify_native_bundle(adapter)
    framework = _framework_name(bundle.descriptor.task_name)
    task_name = _launcher_task_name(bundle.descriptor.task_name)
    command = (
        "python -m "
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.nemo_native_adapter"
        " run"
        ' --adapter-config "{{config.params.extra.adapter_config}}"'
        ' --native-output-dir "{{config.output_dir}}"'
        ' --bundle-root "{{config.params.extra.runtime_bundle_root}}"'
        ' --target-url "{{target.api_endpoint.url}}"'
        ' --target-model-id "{{target.api_endpoint.model_id}}"'
    )
    return {
        "framework": {"name": framework, "pkg_name": framework},
        "defaults": {
            "config": {
                "params": {
                    "task": task_name,
                    "limit_samples": None,
                    "extra": {
                        "adapter_config": adapter_config_path,
                        "bundle_content_hash": adapter.bundle_content_hash,
                        "runtime_bundle_root": str(adapter.bundle_root),
                    },
                }
            },
            "target": {"api_endpoint": {"adapter_config": {"mode": "client"}}},
            "command": command,
        },
        "evaluations": [
            {
                "name": task_name,
                "description": "BFCL native tool-calling evaluation",
                "defaults": {
                    "config": {
                        "type": f"{framework}.{task_name}",
                        "supported_endpoint_types": ["chat"],
                        "required_capabilities": ["tools"],
                    }
                },
            }
        ],
    }


def install_native_framework(
    adapter: NemoNativeAdapterConfig,
    *,
    adapter_config_path: str | Path,
    install_dir: str | Path,
) -> Path:
    """Build an immutable namespace package for image bake or explicit install."""
    destination = Path(install_dir).expanduser().resolve()
    fdf = native_framework_definition(
        adapter,
        adapter_config_path=str(Path(adapter_config_path).resolve()),
    )
    framework = str(fdf["framework"]["name"])
    package_root = destination / framework
    namespace = package_root / "nemo_evaluator" / framework
    namespace.mkdir(parents=True, exist_ok=True)
    _write_immutable(namespace / "__init__.py", b"")
    _write_immutable(
        namespace / "framework.yml",
        yaml.safe_dump(fdf, sort_keys=False, allow_unicode=True).encode("utf-8"),
    )
    output_module = f'''"""Parse BFCL native adapter output."""
import hashlib
import json
from pathlib import Path
from nemo_evaluator.api.api_dataclasses import EvaluationResult

def parse_output(output_dir: str) -> EvaluationResult:
    path = Path(output_dir) / {NEMO_NATIVE_RESULT_FILE!r}
    manifest_path = Path(output_dir) / {NEMO_NATIVE_MANIFEST_FILE!r}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["artifacts"]["nemo_evaluator_result"]["content_hash"]
    actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("native evaluator result differs from its manifest hash")
    return EvaluationResult.model_validate(json.loads(path.read_text(encoding="utf-8")))
'''
    _write_immutable(namespace / "output.py", output_module.encode("utf-8"))
    pyproject = f"""[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "nemo-evaluator-{framework}"
version = "{NEMO_NATIVE_ADAPTER_VERSION}"
requires-python = ">=3.10"
dependencies = ["nemo-evaluator=={SUPPORTED_NEMO_EVALUATOR_VERSION}"]

[tool.setuptools.packages.find]
where = ["."]
namespaces = true
include = ["nemo_evaluator.{framework}"]

[tool.setuptools.package-data]
"nemo_evaluator.{framework}" = ["framework.yml"]
"""
    _write_immutable(package_root / "pyproject.toml", pyproject.encode("utf-8"))
    return package_root


def launcher_task_entry(
    adapter: NemoNativeAdapterConfig,
    *,
    adapter_config_path: str | Path,
    container: str | None = None,
    dataset_mount_path: str = "/datasets/bfcl",
) -> dict[str, Any]:
    """Return one validated Launcher ``evaluation.tasks`` entry."""
    verify_nemo_runtime(adapter, require_launcher=True)
    bundle = verify_native_bundle(adapter)
    framework = _framework_name(bundle.descriptor.task_name)
    task_name = _launcher_task_name(bundle.descriptor.task_name)
    candidate = load_eval_config(adapter.eval_config_path).candidate(
        adapter.candidate_alias
    )
    entry: dict[str, Any] = {
        "name": f"{framework}.{task_name}",
        "endpoint_type": "chat",
        "env_vars": {
            candidate.api.api_key_env: f"host:{candidate.api.api_key_env}"
        },
        "nemo_evaluator_config": {
            "config": {
                "params": {
                    "extra": {
                        "adapter_config": str(Path(adapter_config_path)),
                        "bundle_content_hash": adapter.bundle_content_hash,
                        "runtime_bundle_root": dataset_mount_path,
                    }
                }
            },
            "target": {"api_endpoint": {"adapter_config": {"mode": "client"}}},
        },
        "dataset_dir": str(bundle.root),
        "dataset_mount_path": dataset_mount_path,
    }
    if container is not None:
        entry["container"] = container
    validation = (
        "import json,sys;"
        "from nemo_evaluator_launcher.common.config_models import TaskModel;"
        "TaskModel.model_validate(json.loads(sys.stdin.read()))"
    )
    with tempfile.TemporaryDirectory(prefix="bfcl-launcher-validation-") as home:
        environment = {**os.environ, "HOME": home}
        completed = subprocess.run(
            [sys.executable, "-c", validation],
            input=json.dumps(entry),
            text=True,
            capture_output=True,
            env=environment,
        )
    if completed.returncode:
        raise NemoNativeAdapterError(
            "native Launcher task entry violates the pinned Launcher schema"
        )
    return entry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BFCL NeMo Evaluator native adapter")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--adapter-config", required=True)
    run.add_argument("--native-output-dir")
    run.add_argument("--bundle-root")
    run.add_argument("--target-url")
    run.add_argument("--target-model-id")
    run.add_argument("--no-probe-oracle", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--adapter-config", required=True)
    install = subparsers.add_parser("install")
    install.add_argument("--adapter-config", required=True)
    install.add_argument("--install-dir", required=True)
    install.add_argument(
        "--adapter-config-in-container",
        help="Absolute adapter config path rendered into the framework command",
    )
    task = subparsers.add_parser("task")
    task.add_argument("--adapter-config", required=True)
    task.add_argument(
        "--adapter-config-in-container",
        help="Absolute path visible inside the evaluation container",
    )
    task.add_argument("--container")
    task.add_argument("--dataset-mount-path", default="/datasets/bfcl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        adapter = load_native_adapter_config(args.adapter_config)
        if args.command == "verify":
            versions = verify_nemo_runtime(adapter)
            bundle = verify_native_bundle(adapter)
            eval_config = load_eval_config(adapter.eval_config_path)
            _validate_eval_boundary(
                adapter,
                bundle,
                eval_config,
                target_url=None,
                target_model_id=None,
            )
            authorized = authorize_bfcl_eval(
                eval_config,
                eval_run_id=None,
                probe_oracle=adapter.probe_oracle,
            )
            _validate_authorization(
                bundle,
                authorized,
                candidate_alias=adapter.candidate_alias,
            )
            print(
                json.dumps(
                    {
                        "adapter_config_hash": adapter.config_hash,
                        "bundle_content_hash": bundle.content_hash,
                        "task_name": bundle.descriptor.task_name,
                        "task_count": len(bundle.records),
                        "authorized_task_count": len(
                            authorized.plan.evaluation_task_ids(
                                adapter.candidate_alias
                            )
                        ),
                        "nemo_versions": versions,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "install":
            package = install_native_framework(
                adapter,
                adapter_config_path=(
                    args.adapter_config_in_container or args.adapter_config
                ),
                install_dir=args.install_dir,
            )
            print(package)
            return 0
        if args.command == "task":
            entry = launcher_task_entry(
                adapter,
                adapter_config_path=(
                    args.adapter_config_in_container or args.adapter_config
                ),
                container=args.container,
                dataset_mount_path=args.dataset_mount_path,
            )
            print(yaml.safe_dump(entry, sort_keys=False, allow_unicode=True))
            return 0
        if args.native_output_dir:
            payload = adapter.model_dump(mode="python")
            payload["native_output_dir"] = Path(args.native_output_dir).resolve()
            adapter = NemoNativeAdapterConfig.model_validate(payload)
        result = run_nemo_native_adapter(
            adapter,
            target_url=args.target_url,
            target_model_id=args.target_model_id,
            # The adapter is the source of truth. The legacy safety flag remains
            # accepted, but a value that contradicts the config is refused.
            probe_oracle=False if args.no_probe_oracle else None,
            bundle_root=args.bundle_root,
        )
    except Exception as exc:
        code = getattr(exc, "code", NemoNativeAdapterError.code)
        print(f"{code}: {exc}", file=sys.stderr)
        return 2
    print(result.result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NEMO_NATIVE_ADAPTER_VERSION",
    "NEMO_NATIVE_FAILURE_FILE",
    "NEMO_NATIVE_MANIFEST_FILE",
    "NEMO_NATIVE_RESULT_FILE",
    "SUPPORTED_NEMO_EVALUATOR_VERSION",
    "SUPPORTED_NEMO_LAUNCHER_VERSION",
    "NemoNativeAdapterConfig",
    "NemoNativeAdapterError",
    "NemoNativeRunResult",
    "VerifiedNemoBundle",
    "install_native_framework",
    "launcher_task_entry",
    "load_native_adapter_config",
    "main",
    "native_evaluation_result_document",
    "native_bundle_tree_hash",
    "native_framework_definition",
    "native_framework_distribution",
    "read_native_bundle_tree",
    "run_nemo_native_adapter",
    "verify_native_bundle",
    "verify_nemo_runtime",
    "write_native_adapter_config",
]
