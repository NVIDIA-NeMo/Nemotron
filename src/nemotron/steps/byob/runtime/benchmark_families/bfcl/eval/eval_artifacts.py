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

"""Deterministic BFCL task-result projections and executable-run artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, TypeVar

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_cache import (
    CandidateIOCache,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_contract import (
    CANDIDATE_IO_CACHE_FILE,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_errors import (
    CandidateCacheError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.contamination_contract import (
    EligibleEvalPlan,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.error_taxonomy import (
    ERROR_TAXONOMY_HASH,
    episode_failure_record,
    gate_failure_record,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_aggregation import (
    ExecutableCandidateScore,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_errors import (
    ToolTraceCacheError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_scoring_contract import (
    ExecutableGateResult,
    ExecutableMetricResult,
    ExecutableTaskScore,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.schemas import BfclEvalConfig
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.tool_trace_cache import (
    ToolTraceCache,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.tool_trace_contract import (
    TOOL_TRACE_CACHE_FILE,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.trace_aggregation import (
    TraceCandidateScore,
    TraceMetricResult,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.trace_scoring_contract import (
    GateResult,
    TraceTaskScore,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import thaw_json
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.benchmark_families.bfcl.runtime_metadata import (
    runtime_metadata,
)

EVAL_ARTIFACT_CONTRACT_VERSION: Final = "1.5"
EVAL_REPORT_FILE: Final = "eval_report.json"
EVAL_TASK_RESULTS_FILE: Final = "eval_task_results.parquet"
EVAL_MANIFEST_FILE: Final = "eval_manifest.json"

# What a published artifact set measured. Trace-only and executable runs report
# different metric taxonomies over the same task set, so a reader that could not
# tell them apart would compare two numbers that answer different questions.
CandidateAggregate = ExecutableCandidateScore | TraceCandidateScore
EvalTaskScore = ExecutableTaskScore | TraceTaskScore
_ScoreT = TypeVar("_ScoreT", ExecutableTaskScore, TraceTaskScore)


class EvalArtifactError(Exception):
    """Final eval evidence is incomplete or crosses an identity boundary."""

    code = "eval_artifact_invalid"

    def __init__(self, problem: str) -> None:
        self.problem = problem
        super().__init__(problem)

    def as_report(self) -> dict[str, str]:
        return {
            "code": self.code,
            "subject": "eval.artifacts",
            "problem": self.problem,
            "expected": "one complete immutable artifact set",
            "recovery": "resume into a new output directory from verified evidence",
        }


@dataclass(frozen=True)
class EvalArtifactSet:
    report_path: Path
    report_hash: str
    task_results_path: Path | None
    task_results_hash: str | None
    manifest_path: Path | None
    manifest_hash: str | None


def _sha256_json(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _metric_document(metric: ExecutableMetricResult | TraceMetricResult) -> dict[str, Any]:
    return {
        "numerator": metric.numerator,
        "denominator": metric.denominator,
        "value": metric.value,
        "not_applicable_reason": metric.not_applicable_reason,
    }


def _gate_bool(gate: ExecutableGateResult | GateResult) -> bool | None:
    if gate.outcome == "not_applicable":
        return None
    return gate.outcome == "passed"


def _metric_bool(metric: ExecutableMetricResult) -> bool | None:
    if metric.denominator == 0:
        return None
    return metric.numerator == metric.denominator


def _contamination_violations(plan: EligibleEvalPlan) -> int:
    """Count scored candidate-task pairs covered by a definite model collision."""
    violations = 0
    for candidate in plan.candidates:
        exposed = {
            task_id
            for collision in candidate.definite_collisions
            for task_id in collision.task_ids
        }
        violations += len(exposed & set(plan.evaluation_task_ids(candidate.alias)))
    return violations


def executable_task_result(score: ExecutableTaskScore) -> dict[str, Any]:
    """Project one score onto the frozen eval-task-results columns."""
    failed = tuple(gate for gate in score.gates if gate.outcome == "failed")
    # The episode record carries the terminal status a gate reason code cannot:
    # every incomplete episode fails the same completion gate, but only the
    # status separates a spent budget from a broken oracle.
    episode = episode_failure_record(score.episode_status, executable=True)
    records = [] if episode is None else [episode]
    records.extend(
        gate_failure_record(
            gate=gate.gate,
            reason_code=gate.reason_code,
            failure_class=gate.failure_class,
        )
        for gate in failed
    )
    return {
        "candidate_alias": score.candidate_alias,
        "candidate_canonical_id": score.canonical_model_identity,
        "task_id": score.task_id,
        "mode": "executable",
        "schema_valid": _gate_bool(score.gate("schema_valid")),
        "tool_name_correct": _metric_bool(score.metric("tool_name_accuracy")),
        "arguments_correct": _metric_bool(score.metric("argument_accuracy")),
        "call_group_correct": _gate_bool(score.gate("call_grouping")),
        "call_order_correct": _gate_bool(score.gate("call_ordering")),
        "required_subset_correct": _gate_bool(score.gate("tool_selection")),
        "milestones_correct": _metric_bool(score.metric("milestone_accuracy")),
        "execution_success": _gate_bool(score.gate("oracle_execution")),
        "assertions_passed": _gate_bool(score.gate("assertions")),
        "final_answer_passed": _metric_bool(
            score.metric("final_answer_success_rate")
        ),
        "episode_status": score.episode_status,
        "non_candidate_stop": score.non_candidate_stop,
        "task_success": score.task_success,
        "failure_codes": [gate.reason_code for gate in failed],
        "failure_records": [record.as_document() for record in records],
    }


def trace_task_result(score: TraceTaskScore) -> dict[str, Any]:
    """Project one trace score onto the shared eval-task-results columns.

    Columns that require live oracle execution, pack assertions, or executable
    milestones remain null. A trace-only row must not imply evidence its scorer
    never observed merely to fill a shared table.
    """
    failed = tuple(gate for gate in score.gates if gate.outcome == "failed")
    return {
        "candidate_alias": score.candidate_alias,
        "candidate_canonical_id": score.canonical_model_identity,
        "task_id": score.task_id,
        "mode": "trace",
        "schema_valid": _gate_bool(score.gate("schema_valid")),
        "tool_name_correct": _gate_bool(score.gate("tool_selection")),
        "arguments_correct": _gate_bool(score.gate("arguments")),
        "call_group_correct": _gate_bool(score.gate("call_grouping")),
        "call_order_correct": _gate_bool(score.gate("call_ordering")),
        "required_subset_correct": _gate_bool(score.gate("tool_selection")),
        "milestones_correct": None,
        "execution_success": None,
        "assertions_passed": None,
        "final_answer_passed": None,
        "episode_status": score.episode_status,
        "non_candidate_stop": score.non_candidate_stop,
        "task_success": score.task_success,
        "failure_codes": [gate.reason_code for gate in failed],
        "failure_records": [
            record.as_document() for record in score.failure_records()
        ],
    }


def _eval_scope(
    candidate_scores: tuple[CandidateAggregate, ...],
) -> Literal["trace", "trace_and_executable"]:
    """Which evaluation an artifact set publishes, as its aggregates declare it."""
    scopes = {score.scope for score in candidate_scores}
    if len(scopes) != 1:
        raise EvalArtifactError("candidate aggregates mix evaluation scopes")
    return next(iter(scopes))


def _task_policy_hash(score: EvalTaskScore) -> str:
    if isinstance(score, ExecutableTaskScore):
        return score.scoring_policy_hash
    return _sha256_json(thaw_json(score.scoring_policy))


def _validate_candidate_aggregates(
    *,
    config: BfclEvalConfig,
    plan: EligibleEvalPlan,
    candidate_scores: tuple[CandidateAggregate, ...],
) -> tuple[Literal["trace", "trace_and_executable"], dict[str, CandidateAggregate]]:
    """Prove aggregate documents still belong to this authorization boundary.

    Aggregators perform the same checks while constructing these values, but report
    and artifact writers are public APIs. They must not trust a deserialized or
    ``model_copy``-restamped aggregate merely because a runner normally supplies it.
    """
    if config.eval_config_hash != plan.eval_config_hash:
        raise EvalArtifactError("eval config and contamination plan identities differ")
    if not candidate_scores:
        raise EvalArtifactError("candidate aggregates do not cover the plan aliases")
    scope = _eval_scope(candidate_scores)
    aliases = tuple(score.candidate_alias for score in candidate_scores)
    if len(set(aliases)) != len(aliases):
        raise EvalArtifactError("candidate aggregates contain a duplicate alias")
    if tuple(sorted(aliases)) != plan.candidate_aliases:
        raise EvalArtifactError("candidate aggregates do not cover the plan aliases")

    by_alias = {score.candidate_alias: score for score in candidate_scores}
    for alias in plan.candidate_aliases:
        aggregate = by_alias[alias]
        candidate = plan.candidate(alias)
        expected = {
            "scope": scope,
            "canonical_model_identity": candidate.canonical_model_identity,
            "plan_identity": plan.plan_identity,
            "eval_config_hash": plan.eval_config_hash,
            "scoring_policy_hash": plan.scoring_policy_hash,
            "source_verification_identity": plan.source_verification_identity,
            "task_ids": plan.evaluation_task_ids(alias),
        }
        actual = {
            "scope": aggregate.scope,
            "canonical_model_identity": aggregate.canonical_model_identity,
            "plan_identity": aggregate.plan_identity,
            "eval_config_hash": aggregate.eval_config_hash,
            "scoring_policy_hash": aggregate.scoring_policy_hash,
            "source_verification_identity": aggregate.source_verification_identity,
            "task_ids": aggregate.task_ids,
        }
        if actual != expected:
            raise EvalArtifactError(
                f"candidate aggregate for {alias!r} crosses its authorization boundary"
            )
    return scope, by_alias


def _validate_complete_run(
    *,
    config: BfclEvalConfig,
    plan: EligibleEvalPlan,
    candidate_scores: tuple[CandidateAggregate, ...],
    task_scores: tuple[_ScoreT, ...],
) -> dict[str, tuple[_ScoreT, ...]]:
    scope, aggregate_by_alias = _validate_candidate_aggregates(
        config=config,
        plan=plan,
        candidate_scores=candidate_scores,
    )
    expected_type = TraceTaskScore if scope == "trace" else ExecutableTaskScore
    if any(not isinstance(score, expected_type) for score in task_scores):
        raise EvalArtifactError(f"{scope} aggregates require {scope} task scores")
    identities = [(score.candidate_alias, score.task_id) for score in task_scores]
    if len(set(identities)) != len(identities):
        raise EvalArtifactError("task scores repeat a candidate and task identity")

    by_candidate: dict[str, tuple[_ScoreT, ...]] = {}
    for alias in plan.candidate_aliases:
        scores = tuple(score for score in task_scores if score.candidate_alias == alias)
        aggregate = aggregate_by_alias[alias]
        if tuple(score.task_id for score in scores) != aggregate.task_ids:
            raise EvalArtifactError(
                f"task scores for {alias!r} do not match its aggregate task order"
            )
        if tuple(score.score_hash for score in scores) != aggregate.task_score_hashes:
            raise EvalArtifactError(
                f"task scores for {alias!r} do not match aggregate score identities"
            )
        for score in scores:
            expected = {
                "scope": aggregate.scope,
                "canonical_model_identity": aggregate.canonical_model_identity,
                "plan_identity": aggregate.plan_identity,
                "eval_config_hash": aggregate.eval_config_hash,
                "scoring_policy_hash": aggregate.scoring_policy_hash,
                "source_verification_identity": aggregate.source_verification_identity,
                "scoring_contract_hash": aggregate.scoring_contract_hash,
            }
            actual = {
                "scope": score.scope,
                "canonical_model_identity": score.canonical_model_identity,
                "plan_identity": score.plan_identity,
                "eval_config_hash": score.eval_config_hash,
                "scoring_policy_hash": _task_policy_hash(score),
                "source_verification_identity": score.source_verification_identity,
                "scoring_contract_hash": score.scoring_contract_hash,
            }
            if actual != expected:
                raise EvalArtifactError(
                    f"task score {alias!r}/{score.task_id!r} crosses its aggregate boundary"
                )
            if isinstance(aggregate, ExecutableCandidateScore):
                if (
                    not isinstance(score, ExecutableTaskScore)
                    or score.oracle_verification_identity
                    != aggregate.oracle_verification_identity
                ):
                    raise EvalArtifactError(
                        f"task score {alias!r}/{score.task_id!r} crosses its oracle boundary"
                    )
        by_candidate[alias] = scores
    known = set(plan.candidate_aliases)
    if any(score.candidate_alias not in known for score in task_scores):
        raise EvalArtifactError("task scores include a candidate outside the plan")
    return by_candidate


def eval_report_document(
    *,
    eval_run_id: str,
    config: BfclEvalConfig,
    plan: EligibleEvalPlan,
    candidate_scores: tuple[CandidateAggregate, ...],
) -> dict[str, Any]:
    """Build the human-facing aggregate report without filesystem identities."""
    if not eval_run_id.strip():
        raise EvalArtifactError("eval_run_id must be non-empty")
    scope, aggregates = _validate_candidate_aggregates(
        config=config,
        plan=plan,
        candidate_scores=candidate_scores,
    )
    candidates: list[dict[str, Any]] = []
    for alias in plan.candidate_aliases:
        candidate = config.candidate(alias)
        aggregate = aggregates[alias]
        candidates.append(
            {
                "alias": alias,
                "scope": aggregate.scope,
                "canonical_id": candidate.canonical_model_identity,
                "aggregate_hash": aggregate.aggregate_hash,
                "task_count": aggregate.task_count,
                "successful_tasks": aggregate.successful_tasks,
                "non_candidate_stops": aggregate.non_candidate_stops,
                "metrics": {
                    metric.metric: _metric_document(metric)
                    for metric in aggregate.metrics
                },
            }
        )
    publication_allowed = config.publication_allowed and plan.publication_allowed
    reasons = sorted(
        set(config.non_publication_reasons) | set(plan.non_publication_reasons)
    )
    return {
        "schema_version": EVAL_ARTIFACT_CONTRACT_VERSION,
        "eval_run_id": eval_run_id,
        "eval_scope": scope,
        "source_run_id": plan.source_run_id,
        "eval_config_hash": config.eval_config_hash,
        "error_taxonomy_hash": ERROR_TAXONOMY_HASH,
        "plan_identity": plan.plan_identity,
        "source_verification_identity": plan.source_verification_identity,
        "comparison_set": {
            "policy": plan.comparison_set,
            "task_count": plan.common.task_count,
            "task_ids_hash": plan.common.task_ids_hash,
        },
        "candidates": candidates,
        "contamination_violations": _contamination_violations(plan),
        "publication_allowed": publication_allowed,
        "non_publication_reasons": reasons,
    }


def _task_results_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("candidate_alias", pa.string(), nullable=False),
            pa.field("candidate_canonical_id", pa.string(), nullable=False),
            pa.field("task_id", pa.string(), nullable=False),
            pa.field("mode", pa.string(), nullable=False),
            pa.field("schema_valid", pa.bool_()),
            pa.field("tool_name_correct", pa.bool_()),
            pa.field("arguments_correct", pa.bool_()),
            pa.field("call_group_correct", pa.bool_()),
            pa.field("call_order_correct", pa.bool_()),
            pa.field("required_subset_correct", pa.bool_()),
            pa.field("milestones_correct", pa.bool_()),
            pa.field("execution_success", pa.bool_()),
            pa.field("assertions_passed", pa.bool_()),
            pa.field("final_answer_passed", pa.bool_()),
            pa.field("episode_status", pa.string(), nullable=False),
            pa.field("non_candidate_stop", pa.bool_(), nullable=False),
            pa.field("task_success", pa.bool_(), nullable=False),
            pa.field("failure_codes", pa.list_(pa.string()), nullable=False),
            pa.field(
                "failure_records",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("layer", pa.string(), nullable=False),
                            pa.field("code", pa.string(), nullable=False),
                            pa.field("attribution", pa.string(), nullable=False),
                            pa.field("subject", pa.string(), nullable=False),
                        ]
                    )
                ),
                nullable=False,
            ),
        ]
    )


def _json_bytes(document: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_immutable_bytes(path: Path, payload: bytes) -> str:
    expected_hash = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    if path.exists():
        if not path.is_file() or _file_hash(path) != expected_hash:
            raise EvalArtifactError(
                f"{path.name} already exists with different immutable evidence"
            )
        return expected_hash
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return expected_hash


def _write_task_results(path: Path, rows: list[dict[str, Any]]) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = _task_results_schema()
    if path.exists():
        # Parquet bytes carry the writer version, so identical rows written by a
        # different pyarrow build differ. Immutability is a property of the rows.
        if not path.is_file():
            raise EvalArtifactError(f"{path.name} already exists and is not a file")
        try:
            existing = pq.read_table(path)
        except Exception as exc:
            raise EvalArtifactError(
                f"{path.name} already exists and is not a readable task-results table"
            ) from exc
        if not existing.schema.equals(schema) or existing.to_pylist() != rows:
            raise EvalArtifactError(
                f"{path.name} already exists with different immutable evidence"
            )
        return _file_hash(path)
    table = pa.Table.from_pylist(rows, schema=schema)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        pq.write_table(table, temporary, compression="zstd")
        result_hash = _file_hash(temporary)
        temporary.replace(path)
        return result_hash
    finally:
        temporary.unlink(missing_ok=True)


def _required_cache(
    path: Path | None,
    *,
    enabled: bool,
    expected_name: str,
    output_dir: Path,
) -> tuple[Path, str] | None:
    if not enabled:
        return None
    if path is None or not path.is_file():
        raise EvalArtifactError(
            f"{expected_name} is required by the eval output contract"
        )
    if path.name != expected_name:
        raise EvalArtifactError(f"expected cache file name {expected_name!r}")
    if path.resolve().parent != output_dir.resolve():
        raise EvalArtifactError(
            f"{expected_name} must live beside the eval artifacts it is hashed into"
        )
    return path, _file_hash(path)


def _validate_candidate_cache(path: Path | None) -> None:
    """Prove the weakest property a candidate cache always has to have.

    Without episode evidence there is nothing to compare the recorded turns
    against, so what remains provable is that no claimed request was left without
    a durable completion — the shape a crashed run leaves behind.
    """
    if path is None:
        return
    try:
        CandidateIOCache(path).validate_complete()
    except CandidateCacheError as exc:
        raise EvalArtifactError(
            f"replay cache failed publication validation: {exc.code}"
        ) from exc


def _validate_cache_evidence(
    *,
    candidate_cache_path: Path | None,
    tool_cache_path: Path | None,
    task_scores: tuple[ExecutableTaskScore, ...],
) -> None:
    """Refuse to hash a replay cache into a manifest it cannot account for."""
    if tool_cache_path is None:
        _validate_candidate_cache(candidate_cache_path)
        return
    try:
        candidate_cache = (
            CandidateIOCache(candidate_cache_path)
            if candidate_cache_path is not None
            else None
        )
        expected_episodes = {
            (score.candidate_alias, score.task_id): score.episode_hash
            for score in task_scores
        }
        if len(expected_episodes) != len(task_scores):
            raise EvalArtifactError("task scores repeat a candidate and task identity")
        cached_episodes: dict[tuple[str, str], str] = {}
        expected_turns: dict[str, tuple[str, str | None]] = {}
        for episode in ToolTraceCache(tool_cache_path).publication_evidence():
            identity = (episode.candidate_alias, episode.task_id)
            if identity in cached_episodes:
                raise EvalArtifactError(
                    "tool trace cache holds two episodes for one candidate and task"
                )
            cached_episodes[identity] = episode.episode_hash
            for turn in episode.turns:
                # Only semantic provider completions are replayable and therefore
                # durable in CandidateIOCache.  Transport/provider failures carry no
                # response hash by design; the tool-trace episode is their evidence and
                # they must remain retryable rather than being hardened into the
                # candidate response cache.
                if turn.response_hash is None:
                    continue
                observation = (turn.call_status, turn.response_hash)
                existing = expected_turns.get(turn.request_hash)
                if existing is not None and existing != observation:
                    raise EvalArtifactError(
                        "one candidate request hash maps to conflicting episode observations"
                    )
                expected_turns[turn.request_hash] = observation
        if cached_episodes != expected_episodes:
            raise EvalArtifactError(
                "tool trace cache does not exactly match the published task-score episodes"
            )
        if candidate_cache is not None:
            candidate_cache.validate_for_publication(expected_turns)
    except (CandidateCacheError, ToolTraceCacheError) as exc:
        raise EvalArtifactError(
            f"replay cache failed publication validation: {exc.code}"
        ) from exc


def _publish(
    *,
    eval_run_id: str,
    config: BfclEvalConfig,
    plan: EligibleEvalPlan,
    candidate_scores: tuple[CandidateAggregate, ...],
    rows: list[dict[str, Any]] | None,
    caches: dict[str, tuple[Path, str]],
) -> EvalArtifactSet:
    """Write report, task table, and manifest as one immutable artifact set.

    Both evaluation modes publish the same three files under the same identities;
    what differs is which task-result projection fills the table and which replay
    caches the manifest binds. Sharing the writer is what keeps a trace-only run
    from acquiring a differently shaped report than an executable one.
    """
    output_dir = config.outputs.output_dir
    report = eval_report_document(
        eval_run_id=eval_run_id,
        config=config,
        plan=plan,
        candidate_scores=candidate_scores,
    )
    report_path = output_dir / EVAL_REPORT_FILE
    report_hash = _write_immutable_bytes(report_path, _json_bytes(report))

    task_path: Path | None = None
    task_hash: str | None = None
    if rows is not None:
        task_path = output_dir / EVAL_TASK_RESULTS_FILE
        task_hash = _write_task_results(task_path, rows)

    manifest_path: Path | None = None
    manifest_hash: str | None = None
    if config.outputs.write_eval_manifest:
        artifacts: dict[str, Any] = {
            "eval_report": {
                "file": EVAL_REPORT_FILE,
                "content_hash": report_hash,
            }
        }
        if task_path is not None:
            artifacts["eval_task_results"] = {
                "file": EVAL_TASK_RESULTS_FILE,
                "content_hash": task_hash,
            }
        for name, (path, content_hash) in caches.items():
            artifacts[name] = {"file": path.name, "content_hash": content_hash}
        manifest = {
            "schema_version": EVAL_ARTIFACT_CONTRACT_VERSION,
            "eval_run_id": eval_run_id,
            "eval_scope": _eval_scope(candidate_scores),
            "source_run_id": plan.source_run_id,
            "eval_config_hash": config.eval_config_hash,
            "error_taxonomy_hash": ERROR_TAXONOMY_HASH,
            "plan_identity": plan.plan_identity,
            "scoring_policy_hash": plan.scoring_policy_hash,
            "source_verification_identity": plan.source_verification_identity,
            "runtime": runtime_metadata(),
            "source": config.source.semantic_payload(),
            "candidates": [
                config.candidate(alias).semantic_payload()
                for alias in plan.candidate_aliases
            ],
            "comparison_set": {
                "policy": plan.comparison_set,
                "task_count": plan.common.task_count,
                "task_ids_hash": plan.common.task_ids_hash,
            },
            "candidate_aggregates": {
                score.candidate_alias: score.aggregate_hash
                for score in candidate_scores
            },
            "artifacts": artifacts,
            "publication_allowed": (
                config.publication_allowed and plan.publication_allowed
            ),
        }
        manifest_path = output_dir / EVAL_MANIFEST_FILE
        manifest_hash = _write_immutable_bytes(
            manifest_path,
            _json_bytes(manifest),
        )
    return EvalArtifactSet(
        report_path=report_path,
        report_hash=report_hash,
        task_results_path=task_path,
        task_results_hash=task_hash,
        manifest_path=manifest_path,
        manifest_hash=manifest_hash,
    )


def write_executable_eval_artifacts(
    *,
    eval_run_id: str,
    config: BfclEvalConfig,
    plan: EligibleEvalPlan,
    candidate_scores: tuple[ExecutableCandidateScore, ...],
    task_scores: tuple[ExecutableTaskScore, ...],
    candidate_io_cache_path: Path | None,
    tool_trace_cache_path: Path | None,
) -> EvalArtifactSet:
    """Write report, task table, and manifest as one identity-checked artifact set."""
    grouped = _validate_complete_run(
        config=config,
        plan=plan,
        candidate_scores=candidate_scores,
        task_scores=task_scores,
    )
    output_dir = config.outputs.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_cache = _required_cache(
        candidate_io_cache_path,
        enabled=config.outputs.cache_candidate_responses,
        expected_name=CANDIDATE_IO_CACHE_FILE,
        output_dir=output_dir,
    )
    tool_cache = _required_cache(
        tool_trace_cache_path,
        enabled=config.outputs.cache_tool_results,
        expected_name=TOOL_TRACE_CACHE_FILE,
        output_dir=output_dir,
    )
    _validate_cache_evidence(
        candidate_cache_path=candidate_cache[0] if candidate_cache is not None else None,
        tool_cache_path=tool_cache[0] if tool_cache is not None else None,
        task_scores=task_scores,
    )
    caches: dict[str, tuple[Path, str]] = {}
    if candidate_cache is not None:
        caches["candidate_io_cache"] = candidate_cache
    if tool_cache is not None:
        caches["tool_trace_cache"] = tool_cache
    return _publish(
        eval_run_id=eval_run_id,
        config=config,
        plan=plan,
        candidate_scores=candidate_scores,
        rows=(
            [
                executable_task_result(score)
                for alias in plan.candidate_aliases
                for score in grouped[alias]
            ]
            if config.outputs.write_task_results
            else None
        ),
        caches=caches,
    )


def write_trace_eval_artifacts(
    *,
    eval_run_id: str,
    config: BfclEvalConfig,
    plan: EligibleEvalPlan,
    candidate_scores: tuple[TraceCandidateScore, ...],
    task_scores: tuple[TraceTaskScore, ...],
    candidate_io_cache_path: Path | None,
) -> EvalArtifactSet:
    """Publish one trace-only run's report, task table, and manifest.

    There is no tool-trace cache to bind: trace evaluation executes no tool, and
    the results it released are the ones the benchmark recorded, which source
    verification already proved by content hash. The candidate cache is therefore
    the whole of a trace run's replay evidence, and it is required to prove it
    left no claimed request without a completion.
    """
    grouped = _validate_complete_run(
        config=config,
        plan=plan,
        candidate_scores=candidate_scores,
        task_scores=task_scores,
    )
    output_dir = config.outputs.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_cache = _required_cache(
        candidate_io_cache_path,
        enabled=config.outputs.cache_candidate_responses,
        expected_name=CANDIDATE_IO_CACHE_FILE,
        output_dir=output_dir,
    )
    _validate_candidate_cache(
        candidate_cache[0] if candidate_cache is not None else None
    )
    return _publish(
        eval_run_id=eval_run_id,
        config=config,
        plan=plan,
        candidate_scores=candidate_scores,
        rows=(
            [
                trace_task_result(score)
                for alias in plan.candidate_aliases
                for score in grouped[alias]
            ]
            if config.outputs.write_task_results
            else None
        ),
        caches=(
            {"candidate_io_cache": candidate_cache}
            if candidate_cache is not None
            else {}
        ),
    )


__all__ = [
    "EVAL_ARTIFACT_CONTRACT_VERSION",
    "EVAL_MANIFEST_FILE",
    "EVAL_REPORT_FILE",
    "EVAL_TASK_RESULTS_FILE",
    "CandidateAggregate",
    "EvalArtifactError",
    "EvalArtifactSet",
    "EvalTaskScore",
    "eval_report_document",
    "executable_task_result",
    "trace_task_result",
    "write_executable_eval_artifacts",
    "write_trace_eval_artifacts",
]
