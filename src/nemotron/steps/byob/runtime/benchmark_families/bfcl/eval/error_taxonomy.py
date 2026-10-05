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

"""Stable machine-readable attribution for BFCL evaluation failures."""

from __future__ import annotations

import hashlib
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

ERROR_TAXONOMY_CONTRACT_VERSION: Final = "1.2"
ErrorAttribution = Literal[
    "success",
    "candidate",
    "infrastructure",
    "evidence",
    "fatal_setup",
    "not_applicable",
]

TRACE_EPISODE_ATTRIBUTION: Final[dict[str, ErrorAttribution]] = {
    "completed": "success",
    "candidate_mismatch": "candidate",
    "malformed_response": "candidate",
    "candidate_call_failed": "infrastructure",
    "unusable_tool_call_ids": "candidate",
    "max_turns_exceeded": "infrastructure",
    "episode_timeout": "infrastructure",
}
EXECUTABLE_EPISODE_ATTRIBUTION: Final[dict[str, ErrorAttribution]] = {
    **TRACE_EPISODE_ATTRIBUTION,
    "oracle_reset_failed": "infrastructure",
    "oracle_call_failed": "infrastructure",
    "oracle_timeout": "infrastructure",
    "oracle_result_malformed": "infrastructure",
    "oracle_state_failed": "infrastructure",
    "oracle_session_failed": "infrastructure",
    "unknown_commit_state": "infrastructure",
    "dependency_resolution_failed": "infrastructure",
    "assertion_infrastructure_failed": "infrastructure",
    "confirmation_not_earned": "candidate",
}
TRACE_NON_CANDIDATE_STOPS: Final = frozenset(
    status
    for status, attribution in TRACE_EPISODE_ATTRIBUTION.items()
    if attribution == "infrastructure"
)
EXECUTABLE_NON_CANDIDATE_STOPS: Final = frozenset(
    status
    for status, attribution in EXECUTABLE_EPISODE_ATTRIBUTION.items()
    if attribution == "infrastructure"
)

# Exception codes are fatal setup/run failures: they produce no task score. Pack
# business errors deliberately do not appear here; they are valid oracle data.
FATAL_EVAL_ERROR_CODES: Final = frozenset(
    {
        "eval_config_invalid",
        "eval_config_schema_invalid",
        "eval_config_path_invalid",
        "candidate_identity_invalid",
        "candidate_revision_mutable",
        "secret_in_eval_config",
        "eval_publication_policy_violation",
        "unsupported_eval_mode",
        "eval_source_invalid",
        "eval_source_manifest_invalid",
        "eval_source_manifest_drift",
        "eval_source_benchmark_hash_mismatch",
        "eval_source_benchmark_schema_mismatch",
        "eval_source_publication_invalid",
        "eval_source_task_index_invalid",
        "eval_source_model_exposure_invalid",
        "eval_source_oracle_pack_drift",
        "eval_source_oracle_resource_mismatch",
        "eval_source_translation_lineage_invalid",
        "eval_source_changed_during_eval",
        "eval_contamination_invalid",
        "eval_contamination_candidate_exposed",
        "eval_contamination_unresolved",
        "eval_contamination_empty_task_set",
        "eval_contamination_task_set_inconsistent",
        "eval_contamination_plan_drift",
        "eval_candidate_client_invalid",
        "eval_candidate_credentials_missing",
        "eval_candidate_authentication_failed",
        "eval_candidate_request_invalid",
        "eval_candidate_provider_extension_invalid",
        "eval_candidate_response_invalid",
        "eval_candidate_cache_invalid",
        "eval_candidate_cache_conflict",
        "eval_conversation_invalid",
        "eval_conversation_script_invalid",
        "eval_conversation_unauthorized",
        "eval_conversation_answer_key_leak",
        "eval_conversation_transition_invalid",
        "eval_executable_invalid",
        "eval_executable_projection_invalid",
        "eval_executable_unauthorized",
        "eval_oracle_session_failed",
        "eval_oracle_reset_failed",
        "eval_oracle_call_failed",
        "eval_oracle_state_failed",
        "eval_assertion_infrastructure_failed",
        "eval_tool_trace_cache_invalid",
        "eval_tool_trace_cache_conflict",
        "eval_executable_scoring_invalid",
        "eval_executable_evidence_mismatch",
        "eval_executable_scoring_policy_unsupported",
        "eval_executable_aggregation_invalid",
        "eval_trace_scoring_invalid",
        "eval_trace_evidence_mismatch",
        "eval_trace_scoring_policy_unsupported",
        "eval_trace_aggregation_invalid",
        "eval_artifact_invalid",
        "eval_runner_invalid",
        "eval_runner_mode_unsupported",
        "eval_nemo_adapter_invalid",
        "eval_cli_invalid",
        "eval_cli_runtime_failed",
        "eval_cli_artifact_conflict",
        "eval_cli_framework_not_installed",
        "eval_cli_framework_version_mismatch",
    }
)
METRIC_NOT_APPLICABLE_CODES: Final = frozenset(
    {
        "metric.all_assertions_not_applicable",
        "metric.assertion_evidence_incomplete",
        "metric.gate_not_applicable",
        "metric.no_applicable_task",
        "metric.no_attempted_call",
        "metric.no_declared_assertion",
        "metric.no_evaluated_turn",
        "metric.no_expected_call",
        "metric.no_final_answer_assertion",
        "metric.no_state_assertion",
        "metric.no_text_milestone",
        "metric.path_evidence_incomplete",
    }
)
REASON_CODE_NAMESPACES: Final = frozenset(
    {
        "arguments",
        "assertion",
        "assertions",
        "call_grouping",
        "call_ordering",
        "candidate",
        "commit_state",
        "commit_state_known",
        "conversation",
        "dependency",
        "dependency_resolution",
        "episode",
        "executable",
        "executable_completion",
        "metric",
        "oracle",
        "oracle_execution",
        "schema_valid",
        "text_turn",
        "tool_execution",
        "tool_selection",
        "trace",
        "trace_completion",
        "turn",
    }
)


class EvalFailureRecord(BaseModel):
    """One normalized failure projected into a task-results artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    layer: Literal["gate", "episode", "setup"]
    code: StrictStr
    attribution: ErrorAttribution
    subject: StrictStr

    @model_validator(mode="after")
    def _known_shape(self) -> EvalFailureRecord:
        if not self.code.strip() or not self.subject.strip():
            raise ValueError("failure code and subject must be non-empty")
        if self.layer == "setup" and self.code not in FATAL_EVAL_ERROR_CODES:
            raise ValueError("setup failures use a registered eval error code")
        if self.layer == "setup" and self.attribution != "fatal_setup":
            raise ValueError("setup failures have fatal_setup attribution")
        return self

    def as_document(self) -> dict[str, str]:
        return self.model_dump(mode="json")


def episode_attribution(
    status: str,
    *,
    executable: bool,
) -> ErrorAttribution:
    mapping = (
        EXECUTABLE_EPISODE_ATTRIBUTION if executable else TRACE_EPISODE_ATTRIBUTION
    )
    try:
        return mapping[status]
    except KeyError as exc:
        raise ValueError(f"unregistered episode status {status!r}") from exc


def episode_failure_record(
    status: str,
    *,
    executable: bool,
) -> EvalFailureRecord | None:
    """Attribute a terminal episode status, or None when it completed."""
    attribution = episode_attribution(status, executable=executable)
    if attribution == "success":
        return None
    return EvalFailureRecord(
        layer="episode",
        code=f"episode.{status}",
        attribution=attribution,
        subject="episode",
    )


def gate_failure_record(
    *,
    gate: str,
    reason_code: str,
    failure_class: str,
) -> EvalFailureRecord:
    if failure_class not in {"candidate", "infrastructure", "evidence"}:
        raise ValueError(
            f"failed gate {gate!r} has invalid failure class {failure_class!r}"
        )
    namespace = reason_code.partition(".")[0]
    if namespace not in REASON_CODE_NAMESPACES:
        raise ValueError(f"unregistered reason-code namespace {namespace!r}")
    return EvalFailureRecord(
        layer="gate",
        code=reason_code,
        attribution=failure_class,
        subject=gate,
    )


def taxonomy_payload() -> dict[str, Any]:
    return {
        "schema_version": ERROR_TAXONOMY_CONTRACT_VERSION,
        "episode_attribution": {
            "trace": TRACE_EPISODE_ATTRIBUTION,
            "executable": EXECUTABLE_EPISODE_ATTRIBUTION,
        },
        "fatal_eval_error_codes": sorted(FATAL_EVAL_ERROR_CODES),
        "metric_not_applicable_codes": sorted(METRIC_NOT_APPLICABLE_CODES),
        "reason_code_namespaces": sorted(REASON_CODE_NAMESPACES),
    }


ERROR_TAXONOMY_HASH: Final = (
    "sha256:"
    + hashlib.sha256(canonical_json(taxonomy_payload()).encode("utf-8")).hexdigest()
)


__all__ = [
    "ERROR_TAXONOMY_CONTRACT_VERSION",
    "ERROR_TAXONOMY_HASH",
    "EXECUTABLE_EPISODE_ATTRIBUTION",
    "EXECUTABLE_NON_CANDIDATE_STOPS",
    "ErrorAttribution",
    "EvalFailureRecord",
    "FATAL_EVAL_ERROR_CODES",
    "METRIC_NOT_APPLICABLE_CODES",
    "REASON_CODE_NAMESPACES",
    "TRACE_EPISODE_ATTRIBUTION",
    "TRACE_NON_CANDIDATE_STOPS",
    "episode_attribution",
    "episode_failure_record",
    "gate_failure_record",
    "taxonomy_payload",
]
