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

"""Identity contract for one replayable executable tool-trace observation."""

from __future__ import annotations

import hashlib
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, StrictStr, model_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.contamination_contract import (
    EligibleEvalPlan,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_contract import (
    EXECUTABLE_CONTRACT_VERSION,
    ExecutableEpisode,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_projection import (
    ExecutableTaskSpec,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.schemas import EvalCandidate
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_contract import (
    VerifiedEvalSource,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import ContentHash
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

TOOL_TRACE_CACHE_CONTRACT_VERSION: Final = "1.0"
TOOL_TRACE_CACHE_FILE: Final = "tool_trace_cache.jsonl"


def _sha256_json(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


class ToolTraceRequest(BaseModel):
    """Everything that must be identical before an episode may be replayed.

    The eval-config hash binds limits and continuation policy. The task-spec hash
    binds model-facing tools, scripted turns, dependency declarations, assertion
    inputs, and mutation policy. Therefore a hit is the same executable question,
    not merely the same tool name and arguments.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = TOOL_TRACE_CACHE_CONTRACT_VERSION
    executable_contract_version: Literal["1.2"] = EXECUTABLE_CONTRACT_VERSION
    candidate_alias: StrictStr
    canonical_model_identity: StrictStr
    task_id: StrictStr
    plan_identity: ContentHash
    eval_config_hash: ContentHash
    scoring_policy_hash: ContentHash
    source_verification_identity: ContentHash
    oracle_verification_identity: ContentHash
    script_hash: ContentHash
    task_spec_hash: ContentHash

    @model_validator(mode="after")
    def _non_empty(self) -> ToolTraceRequest:
        for field in ("candidate_alias", "canonical_model_identity", "task_id"):
            if not getattr(self, field).strip():
                raise ValueError(f"{field} must be non-empty")
        return self

    def semantic_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @property
    def trace_key(self) -> str:
        return _sha256_json(self.semantic_payload())

    def as_document(self) -> dict[str, Any]:
        return {**self.semantic_payload(), "trace_key": self.trace_key}

    def accepts(self, episode: ExecutableEpisode) -> bool:
        return (
            episode.candidate_alias == self.candidate_alias
            and episode.canonical_model_identity == self.canonical_model_identity
            and episode.task_id == self.task_id
            and episode.plan_identity == self.plan_identity
            and episode.eval_config_hash == self.eval_config_hash
            and episode.source_verification_identity
            == self.source_verification_identity
            and episode.oracle_verification_identity
            == self.oracle_verification_identity
            and episode.script_hash == self.script_hash
            and episode.task_spec_hash == self.task_spec_hash
        )


def build_tool_trace_request(
    *,
    candidate: EvalCandidate,
    task: ExecutableTaskSpec,
    source: VerifiedEvalSource,
    plan: EligibleEvalPlan,
) -> ToolTraceRequest:
    """Bind a cache lookup to the same identities the driver authorizes."""
    return ToolTraceRequest(
        candidate_alias=candidate.alias,
        canonical_model_identity=candidate.canonical_model_identity,
        task_id=task.task_id,
        plan_identity=plan.plan_identity,
        eval_config_hash=plan.eval_config_hash,
        scoring_policy_hash=plan.scoring_policy_hash,
        source_verification_identity=source.verification_identity,
        oracle_verification_identity=task.oracle_verification_identity,
        script_hash=task.script.script_hash,
        task_spec_hash=task.task_spec_hash,
    )


__all__ = [
    "TOOL_TRACE_CACHE_CONTRACT_VERSION",
    "TOOL_TRACE_CACHE_FILE",
    "ToolTraceRequest",
    "build_tool_trace_request",
]
