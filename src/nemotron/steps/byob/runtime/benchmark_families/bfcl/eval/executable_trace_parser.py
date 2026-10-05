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

"""Project executable evidence into the shared normalized trace scoring view."""

from __future__ import annotations

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.error_taxonomy import (
    EXECUTABLE_NON_CANDIDATE_STOPS,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_contract import (
    ExecutableEpisode,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_projection import (
    ExecutableTaskSpec,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_scoring_errors import (
    ExecutableEvidenceError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.trace_parser import (
    ParsedCall,
    ParsedTrace,
    ParsedTurn,
)


def parse_executable_trace(
    episode: ExecutableEpisode,
    task: ExecutableTaskSpec,
) -> ParsedTrace:
    """Read canonical turns and calls without re-parsing provider or oracle bytes."""

    _same_replay(episode, task)
    turns = tuple(
        ParsedTurn(
            turn_index=observed.turn_index,
            kind=(
                "tool_calls"
                if task.script.turn(observed.turn_index).expects_tool_calls
                else "text"
            ),
            call_status=observed.call_status,
            advanced=observed.advanced,
            finish_reason=observed.finish_reason,
            assistant_content=observed.assistant_content,
            calls=tuple(
                ParsedCall(
                    turn_index=observed.turn_index,
                    position_in_turn=position,
                    provider_index=call.index,
                    id=call.id,
                    type=call.type,
                    function_name=call.function_name,
                    arguments_status=call.arguments_status,
                    parsed_arguments=call.parsed_arguments,
                )
                for position, call in enumerate(observed.tool_calls)
            ),
            detail=observed.detail,
        )
        for observed in episode.observed
    )
    return ParsedTrace(
        task_id=episode.task_id,
        candidate_alias=episode.candidate_alias,
        canonical_model_identity=episode.canonical_model_identity,
        plan_identity=episode.plan_identity,
        source_verification_identity=episode.source_verification_identity,
        script_hash=episode.script_hash,
        episode_hash=episode.episode_hash,
        status=episode.status,
        non_candidate_stop=episode.status in EXECUTABLE_NON_CANDIDATE_STOPS,
        scripted_turns=len(task.script.turns),
        turns=turns,
        unsent_turn_indexes=tuple(range(len(turns), len(task.script.turns))),
    )


def _same_replay(episode: ExecutableEpisode, task: ExecutableTaskSpec) -> None:
    """Refuse evidence and a task spec that do not identify one live replay."""

    checks = (
        ("task_id", episode.task_id, task.task_id),
        ("candidate_alias", episode.candidate_alias, task.candidate_alias),
        (
            "canonical_model_identity",
            episode.canonical_model_identity,
            task.canonical_model_identity,
        ),
        ("plan_identity", episode.plan_identity, task.plan_identity),
        ("eval_config_hash", episode.eval_config_hash, task.eval_config_hash),
        (
            "source_verification_identity",
            episode.source_verification_identity,
            task.source_verification_identity,
        ),
        (
            "oracle_verification_identity",
            episode.oracle_verification_identity,
            task.oracle_verification_identity,
        ),
        ("script_hash", episode.script_hash, task.script.script_hash),
        ("task_spec_hash", episode.task_spec_hash, task.task_spec_hash),
    )
    for field, actual, expected in checks:
        if actual != expected:
            raise ExecutableEvidenceError(
                f"eval.episode.{field}",
                "does not identify the executable task supplied to the parser",
                actual=actual,
                expected=str(expected),
                recovery=(
                    "parse the episode with the exact ExecutableTaskSpec used to "
                    "drive it"
                ),
            )
    if len(episode.observed) > len(task.script.turns):
        raise ExecutableEvidenceError(
            "eval.episode.observed",
            "holds more assistant turns than the executable script has",
            actual=len(episode.observed),
            expected=f"at most {len(task.script.turns)} turns",
            recovery="re-drive the task and retain only canonical episode evidence",
        )
    if episode.status == "completed" and len(episode.observed) != len(
        task.script.turns
    ):
        raise ExecutableEvidenceError(
            "eval.episode.status",
            "claims completion before every executable turn was observed",
            actual=len(episode.observed),
            expected=f"exactly {len(task.script.turns)} observed turns",
            recovery=(
                "retain the original incomplete terminal status, or re-drive the "
                "episode to the end of the executable script"
            ),
        )


__all__ = ["EXECUTABLE_NON_CANDIDATE_STOPS", "parse_executable_trace"]
