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

"""The one path from ``benchmark.parquet`` to an export writer.

Every format Stage 12 emits is a re-encoding of the same published rows, so the
parquet is decoded exactly once, here:

.. code-block:: text

    benchmark.parquet row
    -> decode canonical tools/arguments   (CanonicalExportRow.from_benchmark_row)
    -> canonical export object            (CanonicalExportProjection)
    -> BFCL writer or NeMo writer

A writer that opened the parquet itself would be a second decoder, and two
decoders are two chances to disagree about what the benchmark asks: one could
read an ``arguments`` map as a JSON object and the other as a list of pairs, and
the resulting exports would score the same model differently. Writers therefore
receive :class:`CanonicalExportProjection` and never a path.

The projection also derives the structure a writer would otherwise have to
reconstruct. ``expected_tool_calls`` carries ``turn_index``, ``call_group``, and
``position_in_group`` per call, but a BFCL multi-turn record needs those calls
*grouped* — one group per assistant message, in turn order. Deriving that once,
and checking it against the rendered ``messages``, is what keeps a parallel group
from being flattened into a sequence by one writer and not the other.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr, model_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    CanonicalExportRow,
    ContentHash,
    ExportedToolCall,
    NonNegativeInt,
    PositiveInt,
    decode_canonical_json,
    json_equal,
    tool_names,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.publication_contract import (
    PUBLICATION_BENCHMARK_TABLE,
)

EXPORT_PROJECTION_VERSION = "1.0"


class ExportProjectionError(ValueError):
    """The published benchmark cannot be projected into a canonical export object."""


class CanonicalCallGroup(BaseModel):
    """Every expected call one assistant message issues at once.

    ``turn_index`` is the ordinal of that assistant message among all assistant
    messages in the rendered conversation, which is how Stage 7 assigned it. A
    group with more than one call is a parallel group: the model is expected to
    issue them together, so a writer that emitted them as consecutive turns would
    turn a parallel-calling task into a sequential one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_index: NonNegativeInt
    call_group: NonNegativeInt
    # Which user request this group answers. BFCL's multi-turn corpus keys its
    # possible answers by user turn, not by assistant turn, so a writer needs both.
    user_turn_index: NonNegativeInt
    calls: tuple[ExportedToolCall, ...]
    is_parallel: StrictBool

    @model_validator(mode="after")
    def validate_group(self) -> CanonicalCallGroup:
        if not self.calls:
            raise ValueError("a call group with no call is not a group")
        if self.is_parallel != (len(self.calls) > 1):
            raise ValueError("a group is parallel exactly when it issues more than one call")
        if any(call.turn_index != self.turn_index for call in self.calls):
            raise ValueError("every call in a group is issued by the same assistant turn")
        if any(call.call_group != self.call_group for call in self.calls):
            raise ValueError("every call in a group carries the same call_group label")
        if [call.position_in_group for call in self.calls] != list(range(len(self.calls))):
            raise ValueError("a group's calls must occupy positions 0..n-1 in order")
        return self


class CanonicalConversationPlan(BaseModel):
    """How one task's expected calls are distributed over its conversation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = EXPORT_PROJECTION_VERSION
    task_id: StrictStr
    user_turns: PositiveInt
    assistant_turns: PositiveInt
    is_multi_turn: StrictBool
    groups: tuple[CanonicalCallGroup, ...] = ()

    @model_validator(mode="after")
    def validate_plan(self) -> CanonicalConversationPlan:
        if self.is_multi_turn != (self.user_turns > 1):
            raise ValueError(
                f"task {self.task_id!r} is multi-turn exactly when the conversation asks more than one user turn"
            )
        turns = [group.turn_index for group in self.groups]
        if turns != sorted(turns):
            raise ValueError(f"task {self.task_id!r} call groups must stay in assistant-turn order")
        if len(set(turns)) != len(turns):
            raise ValueError(f"task {self.task_id!r} gives one assistant turn two call groups")
        if turns and max(turns) >= self.assistant_turns:
            raise ValueError(f"task {self.task_id!r} places a call group past its last assistant turn")
        user_turns = [group.user_turn_index for group in self.groups]
        if user_turns != sorted(user_turns):
            raise ValueError(f"task {self.task_id!r} answers an earlier user turn after a later one")
        if user_turns and max(user_turns) >= self.user_turns:
            raise ValueError(f"task {self.task_id!r} answers a user turn the conversation never asks")
        return self

    @property
    def call_count(self) -> int:
        return sum(len(group.calls) for group in self.groups)

    @property
    def parallel_groups(self) -> tuple[CanonicalCallGroup, ...]:
        return tuple(group for group in self.groups if group.is_parallel)

    @property
    def call_group_payload(self) -> tuple[dict[str, Any], ...]:
        """Describe the grouping for a writer, by index into ``expected_tool_calls``.

        Indexes rather than repeated call bodies: a writer that restated the calls
        here could restate them differently from the record's own list, and the
        export would then carry two disagreeing answers for one task.
        """
        payload: list[dict[str, Any]] = []
        index = 0
        for group in self.groups:
            indexes = list(range(index, index + len(group.calls)))
            index += len(group.calls)
            payload.append(
                {
                    "turn_index": group.turn_index,
                    "call_group": group.call_group,
                    "user_turn_index": group.user_turn_index,
                    "is_parallel": group.is_parallel,
                    "calls": indexes,
                }
            )
        return tuple(payload)

    @property
    def calls_by_user_turn(self) -> tuple[tuple[ExportedToolCall, ...], ...]:
        """Expected calls per user turn, which is how BFCL keys its answers.

        Every user turn gets an entry, empty included: a clarifying turn that
        triggers no call is part of the conversation a scorer replays, and dropping
        it would shift every later turn's answers onto the wrong request.
        """
        turns: list[list[ExportedToolCall]] = [[] for _ in range(self.user_turns)]
        for group in self.groups:
            turns[group.user_turn_index].extend(group.calls)
        return tuple(tuple(turn) for turn in turns)


class ProjectionSource(BaseModel):
    """The published file this projection decoded, so an export can cite it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = EXPORT_PROJECTION_VERSION
    file: StrictStr
    content_hash: ContentHash
    rows: NonNegativeInt

    @model_validator(mode="after")
    def validate_source(self) -> ProjectionSource:
        if self.file != PUBLICATION_BENCHMARK_TABLE:
            raise ValueError(f"an export projects {PUBLICATION_BENCHMARK_TABLE}, not {self.file!r}")
        return self


class ProjectionProvenance(BaseModel):
    """What every projected row agrees on, for an export to declare once.

    A bundle descriptor has to name the pack, version, and tier it came from, and
    a writer that read those off the first row would mislabel a projection whose
    rows disagreed. Deriving them here means a disagreement stops the export
    instead of being silently resolved in favour of row zero.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = EXPORT_PROJECTION_VERSION
    pack_id: StrictStr
    pack_version: StrictStr
    expt_name: StrictStr
    tier: StrictStr
    gold_eligible: StrictBool
    system_prompt_ids: tuple[StrictStr, ...]
    languages: tuple[StrictStr, ...]
    turn_policies: tuple[StrictStr, ...]
    paraphrase_models: tuple[StrictStr, ...] = ()

    @model_validator(mode="after")
    def validate_provenance(self) -> ProjectionProvenance:
        for name in ("system_prompt_ids", "languages", "turn_policies", "paraphrase_models"):
            values: tuple[str, ...] = getattr(self, name)
            if list(values) != sorted(set(values)):
                raise ValueError(f"{name} must be the sorted distinct values the rows carry")
        if not self.system_prompt_ids:
            raise ValueError("a projection with no system prompt cannot describe what the model was told")
        if not self.languages:
            raise ValueError("a projection with no language cannot be localized or reported on")
        if not self.turn_policies:
            raise ValueError("a projection with no turn policy cannot be sliced by conversation shape")
        return self


class CanonicalExportProjection(BaseModel):
    """The published benchmark, decoded once, for every writer to re-encode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = EXPORT_PROJECTION_VERSION
    source: ProjectionSource
    provenance: ProjectionProvenance
    rows: tuple[CanonicalExportRow, ...]
    plans: tuple[CanonicalConversationPlan, ...]

    @model_validator(mode="after")
    def validate_projection(self) -> CanonicalExportProjection:
        if not self.rows:
            raise ValueError("an export projection of no row would describe an empty benchmark")
        task_ids = [row.task_id for row in self.rows]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("a projection may carry one row per task")
        if [plan.task_id for plan in self.plans] != task_ids:
            raise ValueError("every projected row needs its conversation plan, in publication order")
        if self.source.rows != len(self.rows):
            raise ValueError("the projection must account for every row of the file it read")
        for row, plan in zip(self.rows, self.plans, strict=True):
            if row.held_out_hit is True:
                raise ValueError(f"task {row.task_id!r} is a held-out hit and cannot enter a compatibility export")
            if plan.call_count != len(row.expected_tool_calls):
                raise ValueError(f"task {row.task_id!r} conversation plan loses an expected call")
        return self

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Publication order, which every writer must preserve."""
        return tuple(row.task_id for row in self.rows)

    def row(self, task_id: str) -> CanonicalExportRow:
        for row in self.rows:
            if row.task_id == task_id:
                return row
        raise KeyError(task_id)

    def plan(self, task_id: str) -> CanonicalConversationPlan:
        for plan in self.plans:
            if plan.task_id == task_id:
                return plan
        raise KeyError(task_id)


def conversation_plan(row: CanonicalExportRow) -> CanonicalConversationPlan:
    """Group one row's expected calls by the assistant message that issues them.

    The rendered ``messages`` are the check, not a second opinion: each assistant
    message with tool calls must be claimed by exactly the expected calls whose
    ``turn_index`` is that message's ordinal, with the same names and arguments in
    the same order. A row where the two disagree would export a turn index that
    points at the wrong turn, and a multi-turn scorer would grade the wrong reply.
    """
    calls_by_turn: dict[int, list[ExportedToolCall]] = {}
    for call in row.expected_tool_calls:
        calls_by_turn.setdefault(call.turn_index, []).append(call)

    user_turns = 0
    assistant_turns = 0
    groups: list[CanonicalCallGroup] = []
    for message in row.messages:
        if message.role == "user":
            user_turns += 1
            continue
        if message.role != "assistant":
            continue
        ordinal = assistant_turns
        assistant_turns += 1
        expected = calls_by_turn.pop(ordinal, None)
        if not message.tool_calls:
            if expected is not None:
                raise ExportProjectionError(
                    f"task {row.task_id!r} expects {len(expected)} call(s) at assistant turn {ordinal}, "
                    "but that turn issues none"
                )
            continue
        if expected is None:
            raise ExportProjectionError(
                f"task {row.task_id!r} issues {len(message.tool_calls)} call(s) at assistant turn {ordinal} "
                "that no expected call claims"
            )
        if len(expected) != len(message.tool_calls):
            raise ExportProjectionError(
                f"task {row.task_id!r} assistant turn {ordinal} issues {len(message.tool_calls)} call(s) "
                f"but expects {len(expected)}"
            )
        for wire, call in zip(message.tool_calls, expected, strict=True):
            arguments = decode_canonical_json(
                wire.function.arguments,
                label=f"task {row.task_id!r} assistant turn {ordinal} arguments",
            )
            if wire.function.name != call.function_name or not json_equal(arguments, call.arguments):
                raise ExportProjectionError(
                    f"task {row.task_id!r} assistant turn {ordinal} does not issue the call it expects"
                )
        labels = {call.call_group for call in expected}
        if len(labels) != 1:
            raise ExportProjectionError(
                f"task {row.task_id!r} assistant turn {ordinal} mixes call groups {sorted(labels)}; "
                "one assistant message issues one group"
            )
        if not user_turns:
            raise ExportProjectionError(
                f"task {row.task_id!r} calls a tool at assistant turn {ordinal}, before it has been asked anything"
            )
        groups.append(
            CanonicalCallGroup(
                turn_index=ordinal,
                call_group=labels.pop(),
                user_turn_index=user_turns - 1,
                calls=tuple(expected),
                is_parallel=len(expected) > 1,
            )
        )
    if calls_by_turn:
        # ``CanonicalExportRow`` already pairs every wire call with an expected one,
        # so a leftover turn means that pairing changed; say so rather than export a
        # call the conversation does not issue.
        raise ExportProjectionError(
            f"task {row.task_id!r} expects calls at assistant turn(s) {sorted(calls_by_turn)} "
            "that its conversation never reaches"
        )
    try:
        return CanonicalConversationPlan(
            task_id=row.task_id,
            user_turns=user_turns,
            assistant_turns=assistant_turns,
            is_multi_turn=row.is_multi_turn,
            groups=tuple(groups),
        )
    except ValueError as exc:
        raise ExportProjectionError(str(exc)) from exc


def derive_provenance(rows: Sequence[CanonicalExportRow]) -> ProjectionProvenance:
    """Collect what the projection may claim about its source, or refuse to claim it."""
    if not rows:
        raise ExportProjectionError("no row was projected, so there is no provenance to declare")
    for field in ("pack_id", "pack_version", "tier"):
        values = sorted({str(getattr(row, field)) for row in rows})
        if len(values) != 1:
            raise ExportProjectionError(
                f"one export describes one published benchmark, but its rows carry {field} {values}"
            )
    languages = sorted({str(row.metadata["language"]) for row in rows})
    if any(not language for language in languages):
        raise ExportProjectionError("a projected row declares no language")
    # The run that produced the rows, which an export descriptor cites as its
    # lineage. Two runs' rows in one projection would make that citation false.
    expt_names = sorted({str(row.metadata["expt_name"]) for row in rows})
    if len(expt_names) != 1:
        raise ExportProjectionError(f"one export describes one run, but its rows carry expt_name {expt_names}")
    try:
        return ProjectionProvenance(
            pack_id=rows[0].pack_id,
            pack_version=rows[0].pack_version,
            expt_name=expt_names[0],
            tier=rows[0].tier,
            # Only true when it holds for the whole projection: an export is
            # consumed as one artifact, and one non-gold row makes it non-gold.
            gold_eligible=all(row.gold_eligible for row in rows),
            system_prompt_ids=tuple(sorted({row.system_prompt_id for row in rows})),
            languages=tuple(languages),
            turn_policies=tuple(sorted({row.turn_policy for row in rows})),
            paraphrase_models=tuple(
                sorted({row.paraphrase_model_canonical for row in rows if row.paraphrase_model_canonical})
            ),
        )
    except ValueError as exc:
        raise ExportProjectionError(str(exc)) from exc


def project_benchmark_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source: ProjectionSource,
) -> CanonicalExportProjection:
    """Decode already-read parquet rows, preserving publication order."""
    canonical: list[CanonicalExportRow] = []
    for position, row in enumerate(rows):
        try:
            canonical.append(CanonicalExportRow.from_benchmark_row(row))
        except ValueError as exc:
            raise ExportProjectionError(f"{source.file} row {position} cannot be projected: {exc}") from exc
    plans = [conversation_plan(row) for row in canonical]
    try:
        return CanonicalExportProjection(
            source=source,
            provenance=derive_provenance(canonical),
            rows=tuple(canonical),
            plans=tuple(plans),
        )
    except ValueError as exc:
        raise ExportProjectionError(str(exc)) from exc


def project_published_benchmark(
    benchmark_path: Path,
    *,
    expected_content_hash: str | None = None,
    expected_task_ids: Sequence[str] | None = None,
) -> CanonicalExportProjection:
    """Read the published benchmark and project it, once, for every writer.

    ``expected_content_hash`` binds the projection to the file Stage 12 verified,
    so an export cannot be built from a parquet that was replaced between
    publication and writing. ``expected_task_ids`` binds it to the publication
    order the manifest reports.
    """
    import pyarrow.parquet as pq

    from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import benchmark_schema

    if not benchmark_path.is_file():
        raise ExportProjectionError(f"there is no published benchmark at {benchmark_path}")
    table = pq.read_table(benchmark_path)
    if not table.schema.equals(benchmark_schema()):
        raise ExportProjectionError(
            f"{benchmark_path.name} was not written with the published benchmark schema, so it cannot be projected"
        )
    content_hash = f"sha256:{hashlib.sha256(benchmark_path.read_bytes()).hexdigest()}"
    if expected_content_hash is not None and content_hash != expected_content_hash:
        raise ExportProjectionError(
            f"{benchmark_path.name} changed after publication verified it; refusing to export a different benchmark"
        )
    projection = project_benchmark_rows(
        table.to_pylist(),
        source=ProjectionSource(
            file=PUBLICATION_BENCHMARK_TABLE,
            content_hash=content_hash,
            rows=table.num_rows,
        ),
    )
    if expected_task_ids is not None and projection.task_ids != tuple(str(task_id) for task_id in expected_task_ids):
        raise ExportProjectionError(
            "the projected rows are not the published rows, in publication order; "
            "an export written from these would not be the benchmark the manifest describes"
        )
    return projection


def projection_lineage(projection: CanonicalExportProjection) -> dict[str, Any]:
    """Describe the projected source for an export descriptor or the manifest."""
    return {
        "schema_version": EXPORT_PROJECTION_VERSION,
        "source": {
            "file": projection.source.file,
            "content_hash": projection.source.content_hash,
            "rows": projection.source.rows,
        },
        "pack_id": projection.provenance.pack_id,
        "pack_version": projection.provenance.pack_version,
        "expt_name": projection.provenance.expt_name,
        "tier": projection.provenance.tier,
        "gold_eligible": projection.provenance.gold_eligible,
        "system_prompt_ids": list(projection.provenance.system_prompt_ids),
        "languages": list(projection.provenance.languages),
        "turn_policies": list(projection.provenance.turn_policies),
        "paraphrase_models": list(projection.provenance.paraphrase_models),
        "tools_exposed": sorted({name for row in projection.rows for name in tool_names(row.tools)}),
        "multi_turn_rows": sum(plan.is_multi_turn for plan in projection.plans),
        "parallel_call_rows": sum(bool(plan.parallel_groups) for plan in projection.plans),
    }
