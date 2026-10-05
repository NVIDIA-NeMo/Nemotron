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

"""The ``bfcl_json`` writer: canonical projection to BFCL V4 JSONL on disk.

Four format decisions are settled here, because leaving them to the reader is how
two consumers end up scoring the same export differently.

**JSONL, not a JSON array.** One record per line streams without loading the
whole benchmark, and a harness that fails on one task can retry that line instead
of the file. An array would also make the two files below impossible to append to
independently.

**Two files, joined by ``id``.** ``question`` and ``possible_answer`` are written
separately, mirroring how BFCL keeps its prompts apart from its answers. The split
is not cosmetic: a single file holding both invites a runner to hand the model the
record it is being scored against.

**Parallel calls stay grouped.** Upstream's per-turn answer list is flat, so a
turn expecting two calls at once and a turn expecting two in sequence look the
same there. The Nemotron extension carries the explicit grouping, and a consumer
that only reads the upstream fields still gets the calls in trace order.

**Expected calls are the answer; oracle results are only provenance.** The
upstream ``question`` and ``ground_truth`` fields carry prompts and expected calls
and nothing else. The recorded tool results stay under ``x-nemotron.messages``, so
a run can be audited without inviting a scorer to compare a model's live tool
output against a snapshot of one backend revision.

Bytes are deterministic: sorted keys, no incidental whitespace, ``\\n`` endings,
and UTF-8 without escaping, so a Vietnamese surface is readable in the file and
two runs of one config produce the same digest.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictStr, field_validator, model_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    BFCL_JSON_SCHEMA_VERSION,
    BFCL_UPSTREAM_SCHEMA_VERSION,
    EXPORT_DIRECTORY,
    BfclJsonRecord,
    ContentHash,
    NonNegativeInt,
    export_tree_hash,
    relative_export_path,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_projection import (
    CanonicalConversationPlan,
    CanonicalExportProjection,
    ProjectionSource,
)

BFCL_JSON_ROOT = f"{EXPORT_DIRECTORY}/bfcl_json"
BFCL_JSON_QUESTION_FILE = f"{BFCL_JSON_ROOT}/{BFCL_UPSTREAM_SCHEMA_VERSION}.jsonl"
BFCL_JSON_ANSWER_FILE = f"{BFCL_JSON_ROOT}/possible_answer/{BFCL_UPSTREAM_SCHEMA_VERSION}.jsonl"


class BfclJsonWriteError(RuntimeError):
    """The BFCL JSON export could not be written from the canonical projection."""


class BfclJsonArtifact(BaseModel):
    """What the writer put on disk, for validation and the manifest to cite."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = BFCL_JSON_SCHEMA_VERSION
    format: Literal["bfcl_json"] = "bfcl_json"
    upstream_schema_version: Literal["BFCL_v4_multi_turn"] = BFCL_UPSTREAM_SCHEMA_VERSION
    root: StrictStr = BFCL_JSON_ROOT
    question_file: StrictStr = BFCL_JSON_QUESTION_FILE
    answer_file: StrictStr = BFCL_JSON_ANSWER_FILE
    rows: NonNegativeInt
    content_hash: ContentHash
    source: ProjectionSource

    @field_validator("root", "question_file", "answer_file")
    @classmethod
    def validate_relative_path(cls, value: str, info: Any) -> str:
        return relative_export_path(value, label=info.field_name)

    @model_validator(mode="after")
    def validate_artifact(self) -> BfclJsonArtifact:
        if self.question_file == self.answer_file:
            raise ValueError("questions and answers must be written to separate files")
        root = PurePosixPath(self.root)
        for name in ("question_file", "answer_file"):
            path: str = getattr(self, name)
            try:
                relative = PurePosixPath(path).relative_to(root)
            except ValueError:
                raise ValueError(f"{name} must live under {self.root!r}")
            if relative == PurePosixPath("."):
                raise ValueError(f"{name} must name a file below {self.root!r}")
        if not self.rows:
            raise ValueError("an export of no row is not a benchmark")
        if self.rows != self.source.rows:
            raise ValueError("the export must carry every row of the benchmark it projected")
        return self

    @property
    def files(self) -> tuple[str, ...]:
        return (self.question_file, self.answer_file)


def bfcl_json_record_pair(
    record: BfclJsonRecord,
    plan: CanonicalConversationPlan,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one task's question and answer records from the canonical projection."""
    if record.id != plan.task_id:
        raise BfclJsonWriteError(f"record {record.id!r} was given the conversation plan of {plan.task_id!r}")
    try:
        return (
            record.question_record(plan.call_group_payload),
            record.ground_truth_record(plan.calls_by_user_turn),
        )
    except ValueError as exc:
        raise BfclJsonWriteError(f"task {record.id!r} cannot be written as BFCL JSON: {exc}") from exc


def write_bfcl_json(projection: CanonicalExportProjection, run_directory: Path) -> BfclJsonArtifact:
    """Write both JSONL files and describe what was written.

    The artifact reports counts and a tree hash; it does not claim the export is
    equivalent to the benchmark. Reading the files back and proving that is a
    separate step, so a writer cannot certify its own output.
    """
    questions: list[str] = []
    answers: list[str] = []
    for row, plan in zip(projection.rows, projection.plans, strict=True):
        question, answer = bfcl_json_record_pair(BfclJsonRecord.from_canonical(row), plan)
        questions.append(_jsonl_line(question))
        answers.append(_jsonl_line(answer))

    _write_lines(run_directory / BFCL_JSON_QUESTION_FILE, questions)
    _write_lines(run_directory / BFCL_JSON_ANSWER_FILE, answers)
    return BfclJsonArtifact(
        rows=len(projection.rows),
        content_hash=export_tree_hash(run_directory, (BFCL_JSON_QUESTION_FILE, BFCL_JSON_ANSWER_FILE)),
        source=projection.source,
    )


def read_bfcl_json(
    run_directory: Path, artifact: BfclJsonArtifact
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read both files back, so the format is decoded the way it was encoded."""
    questions = _read_lines(run_directory / artifact.question_file)
    answers = _read_lines(run_directory / artifact.answer_file)
    if len(questions) != len(answers):
        raise BfclJsonWriteError(
            f"the export holds {len(questions)} question(s) and {len(answers)} answer(s); "
            "every task is written to both files"
        )
    # Last, so a blank line or a lost task reports itself precisely; this catches
    # what those checks cannot, an edit that leaves both files well formed.
    if export_tree_hash(run_directory, artifact.files) != artifact.content_hash:
        raise BfclJsonWriteError("the BFCL JSON export no longer matches the tree hash the writer reported")
    return questions, answers


def _jsonl_line(record: dict[str, Any]) -> str:
    """Encode one record deterministically, keeping non-ASCII text readable."""
    return json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" keeps the "\n" endings written here from becoming "\r\n" on a
    # host whose default differs, which would change the export's digest.
    with path.open("w", encoding="utf-8", newline="") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")


def _read_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BfclJsonWriteError(f"the BFCL JSON export is missing {path.name}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                raise BfclJsonWriteError(f"{path.name} line {number} is blank; JSONL carries one record per line")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BfclJsonWriteError(f"{path.name} line {number} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise BfclJsonWriteError(f"{path.name} line {number} is not a JSON object")
            records.append(record)
    return records
