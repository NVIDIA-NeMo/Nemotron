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

"""Read compatibility exports back and prove they equal the published projection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypeAlias

from nemotron.steps.byob.runtime.benchmark_families.bfcl.bfcl_json_export import (
    BfclJsonArtifact,
    bfcl_json_record_pair,
    read_bfcl_json,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    BFCL_JSON_SCHEMA_VERSION,
    NEMO_EVALUATOR_SCHEMA_VERSION,
    BfclJsonRecord,
    ExportFormatReport,
    ExportRowFailure,
    ExportValidationReport,
    NemoEvaluatorRecord,
    validate_export_equivalence,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_projection import (
    CanonicalExportProjection,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.nemo_evaluator_export import (
    NemoEvaluatorArtifact,
    read_nemo_evaluator_bundle,
)

EXPORT_VALIDATION_REPORT_FILE = "exports/export_validation_report.json"

ExportArtifact: TypeAlias = BfclJsonArtifact | NemoEvaluatorArtifact


class ExportValidationError(RuntimeError):
    """A written compatibility export is not the benchmark it claims to encode."""


def validate_and_write_export_report(
    projection: CanonicalExportProjection,
    artifacts: dict[str, ExportArtifact],
    run_directory: Path,
) -> tuple[ExportValidationReport, str]:
    """Validate every enabled writer from disk and persist deterministic evidence."""
    reports: list[ExportFormatReport] = []
    if artifact := artifacts.get("bfcl_json"):
        if not isinstance(artifact, BfclJsonArtifact):
            raise ExportValidationError("bfcl_json writer returned the wrong artifact type")
        reports.append(_validate_bfcl(projection, artifact, run_directory))
    if artifact := artifacts.get("nemo_evaluator_bundle"):
        if not isinstance(artifact, NemoEvaluatorArtifact):
            raise ExportValidationError("nemo_evaluator_bundle writer returned the wrong artifact type")
        reports.append(_validate_nemo(projection, artifact, run_directory))

    report = ExportValidationReport(
        benchmark_rows=len(projection.rows),
        benchmark_content_hash=projection.source.content_hash,
        formats=tuple(sorted(reports, key=lambda item: item.format)),
        status="passed" if all(item.equivalent for item in reports) else "failed",
    )
    path = run_directory / EXPORT_VALIDATION_REPORT_FILE
    payload = (
        json.dumps(
            report.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    content_hash = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    if report.status != "passed":
        raise ExportValidationError(f"compatibility export validation failed; inspect {EXPORT_VALIDATION_REPORT_FILE}")
    return report, content_hash


def _validate_bfcl(
    projection: CanonicalExportProjection,
    artifact: BfclJsonArtifact,
    run_directory: Path,
) -> ExportFormatReport:
    questions, answers = read_bfcl_json(run_directory, artifact)
    failures: list[ExportRowFailure] = []
    records: list[BfclJsonRecord] = []
    for position, (row, plan) in enumerate(zip(projection.rows, projection.plans, strict=True)):
        if position >= len(questions) or position >= len(answers):
            break
        expected_record = BfclJsonRecord.from_canonical(row)
        expected_question, expected_answer = bfcl_json_record_pair(expected_record, plan)
        if questions[position] != expected_question or answers[position] != expected_answer:
            failures.append(ExportRowFailure(task_id=row.task_id, reason="truth_field_changed", field="messages"))
        extension = questions[position].get("x-nemotron")
        if not isinstance(extension, dict):
            failures.append(ExportRowFailure(task_id=row.task_id, reason="truth_field_changed", field="messages"))
            continue
        try:
            records.append(
                BfclJsonRecord.model_validate(
                    {
                        "schema_version": extension.get("schema_version"),
                        "upstream_schema_version": extension.get("upstream_schema_version"),
                        "id": questions[position].get("id"),
                        "messages": extension.get("messages"),
                        "tools": extension.get("tools"),
                        "expected_tool_calls": extension.get("expected_tool_calls"),
                        "success_assertions": extension.get("success_assertions"),
                        "call_order": extension.get("call_order"),
                        "call_order_prefix": extension.get("call_order_prefix"),
                        "metadata": extension.get("metadata"),
                    }
                )
            )
        except ValueError:
            failures.append(ExportRowFailure(task_id=row.task_id, reason="truth_field_changed", field="messages"))
    failures.extend(validate_export_equivalence(projection.rows, records))
    return ExportFormatReport(
        format="bfcl_json",
        format_schema_version=BFCL_JSON_SCHEMA_VERSION,
        path=artifact.root,
        rows=len(questions),
        content_hash=artifact.content_hash,
        equivalent=not failures,
        failures=tuple(_deduplicate(failures)),
    )


def _validate_nemo(
    projection: CanonicalExportProjection,
    artifact: NemoEvaluatorArtifact,
    run_directory: Path,
) -> ExportFormatReport:
    _, payloads = read_nemo_evaluator_bundle(run_directory, artifact)
    records: list[NemoEvaluatorRecord] = []
    failures: list[ExportRowFailure] = []
    for position, payload in enumerate(payloads):
        task_id = str(payload.get("task_id") or f"<row-{position}>")
        try:
            record = NemoEvaluatorRecord.model_validate(payload)
            records.append(record)
            if position >= len(projection.rows) or record != NemoEvaluatorRecord.from_canonical(
                projection.rows[position]
            ):
                failures.append(ExportRowFailure(task_id=task_id, reason="truth_field_changed", field="messages"))
        except ValueError:
            failures.append(ExportRowFailure(task_id=task_id, reason="truth_field_changed", field="messages"))
    failures.extend(validate_export_equivalence(projection.rows, records))
    return ExportFormatReport(
        format="nemo_evaluator_bundle",
        format_schema_version=NEMO_EVALUATOR_SCHEMA_VERSION,
        path=artifact.root,
        rows=len(payloads),
        content_hash=artifact.content_hash,
        equivalent=not failures,
        failures=tuple(_deduplicate(failures)),
    )


def _deduplicate(failures: list[ExportRowFailure]) -> list[ExportRowFailure]:
    unique = {(item.task_id, item.reason, item.field): item for item in failures}
    return [unique[key] for key in sorted(unique, key=lambda item: (item[0], item[1], item[2] or ""))]
