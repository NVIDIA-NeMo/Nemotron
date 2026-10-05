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

"""Deterministic, read-only BFCL bias audit for the B1-B16 contract."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final, cast

import yaml  # type: ignore[import-untyped]

from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    export_content_hash,
    relative_export_path,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import (
    canonical_json,
)

BIAS_AUDIT_CONTRACT_VERSION: Final = "1.0"
BIAS_IDS: Final = tuple(f"B{index}" for index in range(1, 17))
ABSOLUTE_MIX_TOLERANCE: Final = 0.05
EDGE_IDS: Final = (
    "B3.single_turn",
    "B3.missing_slot",
    "B3.correction",
    "B3.confirmation",
    "B3.dependent_call",
    "B3.multi_tool",
    "B3.negative_path",
    "B3.clarify_only",
    "B3.irrelevant",
    "B3.distractor_present",
)
BFCL_JSON_FILES: Final = (
    "BFCL_v4_multi_turn.jsonl",
    "possible_answer/BFCL_v4_multi_turn.jsonl",
)
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_TOOL_TOKEN = re.compile(r"(?<![A-Za-z0-9_]){}(?![A-Za-z0-9_])")
_TRUTH_CREEP_INCIDENT_TYPES = frozenset(
    {
        "tool_correctness_label",
        "gold_rewrite",
        "truth_based_keep_drop",
    }
)


class BiasAuditError(ValueError):
    """An input cannot support a trustworthy, read-only bias audit."""


@dataclass(frozen=True)
class AuditInputs:
    run_manifest: Path
    output_dir: Path
    published: Path | None = None
    raw: Path | None = None
    expanded: Path | None = None
    pack_manifest: Path | None = None
    contamination_reports: tuple[Path, ...] = ()
    distractor_evidence: Path | None = None
    judge_evidence: Path | None = None
    portability_evidence: Path | None = None
    exceptions: Path | None = None


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _semantic_hash(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


def _published_export_hash(
    root: Path,
    files: Sequence[str],
    logical_root: str,
) -> str:
    normalized_root = relative_export_path(
        logical_root,
        label="published export path",
    ).rstrip("/")
    return cast(
        str,
        export_content_hash(
            {
                f"{normalized_root}/{relative_export_path(path, label='export file')}": (root / path).read_bytes()
                for path in files
            }
        ),
    )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BiasAuditError(f"{label} must be a mapping")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise BiasAuditError(f"{label} must be a list")
    return value


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise BiasAuditError(f"{label} must be an integer")
    return value


def _load_document(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise BiasAuditError(f"{label} does not exist: {path}")
    try:
        if path.suffix.casefold() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise BiasAuditError(f"{label} is not valid JSON/YAML: {path}") from exc
    return _mapping(value, label)


def _json_value(value: Any, label: str) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise BiasAuditError(f"{label} is not valid JSON text") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise BiasAuditError(f"cannot read JSONL artifact: {path}") from exc
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            raise BiasAuditError(f"{path}:{index} is blank")
        try:
            rows.append(_mapping(json.loads(line), f"{path}:{index}"))
        except json.JSONDecodeError as exc:
            raise BiasAuditError(f"{path}:{index} is not valid JSON") from exc
    return rows


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        return cast(list[dict[str, Any]], pq.read_table(path).to_pylist())
    except Exception as exc:  # noqa: BLE001 - normalize parser failures at boundary
        raise BiasAuditError(f"cannot read Parquet artifact: {path}") from exc


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BiasAuditError(f"row artifact does not exist: {path}")
    if path.suffix.casefold() == ".parquet":
        return _read_parquet(path)
    if path.suffix.casefold() in {".jsonl", ".ndjson"}:
        return _read_jsonl(path)
    raise BiasAuditError(f"unsupported row artifact: {path}")


def _canonical_row(row: Mapping[str, Any]) -> dict[str, Any]:
    outer_row = row
    exported = row.get("x-nemotron")
    if isinstance(exported, dict):
        row = exported
    metadata = _json_value(row.get("metadata") or {}, "row.metadata")
    metadata = _mapping(metadata, "row.metadata")
    messages = _json_value(
        row.get("messages") or row.get("reference_trace") or [],
        "row.messages",
    )
    if not messages and row.get("turns"):
        steps = _list(
            _json_value(row["turns"], "row.turns"),
            "row.turns",
        )
        messages = [
            {
                "role": ("user" if step.get("kind") == "user" else "assistant"),
                "content": step.get("content"),
            }
            for value in steps
            for step in [_mapping(value, "row turn")]
            if step.get("kind") in {"user", "assistant_text"}
        ]
    expected = _json_value(
        row.get("expected_tool_calls") or metadata.get("expected_tool_calls") or [],
        "row.expected_tool_calls",
    )
    tools = _json_value(row.get("tools") or [], "row.tools")
    task_id = row.get("task_id") or row.get("id") or outer_row.get("task_id") or outer_row.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise BiasAuditError("every audited row needs a non-empty task_id")
    required = metadata.get("required_tools")
    if required is None:
        required = sorted(
            {
                str(call.get("function_name") or call.get("name"))
                for call in expected
                if isinstance(call, dict) and (call.get("function_name") or call.get("name"))
            }
        )
    user_text = [
        str(message.get("content"))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user" and isinstance(message.get("content"), str)
    ]
    return {
        "task_id": task_id,
        "base_task_id": metadata.get("surface", {}).get("base_task_id")
        if isinstance(metadata.get("surface"), dict)
        else metadata.get("base_task_id") or task_id,
        "variant_index": _integer(
            metadata.get("variant_index")
            if metadata.get("variant_index") is not None
            else row.get("variant_index") or 0,
            "row.variant_index",
        ),
        "category": metadata.get("category") or row.get("category"),
        "difficulty": metadata.get("difficulty") or row.get("difficulty"),
        "intent": metadata.get("intent") or row.get("intent"),
        "turn_policy": metadata.get("turn_policy") or row.get("turn_policy"),
        "required_tools": [str(item) for item in required or []],
        "tools_present": [
            str(item)
            for item in metadata.get("tools_present")
            or [tool.get("function", {}).get("name") for tool in tools if isinstance(tool, dict)]
            if item
        ],
        "num_tool_calls": _integer(
            metadata.get("num_tool_calls") if metadata.get("num_tool_calls") is not None else len(expected),
            "row.num_tool_calls",
        ),
        "is_multi_turn": bool(metadata.get("is_multi_turn", len(user_text) > 1)),
        "fixture_refs": [str(item) for item in metadata.get("fixture_refs") or row.get("fixture_refs") or []],
        "held_out_hit": metadata.get("held_out_hit"),
        "paraphrase_model": metadata.get("paraphrase_model") or row.get("paraphrase_model"),
        "user_text": user_text,
        "messages": messages,
        "expected_tool_calls": expected,
        "tools": tools,
    }


def _load_published(path: Path, manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    if path.is_dir():
        declared = _mapping(manifest.get("exports") or {}, "run_manifest.exports").get("formats", {}).get("bfcl_json")
        declared = _mapping(declared, "run_manifest.exports.formats.bfcl_json")
        declared_path = declared.get("path")
        if not isinstance(declared_path, str):
            raise BiasAuditError("BFCL JSON export needs a declared path")
        actual_hash = _published_export_hash(
            path,
            BFCL_JSON_FILES,
            declared_path,
        )
        if actual_hash != declared.get("content_hash"):
            raise BiasAuditError("BFCL JSON export hash does not match run_manifest.json")
        rows = _read_jsonl(path / BFCL_JSON_FILES[0])
        return [_canonical_row(row) for row in rows], actual_hash
    expected = (
        _mapping(manifest.get("artifacts") or {}, "run_manifest.artifacts")
        .get("benchmark_parquet", {})
        .get("content_hash")
    )
    actual_hash = _file_hash(path)
    if actual_hash != expected:
        raise BiasAuditError("published benchmark hash does not match run_manifest.json")
    return [_canonical_row(row) for row in _read_rows(path)], actual_hash


def _load_optional_layer(
    path: Path | None,
    *,
    manifest: Mapping[str, Any],
    artifact_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path is None:
        return [], {
            "available": False,
            "rows": 0,
            "content_hash": None,
            "held_out_rescan": False,
            "paraphrase_rescan": False,
        }
    expected = (
        _mapping(manifest.get("artifacts") or {}, "run_manifest.artifacts").get(artifact_name, {}).get("content_hash")
    )
    actual = _file_hash(path)
    if expected != actual:
        raise BiasAuditError(f"{artifact_name} hash does not match run_manifest.json")
    rows = [_canonical_row(row) for row in _read_rows(path)]
    return rows, {
        "available": True,
        "rows": len(rows),
        "content_hash": actual,
        "held_out_rescan": True,
        "paraphrase_rescan": True,
    }


def _load_expanded_layer(
    path: Path | None,
    *,
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path is None or path.is_file():
        return _load_optional_layer(
            path,
            manifest=manifest,
            artifact_name="rendered_conversations",
        )
    if not path.is_dir():
        raise BiasAuditError(f"expanded layer does not exist: {path}")
    task_path = path / "task_instances.parquet"
    surface_path = path / "rendered_conversations.parquet"
    artifacts = _mapping(
        manifest.get("artifacts") or {},
        "run_manifest.artifacts",
    )
    hashes: dict[str, str] = {}
    for name, source in (
        ("task_instances", task_path),
        ("rendered_conversations", surface_path),
    ):
        if not source.is_file():
            raise BiasAuditError(f"expanded stage cache is missing {source.name}")
        actual = _file_hash(source)
        expected = (artifacts.get(name) or {}).get("content_hash")
        if actual != expected:
            raise BiasAuditError(f"{name} hash does not match run_manifest.json")
        hashes[name] = actual
    tasks = {str(row["task_id"]): _canonical_row(row) for row in _read_parquet(task_path)}
    merged: list[dict[str, Any]] = []
    for raw_surface in _read_parquet(surface_path):
        surface = _canonical_row(raw_surface)
        task = tasks.get(str(surface["task_id"]))
        if task is None:
            raise BiasAuditError(f"expanded surface {surface['task_id']!r} has no task instance")
        merged.append(
            {
                **task,
                **surface,
                "fixture_refs": task["fixture_refs"],
                "required_tools": task["required_tools"],
                "tools_present": task["tools_present"],
                "category": task["category"],
                "difficulty": task["difficulty"],
                "intent": task["intent"],
                "turn_policy": task["turn_policy"],
            }
        )
    return merged, {
        "available": True,
        "rows": len(merged),
        "content_hash": _semantic_hash(hashes),
        "artifact_hashes": hashes,
        "held_out_rescan": True,
        "paraphrase_rescan": True,
    }


def _auto_inputs(inputs: AuditInputs, manifest: Mapping[str, Any]) -> AuditInputs:
    root = inputs.run_manifest.parent
    published = inputs.published
    if published is None:
        parquet = root / "benchmark.parquet"
        release_parquet = root / "benchmark" / "benchmark.parquet"
        export = root / "exports" / "bfcl_json"
        published = (
            parquet
            if parquet.is_file()
            else release_parquet
            if release_parquet.is_file()
            else export
            if export.is_dir()
            else None
        )
    raw = inputs.raw
    if raw is None:
        root_raw = root / "benchmark_raw.parquet"
        release_raw = root / "benchmark" / "benchmark_raw.parquet"
        raw = root_raw if root_raw.is_file() else release_raw if release_raw.is_file() else None
    expanded = inputs.expanded
    if expanded is None:
        candidate = root / "stage_cache"
        expanded = candidate if (candidate / "rendered_conversations.parquet").is_file() else None
    if published is None:
        raise BiasAuditError("no published benchmark or verified BFCL JSON export is available")
    return AuditInputs(
        run_manifest=inputs.run_manifest,
        output_dir=inputs.output_dir,
        published=published,
        raw=raw,
        expanded=expanded,
        pack_manifest=inputs.pack_manifest,
        contamination_reports=inputs.contamination_reports,
        distractor_evidence=inputs.distractor_evidence,
        judge_evidence=inputs.judge_evidence,
        portability_evidence=inputs.portability_evidence,
        exceptions=inputs.exceptions,
    )


def _validate_applicability(manifest: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    source = _mapping(
        manifest.get("bias_applicability"),
        "run_manifest.bias_applicability",
    )
    required = set(BIAS_IDS) | set(EDGE_IDS)
    missing = sorted(required - set(source))
    unknown = sorted(set(source) - required)
    if missing or unknown:
        raise BiasAuditError(f"bias_applicability keys mismatch; missing={missing!r}, unknown={unknown!r}")
    result: dict[str, dict[str, str]] = {}
    for bias_id in sorted(required):
        entry = _mapping(source[bias_id], f"bias_applicability.{bias_id}")
        status = entry.get("status")
        if status not in {"applicable", "na"}:
            raise BiasAuditError(f"bias_applicability.{bias_id}.status must be applicable or na")
        reason = entry.get("reason")
        if status == "na" and (not isinstance(reason, str) or not reason.strip()):
            raise BiasAuditError(f"bias_applicability.{bias_id} needs a stable N/A reason")
        if status == "applicable" and reason is not None:
            raise BiasAuditError(f"bias_applicability.{bias_id} cannot carry an N/A reason")
        result[bias_id] = (
            {"status": status, "reason": reason.strip()} if isinstance(reason, str) else {"status": status}
        )
    return result


def _gini(counts: Iterable[int]) -> float:
    values = [int(value) for value in counts]
    if not values or any(value < 0 for value in values):
        raise BiasAuditError("Gini requires a non-empty non-negative count vector")
    total = sum(values)
    if total == 0:
        return 0.0
    numerator = sum(abs(left - right) for left in values for right in values)
    return numerator / (2 * len(values) * total)


def _rates(counts: Mapping[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        raise BiasAuditError("a distribution metric has an empty denominator")
    return {key: value / total for key, value in sorted(counts.items())}


def _metric(
    bias_id: str,
    name: str,
    value: Any,
    threshold: str,
    passed: bool,
    diagnostics: Mapping[str, Any],
    sources: Sequence[str],
    *,
    applicability: Mapping[str, str],
    evidence_complete: bool = True,
) -> dict[str, Any]:
    applicable = applicability["status"] == "applicable"
    effective_pass = bool(passed and evidence_complete) if applicable else True
    return {
        "bias_id": bias_id,
        "applicability": dict(applicability),
        "primary_metric": {
            "name": name,
            "value": value if applicable else None,
            "threshold": threshold,
            "passed": effective_pass,
        },
        "supporting_diagnostics": dict(diagnostics),
        "source_evidence": list(sources),
        "evidence_complete": evidence_complete if applicable else True,
        "exceptions": [],
    }


def _counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    values = [row.get(field) for row in rows]
    if any(not isinstance(value, str) or not value for value in values):
        raise BiasAuditError(f"published rows have missing {field}")
    return dict(sorted(Counter(str(value) for value in values).items()))


def _tool_definitions(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    definitions: dict[str, dict[str, Any]] = {}
    for row in rows:
        for tool in row.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function") or {}
            name = function.get("name") if isinstance(function, dict) else None
            if isinstance(name, str):
                if name in definitions and definitions[name] != tool:
                    raise BiasAuditError(f"tool {name!r} changes across published rows")
                definitions[name] = tool
    if not definitions:
        for name in sorted({tool for row in rows for tool in row.get("tools_present") or []}):
            definitions[name] = {}
    return definitions


def _parse_fixture_ref(value: str) -> tuple[str, str] | None:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    if (
        isinstance(decoded, list)
        and len(decoded) == 2
        and all(isinstance(item, (str, int, float)) for item in decoded)
    ):
        return str(decoded[0]), str(decoded[1])
    return None


def _load_pack(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    manifest = _load_document(path, "pack manifest")
    root = path.parent
    paths = _mapping(manifest.get("paths"), "pack manifest paths")
    result: dict[str, Any] = {"manifest": manifest, "manifest_hash": _file_hash(path)}
    for key, label in (("tools", "tools"), ("templates", "templates")):
        relative = paths.get(key)
        if not isinstance(relative, str):
            raise BiasAuditError(f"pack manifest paths.{key} is required")
        source = root / relative
        if key == "tools":
            result[label] = _list(
                json.loads(source.read_text(encoding="utf-8")),
                "pack tools",
            )
        else:
            result[label] = _list(
                yaml.safe_load(source.read_text(encoding="utf-8")),
                "pack templates",
            )
        result[f"{label}_hash"] = _file_hash(source)
    held_out_source = manifest.get("held_out")
    if held_out_source is not None:
        if not isinstance(held_out_source, str):
            raise BiasAuditError("pack manifest held_out must be a path")
        held_path = root / held_out_source
        result["held_out"] = _load_document(held_path, "held-out policy")
        result["held_out_hash"] = _file_hash(held_path)
    return result


def _input_artifact_hashes(
    inputs: AuditInputs,
    pack: Mapping[str, Any],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for label, path in (
        ("raw", inputs.raw),
        ("expanded", inputs.expanded),
        ("distractor_evidence", inputs.distractor_evidence),
        ("judge_evidence", inputs.judge_evidence),
        ("portability_evidence", inputs.portability_evidence),
        ("exceptions", inputs.exceptions),
    ):
        if path is not None:
            if label == "expanded" and path.is_dir():
                for artifact in (
                    "task_instances.parquet",
                    "rendered_conversations.parquet",
                ):
                    source = path / artifact
                    if not source.is_file():
                        raise BiasAuditError(f"expanded stage cache is missing {artifact}")
                    hashes[f"expanded_{artifact.removesuffix('.parquet')}"] = _file_hash(source)
                continue
            if not path.is_file():
                raise BiasAuditError(f"{label} must be a file: {path}")
            hashes[label] = _file_hash(path)
    for index, path in enumerate(inputs.contamination_reports):
        if not path.is_file():
            raise BiasAuditError(f"contamination report does not exist: {path}")
        hashes[f"contamination_report_{index}"] = _file_hash(path)
    for label in (
        "manifest_hash",
        "tools_hash",
        "templates_hash",
        "held_out_hash",
    ):
        value = pack.get(label)
        if value is not None:
            if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
                raise BiasAuditError(f"pack {label} is not a content hash")
            hashes[f"pack_{label.removesuffix('_hash')}"] = value
    return dict(sorted(hashes.items()))


def _held_out_refs(pack: Mapping[str, Any]) -> set[tuple[str, str]]:
    policy = pack.get("held_out") or {}
    fixtures = (policy.get("fixtures") or {}) if isinstance(policy, dict) else {}
    if not isinstance(fixtures, dict):
        raise BiasAuditError("held_out.fixtures must be a mapping")
    return {
        (str(collection), str(identifier))
        for collection, identifiers in fixtures.items()
        for identifier in _list(
            identifiers,
            f"held_out.fixtures.{collection}",
        )
    }


def _held_out_hits(
    rows: Sequence[Mapping[str, Any]],
    held_out: set[tuple[str, str]],
) -> list[str]:
    hits: list[str] = []
    for row in rows:
        explicit = row.get("held_out_hit")
        refs = {
            parsed for value in row.get("fixture_refs") or [] if (parsed := _parse_fixture_ref(str(value))) is not None
        }
        if explicit is True or refs & held_out:
            hits.append(str(row["task_id"]))
    return sorted(set(hits))


def _scalar_strings(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {item for child in value.values() for item in _scalar_strings(child)}
    if isinstance(value, list):
        return {item for child in value for item in _scalar_strings(child)}
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        text = str(value).strip()
        return {text} if len(text) >= 3 else set()
    return set()


def _paraphrase_leaks(
    rows: Sequence[Mapping[str, Any]],
    fallback_tool_names: Sequence[str] = (),
) -> list[dict[str, str]]:
    canonical_by_base = {
        str(row["base_task_id"]): "\n".join(row.get("user_text") or [])
        for row in rows
        if int(row.get("variant_index") or 0) == 0
    }
    leaks: list[dict[str, str]] = []
    for row in rows:
        if not row.get("paraphrase_model"):
            continue
        task_id = str(row["task_id"])
        candidate = "\n".join(row.get("user_text") or [])
        for tool in row.get("tools_present") or fallback_tool_names:
            if _TOOL_TOKEN.pattern.format(re.escape(tool)) and re.search(
                _TOOL_TOKEN.pattern.format(re.escape(tool)),
                candidate,
                flags=re.IGNORECASE,
            ):
                leaks.append({"task_id": task_id, "reason": "tool_name", "value": str(tool)})
        canonical = canonical_by_base.get(str(row["base_task_id"]))
        if canonical is None:
            continue
        hidden_candidates = _scalar_strings(row.get("expected_tool_calls") or [])
        for value in sorted(hidden_candidates):
            if value not in canonical and value in candidate:
                leaks.append(
                    {
                        "task_id": task_id,
                        "reason": "hidden_or_novel_value",
                        "value": value,
                    }
                )
    return sorted(
        {canonical_json(item): item for item in leaks}.values(),
        key=lambda item: (item["task_id"], item["reason"], item["value"]),
    )


def _negative_families(row: Mapping[str, Any]) -> set[str]:
    families: set[str] = set()
    if row.get("turn_policy") == "irrelevant":
        families.add("decline")
    if row.get("turn_policy") != "negative_path":
        return families
    for message in row.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            error = value.get("error")
            if isinstance(error, dict) and isinstance(error.get("code"), str):
                families.add(str(error["code"]))
            elif isinstance(value.get("status"), str):
                families.add(str(value["status"]))
    return families


def _load_review_evidence(
    path: Path | None,
    *,
    kind: str,
    manifest_hash: str,
) -> dict[str, Any] | None:
    if path is None:
        return None
    document = _load_document(path, f"{kind} evidence")
    if document.get("schema_version") != "1.0":
        raise BiasAuditError(f"{kind} evidence schema_version must be 1.0")
    if document.get("kind") != kind:
        raise BiasAuditError(f"{kind} evidence kind mismatch")
    if document.get("run_manifest_hash") != manifest_hash:
        raise BiasAuditError(f"{kind} evidence is bound to another run manifest")
    reviewers = document.get("reviewers")
    if (
        not isinstance(reviewers, list)
        or len(set(reviewers)) < 2
        or any(not isinstance(item, str) or not item.strip() for item in reviewers)
    ):
        raise BiasAuditError(f"{kind} evidence needs at least two reviewers")
    return document


def _distractor_metric(
    rows: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any] | None,
    manifest_hash: str,
) -> tuple[Any, bool, dict[str, Any], bool]:
    eligible = [row for row in rows if set(row.get("tools_present") or []) > set(row.get("required_tools") or [])]
    fingerprints = {canonical_json(sorted(row.get("tools_present") or [])) for row in eligible}
    diagnostics: dict[str, Any] = {
        "eligible_rows": len(eligible),
        "distractor_presence_rate": len(eligible) / len(rows),
        "tools_present_pattern_count": len(fingerprints),
    }
    if evidence is None:
        diagnostics["missing_evidence"] = "reviewed B10 distractor evidence"
        return None, False, diagnostics, False
    seed = evidence.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise BiasAuditError("B10 evidence seed must be an integer")
    expected = [
        row["task_id"]
        for row in sorted(
            eligible,
            key=lambda row: (
                hashlib.sha256(f"{seed}\0{row['task_id']}".encode()).hexdigest(),
                row["task_id"],
            ),
        )[: min(30, len(eligible))]
    ]
    reviews = _list(evidence.get("rows"), "B10 evidence rows")
    reviewed = [item.get("task_id") for item in reviews if isinstance(item, dict)]
    if reviewed != expected:
        raise BiasAuditError("B10 evidence does not contain the exact seeded sample")
    row_by_id = {str(row["task_id"]): row for row in eligible}
    agreements = 0
    annotator_agreements: list[float] = []
    disagreement_count = 0
    adjudicated_count = 0
    declared_reviewers = set(_list(evidence.get("reviewers"), "B10 reviewers"))
    for item in reviews:
        item = _mapping(item, "B10 evidence row")
        task_id = str(item["task_id"])
        annotations = _list(item.get("annotations"), f"B10 {task_id} annotations")
        if len(annotations) < 2:
            raise BiasAuditError(f"B10 {task_id} needs two annotations")
        normalized_annotations = [_mapping(annotation, f"B10 {task_id} annotation") for annotation in annotations]
        reviewers = [annotation.get("reviewer") for annotation in normalized_annotations]
        if any(reviewer not in declared_reviewers for reviewer in reviewers) or len(set(reviewers)) != len(reviewers):
            raise BiasAuditError(f"B10 {task_id} annotations need distinct declared reviewers")
        selections = [
            tuple(
                sorted(
                    str(tool)
                    for tool in _list(
                        annotation.get("selected_tools"),
                        "selected_tools",
                    )
                )
            )
            for annotation in normalized_annotations
        ]
        allowed = set(row_by_id[task_id]["tools_present"])
        if any(not set(selection) <= allowed for selection in selections):
            raise BiasAuditError(f"B10 {task_id} selects a tool outside tools_present")
        winner, count = Counter(selections).most_common(1)[0]
        if len(set(selections)) > 1:
            disagreement_count += 1
            adjudicated = tuple(
                sorted(
                    str(tool)
                    for tool in _list(
                        item.get("adjudicated_tools"),
                        f"B10 {task_id} adjudicated_tools",
                    )
                )
            )
            rationale = item.get("adjudication_rationale")
            if not isinstance(rationale, str) or not rationale.strip():
                raise BiasAuditError(f"B10 {task_id} disagreement needs an adjudication rationale")
            if not set(adjudicated) <= allowed:
                raise BiasAuditError(f"B10 {task_id} adjudication selects a tool outside tools_present")
            if count > len(selections) / 2 and adjudicated != winner:
                raise BiasAuditError(f"B10 {task_id} adjudication contradicts the annotation majority")
            if count <= len(selections) / 2:
                winner = adjudicated
            adjudicated_count += 1
        gold = tuple(sorted(row_by_id[task_id]["required_tools"]))
        agreements += int(winner == gold)
        annotator_agreements.append(count / len(selections))
    score = agreements / len(reviews) if reviews else 0.0
    diagnostics.update(
        {
            "sample_seed": seed,
            "sample_task_ids": expected,
            "sample_hash": _semantic_hash(expected),
            "inter_annotator_exact_agreement": (
                sum(annotator_agreements) / len(annotator_agreements) if annotator_agreements else 0.0
            ),
            "disagreement_count": disagreement_count,
            "adjudicated_count": adjudicated_count,
            "evidence_hash": _semantic_hash(evidence),
            "run_manifest_hash": manifest_hash,
        }
    )
    return score, score >= 0.90, diagnostics, True


def _contamination_metric(
    paths: Sequence[Path],
    *,
    run_id: str,
    task_ids_hash: str,
) -> tuple[int, bool, dict[str, Any], bool]:
    violations = 0
    unresolved = 0
    candidates = 0
    hashes: list[dict[str, str]] = []
    for path in paths:
        document = _load_document(path, "contamination report")
        if document.get("source_run_id") != run_id:
            raise BiasAuditError(f"contamination report is bound to another source run: {path}")
        if document.get("source_task_ids_hash") != task_ids_hash:
            raise BiasAuditError(f"contamination report is bound to another source task set: {path}")
        if document.get("schema_version") != "1.0":
            raise BiasAuditError(f"contamination report has an unsupported schema: {path}")
        hashes.append({"name": path.name, "content_hash": _file_hash(path)})
        for candidate in _list(document.get("candidates"), "contamination candidates"):
            candidate = _mapping(candidate, "contamination candidate")
            candidates += 1
            violations += len(_list(candidate.get("collisions") or [], "collisions"))
            unresolved += int(bool(candidate.get("unresolved")))
        if document.get("publication_allowed") is False:
            violations += 1
    complete = bool(paths)
    return (
        violations,
        violations == 0 and unresolved == 0,
        {
            "reports_scanned": len(paths),
            "candidates_scanned": candidates,
            "unresolved_candidates": unresolved,
            "report_hashes": sorted(hashes, key=lambda item: (item["name"], item["content_hash"])),
            **({} if complete else {"missing_evidence": "release-bound contamination report"}),
        },
        complete,
    )


def _portability_metric(
    evidence: Mapping[str, Any] | None,
    manifest_hash: str,
) -> tuple[Any, bool, dict[str, Any], bool]:
    if evidence is None:
        return (
            None,
            False,
            {"missing_evidence": "B12 portability evidence"},
            False,
        )
    if evidence.get("schema_version") != "1.0":
        raise BiasAuditError("B12 portability evidence schema_version must be 1.0")
    if evidence.get("kind") != "agnostic_portability":
        raise BiasAuditError("B12 portability evidence kind mismatch")
    if evidence.get("run_manifest_hash") != manifest_hash:
        raise BiasAuditError("B12 portability evidence is bound to another run")
    hits = evidence.get("agnostic_grep_hits")
    smoke = evidence.get("smoke_pack_gold_eligible")
    no_edits = evidence.get("no_core_edits_required")
    if not isinstance(hits, int) or isinstance(hits, bool) or hits < 0:
        raise BiasAuditError("B12 agnostic_grep_hits must be a non-negative integer")
    if not isinstance(smoke, bool) or not isinstance(no_edits, bool):
        raise BiasAuditError("B12 portability booleans are required")
    passed = hits == 0 and smoke and no_edits
    return (
        passed,
        passed,
        {
            "agnostic_grep_hits": hits,
            "smoke_pack_gold_eligible": smoke,
            "no_core_edits_required": no_edits,
            "evidence_hash": _semantic_hash(evidence),
        },
        True,
    )


def _judge_metric(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, Any] | None,
) -> tuple[Any, bool, dict[str, Any], bool]:
    judge = _mapping(manifest.get("models") or {}, "run_manifest.models").get("surface_judge") or {}
    enabled = bool(judge.get("enabled")) if isinstance(judge, dict) else False
    report = (
        _mapping(
            manifest.get("surface_quality_validation") or {},
            "surface_quality_validation",
        ).get("report")
        or {}
    )
    prompt_hash = report.get("judge_prompt_hash") if isinstance(report, dict) else None
    if not enabled:
        return (
            0,
            True,
            {
                "judge_enabled": False,
                "judge_prompt_hash": prompt_hash,
                "rationales_sampled": 0,
            },
            True,
        )
    if evidence is None:
        return (
            None,
            False,
            {"judge_enabled": True, "missing_evidence": "reviewed B13 evidence"},
            False,
        )
    prompt_hashes = _list(evidence.get("prompt_hashes"), "B13 prompt_hashes")
    if (
        not prompt_hashes
        or len(set(prompt_hashes)) != len(prompt_hashes)
        or any(not isinstance(item, str) or _DIGEST.fullmatch(item) is None for item in prompt_hashes)
    ):
        raise BiasAuditError("B13 prompt_hashes must contain distinct sha256 content hashes")
    if prompt_hash not in prompt_hashes:
        raise BiasAuditError("B13 evidence omits the manifest judge prompt")
    seed = evidence.get("seed")
    sample_size = evidence.get("sample_size")
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not isinstance(sample_size, int)
        or isinstance(sample_size, bool)
        or sample_size < 1
    ):
        raise BiasAuditError("B13 evidence needs integer seed and positive sample_size")
    expected = [
        row["task_id"]
        for row in sorted(
            rows,
            key=lambda row: (
                hashlib.sha256(f"{seed}\0{row['task_id']}".encode()).hexdigest(),
                row["task_id"],
            ),
        )[: min(sample_size, len(rows))]
    ]
    reviews = _list(evidence.get("rows"), "B13 rows")
    if [item.get("task_id") for item in reviews if isinstance(item, dict)] != expected:
        raise BiasAuditError("B13 evidence does not contain the exact seeded sample")
    reviewers = set(_list(evidence.get("reviewers"), "B13 reviewers"))
    incidents = 0
    incident_counts: Counter[str] = Counter()
    for item in reviews:
        row = _mapping(item, "B13 row")
        for incident in _list(row.get("incidents") or [], "B13 incidents"):
            incident = _mapping(incident, "B13 incident")
            incident_type = incident.get("type")
            reviewer = incident.get("reviewer")
            detail = incident.get("detail")
            if incident_type not in _TRUTH_CREEP_INCIDENT_TYPES:
                raise BiasAuditError("B13 incident has an unsupported type")
            if reviewer not in reviewers:
                raise BiasAuditError("B13 incident reviewer is not declared")
            if not isinstance(detail, str) or not detail.strip():
                raise BiasAuditError("B13 incident needs a non-empty detail")
            incident_counts[str(incident_type)] += 1
            incidents += 1
    return (
        incidents,
        incidents == 0,
        {
            "judge_enabled": True,
            "judge_prompt_hashes": prompt_hashes,
            "sample_seed": seed,
            "sample_task_ids": expected,
            "incident_counts": dict(sorted(incident_counts.items())),
            "evidence_hash": _semantic_hash(evidence),
        },
        True,
    )


def _load_exceptions(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    document = _load_document(path, "bias exceptions")
    if document.get("schema_version") != "1.0":
        raise BiasAuditError("bias exceptions schema_version must be 1.0")
    entries = _list(document.get("exceptions"), "bias exceptions")
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        entry = _mapping(entry, "bias exception")
        metric = entry.get("affected_metric")
        owner = entry.get("owner")
        rationale = entry.get("rationale")
        approved = entry.get("approval_date")
        if metric not in BIAS_IDS:
            raise BiasAuditError("exception affected_metric must be B1-B16")
        if any(not isinstance(value, str) or not value.strip() for value in (owner, rationale, approved)):
            raise BiasAuditError("exception needs owner, rationale, and approval_date")
        owner = cast(str, owner)
        rationale = cast(str, rationale)
        approved = cast(str, approved)
        try:
            date.fromisoformat(approved)
        except ValueError as exc:
            raise BiasAuditError("exception approval_date must be ISO YYYY-MM-DD") from exc
        normalized.append(
            {
                "affected_metric": metric,
                "owner": owner.strip(),
                "rationale": rationale.strip(),
                "approval_date": approved,
            }
        )
    return sorted(
        normalized,
        key=lambda item: (
            item["affected_metric"],
            item["approval_date"],
            item["owner"],
        ),
    )


def _build_metrics(
    manifest: Mapping[str, Any],
    applicability: Mapping[str, Mapping[str, str]],
    published: Sequence[Mapping[str, Any]],
    raw: Sequence[Mapping[str, Any]],
    expanded: Sequence[Mapping[str, Any]],
    layers: Mapping[str, Mapping[str, Any]],
    pack: Mapping[str, Any],
    manifest_hash: str,
    distractor_evidence: Mapping[str, Any] | None,
    judge_evidence: Mapping[str, Any] | None,
    portability_evidence: Mapping[str, Any] | None,
    contamination_reports: Sequence[Path],
) -> list[dict[str, Any]]:
    if not published:
        raise BiasAuditError("published benchmark is empty")
    sources = ["run_manifest.json", "published"]
    targets = _mapping(manifest.get("bias_targets") or {}, "bias_targets")
    category = _counts(published, "category")
    category_median = statistics.median(category.values())
    category_balance = 1 - _gini(category.values())
    min_category = min(category.values())
    min_target = int(targets.get("tasks_per_category", 1))
    max_category_ratio = max(category.values()) / category_median
    metrics = [
        _metric(
            "B1",
            "category_balance_score",
            category_balance,
            ">=0.70; min category >= tasks_per_category; max/median <=1.5",
            category_balance >= 0.70 and min_category >= min_target and max_category_ratio <= 1.5,
            {
                "counts": category,
                "minimum_count": min_category,
                "minimum_target": min_target,
                "max_median_ratio": max_category_ratio,
            },
            sources,
            applicability=applicability["B1"],
        )
    ]

    difficulty = _counts(published, "difficulty")
    difficulty_rates = _rates(difficulty)
    difficulty_target = {
        str(key): float(value)
        for key, value in _mapping(
            targets.get("difficulty_mix"),
            "bias_targets.difficulty_mix",
        ).items()
    }
    difficulty_errors = {
        key: abs(difficulty_rates.get(key, 0.0) - target) for key, target in sorted(difficulty_target.items())
    }
    difficulty_max = max(difficulty_errors.values())
    metrics.append(
        _metric(
            "B2",
            "difficulty_mix_max_abs",
            difficulty_max,
            "<=0.05",
            difficulty_max <= 0.05,
            {
                "counts": difficulty,
                "empirical": difficulty_rates,
                "target": difficulty_target,
                "absolute_errors": difficulty_errors,
                "difficulty_mix_l1": sum(difficulty_errors.values()),
            },
            sources,
            applicability=applicability["B2"],
        )
    )

    turn_policy = _counts(published, "turn_policy")
    edge_hits: dict[str, bool] = {}
    for edge in EDGE_IDS:
        suffix = edge.removeprefix("B3.")
        if suffix == "distractor_present":
            edge_hits[edge] = any(set(row["tools_present"]) > set(row["required_tools"]) for row in published)
        elif suffix == "multi_tool":
            edge_hits[edge] = turn_policy.get("multi_tool", 0) > 0 or any(
                row["num_tool_calls"] >= 2 for row in published
            )
        else:
            edge_hits[edge] = turn_policy.get(suffix, 0) > 0
    applicable_edges = [edge for edge in EDGE_IDS if applicability[edge]["status"] == "applicable"]
    edge_coverage = (
        sum(edge_hits[edge] for edge in applicable_edges) / len(applicable_edges) if applicable_edges else 1.0
    )
    non_single = 1 - turn_policy.get("single_turn", 0) / len(published)
    metrics.append(
        _metric(
            "B3",
            "edge_policy_coverage",
            edge_coverage,
            "=1.0; non-single-turn-policy share >=0.35",
            edge_coverage == 1.0 and non_single >= 0.35,
            {
                "counts": turn_policy,
                "edge_hits": edge_hits,
                "applicable_edges": applicable_edges,
                "non_single_turn_policy_share": non_single,
            },
            sources,
            applicability=applicability["B3"],
        )
    )

    tool_defs = _tool_definitions(published)
    exempt = {name for name, definition in tool_defs.items() if definition.get("x-eval-only-distractor") is True}
    individual = Counter(
        tool for row in published for tool in set(row.get("required_tools") or []) if tool not in exempt
    )
    for tool in set(tool_defs) - exempt:
        individual.setdefault(tool, 0)
    tool_balance = 1 - _gini(individual.values())
    orphan = [tool for tool, count in individual.items() if count == 0]
    nonzero_tool_counts = [count for count in individual.values() if count > 0]
    tool_ratio = max(nonzero_tool_counts) / statistics.median(nonzero_tool_counts) if nonzero_tool_counts else math.inf
    tool_sets = Counter(canonical_json(sorted(row.get("required_tools") or [])) for row in published)
    metrics.append(
        _metric(
            "B4",
            "tool_usage_balance_score",
            tool_balance,
            ">=0.60; orphan_tool_rate=0; max/median <=2",
            tool_balance >= 0.60 and not orphan and tool_ratio <= 2,
            {
                "individual_tool_counts": dict(sorted(individual.items())),
                "exempt_tools": sorted(exempt),
                "orphan_tools": sorted(orphan),
                "orphan_tool_rate": len(orphan) / len(individual) if individual else 0,
                "max_median_ratio": tool_ratio,
                "required_tools_fingerprints": dict(sorted(tool_sets.items())),
            },
            sources,
            applicability=applicability["B4"],
        )
    )

    call_buckets = Counter(
        "1" if row["num_tool_calls"] == 1 else "2" if row["num_tool_calls"] == 2 else "3+"
        for row in published
        if row["num_tool_calls"] > 0
    )
    for bucket in ("1", "2", "3+"):
        call_buckets.setdefault(bucket, 0)
    call_total = sum(call_buckets.values())
    call_rates = _rates(call_buckets) if call_total else {bucket: 0.0 for bucket in ("1", "2", "3+")}
    call_target_source = targets.get("tool_call_count_mix")
    call_target: dict[str, float] | None = None
    call_errors: dict[str, float] = {}
    call_max: float | None = None
    target_complete = call_target_source is not None
    if call_target_source is not None:
        target_mapping = _mapping(
            call_target_source,
            "bias_targets.tool_call_count_mix",
        )
        if set(target_mapping) != {"1", "2", "3+"}:
            raise BiasAuditError("bias_targets.tool_call_count_mix must contain exactly '1', '2', and '3+'")
        call_target = {str(key): float(value) for key, value in target_mapping.items()}
        if any(not math.isfinite(value) or value < 0 for value in call_target.values()) or not math.isclose(
            sum(call_target.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise BiasAuditError("bias_targets.tool_call_count_mix must be non-negative and sum to 1")
        call_errors = {bucket: abs(call_rates[bucket] - call_target[bucket]) for bucket in ("1", "2", "3+")}
        call_max = max(call_errors.values())
    multi_call_count = sum(1 for row in published if int(row["num_tool_calls"]) >= 2)
    metrics.append(
        _metric(
            "B5",
            "tool_call_count_mix_max_abs",
            call_max,
            "<=0.05; at least one multi-call row",
            call_max is not None
            and call_max <= ABSOLUTE_MIX_TOLERANCE
            and multi_call_count > 0,
            {
                "counts_excluding_zero_call_rows": dict(sorted(call_buckets.items())),
                "zero_call_rows": sum(1 for row in published if row["num_tool_calls"] == 0),
                "empirical": call_rates,
                "target": call_target,
                "absolute_errors": call_errors,
                "absolute_tolerance": ABSOLUTE_MIX_TOLERANCE,
                "multi_call_rate": multi_call_count / len(published),
                **({} if target_complete else {"missing_evidence": ("run_manifest.bias_targets.tool_call_count_mix")}),
            },
            sources,
            applicability=applicability["B5"],
            evidence_complete=target_complete,
        )
    )

    fixture_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in published:
        for value in row.get("fixture_refs") or []:
            parsed = _parse_fixture_ref(str(value))
            if parsed is not None:
                fixture_counts[parsed[0]][parsed[1]] += 1
    concentrations = {
        collection: max(counts.values()) / sum(counts.values())
        for collection, counts in fixture_counts.items()
        if counts
    }
    id_concentration = max(concentrations.values(), default=0.0)
    fixture_floor = all(len(counts) >= 3 for counts in fixture_counts.values())
    metrics.append(
        _metric(
            "B6",
            "id_concentration_max",
            id_concentration,
            "<=0.25; >=3 distinct ids per applicable collection",
            id_concentration <= 0.25 and fixture_floor,
            {
                "top_id_share_by_collection": concentrations,
                "distinct_ids_by_collection": {key: len(value) for key, value in sorted(fixture_counts.items())},
            },
            sources,
            applicability=applicability["B6"],
            evidence_complete=bool(fixture_counts),
        )
    )

    held_out = _held_out_refs(pack)
    layer_rows = {
        "expanded": expanded,
        "raw": raw,
        "published": published,
    }
    held_hits = {name: _held_out_hits(rows, held_out) for name, rows in layer_rows.items()}
    held_complete = all(layers[name]["available"] for name in ("expanded", "raw")) and bool(pack)
    held_count = sum(len(value) for value in held_hits.values())
    held_missing = [name for name in ("expanded", "raw") if not layers[name]["available"]]
    if not pack:
        held_missing.append("held-out policy")
    metrics.append(
        _metric(
            "B7",
            "held_out_leak_count",
            held_count,
            "=0 across expanded, raw, and published",
            held_count == 0,
            {
                "hits_by_layer": held_hits,
                "held_out_reference_count": len(held_out),
                "layers": dict(layers),
                **({} if held_complete else {"missing_evidence": ", ".join(held_missing)}),
            },
            ["run_manifest.json", "expanded", "raw", "published", "held_out.yaml"],
            applicability=applicability["B7"],
            evidence_complete=held_complete,
        )
    )

    pack_tool_names: list[str] = []
    for tool in pack.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        name = function.get("name") if isinstance(function, dict) else tool.get("name")
        if isinstance(name, str) and name:
            pack_tool_names.append(name)
    pack_tool_names = sorted(set(pack_tool_names))
    paraphrase_hits = {name: _paraphrase_leaks(rows, pack_tool_names) for name, rows in layer_rows.items()}
    paraphrase_unique = {canonical_json(item) for values in paraphrase_hits.values() for item in values}
    paraphrase_complete = all(layers[name]["available"] for name in ("expanded", "raw"))
    paraphrase_missing = [name for name in ("expanded", "raw") if not layers[name]["available"]]
    rejection = manifest.get("paraphrase_rejections") or {}
    attempted = int(rejection.get("requested_candidates") or 0)
    rejected = int(rejection.get("rejected_candidates") or 0)
    metrics.append(
        _metric(
            "B8",
            "paraphrase_leak_escape_count",
            len(paraphrase_unique),
            "=0 across expanded, raw, and published",
            not paraphrase_unique,
            {
                "leaks_by_layer": paraphrase_hits,
                "layers": dict(layers),
                "paraphrase_reject_rate": rejected / attempted if attempted else None,
                "rejections_by_guard": rejection.get("by_reason") or {},
                **(
                    {}
                    if paraphrase_complete
                    else {"missing_evidence": ", ".join(paraphrase_missing) + " paraphrase rows"}
                ),
            },
            ["run_manifest.json", "expanded", "raw", "published"],
            applicability=applicability["B8"],
            evidence_complete=paraphrase_complete,
        )
    )

    contamination = _contamination_metric(
        contamination_reports,
        run_id=str(manifest.get("run_id")),
        task_ids_hash=_semantic_hash([str(row["task_id"]) for row in published]),
    )
    metrics.append(
        _metric(
            "B9",
            "contamination_violations",
            contamination[0],
            "=0 with release-bound evaluation evidence",
            contamination[1],
            contamination[2],
            ["run_manifest.json", "contamination_report.json"],
            applicability=applicability["B9"],
            evidence_complete=contamination[3],
        )
    )

    distractor = _distractor_metric(published, distractor_evidence, manifest_hash)
    metrics.append(
        _metric(
            "B10",
            "distractor_gold_agreement",
            distractor[0],
            ">=0.90 on exact seeded min(30, eligible) sample",
            distractor[1],
            distractor[2],
            ["published", "reviewed B10 evidence"],
            applicability=applicability["B10"],
            evidence_complete=distractor[3],
        )
    )

    dedup = _mapping(
        manifest.get("semantic_deduplication") or {},
        "semantic_deduplication",
    )
    dedup_report = dedup.get("report") or {}
    dedup_counts = dedup_report.get("counts") or {} if isinstance(dedup_report, dict) else {}
    dedup_input = int(dedup_counts.get("stage_ten_survivors") or len(raw) or len(published))
    dedup_dropped = int(dedup_counts.get("semantic_duplicate_drops") or 0)
    metrics.append(
        _metric(
            "B11",
            "post_dedup_edge_coverage",
            edge_coverage,
            "=1.0",
            edge_coverage == 1.0,
            {
                "dedup_enabled": bool(dedup.get("enabled")),
                "dedup_drop_rate": dedup_dropped / dedup_input if dedup_input else None,
                "dedup_drop_count": dedup_dropped,
                "edge_hits": edge_hits,
            },
            ["run_manifest.json", "published"],
            applicability=applicability["B11"],
        )
    )

    portability = _portability_metric(portability_evidence, manifest_hash)
    metrics.append(
        _metric(
            "B12",
            "agnostic_portability_pass",
            portability[0],
            "=true",
            portability[1],
            portability[2],
            ["reviewed B12 portability evidence"],
            applicability=applicability["B12"],
            evidence_complete=portability[3],
        )
    )

    judge = _judge_metric(manifest, published, judge_evidence)
    metrics.append(
        _metric(
            "B13",
            "truth_creep_incidents",
            judge[0],
            "=0",
            judge[1],
            judge[2],
            ["run_manifest.json", "surface-quality report", "reviewed B13 evidence"],
            applicability=applicability["B13"],
            evidence_complete=judge[3],
        )
    )

    negative_count = sum(1 for row in published if row["turn_policy"] in {"negative_path", "irrelevant"})
    negative_rate = negative_count / len(published)
    families = sorted({family for row in published for family in _negative_families(row) if family != "decline"})
    metrics.append(
        _metric(
            "B14",
            "negative_or_irrelevant_rate",
            negative_rate,
            ">=0.10; >=2 negative families when defined",
            negative_rate >= 0.10 and len(families) >= 2,
            {
                "negative_or_irrelevant_count": negative_count,
                "negative_families": families,
                "distinct_negative_families": len(families),
            },
            sources,
            applicability=applicability["B14"],
        )
    )

    category_intents: dict[str, Counter[str]] = defaultdict(Counter)
    for row in published:
        category_intents[str(row["category"])][str(row["intent"])] += 1
    multi_intent = {category: counts for category, counts in category_intents.items() if len(counts) >= 2}
    mean_gini = (
        sum(_gini(counts.values()) for counts in multi_intent.values()) / len(multi_intent) if multi_intent else 0.0
    )
    intent_balance = 1 - mean_gini
    declared_intents = {
        str(template.get("intent"))
        for template in pack.get("templates") or []
        if isinstance(template, dict) and template.get("intent")
    }
    observed_intents = {str(row["intent"]) for row in published}
    orphan_intents = sorted(declared_intents - observed_intents)
    max_shares = {
        category: max(counts.values()) / sum(counts.values()) for category, counts in category_intents.items()
    }
    max_intent_share_source = targets.get("max_intent_share")
    max_intent_share: float | None = None
    if max_intent_share_source is not None:
        if (
            not isinstance(max_intent_share_source, (int, float))
            or isinstance(max_intent_share_source, bool)
            or not math.isfinite(float(max_intent_share_source))
            or not 0.0 < float(max_intent_share_source) <= 1.0
        ):
            raise BiasAuditError(
                "bias_targets.max_intent_share must be finite, greater than 0, "
                "and at most 1"
            )
        max_intent_share = float(max_intent_share_source)
    allowed_max_intent_share = (
        min(1.0, max_intent_share + ABSOLUTE_MIX_TOLERANCE)
        if max_intent_share is not None
        else None
    )
    missing_intent_evidence = [
        name
        for available, name in (
            (bool(pack), "pack task_templates.yaml"),
            (max_intent_share is not None, "run_manifest.bias_targets.max_intent_share"),
        )
        if not available
    ]
    intent_complete = not missing_intent_evidence
    metrics.append(
        _metric(
            "B15",
            "intent_balance_score",
            intent_balance,
            ">=0.65; orphan_intent_rate=0; max intent share <= configured target +0.05",
            intent_complete
            and intent_balance >= 0.65
            and not orphan_intents
            and allowed_max_intent_share is not None
            and all(value <= allowed_max_intent_share for value in max_shares.values()),
            {
                "counts_by_category": {
                    category: dict(sorted(counts.items())) for category, counts in sorted(category_intents.items())
                },
                "single_intent_categories": sorted(set(category_intents) - set(multi_intent)),
                "max_intent_share_by_category": max_shares,
                "max_intent_share_target": max_intent_share,
                "absolute_tolerance": ABSOLUTE_MIX_TOLERANCE,
                "allowed_max_intent_share": allowed_max_intent_share,
                "orphan_intents": orphan_intents,
                "orphan_intent_rate": (len(orphan_intents) / len(declared_intents) if declared_intents else None),
                **({} if intent_complete else {"missing_evidence": missing_intent_evidence}),
            },
            ["published", "pack task_templates.yaml"],
            applicability=applicability["B15"],
            evidence_complete=intent_complete,
        )
    )

    multi_count = sum(bool(row["is_multi_turn"]) for row in published)
    multi_share = multi_count / len(published)
    turn_target = float(
        _mapping(targets.get("turn_mix"), "bias_targets.turn_mix").get(
            "multi_turn",
            0.4,
        )
    )
    turn_error = abs(multi_share - turn_target)
    metrics.append(
        _metric(
            "B16",
            "turn_mix_abs_error",
            turn_error,
            "<=0.05",
            turn_error <= 0.05,
            {
                "single_turn_count": len(published) - multi_count,
                "multi_turn_count": multi_count,
                "multi_turn_share": multi_share,
                "target_multi_turn_share": turn_target,
            },
            sources,
            applicability=applicability["B16"],
        )
    )
    if [metric["bias_id"] for metric in metrics] != list(BIAS_IDS):
        raise AssertionError("internal B1-B16 ordering mismatch")
    return metrics


def build_bias_audit_report(inputs: AuditInputs) -> dict[str, Any]:
    """Read and verify frozen artifacts, then recompute the B1-B16 report."""
    manifest_path = inputs.run_manifest.expanduser().resolve()
    manifest = _load_document(manifest_path, "run manifest")
    manifest_hash = _file_hash(manifest_path)
    resolved = _auto_inputs(
        AuditInputs(
            **{
                **inputs.__dict__,
                "run_manifest": manifest_path,
                "output_dir": inputs.output_dir.expanduser().resolve(),
            }
        ),
        manifest,
    )
    applicability = _validate_applicability(manifest)
    pack = _load_pack(resolved.pack_manifest.expanduser().resolve() if resolved.pack_manifest is not None else None)
    input_artifact_hashes = _input_artifact_hashes(resolved, pack)
    published, published_hash = _load_published(resolved.published, manifest)  # type: ignore[arg-type]
    expected_rows = int(
        _mapping(manifest.get("publication"), "run_manifest.publication").get("published", {}).get("rows")
    )
    if len(published) != expected_rows:
        raise BiasAuditError(f"published row count {len(published)} != manifest {expected_rows}")
    task_ids = [str(row["task_id"]) for row in published]
    if len(task_ids) != len(set(task_ids)):
        raise BiasAuditError("published task ids are not unique")
    raw, raw_layer = _load_optional_layer(
        resolved.raw,
        manifest=manifest,
        artifact_name="benchmark_raw_parquet",
    )
    expanded, expanded_layer = _load_expanded_layer(
        resolved.expanded,
        manifest=manifest,
    )
    layers = {
        "expanded": expanded_layer,
        "raw": raw_layer,
        "published": {
            "available": True,
            "rows": len(published),
            "content_hash": published_hash,
            "held_out_rescan": True,
            "paraphrase_rescan": True,
        },
    }
    distractor = _load_review_evidence(
        resolved.distractor_evidence,
        kind="distractor_gold_agreement",
        manifest_hash=manifest_hash,
    )
    judge = _load_review_evidence(
        resolved.judge_evidence,
        kind="judge_truth_creep",
        manifest_hash=manifest_hash,
    )
    portability = (
        _load_document(resolved.portability_evidence, "B12 portability evidence")
        if resolved.portability_evidence is not None
        else None
    )
    metrics = _build_metrics(
        manifest,
        applicability,
        published,
        raw,
        expanded,
        layers,
        pack,
        manifest_hash,
        distractor,
        judge,
        portability,
        resolved.contamination_reports,
    )
    exceptions = _load_exceptions(resolved.exceptions)
    by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for exception in exceptions:
        by_metric[exception["affected_metric"]].append(exception)
    for metric in metrics:
        metric["exceptions"] = by_metric.get(metric["bias_id"], [])
    failed = [
        metric["bias_id"]
        for metric in metrics
        if metric["applicability"]["status"] == "applicable" and not metric["primary_metric"]["passed"]
    ]
    unexcepted = [bias_id for bias_id in failed if not by_metric.get(bias_id)]
    report: dict[str, Any] = {
        "schema_version": BIAS_AUDIT_CONTRACT_VERSION,
        "report_hash": None,
        "source": {
            "run_id": manifest.get("run_id"),
            "run_manifest": manifest_path.name,
            "run_manifest_hash": manifest_hash,
            "published_content_hash": published_hash,
            "published_task_ids_hash": _semantic_hash(task_ids),
            "published_rows": len(published),
            "pack_id": _mapping(manifest.get("pack"), "run_manifest.pack").get("pack_id"),
            "pack_version": _mapping(
                manifest.get("pack"),
                "run_manifest.pack",
            ).get("version"),
            "input_artifact_hashes": input_artifact_hashes,
        },
        "layers": layers,
        "metrics": metrics,
        "summary": {
            "applicable": sum(metric["applicability"]["status"] == "applicable" for metric in metrics),
            "not_applicable": sum(metric["applicability"]["status"] == "na" for metric in metrics),
            "passed": sum(
                metric["applicability"]["status"] == "applicable" and metric["primary_metric"]["passed"]
                for metric in metrics
            ),
            "failed_bias_ids": failed,
            "approved_exception_bias_ids": [
                bias_id for bias_id in failed if bias_id not in unexcepted
            ],
            "unexcepted_failure_bias_ids": unexcepted,
            "status": ("passed" if not failed else "passed_with_exceptions" if not unexcepted else "failed"),
        },
    }
    report["report_hash"] = _semantic_hash({**report, "report_hash": None})
    validate_bias_audit_report(report)

    # Inputs are held stable until the complete semantic report exists.
    if _file_hash(manifest_path) != manifest_hash:
        raise BiasAuditError("run_manifest.json changed during the audit")
    if resolved.published.is_dir():  # type: ignore[union-attr]
        declared = _mapping(manifest.get("exports") or {}, "run_manifest.exports").get("formats", {}).get("bfcl_json")
        declared = _mapping(
            declared,
            "run_manifest.exports.formats.bfcl_json",
        )
        declared_path = declared.get("path")
        if not isinstance(declared_path, str):
            raise BiasAuditError("BFCL JSON export needs a declared path")
        current = _published_export_hash(
            resolved.published,  # type: ignore[arg-type]
            BFCL_JSON_FILES,
            declared_path,
        )
    else:
        current = _file_hash(resolved.published)  # type: ignore[arg-type]
    if current != published_hash:
        raise BiasAuditError("published benchmark changed during the audit")
    current_pack = _load_pack(
        resolved.pack_manifest.expanduser().resolve() if resolved.pack_manifest is not None else None
    )
    if _input_artifact_hashes(resolved, current_pack) != input_artifact_hashes:
        raise BiasAuditError("an audit evidence artifact changed during the audit")
    return report


def validate_bias_audit_report(report: Mapping[str, Any]) -> None:
    """Validate the strict public shape and its semantic content address."""
    allowed = {
        "schema_version",
        "report_hash",
        "source",
        "layers",
        "metrics",
        "summary",
    }
    if set(report) != allowed:
        raise BiasAuditError("bias audit report has unknown or missing top-level keys")
    if report.get("schema_version") != BIAS_AUDIT_CONTRACT_VERSION:
        raise BiasAuditError("unsupported bias audit report schema_version")
    digest = report.get("report_hash")
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        raise BiasAuditError("bias audit report_hash must be sha256:<64 hex>")
    expected = _semantic_hash({**dict(report), "report_hash": None})
    if digest != expected:
        raise BiasAuditError("bias audit report_hash does not match report content")
    source = _mapping(report.get("source"), "bias audit source")
    if set(source) != {
        "run_id",
        "run_manifest",
        "run_manifest_hash",
        "published_content_hash",
        "published_task_ids_hash",
        "published_rows",
        "pack_id",
        "pack_version",
        "input_artifact_hashes",
    }:
        raise BiasAuditError("bias audit source has an invalid shape")
    for field in (
        "run_manifest_hash",
        "published_content_hash",
        "published_task_ids_hash",
    ):
        if not isinstance(source[field], str) or _DIGEST.fullmatch(source[field]) is None:
            raise BiasAuditError(f"bias audit source {field} is not a content hash")
    if (
        not isinstance(source["published_rows"], int)
        or isinstance(source["published_rows"], bool)
        or source["published_rows"] < 1
    ):
        raise BiasAuditError("bias audit source published_rows must be positive")
    input_hashes = _mapping(
        source["input_artifact_hashes"],
        "source.input_artifact_hashes",
    )
    if any(not isinstance(value, str) or _DIGEST.fullmatch(value) is None for value in input_hashes.values()):
        raise BiasAuditError("source input artifact hashes are invalid")
    layers = _mapping(report.get("layers"), "bias audit layers")
    if set(layers) != {"expanded", "raw", "published"}:
        raise BiasAuditError("bias audit layers must be expanded, raw, and published")
    for name, layer_value in layers.items():
        layer = _mapping(layer_value, f"bias audit layer {name}")
        required_layer = {
            "available",
            "rows",
            "content_hash",
            "held_out_rescan",
            "paraphrase_rescan",
        }
        if not required_layer <= set(layer) or set(layer) - required_layer - {"artifact_hashes"}:
            raise BiasAuditError(f"bias audit layer {name} has an invalid shape")
        if not isinstance(layer["available"], bool):
            raise BiasAuditError(f"bias audit layer {name} availability is invalid")
        if not isinstance(layer["rows"], int) or isinstance(layer["rows"], bool) or layer["rows"] < 0:
            raise BiasAuditError(f"bias audit layer {name} row count is invalid")
        expected_rescan = layer["available"]
        if layer["held_out_rescan"] is not expected_rescan or layer["paraphrase_rescan"] is not expected_rescan:
            raise BiasAuditError(f"bias audit layer {name} rescan status is invalid")
        content_hash = layer["content_hash"]
        if expected_rescan != (isinstance(content_hash, str) and _DIGEST.fullmatch(content_hash) is not None):
            raise BiasAuditError(f"bias audit layer {name} content hash is invalid")
    metrics = _list(report.get("metrics"), "bias audit metrics")
    if [item.get("bias_id") for item in metrics if isinstance(item, dict)] != list(BIAS_IDS):
        raise BiasAuditError("bias audit report must contain B1-B16 exactly once")
    for item in metrics:
        item = _mapping(item, "bias metric")
        if set(item) != {
            "bias_id",
            "applicability",
            "primary_metric",
            "supporting_diagnostics",
            "source_evidence",
            "evidence_complete",
            "exceptions",
        }:
            raise BiasAuditError(f"{item.get('bias_id')} has an invalid metric record shape")
        primary = _mapping(item["primary_metric"], "primary_metric")
        if set(primary) != {"name", "value", "threshold", "passed"}:
            raise BiasAuditError(f"{item['bias_id']} must have exactly one primary metric")
        if (
            not isinstance(primary["name"], str)
            or not primary["name"].strip()
            or not isinstance(primary["threshold"], str)
            or not primary["threshold"].strip()
            or not isinstance(primary["passed"], bool)
        ):
            raise BiasAuditError(f"{item['bias_id']} primary metric is invalid")
        applicability = _mapping(item["applicability"], "metric applicability")
        if applicability.get("status") not in {"applicable", "na"}:
            raise BiasAuditError(f"{item['bias_id']} applicability is invalid")
        if applicability["status"] == "na":
            if (
                not isinstance(applicability.get("reason"), str)
                or not applicability["reason"].strip()
                or primary["value"] is not None
                or primary["passed"] is not True
                or item["evidence_complete"] is not True
            ):
                raise BiasAuditError(f"{item['bias_id']} N/A record is invalid")
        elif set(applicability) != {"status"}:
            raise BiasAuditError(f"{item['bias_id']} applicable record cannot carry a reason")
        if not isinstance(item["evidence_complete"], bool):
            raise BiasAuditError(f"{item['bias_id']} evidence status is invalid")
        _mapping(item["supporting_diagnostics"], "supporting diagnostics")
        sources = _list(item["source_evidence"], "source evidence")
        if any(not isinstance(value, str) or not value for value in sources):
            raise BiasAuditError(f"{item['bias_id']} source evidence is invalid")
        exceptions = _list(item["exceptions"], "metric exceptions")
        for exception_value in exceptions:
            exception = _mapping(exception_value, "metric exception")
            if (
                set(exception)
                != {
                    "affected_metric",
                    "owner",
                    "rationale",
                    "approval_date",
                }
                or exception["affected_metric"] != item["bias_id"]
            ):
                raise BiasAuditError(f"{item['bias_id']} exception is invalid")
    summary = _mapping(report.get("summary"), "bias audit summary")
    if set(summary) != {
        "applicable",
        "not_applicable",
        "passed",
        "failed_bias_ids",
        "approved_exception_bias_ids",
        "unexcepted_failure_bias_ids",
        "status",
    }:
        raise BiasAuditError("bias audit summary has an invalid shape")
    applicable_metrics = [item for item in metrics if item["applicability"]["status"] == "applicable"]
    failed = [item["bias_id"] for item in applicable_metrics if not item["primary_metric"]["passed"]]
    excepted = [
        item["bias_id"] for item in applicable_metrics if not item["primary_metric"]["passed"] and item["exceptions"]
    ]
    unexcepted = [bias_id for bias_id in failed if bias_id not in excepted]
    expected_status = "passed" if not failed else "passed_with_exceptions" if not unexcepted else "failed"
    if summary != {
        "applicable": len(applicable_metrics),
        "not_applicable": len(metrics) - len(applicable_metrics),
        "passed": sum(item["primary_metric"]["passed"] for item in applicable_metrics),
        "failed_bias_ids": failed,
        "approved_exception_bias_ids": excepted,
        "unexcepted_failure_bias_ids": unexcepted,
        "status": expected_status,
    }:
        raise BiasAuditError("bias audit summary is inconsistent with metrics")


def render_bias_audit_markdown(report: Mapping[str, Any]) -> str:
    """Render a deterministic reviewer summary from a validated JSON report."""
    validate_bias_audit_report(report)
    source = _mapping(report["source"], "report source")
    summary = _mapping(report["summary"], "report summary")
    lines = [
        "# BFCL B1-B16 bias audit",
        "",
        f"- Report hash: `{report['report_hash']}`",
        f"- Source run: `{source['run_id']}`",
        f"- Manifest hash: `{source['run_manifest_hash']}`",
        f"- Published rows: {source['published_rows']}",
        f"- Status: **{summary['status']}**",
        "",
        "| ID | Applicability | Primary metric | Value | Threshold | Pass | Evidence |",
        "|---|---|---|---:|---|---|---|",
    ]
    for metric in report["metrics"]:
        applicability = metric["applicability"]
        primary = metric["primary_metric"]
        value = json.dumps(
            primary["value"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        reason = applicability.get("reason")
        applicability_text = applicability["status"] + (f": {reason}" if reason else "")
        lines.append(
            "| {bias_id} | {applicability} | `{name}` | `{value}` | {threshold} | {passed} | {evidence} |".format(
                bias_id=metric["bias_id"],
                applicability=applicability_text.replace("|", "\\|"),
                name=primary["name"],
                value=value.replace("|", "\\|"),
                threshold=str(primary["threshold"]).replace("|", "\\|"),
                passed="yes" if primary["passed"] else "no",
                evidence="complete" if metric["evidence_complete"] else "incomplete",
            )
        )
    lines.extend(["", "## Metric diagnostics", ""])
    for metric in report["metrics"]:
        lines.extend(
            [
                f"### {metric['bias_id']}",
                "- Source evidence: " + ", ".join(f"`{value}`" for value in metric["source_evidence"]),
                "- Supporting diagnostics:",
                "```json",
                json.dumps(
                    metric["supporting_diagnostics"],
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                ),
                "```",
                "",
            ]
        )
    lines.extend(["## Failures and exceptions", ""])
    if not summary["failed_bias_ids"]:
        lines.append("No applicable metric failed.")
    for metric in report["metrics"]:
        if metric["bias_id"] not in summary["failed_bias_ids"]:
            continue
        lines.append(f"### {metric['bias_id']}")
        missing = metric["supporting_diagnostics"].get("missing_evidence")
        if missing:
            lines.append(f"- Missing evidence: {missing}")
        if metric["exceptions"]:
            for exception in metric["exceptions"]:
                lines.append(
                    "- Approved exception: {rationale} — {owner}, {date}".format(
                        rationale=exception["rationale"],
                        owner=exception["owner"],
                        date=exception["approval_date"],
                    )
                )
        else:
            lines.append("- No approved exception.")
        lines.append("")
    lines.extend(["## Input layers", ""])
    for name, layer in report["layers"].items():
        lines.append(
            f"- `{name}`: available={str(layer['available']).lower()}, "
            f"rows={layer['rows']}, hash={layer['content_hash']}"
        )
    return "\n".join(lines) + "\n"


def write_bias_audit_reports(
    report: Mapping[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write JSON and Markdown atomically; never replace different audit bytes."""
    validate_bias_audit_report(report)
    root = output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "bias_audit_report.json"
    markdown_path = root / "bias_audit_report.md"
    json_bytes = (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    markdown_bytes = render_bias_audit_markdown(report).encode("utf-8")
    for path, content in (
        (json_path, json_bytes),
        (markdown_path, markdown_bytes),
    ):
        if path.exists():
            if not path.is_file() or path.read_bytes() != content:
                raise BiasAuditError(f"refusing to replace a different report: {path}")
            continue
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_bytes(content)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return json_path, markdown_path


__all__ = [
    "AuditInputs",
    "BIAS_AUDIT_CONTRACT_VERSION",
    "BIAS_IDS",
    "BiasAuditError",
    "build_bias_audit_report",
    "render_bias_audit_markdown",
    "validate_bias_audit_report",
    "write_bias_audit_reports",
]
