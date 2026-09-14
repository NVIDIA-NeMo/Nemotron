#!/usr/bin/env python3
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

"""Restore a hash-verified BFCL rendered-conversation cache from the raw table."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import (
    benchmark_schema,
    canonical_json,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stage_tables import (
    rendered_conversations_schema,
    write_stage_table,
)


class RestoreRenderedConversationsError(ValueError):
    """The frozen raw table cannot reproduce its committed rendered cache."""


def _hash(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RestoreRenderedConversationsError(f"{label} must be a mapping")
    return value


def _steps(row: dict[str, Any]) -> list[dict[str, Any]]:
    expected = list(row.get("expected_tool_calls") or [])
    expected_index = 0
    steps: list[dict[str, Any]] = []
    for message in row.get("messages") or []:
        role = message.get("role")
        if role == "user":
            steps.append({"kind": "user", "content": str(message["content"])})
        elif role == "assistant" and message.get("tool_calls"):
            tool_calls = list(message["tool_calls"])
            group = expected[expected_index : expected_index + len(tool_calls)]
            if len(group) != len(tool_calls):
                raise RestoreRenderedConversationsError(
                    f"{row['task_id']}: assistant call group exceeds expected trace"
                )
            groups = {int(call["call_group"]) for call in group}
            if len(groups) != 1:
                raise RestoreRenderedConversationsError(f"{row['task_id']}: assistant message spans call groups")
            expected_names = [str(call["function_name"]) for call in group]
            actual_names = [str(_mapping(call.get("function"), "tool call function")["name"]) for call in tool_calls]
            if actual_names != expected_names:
                raise RestoreRenderedConversationsError(f"{row['task_id']}: message calls do not match expected trace")
            steps.append({"kind": "calls", "call_group": groups.pop()})
            expected_index += len(tool_calls)
        elif role == "assistant" and message.get("content") is not None:
            steps.append({"kind": "assistant_text", "content": str(message["content"])})
        elif role not in {"system", "tool"}:
            raise RestoreRenderedConversationsError(f"{row['task_id']}: unsupported message role {role!r}")
    if expected_index != len(expected):
        raise RestoreRenderedConversationsError(f"{row['task_id']}: expected trace has unconsumed calls")
    return steps


def _rendered_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = _mapping(
        json.loads(str(row.get("metadata") or "{}")),
        "row metadata",
    )
    steps = _steps(row)
    task_id = str(row["task_id"])
    return {
        "task_id": task_id,
        "base_task_id": str(metadata.get("base_task_id") or task_id),
        "variant_index": int(row.get("variant_index") or 0),
        "source": str(metadata.get("surface_source") or "template"),
        "language": str(metadata["language"]),
        "system_prompt_id": str(row["system_prompt_id"]),
        "paraphrase_model": row.get("paraphrase_model"),
        "paraphrase_model_canonical": row.get("paraphrase_model_canonical"),
        "profile_hash": metadata.get("profile_hash"),
        "num_user_turns": sum(step["kind"] == "user" for step in steps),
        "accepted": True,
        "guard_violations": canonical_json([]),
        "turns": canonical_json(steps),
    }


def restore_rendered_conversations(
    *,
    run_manifest: Path,
    raw_benchmark: Path,
    output: Path,
) -> Path:
    """Reconstruct the cache and publish it only if its committed hash matches."""
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    manifest = _mapping(
        json.loads(run_manifest.read_text(encoding="utf-8")),
        "run manifest",
    )
    artifacts = _mapping(manifest.get("artifacts"), "run_manifest.artifacts")
    expected_raw = _mapping(
        artifacts.get("benchmark_raw_parquet"),
        "artifacts.benchmark_raw_parquet",
    ).get("content_hash")
    if _hash(raw_benchmark) != expected_raw:
        raise RestoreRenderedConversationsError("raw benchmark hash does not match run_manifest.json")
    table = pq.read_table(raw_benchmark)
    if not table.schema.equals(benchmark_schema()):
        raise RestoreRenderedConversationsError("raw benchmark does not use the BFCL benchmark schema")
    rows = [_rendered_row(row) for row in table.to_pylist()]
    expected = _mapping(
        artifacts.get("rendered_conversations"),
        "artifacts.rendered_conversations",
    ).get("content_hash")
    if output.exists():
        if output.is_file() and _hash(output) == expected:
            return output
        raise RestoreRenderedConversationsError(f"refusing to replace existing output: {output}")
    candidate = output.with_name(f".{output.name}.restore")
    candidate.unlink(missing_ok=True)
    try:
        write_stage_table(candidate, rows, rendered_conversations_schema())
        actual = _hash(candidate)
        if actual != expected:
            raise RestoreRenderedConversationsError(f"reconstructed hash {actual} does not match manifest {expected}")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise RestoreRenderedConversationsError(f"output appeared during restore: {output}")
        candidate.replace(output)
    finally:
        candidate.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--raw-benchmark", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        output = restore_rendered_conversations(
            run_manifest=args.run_manifest,
            raw_benchmark=args.raw_benchmark,
            output=args.output,
        )
    except (OSError, json.JSONDecodeError, RestoreRenderedConversationsError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "status": "fail",
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    print(json.dumps({"output": str(output), "content_hash": _hash(output)}))


if __name__ == "__main__":
    main()
