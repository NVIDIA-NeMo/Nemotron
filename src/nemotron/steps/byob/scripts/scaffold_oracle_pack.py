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

"""Create a runnable, transport-specific BFCL Oracle-pack starter."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]

_PACK_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_ZERO_DIGEST = "sha256:" + "0" * 64
OPERATOR_REFERENCE_SLUGS = (
    "oracle-pack-inputs",
    "manifest",
    "tools-and-fixtures",
    "python-backend",
    "task-templates",
    "assertions",
    "validation-cases",
    "endpoint-config",
    "held-out-policy",
)


def _pack_id(domain: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", domain.strip().casefold()).strip("_")
    if not _PACK_ID.fullmatch(value):
        raise ValueError("domain must produce a pack id matching [a-z][a-z0-9_]*")
    return value


def _manifest(
    *,
    pack_id: str,
    version: str,
    language: str,
    transport: Literal["python", "endpoint"],
    include_held_out: bool,
) -> dict:
    paths = {
        "tools": "tools.json",
        "fixtures": "fixtures.json",
        "templates": "task_templates.yaml",
        "assertions": "assertions.py",
        "validation_cases": "validation_cases.yaml",
        "backend" if transport == "python" else "endpoint": (
            "backend.py" if transport == "python" else "endpoint_config.yaml"
        ),
    }
    manifest: dict = {
        "pack_id": pack_id,
        "version": version,
        "description": f"Runnable BFCL starter pack for {pack_id}.",
        "languages": [language],
        "clock": "2026-01-01T00:00:00+00:00",
        "paths": paths,
        "primary_keys": {"records": "record_id"},
        "absent_ids": {"records": ["REC-ABSENT-1"]},
        "assistant_turn_templates": {
            "ask_for_slot": {language: "Please provide {slot_name}."},
            "ask_confirm": {language: "Please confirm the requested operation."},
            "decline": {language: "That request is outside the available tools."},
            "final_answer": {language: "Done."},
        },
    }
    if include_held_out:
        manifest["held_out"] = "held_out.yaml"
    return manifest


def _tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_record",
                "description": "Return one record by its stable identifier.",
                "parameters": {
                    "type": "object",
                    "properties": {"record_id": {"type": "string"}},
                    "required": ["record_id"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _fixtures(include_held_out: bool) -> dict:
    records = [{"record_id": "REC-001", "name": "Example record", "status": "active"}]
    if include_held_out:
        records.append(
            {
                "record_id": "REC-HELD-OUT-1",
                "name": "Reserved record",
                "status": "active",
            }
        )
    return {"records": records}


def _templates(language: str) -> list[dict]:
    return [
        {
            "template_id": "starter_get_record",
            "intent": "get_record",
            "category": "records",
            "difficulty": "easy",
            "turn_policy": "single_turn",
            "call_order": "strict",
            "required_tools": ["get_record"],
            "tools_present": ["get_record"],
            "slots": {
                "record_id": {
                    "source": "fixture:records.record_id",
                    "visible_in_first_turn": True,
                }
            },
            "success_assertions": ["assert_record_reported"],
            "paraphrase": {"allowed": False},
            "user_turn_templates": {language: "Show me record {record_id}."},
            "assistant_milestones": [
                {"type": "tool_call", "tool": "get_record"},
                {"type": "final_answer"},
            ],
        },
        {
            "template_id": "starter_missing_record",
            "intent": "get_missing_record",
            "category": "records",
            "difficulty": "medium",
            "turn_policy": "negative_path",
            "call_order": "strict",
            "required_tools": ["get_record"],
            "tools_present": ["get_record"],
            "slots": {
                "record_id": {
                    "source": "absent:records",
                    "visible_in_first_turn": True,
                }
            },
            "success_assertions": ["assert_record_not_found"],
            "paraphrase": {"allowed": False},
            "user_turn_templates": {language: "Show me record {record_id}."},
            "assistant_milestones": [
                {"type": "tool_call", "tool": "get_record"},
                {"type": "final_answer"},
            ],
        },
        {
            "template_id": "starter_irrelevant",
            "intent": "unsupported_request",
            "category": "records",
            "difficulty": "easy",
            "turn_policy": "irrelevant",
            "call_order": "strict",
            "required_tools": [],
            "tools_present": ["get_record"],
            "slots": {},
            "success_assertions": ["assert_no_tool_called"],
            "paraphrase": {"allowed": False},
            "user_turn_templates": {language: "Perform an operation this pack does not support."},
            "assistant_milestones": [{"type": "decline"}],
        },
    ]


def _backend() -> str:
    return '''"""Deterministic starter backend. Replace the record domain, not the interface."""

from __future__ import annotations

import copy
from typing import Any

_SNAPSHOT: dict[str, Any] | None = None
_STATE: dict[str, Any] = {}


def list_tools() -> list[str]:
    """Return public names that must match tools.json exactly."""
    return ["get_record"]


def reset(*, ctx: Any, fixtures: dict | None = None) -> None:
    """Start one deterministic episode from a defensive fixture snapshot."""
    del ctx
    global _SNAPSHOT, _STATE
    if fixtures is not None:
        _SNAPSHOT = copy.deepcopy(fixtures)
    if _SNAPSHOT is None:
        raise RuntimeError("reset requires fixtures on the first call")
    _STATE = copy.deepcopy(_SNAPSHOT)


def get_state() -> dict:
    """Return a copy of the complete state that assertions may inspect."""
    return copy.deepcopy(_STATE)


def call_tool(name: str, arguments: dict, *, ctx: Any) -> dict:
    """Dispatch one public tool; return expected business failures as data."""
    del ctx
    if name != "get_record":
        return _error("invalid_argument", "name", None, f"unknown tool {name!r}")
    record_id = arguments.get("record_id")
    if not isinstance(record_id, str):
        return _error(
            "invalid_argument", "record_id", None, "record_id must be a string"
        )
    for record in _STATE.get("records", []):
        if record.get("record_id") == record_id:
            return copy.deepcopy(record)
    return _error("not_found", "record_id", record_id, "record was not found")


def _error(code: str, field: str, identifier: str | None, message: str) -> dict:
    return {
        "error": {
            "code": code,
            "entity": "records",
            "id": identifier,
            "field": field,
            "message": message,
        }
    }
'''


def _assertions() -> str:
    return '''"""Assertions for the generated starter pack."""

from __future__ import annotations

from typing import Any


def assert_record_reported(
    *, state: dict, trace: list, task: dict, ctx: Any
) -> None:
    del state, ctx
    expected = (task.get("slots") or {}).get("record_id")
    results = [
        item.get("result")
        for item in trace
        if item.get("tool") == "get_record"
    ]
    if not any(
        isinstance(result, dict) and result.get("record_id") == expected
        for result in results
    ):
        raise AssertionError(f"record {expected!r} was not returned")


def assert_record_not_found(
    *, state: dict, trace: list, task: dict, ctx: Any
) -> None:
    del state, task, ctx
    if not any(
        isinstance(item.get("result"), dict)
        and (item["result"].get("error") or {}).get("code") == "not_found"
        for item in trace
    ):
        raise AssertionError("expected a structured not_found result")


def assert_no_tool_called(
    *, state: dict, trace: list, task: dict, ctx: Any
) -> None:
    del state, task, ctx
    if trace:
        raise AssertionError("irrelevant task must not call a tool")


ASSERTIONS = {
    "assert_record_reported": assert_record_reported,
    "assert_record_not_found": assert_record_not_found,
    "assert_no_tool_called": assert_no_tool_called,
}

ASSERTION_CAPABILITIES = {
    "assert_record_reported": {
        "trace": True,
        "executable": True,
        "category": "result",
    },
    "assert_record_not_found": {
        "trace": True,
        "executable": True,
        "category": "result",
    },
    "assert_no_tool_called": {
        "trace": True,
        "executable": True,
        "category": "path",
    },
}
'''


def _validation_cases() -> list[dict]:
    return [
        {
            "id": "success_get_record",
            "tool": "get_record",
            "arguments": {"record_id": "REC-001"},
            "expect": {"result_class": "success", "error_code": None},
            "reset_before": True,
        },
        {
            "id": "missing_get_record",
            "tool": "get_record",
            "arguments": {"record_id": "REC-ABSENT-1"},
            "expect": {
                "result_class": "structured_error",
                "error_code": "not_found",
            },
            "reset_before": True,
        },
        {
            "id": "invalid_get_record",
            "tool": "get_record",
            "arguments": {"record_id": 7},
            "expect": {
                "result_class": "structured_error",
                "error_code": "invalid_argument",
            },
            "reset_before": True,
        },
    ]


def _endpoint(pack_id: str, version: str) -> dict:
    return {
        "protocol_version": "bfcl-oracle-http-v1",
        "base_url": "https://oracle.example.invalid",
        "auth": {"bearer_token_env": "BFCL_ORACLE_TOKEN"},
        "expected": {
            "oracle_id": pack_id,
            "oracle_version": version,
            "content_digest": _ZERO_DIGEST,
        },
        "max_request_bytes": 1048576,
        "max_response_bytes": 1048576,
    }


def _config(target: Path, pack_id: str) -> dict:
    return {
        "family": "bfcl",
        "expt_name": f"bfcl_{pack_id}_starter",
        "stage": "prepare",
        "random_seed": 7,
        "config_status": "resolved",
        "output_dir": str(target.parent / f"{target.name}_output"),
        "oracle_pack": {"manifest_path": str(target / "manifest.yaml")},
        "oracle_runtime": {
            "clock": "2026-01-01T00:00:00+00:00",
            "tool_timeout_s": 5.0,
            "assertion_timeout_s": 5.0,
            "import_timeout_s": 10.0,
            "reset_timeout_s": 5.0,
            "episode_timeout_s": 60.0,
            "worker": "process",
            "allowed_roots": [str(target)],
        },
        "lineage": {
            "policy": "smoke_no_publication",
            "profile_influenced_surface": False,
            "judge_advisory": None,
            "roles": {
                "profile": {"enabled": False, "model_config": None},
                "paraphrase": {"enabled": False, "model_config": None},
                "surface_judge": {"enabled": False, "model_config": None},
            },
        },
        "surface_generation": {
            "model_paraphrase_enabled": False,
            "paraphrases_per_template": 0,
            "preserve_slot_values": True,
            "prevent_tool_name_leakage": True,
        },
        "surface_quality_validation": {
            "enabled": False,
            "drop_authority": False,
        },
        "task_generation": {"tasks_per_category": 3},
        "semantic_deduplication_config": {"enabled": False},
        "exports": {"bfcl_json": False, "nemo_evaluator_bundle": False},
    }


def _readme(
    pack_id: str,
    transport: Literal["python", "endpoint"],
    include_held_out: bool,
) -> str:
    oracle_file = "backend.py" if transport == "python" else "endpoint_config.yaml"
    endpoint_note = (
        ""
        if transport == "python"
        else "\nBefore validation, replace the endpoint URL and identity pins with "
        "reviewed values from `GET /v1/metadata`, configure the referenced credential "
        "environment variables, and pin conformance evidence from `GET /v1/conformance`. "
        "The placeholder endpoint cannot reach Gold.\n"
    )
    held_out_note = (
        "\n`held_out.yaml` reserves `REC-HELD-OUT-1` and removes it from runtime "
        "state. Replace this example with reviewed reservations.\n"
        if include_held_out
        else ""
    )
    reference_lines = "\n".join(
        f"- `docs/build-benchmarks/function-calling/reference/{slug}.md`" for slug in OPERATOR_REFERENCE_SLUGS
    )
    return f"""# {pack_id} BFCL Oracle pack

This runnable starter deliberately uses a small `get_record` domain. Replace its
business names and values while preserving the contracts demonstrated here.

## Files

- `manifest.yaml`: pack identity, language, deterministic clock, and file map.
- `tools.json`: model-facing JSON schemas; executable names and arguments are truth.
- `fixtures.json`: deterministic reset state and slot inventory.
- `task_templates.yaml`: success, structured-error, and irrelevant task examples.
- `{oracle_file}`: the selected Oracle transport.
- `assertions.py`: trace/executable success conditions and capability declarations.
- `validation_cases.yaml`: direct success and business-error probes.
- `validate.yaml`: standalone validation and smoke-generation configuration.
{endpoint_note}{held_out_note}
## Fill order and contracts

Work in this order: public tool schemas, executable oracle, fixtures, manifest,
conversation templates, assertions, then validation cases. For a Python oracle,
`backend.py` must expose these four functions:

```python
def list_tools() -> list[str]: ...
def reset(*, ctx, fixtures=None) -> None: ...
def call_tool(name: str, arguments: dict, *, ctx) -> dict: ...
def get_state() -> dict: ...
```

`list_tools()` must match the public names in `tools.json`. Tool calls return
JSON-serializable values; expected business errors use
`{{"error": {{"code": "..."}}}}` rather than raising. Keep reset and state
deterministic and return defensive copies.

When working from a Nemotron source checkout, the operator-facing references are:

{reference_lines}

These source-relative paths are not bundled in the Python wheel. For an installed
package, use the hosted documentation matching that package version.

## Validate and generate

Run from this pack directory:

```bash
PACK="$(pwd)"

python -m nemotron.steps.byob.scripts.validate_oracle_pack \\
  --config "$PACK/validate.yaml" \\
  --output-dir "/tmp/bfcl-{pack_id}-validation"

python -m nemotron.steps.byob.scripts.run \\
  --config "$PACK/validate.yaml" --stage generate
```

Inspect
`/tmp/bfcl-{pack_id}-validation/bfcl_{pack_id}_starter/stage_cache/oracle_validation_report.json`
before generation.
Do not publish until every TODO value and, for endpoints, every identity pin is
replaced with reviewed domain evidence.
"""


def scaffold_oracle_pack(
    target: Path,
    *,
    domain: str,
    version: str = "0.1.0",
    language: str = "en",
    transport: Literal["python", "endpoint"] = "python",
    include_held_out: bool = False,
) -> Path:
    """Atomically create a runnable starter without overwriting existing work."""
    pack_id = _pack_id(domain)
    target = target.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}-{uuid.uuid4().hex}.tmp"
    try:
        temporary.mkdir()
        manifest = _manifest(
            pack_id=pack_id,
            version=version,
            language=language,
            transport=transport,
            include_held_out=include_held_out,
        )
        (temporary / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False),
            encoding="utf-8",
        )
        (temporary / "tools.json").write_text(
            json.dumps(_tools(), indent=2) + "\n",
            encoding="utf-8",
        )
        (temporary / "fixtures.json").write_text(
            json.dumps(_fixtures(include_held_out), indent=2) + "\n",
            encoding="utf-8",
        )
        (temporary / "task_templates.yaml").write_text(
            yaml.safe_dump(_templates(language), sort_keys=False),
            encoding="utf-8",
        )
        (temporary / "assertions.py").write_text(
            _assertions(),
            encoding="utf-8",
        )
        (temporary / "validation_cases.yaml").write_text(
            yaml.safe_dump(_validation_cases(), sort_keys=False),
            encoding="utf-8",
        )
        if transport == "python":
            (temporary / "backend.py").write_text(_backend(), encoding="utf-8")
        else:
            (temporary / "endpoint_config.yaml").write_text(
                yaml.safe_dump(_endpoint(pack_id, version), sort_keys=False),
                encoding="utf-8",
            )
        if include_held_out:
            (temporary / "held_out.yaml").write_text(
                yaml.safe_dump(
                    {
                        "version": "1",
                        "fixtures": {"records": ["REC-HELD-OUT-1"]},
                        "templates": [],
                        "policy": {
                            "fixtures_in_backend_state": False,
                            "seed": 7,
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
        (temporary / "README.md").write_text(
            _readme(pack_id, transport, include_held_out),
            encoding="utf-8",
        )
        (temporary / "validate.yaml").write_text(
            yaml.safe_dump(_config(target, pack_id), sort_keys=False),
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--transport",
        choices=("python", "endpoint"),
        default="python",
    )
    parser.add_argument("--include-held-out", action="store_true")
    args = parser.parse_args()
    try:
        target = scaffold_oracle_pack(
            args.target,
            domain=args.domain,
            version=args.version,
            language=args.language,
            transport=args.transport,
            include_held_out=args.include_held_out,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(target)


if __name__ == "__main__":
    main()
