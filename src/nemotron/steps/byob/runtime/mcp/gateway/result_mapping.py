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

"""Total MCP ``tools/call`` result mapping into BFCL result objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jsonschema import SchemaError, ValidationError
from jsonschema.validators import validator_for

from nemotron.steps.byob.runtime.mcp.config import ResultsConfig
from nemotron.steps.byob.runtime.mcp.gateway.errors import GatewayError, upstream_failure

_MISSING = object()
# ``CallToolResult.resultType`` is an open string in the specification: only these two
# values are defined, and any other one names an extension whose semantics the gateway
# has not reviewed.
_COMPLETE_RESULT_TYPE = "complete"
_INPUT_REQUIRED_RESULT_TYPE = "input_required"
# A task-augmented call answers with ``CreateTaskResult``, which carries a handle
# instead of ``structuredContent``.
_TASK_HANDLE_KEYS = ("task", "taskId", "task_id")


def _field(result: Mapping[str, Any], camel: str, snake: str) -> Any:
    if camel in result:
        return result[camel]
    return result.get(snake)


def _reject_unsupported_result(result: Mapping[str, Any], *, operation: str) -> None:
    """Refuse every result shape gateway v1 cannot map to reviewed oracle truth.

    Anything unrecognized is refused rather than read for a ``structuredContent`` that
    would then be published as a business outcome the server never asserted.
    """
    if any(key in result for key in _TASK_HANDLE_KEYS):
        raise upstream_failure(
            "mcp_async_task_unsupported",
            f"MCP {operation} returned a task handle, which gateway v1 does not drive",
        )
    result_type = _field(result, "resultType", "result_type")
    if result_type is None or result_type == _COMPLETE_RESULT_TYPE:
        return
    if result_type == _INPUT_REQUIRED_RESULT_TYPE:
        raise upstream_failure(
            "mcp_input_required_unsupported",
            f"MCP {operation} requested further input, which gateway v1 does not drive",
        )
    raise upstream_failure(
        "mcp_unsupported_result_type",
        f"MCP {operation} declared an unreviewed resultType extension",
    )


def _dotted_value(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for segment in path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            return _MISSING
        current = current[segment]
    return current


def _validate_output_schema(
    structured: dict[str, Any],
    output_schema: dict[str, Any] | None,
) -> None:
    if output_schema is None:
        return
    try:
        validator_type = validator_for(output_schema)
        validator_type.check_schema(output_schema)
        validator_type(output_schema).validate(structured)
    except (SchemaError, ValidationError) as exc:
        raise upstream_failure(
            "mcp_output_schema_mismatch",
            "MCP structuredContent does not satisfy the declared outputSchema",
        ) from exc


def map_call_result(
    result: Mapping[str, Any],
    *,
    config: ResultsConfig,
    output_schema: dict[str, Any] | None,
) -> dict[str, Any]:
    """Map one synchronous MCP result according to the contract's §8 precedence."""
    _reject_unsupported_result(result, operation="tools/call")

    is_error = _field(result, "isError", "is_error")
    if is_error is None:
        is_error = False
    if not isinstance(is_error, bool):
        raise upstream_failure(
            "mcp_protocol_error",
            "MCP tools/call isError must be boolean when present",
        )

    structured = _field(result, "structuredContent", "structured_content")
    if not isinstance(structured, Mapping):
        raise upstream_failure(
            "mcp_result_not_object",
            "MCP tools/call must return structuredContent as a JSON object",
        )
    structured_object = dict(structured)
    _validate_output_schema(structured_object, output_schema)

    error_value = _dotted_value(structured_object, config.error_path)
    if is_error:
        if not isinstance(error_value, Mapping):
            raise upstream_failure(
                "mcp_unstructured_error",
                "MCP tools/call returned isError=true without a structured error object",
            )
        code = error_value.get("code")
        if not isinstance(code, str) or not code.strip():
            raise upstream_failure(
                "mcp_unstructured_error",
                "MCP structured error must contain a non-empty string code",
            )
        return {"error": dict(error_value)}

    if error_value is not _MISSING:
        raise upstream_failure(
            "mcp_error_flag_inconsistent",
            "MCP result contains an error object while isError is false",
        )

    # Pending confirmation and ordinary success both preserve the complete structured
    # object. BFCL classifies pending by its configured status vocabulary.
    return structured_object


def control_result_object(
    result: Mapping[str, Any],
    *,
    operation: str,
) -> dict[str, Any]:
    """Read a control result without ever parsing free-text content."""
    _reject_unsupported_result(result, operation=f"control operation {operation!r}")
    is_error = _field(result, "isError", "is_error")
    if is_error is True:
        raise upstream_failure(
            "mcp_control_failed",
            f"MCP control operation {operation!r} returned isError=true",
        )
    if is_error not in (None, False):
        raise upstream_failure(
            "mcp_protocol_error",
            f"MCP control operation {operation!r} returned non-boolean isError",
        )
    structured = _field(result, "structuredContent", "structured_content")
    if not isinstance(structured, Mapping):
        raise upstream_failure(
            "mcp_result_not_object",
            f"MCP control operation {operation!r} must return object structuredContent",
        )
    return dict(structured)


def ensure_gateway_error(exc: Exception, *, operation: str) -> GatewayError:
    if isinstance(exc, GatewayError):
        return exc
    return upstream_failure(
        "mcp_call_failed",
        f"MCP {operation} failed: {type(exc).__name__}",
    )
