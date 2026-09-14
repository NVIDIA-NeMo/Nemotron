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

"""Deterministic MCP ``tools/list`` to BFCL tool normalization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from nemotron.steps.byob.runtime.benchmark_families.bfcl.json_schema import (
    validate_tool_definition,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.mcp.config import McpOracleConfig
from nemotron.steps.byob.runtime.mcp.errors import (
    McpCatalogError,
    McpNormalizationError,
)

# The rules for third-party text are shared with the authoring side, which cannot import
# this package, so they live in a module neither environment owns.
from nemotron.steps.byob.runtime.pack_authoring.untrusted_text import (
    EMBEDDED_URL,
    ENGLISH_INJECTION_LEXICON,
    MAX_DESCRIPTION_CHARS,
    SMUGGLED_BLOCK,
    invisible_characters,
)


@dataclass(frozen=True)
class NormalizationIssue:
    source_name: str
    code: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {
            "source_name": self.source_name,
            "code": self.code,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class NormalizedTool:
    source_name: str
    published_name: str
    definition: dict[str, Any]
    output_schema: dict[str, Any] | None
    annotations: dict[str, Any] | None
    raw_digest: str
    # "config", "server_annotation", or None; lets review see whether a mutation flag
    # came from the reviewed pack or from the server's own claim about itself.
    mutation_source: str | None

    def evidence(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "published_name": self.published_name,
            "output_schema": deepcopy(self.output_schema),
            "annotations": deepcopy(self.annotations),
            "mutation_source": self.mutation_source,
            "raw_digest": self.raw_digest,
        }


@dataclass(frozen=True)
class NormalizedCatalog:
    tools: tuple[NormalizedTool, ...]
    exclusions: tuple[NormalizationIssue, ...]
    warnings: tuple[NormalizationIssue, ...]

    @property
    def bfcl_tools(self) -> list[dict[str, Any]]:
        return [deepcopy(tool.definition) for tool in self.tools]

    @property
    def source_to_published(self) -> dict[str, str]:
        return {tool.source_name: tool.published_name for tool in self.tools}


def _json_copy(value: Any, *, label: str) -> Any:
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise McpNormalizationError(f"{label} is not strict JSON: {exc}") from exc


def _as_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise McpNormalizationError(f"{label} must be a JSON object")
    copied = _json_copy(dict(value), label=label)
    if not isinstance(copied, dict):
        raise McpNormalizationError(f"{label} did not preserve its object shape")
    return copied


def _tool_mapping(tool: Any, index: int) -> dict[str, Any]:
    if isinstance(tool, Mapping):
        return _as_mapping(tool, label=f"tools[{index}]")
    model_dump = getattr(tool, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json", by_alias=True, exclude_none=True)
        return _as_mapping(dumped, label=f"tools[{index}]")
    raise McpNormalizationError(
        f"tools[{index}] is neither an MCP tool model nor a JSON mapping"
    )


def _strip_mcp_header(value: Any, path: str, removed: list[str]) -> Any:
    if isinstance(value, list):
        return [
            _strip_mcp_header(child, f"{path}[{index}]", removed)
            for index, child in enumerate(value)
        ]
    if not isinstance(value, dict):
        return value
    cleaned: dict[str, Any] = {}
    for key, child in value.items():
        if key == "x-mcp-header":
            removed.append(path or "$")
            continue
        cleaned[key] = _strip_mcp_header(child, f"{path}.{key}", removed)
    return cleaned


def _normalize_parameters(
    source_name: str,
    raw_parameters: Any,
    config: McpOracleConfig,
) -> tuple[dict[str, Any], list[NormalizationIssue]]:
    parameters = _as_mapping(
        raw_parameters,
        label=f"tool {source_name!r} inputSchema",
    )
    removed_headers: list[str] = []
    parameters = _strip_mcp_header(parameters, "$", removed_headers)
    warnings = [
        NormalizationIssue(
            source_name,
            "ignored_mcp_header",
            f"ignored untrusted x-mcp-header annotation at {path}",
        )
        for path in removed_headers
    ]
    if parameters.get("type") != "object":
        raise McpNormalizationError(
            f"selected tool {source_name!r} inputSchema must explicitly declare type=object"
        )
    parameters.setdefault("properties", {})
    if not isinstance(parameters["properties"], dict):
        raise McpNormalizationError(
            f"selected tool {source_name!r} inputSchema.properties must be an object"
        )
    parameters.setdefault("required", [])
    if not isinstance(parameters["required"], list) or not all(
        isinstance(name, str) for name in parameters["required"]
    ):
        raise McpNormalizationError(
            f"selected tool {source_name!r} inputSchema.required must be a string array"
        )
    if len(parameters["required"]) != len(set(parameters["required"])):
        raise McpNormalizationError(
            f"selected tool {source_name!r} inputSchema.required contains duplicates"
        )
    if config.control.episode_binding == "argument":
        argument = config.control.episode_argument
        assert argument is not None
        if argument not in parameters["properties"]:
            raise McpNormalizationError(
                f"selected tool {source_name!r} does not declare configured "
                f"episode_argument {argument!r}"
            )
        episode_schema = parameters["properties"][argument]
        if not isinstance(episode_schema, dict) or episode_schema.get("type") != "string":
            raise McpNormalizationError(
                f"selected tool {source_name!r} episode_argument {argument!r} "
                "must explicitly declare type=string"
            )
        parameters["properties"].pop(argument, None)
        parameters["required"] = [
            name for name in parameters["required"] if name != argument
        ]
    parameters["required"] = sorted(set(parameters["required"]))
    return parameters, warnings


def _normalize_one(
    raw: dict[str, Any],
    config: McpOracleConfig,
) -> tuple[NormalizedTool, tuple[NormalizationIssue, ...]]:
    source_name = raw.get("name")
    if not isinstance(source_name, str) or not source_name.strip():
        raise McpNormalizationError("selected MCP tool has no non-empty name")
    source_name = source_name.strip()
    published_name = config.tools.published_name(source_name)
    description = raw.get("description", "")
    if description is None:
        description = ""
    if not isinstance(description, str):
        raise McpNormalizationError(
            f"selected tool {source_name!r} description must be a string"
        )
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise McpNormalizationError(
            f"selected tool {source_name!r} description exceeds "
            f"{MAX_DESCRIPTION_CHARS} characters"
        )
    invisible = invisible_characters(description)
    if invisible:
        # Fail closed rather than warn: the operator reviewing this description cannot
        # see these code points, so a warning would ask for review that cannot happen.
        raise McpNormalizationError(
            f"selected tool {source_name!r} description contains invisible or "
            f"direction-overriding characters {invisible}, which defeat human review"
        )
    parameters, warning_list = _normalize_parameters(
        source_name,
        raw.get("inputSchema", raw.get("input_schema")),
        config,
    )
    for code, detail, pattern in (
        (
            "suspicious_description",
            "description contains instruction-like language in the English heuristic",
            ENGLISH_INJECTION_LEXICON,
        ),
        (
            "description_embeds_block",
            "description embeds a fenced block or HTML comment",
            SMUGGLED_BLOCK,
        ),
        ("description_embeds_url", "description embeds a URL", EMBEDDED_URL),
    ):
        if description and pattern.search(description):
            warning_list.append(NormalizationIssue(source_name, code, detail))
    definition: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": published_name,
            "description": description,
            "parameters": parameters,
        },
    }
    annotations_raw = raw.get("annotations")
    annotations = (
        _as_mapping(annotations_raw, label=f"tool {source_name!r} annotations")
        if annotations_raw is not None
        else None
    )
    declared_mutating = published_name in config.tools.mutates
    read_only_hint = annotations.get("readOnlyHint") if annotations is not None else None
    mutates = declared_mutating
    mutation_source = "config" if declared_mutating else None
    if config.tools.trust_annotations and read_only_hint is False and not declared_mutating:
        mutates = True
        mutation_source = "server_annotation"
    # Either direction of disagreement means one of the two sides is wrong about the
    # mutation surface that check M1 is built on, so surface it instead of silently
    # letting the reviewed config win.
    if declared_mutating and read_only_hint is True:
        warning_list.append(
            NormalizationIssue(
                source_name,
                "mutation_disagreement",
                "reviewed config declares this tool mutating while the server "
                "annotates it read-only",
            )
        )
    elif not mutates and read_only_hint is False:
        warning_list.append(
            NormalizationIssue(
                source_name,
                "undeclared_mutation_hint",
                "the server annotates this tool as mutating while reviewed config "
                "omits it from tools.mutates",
            )
        )
    if mutates:
        definition["x-mutates"] = True
    if published_name in config.tools.requires_confirmation:
        parameter_name = config.results.confirmation_parameter
        confirmation_schema = parameters["properties"].get(parameter_name)
        if not isinstance(confirmation_schema, dict) or confirmation_schema.get("type") != "boolean":
            raise McpNormalizationError(
                f"confirmation-gated tool {source_name!r} must declare a boolean "
                f"{parameter_name!r} input"
            )
        confirmation_enum = confirmation_schema.get("enum")
        if "const" in confirmation_schema or (
            confirmation_enum is not None
            and (
                not isinstance(confirmation_enum, list)
                or len(confirmation_enum) != 2
                or not any(value is False for value in confirmation_enum)
                or not any(value is True for value in confirmation_enum)
            )
        ):
            raise McpNormalizationError(
                f"confirmation-gated tool {source_name!r} must allow both false and true "
                f"for {parameter_name!r}"
            )
        definition["x-requires-confirmation"] = True
    failures = validate_tool_definition(definition)
    if failures:
        raise McpNormalizationError(
            f"selected tool {source_name!r} is outside BFCL's JSON Schema subset: "
            f"{canonical_json(failures)}"
        )
    output_raw = raw.get("outputSchema", raw.get("output_schema"))
    output_schema = (
        _as_mapping(output_raw, label=f"tool {source_name!r} outputSchema")
        if output_raw is not None
        else None
    )
    return (
        NormalizedTool(
            source_name=source_name,
            published_name=published_name,
            definition=_json_copy(definition, label=f"tool {source_name!r} definition"),
            output_schema=output_schema,
            annotations=annotations,
            mutation_source=mutation_source,
            raw_digest="sha256:"
            + hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest(),
        ),
        tuple(warning_list),
    )


def normalize_catalog(
    tools: Sequence[Any],
    config: McpOracleConfig,
) -> NormalizedCatalog:
    """Normalize the complete paginated catalog and fail closed on selected tools."""
    raw_tools = [_tool_mapping(tool, index) for index, tool in enumerate(tools)]
    by_name: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_tools):
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise McpCatalogError(f"tools[{index}] has no non-empty name")
        if name != name.strip():
            raise McpCatalogError(
                f"tools[{index}] name {name!r} contains surrounding whitespace"
            )
        if name in by_name:
            raise McpCatalogError(f"tools/list returned duplicate tool name {name!r}")
        by_name[name] = raw
    missing = sorted(set(config.tools.include) - set(by_name))
    if missing:
        raise McpCatalogError(f"selected MCP tools are absent from the catalog: {missing}")

    normalized: list[NormalizedTool] = []
    warnings: list[NormalizationIssue] = []
    for source_name in config.tools.include:
        try:
            tool, tool_warnings = _normalize_one(by_name[source_name], config)
        except McpNormalizationError as exc:
            raise McpNormalizationError(
                f"cannot publish selected MCP tool {source_name!r}: {exc}"
            ) from exc
        normalized.append(tool)
        warnings.extend(tool_warnings)

    normalized.sort(key=lambda item: item.published_name)
    published = [tool.published_name for tool in normalized]
    if len(published) != len(set(published)):
        raise McpCatalogError("normalized catalog contains duplicate published names")
    exclusions = [
        NormalizationIssue(name, "not_selected", "tool is outside tools.include")
        for name in sorted(set(by_name) - set(config.tools.include))
    ]
    return NormalizedCatalog(
        tools=tuple(normalized),
        exclusions=tuple(exclusions),
        warnings=tuple(
            sorted(
                warnings,
                key=lambda item: (item.source_name, item.code, item.detail),
            )
        ),
    )
