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

"""The one JSON-Schema subset this pipeline implements, for both sides of it.

Generation certifies a gold call against a tool's declared parameters, and
evaluation certifies a candidate call against the same declaration. Two engines
would eventually disagree, and the disagreement would be invisible: a benchmark
would ship calls its own scorer marks invalid, or a scorer would reject arguments
generation had already accepted. So the subset lives here once, and both callers
import it.

The subset is deliberately narrow. Keywords are allowlisted rather than ignored,
because silently accepting a constraint that is never enforced certifies a call
nobody checked. Comparisons never let ``True`` equal ``1``: a benchmark whose
scorer conflates the two reports wrong answers the model never gave.

Default insertion is here for the same reason. Filling a declared default on
whichever side omitted it is what makes spelling out a default neither an
advantage nor a penalty, and generation and evaluation have to fill it the same
way or a published call would stop matching itself.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Final
from urllib.parse import unquote

from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import thaw_json

_JSON_TYPES: Final[dict[str, tuple[type, ...]]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}
_SCHEMA_TYPES: Final = {*_JSON_TYPES, "null"}
# Branching schemas still need a larger comparison model. Local definitions,
# references, and conjunctions are safe because every referenced constraint is
# resolved and every ``allOf`` branch is enforced by this module.
UNSUPPORTED_SCHEMA_KEYWORDS: Final = {"anyOf", "not", "oneOf"}
SUPPORTED_SCHEMA_KEYWORDS: Final = {
    "$defs",
    "$ref",
    "definitions",
    "allOf",
    "type",
    "description",
    "title",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
    # JSON Schema treats `format` as an annotation rather than an assertion, so a tool
    # may declare it without this module having to enforce it.
    "format",
    "enum",
    "const",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "pattern",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
}


def same_json_value(left: Any, right: Any) -> bool:
    """Compare two JSON values without letting ``True`` equal ``1``."""
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    return left == right


def _duplicates(values: list[Any]) -> list[Any]:
    """Return the values that appear more than once, in declaration order."""
    repeated: list[Any] = []
    for index, value in enumerate(values):
        if any(same_json_value(value, earlier) for earlier in values[:index]) and not any(
            same_json_value(value, seen) for seen in repeated
        ):
            repeated.append(value)
    return repeated


def _type_matches(value: Any, declared: str) -> bool:
    if declared == "null":
        return value is None
    allowed = _JSON_TYPES.get(declared)
    if allowed is None:
        return False
    if declared in {"integer", "number"} and isinstance(value, bool):
        return False
    return isinstance(value, allowed)


def validate_function_schema(function: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the parameter-schema subset implemented by this pipeline."""
    failures: list[dict[str, Any]] = []
    parameters = function.get("parameters")
    if not isinstance(parameters, dict):
        return [{"reason": "parameters_not_object"}]
    if parameters.get("type", "object") != "object":
        failures.append({"reason": "parameters_type_not_object"})

    def visit(schema: dict[str, Any], path: str, ref_stack: tuple[str, ...] = ()) -> None:
        for keyword in sorted(UNSUPPORTED_SCHEMA_KEYWORDS & schema.keys()):
            failures.append({"reason": "unsupported_schema_keyword", "path": path, "keyword": keyword})
        for keyword in sorted(set(schema) - SUPPORTED_SCHEMA_KEYWORDS - UNSUPPORTED_SCHEMA_KEYWORDS):
            # Silently accepting a constraint we do not enforce can certify an invalid
            # expected call. Annotation-only keys are explicitly allowlisted above.
            failures.append({"reason": "unsupported_schema_keyword", "path": path, "keyword": keyword})
        declared = schema.get("type")
        types = [declared] if isinstance(declared, str) else declared
        type_is_valid = declared is None or (
            isinstance(types, list)
            and bool(types)
            and all(isinstance(item, str) and item in _SCHEMA_TYPES for item in types)
        )
        if not type_is_valid:
            failures.append({"reason": "invalid_schema_type", "path": path, "value": declared})
        elif isinstance(types, list) and _duplicates(types):
            # A repeated member of a union says the author meant to allow something else.
            failures.append({"reason": "duplicate_schema_type", "path": path, "values": _duplicates(types)})

        def check_allowed(keyword: str, values: list[Any]) -> None:
            """Refuse fixed values that the declared type can never accept."""
            if declared is None or not type_is_valid or not isinstance(types, list):
                return
            for value in values:
                if not any(_type_matches(value, item) for item in types):
                    failures.append(
                        {
                            "reason": "schema_value_violates_type",
                            "path": path,
                            "keyword": keyword,
                            "value": value,
                        }
                    )

        if "enum" in schema:
            enum = schema["enum"]
            if not isinstance(enum, list):
                failures.append({"reason": "schema_enum_not_array", "path": path})
            elif not enum:
                # Nothing can satisfy an empty enum, so every bound argument would fail.
                failures.append({"reason": "schema_enum_empty", "path": path})
            else:
                repeated = _duplicates(enum)
                if repeated:
                    failures.append({"reason": "duplicate_schema_enum_value", "path": path, "values": repeated})
                check_allowed("enum", enum)
        if "const" in schema:
            check_allowed("const", [schema["const"]])
        if "default" in schema:
            for problem in _check_value(schema, schema["default"], path, root=parameters):
                failures.append(
                    {
                        "reason": "invalid_schema_default",
                        "path": path,
                        "failure": problem["reason"],
                    }
                )
        if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
            failures.append({"reason": "schema_additional_properties_not_boolean", "path": path})
        for keyword in ("minLength", "maxLength", "minItems", "maxItems"):
            value = schema.get(keyword)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                failures.append({"reason": "invalid_schema_bound", "path": path, "keyword": keyword})
        for keyword in ("minimum", "maximum"):
            value = schema.get(keyword)
            if value is not None and (not isinstance(value, int | float) or isinstance(value, bool)):
                failures.append({"reason": "invalid_schema_bound", "path": path, "keyword": keyword})
        if "pattern" in schema:
            if not isinstance(schema["pattern"], str):
                failures.append({"reason": "schema_pattern_not_string", "path": path})
            else:
                try:
                    re.compile(schema["pattern"])
                except re.error:
                    failures.append({"reason": "invalid_schema_pattern", "path": path})

        comparable_bounds = (
            ("minLength", "maxLength"),
            ("minItems", "maxItems"),
            ("minimum", "maximum"),
        )
        for lower_name, upper_name in comparable_bounds:
            lower = schema.get(lower_name)
            upper = schema.get(upper_name)
            if (
                isinstance(lower, int | float)
                and not isinstance(lower, bool)
                and isinstance(upper, int | float)
                and not isinstance(upper, bool)
                and lower > upper
            ):
                failures.append(
                    {
                        "reason": "inconsistent_schema_bounds",
                        "path": path,
                        "lower": lower_name,
                        "upper": upper_name,
                    }
                )

        reference = schema.get("$ref")
        if reference is not None:
            if not isinstance(reference, str) or not reference.startswith("#/"):
                failures.append({"reason": "schema_ref_not_local", "path": path, "value": reference})
            elif reference in ref_stack:
                failures.append({"reason": "cyclic_schema_ref", "path": path, "reference": reference})
            else:
                target = _resolve_local_ref(reference, parameters)
                if target is None:
                    failures.append({"reason": "unresolvable_schema_ref", "path": path, "reference": reference})
                else:
                    visit(dict(target), f"{path}->$ref", (*ref_stack, reference))

        all_of = schema.get("allOf")
        if all_of is not None:
            if not isinstance(all_of, list) or not all_of:
                failures.append({"reason": "schema_all_of_not_nonempty_array", "path": path})
            else:
                for index, branch in enumerate(all_of):
                    if not isinstance(branch, dict):
                        failures.append({"reason": "all_of_schema_not_object", "path": f"{path}.allOf[{index}]"})
                    else:
                        visit(branch, f"{path}.allOf[{index}]", ref_stack)

        for definitions_keyword in ("$defs", "definitions"):
            definitions = schema.get(definitions_keyword)
            if definitions is None:
                continue
            if not isinstance(definitions, dict):
                failures.append(
                    {"reason": "schema_definitions_not_object", "path": path, "keyword": definitions_keyword}
                )
                continue
            for name, definition in definitions.items():
                definition_path = f"{path}.{definitions_keyword}.{name}"
                if not isinstance(definition, dict):
                    failures.append({"reason": "definition_schema_not_object", "path": definition_path})
                else:
                    visit(definition, definition_path, ref_stack)

        properties = schema.get("properties")
        if properties is not None and not isinstance(properties, dict):
            failures.append({"reason": "schema_properties_not_object", "path": path})
            properties = {}
        properties = properties or {}
        required = schema.get("required") or []
        if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
            failures.append({"reason": "schema_required_not_string_array", "path": path})
        else:
            repeated_required = _duplicates(required)
            if repeated_required:
                failures.append({"reason": "duplicate_required_property", "path": path, "values": repeated_required})
            for name in required:
                if name not in properties:
                    failures.append({"reason": "required_property_not_declared", "path": path, "property": name})
        for name, child in properties.items():
            if not isinstance(child, dict):
                failures.append({"reason": "property_schema_not_object", "path": f"{path}.{name}"})
            else:
                visit(child, f"{path}.{name}", ref_stack)
        if "items" in schema:
            if not isinstance(schema["items"], dict):
                failures.append({"reason": "items_schema_not_object", "path": path})
            else:
                visit(schema["items"], f"{path}[]", ref_stack)

    visit(parameters, "$")
    return failures


def validate_tool_definition(tool: Any) -> list[dict[str, Any]]:
    """Validate the model-facing OpenAI function-tool envelope and its schema."""
    if not isinstance(tool, dict):
        return [{"reason": "tool_not_object"}]
    failures: list[dict[str, Any]] = []
    if tool.get("type", "function") != "function":
        failures.append({"reason": "tool_type_not_function", "value": tool.get("type")})
    for extension in ("x-mutates", "x-requires-confirmation"):
        if extension in tool and not isinstance(tool[extension], bool):
            failures.append(
                {
                    "reason": "tool_extension_not_boolean",
                    "field": extension,
                    "value": tool[extension],
                }
            )
    function = tool.get("function")
    if not isinstance(function, dict):
        return [*failures, {"reason": "tool_function_not_object"}]
    if not isinstance(function.get("name"), str) or not function["name"].strip():
        failures.append({"reason": "tool_name_not_nonempty_string"})
    if "description" in function and not isinstance(function["description"], str):
        failures.append({"reason": "tool_description_not_string"})
    if "strict" in function and not isinstance(function["strict"], bool):
        failures.append({"reason": "tool_strict_not_boolean"})
    failures.extend(validate_function_schema(function))
    return failures


def _check_value(
    schema: Mapping[str, Any],
    value: Any,
    path: str,
    *,
    root: Mapping[str, Any] | None = None,
    ref_stack: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Validate the JSON-Schema subset used by OpenAI function parameters.

    ``$ref`` and ``allOf`` are resolved rather than skipped. Skipping them would
    make every constraint behind a shared definition unenforced while default
    insertion still followed it, so the two halves of this module would disagree
    about what a schema says. A reference that cannot be resolved is a failure:
    treating it as "no constraints" would certify any value at all.
    """
    root = root if root is not None else schema
    failures: list[dict[str, Any]] = []
    reference = schema.get("$ref")
    if reference is not None:
        if not isinstance(reference, str) or reference in ref_stack:
            return [{"reason": "unresolvable_schema_ref", "argument": path}]
        resolved = _resolve_local_ref(reference, root)
        if resolved is None:
            return [{"reason": "unresolvable_schema_ref", "argument": path}]
        failures.extend(_check_value(resolved, value, path, root=root, ref_stack=(*ref_stack, reference)))
        schema = {key: child for key, child in schema.items() if key != "$ref"}
    for branch in schema.get("allOf", ()):
        if isinstance(branch, Mapping):
            failures.extend(_check_value(branch, value, path, root=root, ref_stack=ref_stack))
    declared = schema.get("type")
    declared_types = [declared] if isinstance(declared, str) else declared
    if isinstance(declared_types, list):
        known = [item for item in declared_types if isinstance(item, str)]
        if not known or not any(_type_matches(value, item) for item in known):
            failures.append(
                {
                    "reason": "argument_type_mismatch",
                    "argument": path,
                    "expected_type": declared,
                }
            )
            return failures
    elif declared is not None:
        failures.append({"reason": "invalid_schema_type", "argument": path, "value": declared})
        return failures

    enum = schema.get("enum")
    if isinstance(enum, list) and not any(same_json_value(value, candidate) for candidate in enum):
        failures.append({"reason": "argument_not_in_enum", "argument": path})
    if "const" in schema and not same_json_value(value, schema["const"]):
        failures.append({"reason": "argument_not_equal_const", "argument": path})

    if isinstance(value, str):
        if isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
            failures.append({"reason": "string_too_short", "argument": path})
        if isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
            failures.append({"reason": "string_too_long", "argument": path})
        if isinstance(schema.get("pattern"), str):
            try:
                matches = re.search(schema["pattern"], value) is not None
            except re.error:
                failures.append({"reason": "invalid_schema_pattern", "argument": path})
            else:
                if not matches:
                    failures.append({"reason": "string_pattern_mismatch", "argument": path})

    if isinstance(value, int | float) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, int | float) and not isinstance(minimum, bool) and value < minimum:
            failures.append({"reason": "number_below_minimum", "argument": path})
        if isinstance(maximum, int | float) and not isinstance(maximum, bool) and value > maximum:
            failures.append({"reason": "number_above_maximum", "argument": path})

    if isinstance(value, list):
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            failures.append({"reason": "array_too_short", "argument": path})
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            failures.append({"reason": "array_too_long", "argument": path})
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                failures.extend(_check_value(item_schema, item, f"{path}[{index}]", root=root, ref_stack=ref_stack))

    if isinstance(value, dict):
        raw_properties = schema.get("properties")
        properties = raw_properties if isinstance(raw_properties, Mapping) else {}
        raw_required = schema.get("required")
        required = (
            raw_required
            if isinstance(raw_required, list) and all(isinstance(name, str) for name in raw_required)
            else ()
        )
        for name in required:
            if name not in value:
                failures.append({"reason": "missing_required_argument", "argument": f"{path}.{name}"})
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    failures.append({"reason": "unknown_argument", "argument": f"{path}.{name}"})
        for name, child in value.items():
            child_schema = properties.get(name)
            if isinstance(child_schema, Mapping):
                failures.extend(_check_value(child_schema, child, f"{path}.{name}", root=root, ref_stack=ref_stack))
    return failures


def check_arguments(function: Mapping[str, Any], arguments: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate one argument object against a function's parameter schema.

    The declared parameters are the reference root, so a nested ``$ref`` resolves
    against the same document that default insertion resolves against.
    """
    parameters = function.get("parameters") or {}
    return _check_parameters(parameters, dict(arguments), root=parameters)


def _check_parameters(
    schema: Mapping[str, Any],
    arguments: Mapping[str, Any],
    *,
    root: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Check the top-level argument object, naming arguments as the caller wrote them.

    Separate from :func:`_check_value` only so a top-level failure reports
    ``limit`` rather than ``$.limit``: these names appear in generation's reject
    reasons and in a scorer's diagnostics, where the caller's own spelling is what
    a reader is looking for.
    """
    failures = _check_value(schema, dict(arguments), "$", root=root)
    normalized: list[dict[str, Any]] = []
    for failure in failures:
        argument = failure.get("argument")
        if isinstance(argument, str) and argument.startswith("$."):
            failure = {**failure, "argument": argument[2:]}
        normalized.append(failure)
    return normalized


def validate_function_arguments(function: dict[str, Any], arguments: Any) -> list[dict[str, Any]]:
    """Validate one declared call payload against a function's parameter schema."""
    if not isinstance(arguments, dict):
        return [{"reason": "arguments_not_object"}]
    return check_arguments(function, arguments)


def declared_function(tools: Any, function_name: str) -> dict[str, Any] | None:
    """Find the ``function`` object a tool list declares under ``function_name``."""
    for tool in tools or ():
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        if isinstance(function, Mapping) and function.get("name") == function_name:
            return dict(thaw_json(function))
    return None


def parameter_schema(tools: Any, function_name: str) -> dict[str, Any]:
    """The parameter schema declared for one tool, or an empty schema."""
    function = declared_function(tools, function_name)
    if function is None:
        return {}
    parameters = function.get("parameters")
    return dict(parameters) if isinstance(parameters, Mapping) else {}


def apply_declared_defaults(arguments: Mapping[str, Any], schema: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively fill parameters the schema gives a default and this side omits.

    Filling the omitting side — rather than both sides, which would be a no-op —
    is what makes spelling out a default neither an advantage nor a penalty.
    Local ``$ref`` and ``allOf`` are followed because a pack may share nested
    object definitions rather than spelling every parameter inline.
    """
    filled = _defaults_in_value(thaw_json(arguments), schema, root=schema)
    return dict(filled) if isinstance(filled, Mapping) else dict(arguments)


def _resolve_local_ref(reference: str, root: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if not reference.startswith("#/"):
        return None
    current: Any = root
    for token in reference[2:].split("/"):
        key = unquote(token).replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current if isinstance(current, Mapping) else None


_NO_DEFAULT: Final = object()


def _schema_default(
    schema: Mapping[str, Any],
    root: Mapping[str, Any],
    *,
    ref_stack: tuple[str, ...] = (),
) -> Any:
    if "default" in schema:
        return schema["default"]
    reference = schema.get("$ref")
    if isinstance(reference, str) and reference not in ref_stack:
        resolved = _resolve_local_ref(reference, root)
        if resolved is not None:
            inherited = _schema_default(resolved, root, ref_stack=(*ref_stack, reference))
            if inherited is not _NO_DEFAULT:
                return inherited
    for branch in schema.get("allOf", ()):
        if isinstance(branch, Mapping):
            inherited = _schema_default(branch, root, ref_stack=ref_stack)
            if inherited is not _NO_DEFAULT:
                return inherited
    return _NO_DEFAULT


def _defaults_in_value(
    value: Any,
    schema: Mapping[str, Any],
    *,
    root: Mapping[str, Any],
    ref_stack: tuple[str, ...] = (),
) -> Any:
    reference = schema.get("$ref")
    if isinstance(reference, str) and reference not in ref_stack:
        resolved = _resolve_local_ref(reference, root)
        if resolved is not None:
            value = _defaults_in_value(value, resolved, root=root, ref_stack=(*ref_stack, reference))
    for branch in schema.get("allOf", ()):
        if isinstance(branch, Mapping):
            value = _defaults_in_value(value, branch, root=root, ref_stack=ref_stack)

    properties = schema.get("properties")
    if isinstance(value, Mapping) and isinstance(properties, Mapping):
        filled = {str(name): thaw_json(child) for name, child in value.items()}
        for name, child_schema in properties.items():
            if not isinstance(child_schema, Mapping):
                continue
            key = str(name)
            default = _schema_default(child_schema, root)
            if key not in filled and default is not _NO_DEFAULT:
                filled[key] = thaw_json(default)
            if key in filled:
                filled[key] = _defaults_in_value(filled[key], child_schema, root=root, ref_stack=ref_stack)
        return filled

    items = schema.get("items")
    if isinstance(value, list) and isinstance(items, Mapping):
        return [_defaults_in_value(item, items, root=root, ref_stack=ref_stack) for item in value]
    return thaw_json(value)


__all__ = [
    "SUPPORTED_SCHEMA_KEYWORDS",
    "UNSUPPORTED_SCHEMA_KEYWORDS",
    "apply_declared_defaults",
    "check_arguments",
    "declared_function",
    "parameter_schema",
    "same_json_value",
    "validate_function_arguments",
    "validate_function_schema",
    "validate_tool_definition",
]
