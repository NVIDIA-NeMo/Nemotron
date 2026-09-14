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

"""Versioned contract for BFCL Stage 12 publication and compatibility exports.

An export is a re-encoding of the published benchmark, never a second source of
truth. Every writer therefore reads one canonical projection of the parquet row,
so no format can invent a tool, regroup a call, or drop an assertion on its way
to disk.

The models here are strict on purpose. Pydantic's default coercion accepts ``"1"``
for an int and ``1`` for a bool, which is the exact drift an export contract
exists to prevent: a scorer comparing ``{"limit": 1}`` against ``{"limit": "1"}``
reports a wrong answer the model never gave. Tool definitions and call arguments
are pack-declared JSON of arbitrary shape, so they stay opaque mappings — a typed
re-model would re-serialize them and lose whichever optional keys the pack chose
to omit — and they are compared with :func:`json_equal` rather than Python's
``==``, under which ``1 == True`` and ``1 == 1.0`` both hold.

The two compatibility formats carry their own ``schema_version`` because they
answer to consumers this pipeline does not own. ``bfcl_json`` pins the upstream
BFCL V4 multi-turn JSONL envelope (``question`` / ``function`` and the separate
``ground_truth`` record) and retains a Nemotron extension for assertions and
exact call grouping. It does not claim to provide BFCL's domain-specific
executable classes. ``nemo_evaluator_bundle`` is BFCL's own self-describing
layout — dataset file, scoring descriptor, evaluator configuration, prompt
catalog, and source lineage.
"""

from __future__ import annotations

import hashlib
import json
import keyword
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import (
    canonical_json,
    decode_arguments,
    decode_tools,
    encode_arguments,
)

EXPORT_CONTRACT_VERSION = "1.0"
BFCL_JSON_SCHEMA_VERSION = "1.0"
NEMO_EVALUATOR_SCHEMA_VERSION = "1.0"
BFCL_UPSTREAM_SCHEMA_VERSION = "BFCL_v4_multi_turn"

# The keys config accepts under ``exports``. Stage 12 writes one report entry per
# name whether or not it is enabled, so a manifest never leaves the reader unsure
# whether a format was skipped or silently failed.
EXPORT_FORMATS = ("bfcl_json", "nemo_evaluator_bundle")
# Every format writes under one run-relative directory, so the whole set can be
# discarded or renamed as a unit when publication becomes transactional.
EXPORT_DIRECTORY = "exports"
ExportFormatName = Literal["bfcl_json", "nemo_evaluator_bundle"]
EXPORT_FORMAT_SCHEMA_VERSIONS = {
    "bfcl_json": BFCL_JSON_SCHEMA_VERSION,
    "nemo_evaluator_bundle": NEMO_EVALUATOR_SCHEMA_VERSION,
}

# The fields a scorer reads. An export that alters any of them changes what the
# benchmark asks, so Stage 12 compares exactly these between the parquet
# projection and every format it wrote.
EXPORT_TRUTH_FIELDS = (
    "task_id",
    "messages",
    "tools",
    "expected_tool_calls",
    "success_assertions",
    "call_order",
    "call_order_prefix",
)

# What a function-calling scorer is expected to measure from one exported record.
# The names are published in the bundle descriptor rather than inferred by the
# harness, so a pack cannot be scored on a dimension it never declared.
EXPORT_SCORING_METRICS = (
    "tool_selection",
    "arguments",
    "call_ordering",
    "results",
    "task_success",
)
ExportScoringMetric = Literal[
    "tool_selection",
    "arguments",
    "call_ordering",
    "results",
    "task_success",
]

# Mirrors ``benchmark_schema()`` field order. Kept as plain names so this module
# stays importable without pyarrow; a test pins the two lists together.
BENCHMARK_ROW_FIELDS = (
    "task_id",
    "template_id",
    "variant_index",
    "messages",
    "tools",
    "expected_tool_calls",
    "success_assertions",
    "fixture_refs",
    "intent",
    "category",
    "difficulty",
    "required_tools",
    "required_tools_fingerprint",
    "tools_present",
    "turn_policy",
    "is_multi_turn",
    "num_tool_calls",
    "call_order",
    "call_order_prefix",
    "system_prompt_id",
    "tier",
    "gold_eligible",
    "validated_by",
    "pack_id",
    "pack_version",
    "seed",
    "paraphrase_model",
    "paraphrase_model_canonical",
    "held_out_hit",
    "src",
    "metadata",
)

# The row's ``metadata`` column is canonical JSON with exactly these keys. It stays
# an opaque mapping so re-serialization is byte-stable, and the key set is pinned
# so a new surface field cannot ride into an export unversioned.
CANONICAL_METADATA_KEYS = frozenset(
    {
        "language",
        "expt_name",
        "base_task_id",
        "surface_source",
        "profile_hash",
    }
)

CallOrderPolicy = Literal["strict", "any", "prefix"]
ValidationEvidence = Literal["schema", "replay", "assertions"]
MessageRole = Literal["system", "user", "assistant", "tool"]
ExportFailureReason = Literal[
    "missing_row",
    "unexpected_row",
    "duplicate_row",
    "row_order_changed",
    "truth_field_changed",
]

NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
PositiveInt = Annotated[StrictInt, Field(ge=1)]
ContentHash = Annotated[StrictStr, Field(pattern=r"^sha256:[0-9a-f]{64}$")]

JSON_TYPE_TAGS = frozenset({"null", "bool", "int", "float", "str", "object", "array"})
_JSON_SCALAR_TAGS = frozenset({"null", "bool", "int", "str"})


class FrozenDict(dict[str, Any]):
    """A JSON object that Pydantic can serialize but no exporter can mutate."""

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("canonical export JSON is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def freeze_json(value: Any) -> Any:
    """Recursively freeze JSON containers while preserving their JSON shape."""
    if isinstance(value, Mapping):
        return FrozenDict({str(key): freeze_json(child) for key, child in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray | memoryview):
        return tuple(freeze_json(child) for child in value)
    return value


def thaw_json(value: Any) -> Any:
    """Return a fresh mutable JSON value for serialization or an external writer."""
    if isinstance(value, Mapping):
        return {str(key): thaw_json(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray | memoryview):
        return [thaw_json(child) for child in value]
    return value


def json_type_tag(value: Any) -> str:
    """Name the JSON type of ``value``, keeping bool distinct from int.

    ``bool`` is a subclass of ``int`` in Python, so the ordering of these checks
    is the whole point: without it ``True`` and ``1`` are indistinguishable and an
    export could flip a flag argument into a count. Anything outside the JSON model
    gets a tag that no JSON value can share, so it never compares equal to one.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, bytes | bytearray | memoryview):
        # Byte strings are Sequences, so they would otherwise pass as JSON arrays.
        return f"unsupported ({type(value).__name__})"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, Sequence):
        return "array"
    return f"unsupported ({type(value).__name__})"


def validate_json_value(value: Any, *, label: str) -> None:
    """Reject anything the canonical JSON codec cannot round-trip.

    NaN and the infinities are the ones that matter in practice: ``json.loads``
    accepts them, so a malformed oracle result would otherwise reach an export as
    a literal no strict JSON reader can parse back.
    """
    tag = json_type_tag(value)
    if tag == "float":
        if not math.isfinite(value):
            raise ValueError(f"{label} must be a finite number; NaN and infinity are not valid benchmark JSON")
        return
    if tag in _JSON_SCALAR_TAGS:
        return
    if tag == "object":
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} must use string object keys, got {json_type_tag(key)}")
            validate_json_value(child, label=f"{label}.{key}")
        return
    if tag == "array":
        for index, child in enumerate(value):
            validate_json_value(child, label=f"{label}[{index}]")
        return
    raise ValueError(f"{label} is not JSON-representable (type {tag})")


def json_equal(left: Any, right: Any) -> bool:
    """Compare two JSON values by type as well as content.

    Python's ``==`` holds for ``1 == True`` and ``1 == 1.0``, so it cannot detect
    the coercions this contract forbids. Comparing type tags first makes ``"1"``,
    ``1``, ``1.0`` and ``True`` four distinct values.
    """
    tag = json_type_tag(left)
    if tag != json_type_tag(right) or tag not in JSON_TYPE_TAGS:
        return False
    if tag == "object":
        if set(left) != set(right):
            return False
        return all(json_equal(left[key], right[key]) for key in left)
    if tag == "array":
        left_items = list(left)
        right_items = list(right)
        if len(left_items) != len(right_items):
            return False
        return all(json_equal(one, other) for one, other in zip(left_items, right_items, strict=True))
    return left == right


def _strict_json_object(value: Any, *, label: str) -> dict[str, Any]:
    """Validate one JSON object without normalizing the keys it carries."""
    if json_type_tag(value) != "object":
        raise ValueError(f"{label} must be a JSON object, got {json_type_tag(value)}")
    validate_json_value(value, label=label)
    return FrozenDict({str(key): freeze_json(child) for key, child in value.items()})


def require_canonical(value: Any, text: str, *, label: str) -> Any:
    """Prove a decoded column re-serializes to exactly the text it came from.

    Round-tripping through :func:`canonical_json` is what proves the column can be
    republished byte-for-byte: text that re-serializes differently would make the
    parquet and the export disagree on a hash even though they agree on content.
    """
    validate_json_value(value, label=label)
    if canonical_json(value) != text:
        raise ValueError(f"{label} must be canonical JSON so an export cannot silently reformat it")
    return value


def decode_canonical_json(text: Any, *, label: str) -> Any:
    """Decode canonical JSON text, refusing text an export would reformat."""
    if not isinstance(text, str):
        raise ValueError(f"{label} must be canonical JSON text, got {json_type_tag(text)}")
    try:
        decoded = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be valid JSON text") from exc
    return require_canonical(decoded, text, label=label)


def decode_lossless_arguments(encoded: Any, *, label: str) -> dict[str, Any]:
    """Decode one Arrow ``arguments`` map and prove the decode lost nothing.

    Parquet hands the map back as pairs rather than a dict, so duplicate keys are
    representable here even though a JSON object cannot hold them; a duplicate
    would make the decoded object depend on iteration order. Re-encoding and
    comparing byte-for-byte then rules out the type drift the map exists to
    prevent, because each value is canonical JSON text of its own.
    """
    pairs = list(encoded.items()) if hasattr(encoded, "items") else list(encoded or [])
    names = [name for name, _ in pairs]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{label} must use non-empty string argument names")
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"{label} repeats argument name(s) {duplicates}, which no JSON object can express")
    decoded = decode_arguments(dict(pairs))
    validate_json_value(decoded, label=label)
    if encode_arguments(decoded) != dict(pairs):
        raise ValueError(f"{label} does not survive a decode/encode round trip; the stored JSON is not canonical")
    return decoded


def tool_names(tools: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """List the function names one record exposes to the model, in record order."""
    return tuple(str((tool.get("function") or {}).get("name")) for tool in tools)


def validated_tool_definitions(value: Any) -> tuple[dict[str, Any], ...]:
    """Validate decoded tool definitions without re-modelling their JSON Schema.

    The definitions stay opaque mappings on purpose: a typed model would drop the
    optional keys a pack omitted and add the ones it did not declare, so the
    exported tools would no longer re-serialize to the published bytes.
    """
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError("tools must be a decoded sequence of tool definitions")
    definitions = [_strict_json_object(tool, label="tools") for tool in value]
    for index, tool in enumerate(definitions):
        declared_type = tool.get("type", "function")
        if declared_type != "function":
            raise ValueError(f"tools[{index}] declares unsupported type {declared_type!r}")
        function = tool.get("function")
        if json_type_tag(function) != "object":
            raise ValueError(f"tools[{index}] must carry a function object")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"tools[{index}].function requires a non-empty name")
    names = tool_names(definitions)
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"tools exposes duplicate function name(s) {duplicates}")
    return tuple(FrozenDict({str(key): freeze_json(child) for key, child in tool.items()}) for tool in definitions)


class ExportedFunction(BaseModel):
    """The wire form of one tool call inside an assistant message."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: StrictStr
    arguments: StrictStr

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("an exported function call requires a non-empty name")
        return value

    @model_validator(mode="after")
    def validate_arguments(self) -> ExportedFunction:
        decoded = decode_canonical_json(self.arguments, label=f"tool_calls.{self.name}.arguments")
        if json_type_tag(decoded) != "object":
            raise ValueError(f"tool_calls.{self.name}.arguments must encode a JSON object")
        return self


class ExportedMessageToolCall(BaseModel):
    """One entry of an assistant message's ``tool_calls`` array."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: StrictStr
    type: Literal["function"] = "function"
    function: ExportedFunction

    @field_validator("id")
    @classmethod
    def normalize_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("an exported tool call requires a non-empty id")
        return value


class ExportedMessage(BaseModel):
    """One OpenAI-style conversation message, as published and as exported."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: MessageRole
    content: StrictStr | None = None
    tool_calls: tuple[ExportedMessageToolCall, ...] = ()
    tool_call_id: StrictStr | None = None

    @field_validator("tool_calls", mode="before")
    @classmethod
    def absent_tool_calls(cls, value: Any) -> Any:
        # Parquet materializes every struct field, so a user message reads back
        # with tool_calls=None rather than with the key missing.
        return () if value is None else value

    @model_validator(mode="after")
    def validate_message(self) -> ExportedMessage:
        identifiers = [call.id for call in self.tool_calls]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("a message cannot repeat a tool_call id")
        if self.role in {"system", "user"}:
            if self.content is None:
                raise ValueError(f"a {self.role} message requires content")
            if self.tool_calls or self.tool_call_id is not None:
                raise ValueError(f"a {self.role} message cannot carry tool-call detail")
            return self
        if self.role == "tool":
            if self.content is None:
                raise ValueError("a tool message requires the result content it returned")
            if self.tool_call_id is None:
                raise ValueError("a tool message requires the tool_call_id it answers")
            if self.tool_calls:
                raise ValueError("a tool message cannot request further calls")
            return self
        if self.tool_call_id is not None:
            raise ValueError("an assistant message cannot answer a tool call")
        if self.tool_calls:
            if self.content is not None:
                # Replay puts the calls of one step in a single assistant message
                # with no prose. Allowing both would let an export decide which of
                # the two a scorer should read.
                raise ValueError("an assistant message that requests tool calls cannot also carry content")
            return self
        if self.content is None:
            raise ValueError("an assistant message requires either content or tool calls")
        return self


class ExportedToolCall(BaseModel):
    """One expected (gold) call, with the grouping a scorer needs to order it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_index: NonNegativeInt
    call_group: NonNegativeInt
    position_in_group: NonNegativeInt
    function_name: StrictStr
    arguments: dict[str, Any]

    @field_validator("function_name")
    @classmethod
    def normalize_function_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("an expected tool call requires a non-empty function_name")
        return value

    @field_validator("arguments")
    @classmethod
    def validate_decoded_arguments(cls, value: Any) -> Any:
        return _strict_json_object(value, label="expected_tool_calls.arguments")

    @property
    def trace_position(self) -> tuple[int, int, int]:
        return (self.turn_index, self.call_group, self.position_in_group)


class CanonicalExportRow(BaseModel):
    """One published benchmark row, decoded once for every export writer.

    This is the only place the parquet encoding is undone. Writers receive
    already-decoded tools and arguments, so a format cannot differ from another
    format by decoding the same column differently.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = EXPORT_CONTRACT_VERSION
    task_id: StrictStr
    template_id: StrictStr
    variant_index: NonNegativeInt
    messages: tuple[ExportedMessage, ...]
    tools: tuple[dict[str, Any], ...]
    expected_tool_calls: tuple[ExportedToolCall, ...]
    success_assertions: tuple[StrictStr, ...] = ()
    fixture_refs: tuple[StrictStr, ...] = ()
    intent: StrictStr | None = None
    category: StrictStr | None = None
    difficulty: StrictStr | None = None
    required_tools: tuple[StrictStr, ...] = ()
    required_tools_fingerprint: StrictStr
    tools_present: tuple[StrictStr, ...] = ()
    turn_policy: StrictStr
    is_multi_turn: StrictBool
    num_tool_calls: NonNegativeInt
    call_order: CallOrderPolicy
    call_order_prefix: PositiveInt | None = None
    system_prompt_id: StrictStr
    tier: StrictStr
    gold_eligible: StrictBool
    validated_by: tuple[ValidationEvidence, ...]
    pack_id: StrictStr
    pack_version: StrictStr
    seed: NonNegativeInt
    paraphrase_model: StrictStr | None = None
    paraphrase_model_canonical: StrictStr | None = None
    held_out_hit: StrictBool | None = None
    src: StrictStr
    metadata: dict[str, Any]

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, value: Any) -> Any:
        return validated_tool_definitions(value)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: Any) -> Any:
        decoded = _strict_json_object(value, label="metadata")
        if set(decoded) != CANONICAL_METADATA_KEYS:
            missing = sorted(CANONICAL_METADATA_KEYS - set(decoded))
            extra = sorted(set(decoded) - CANONICAL_METADATA_KEYS)
            raise ValueError(f"metadata must carry exactly the canonical keys (missing={missing}, extra={extra})")
        return decoded

    @model_validator(mode="after")
    def validate_row(self) -> CanonicalExportRow:
        if not self.messages:
            raise ValueError(f"task {self.task_id!r} exports no messages, so it asks the model nothing")
        user_turn_count = sum(message.role == "user" for message in self.messages)
        if not user_turn_count:
            raise ValueError(f"task {self.task_id!r} exports no user turn")
        positions = [call.trace_position for call in self.expected_tool_calls]
        if positions != sorted(positions):
            raise ValueError(
                f"task {self.task_id!r} expected_tool_calls must stay in trace order "
                "(turn_index, call_group, position_in_group)"
            )
        if len(set(positions)) != len(positions):
            raise ValueError(f"task {self.task_id!r} expected_tool_calls repeat a trace position")
        wire_calls = [call for message in self.messages if message.role == "assistant" for call in message.tool_calls]
        if len(wire_calls) != len(self.expected_tool_calls):
            raise ValueError(
                f"task {self.task_id!r} messages carry {len(wire_calls)} tool calls but expected_tool_calls "
                f"carries {len(self.expected_tool_calls)}"
            )
        for wire, expected in zip(wire_calls, self.expected_tool_calls, strict=True):
            wire_arguments = decode_canonical_json(
                wire.function.arguments,
                label=f"task {self.task_id!r} message tool-call arguments",
            )
            if wire.function.name != expected.function_name or not json_equal(wire_arguments, expected.arguments):
                raise ValueError(
                    f"task {self.task_id!r} message tool calls do not match expected_tool_calls in trace order"
                )
        if self.num_tool_calls != len(self.expected_tool_calls):
            raise ValueError(
                f"task {self.task_id!r} reports {self.num_tool_calls} tool calls but exports "
                f"{len(self.expected_tool_calls)}"
            )
        exposed = set(tool_names(self.tools))
        if unexposed := sorted({call.function_name for call in self.expected_tool_calls} - exposed):
            raise ValueError(
                f"task {self.task_id!r} expects call(s) to {unexposed} that the exported tools do not expose"
            )
        if missing_required := sorted(set(self.required_tools) - exposed):
            raise ValueError(f"task {self.task_id!r} requires tool(s) {missing_required} that it does not expose")
        if self.required_tools_fingerprint != canonical_json(sorted(self.required_tools)):
            raise ValueError(f"task {self.task_id!r} required_tools_fingerprint does not match required_tools")
        if self.call_order == "prefix":
            if self.call_order_prefix is None:
                raise ValueError(f"task {self.task_id!r} declares call_order: prefix without call_order_prefix")
            if self.call_order_prefix > len(self.required_tools):
                # A prefix past the required tools describes an order the task
                # cannot have, and a scorer would read it as strict instead.
                raise ValueError(
                    f"task {self.task_id!r} call_order_prefix {self.call_order_prefix} exceeds its "
                    f"{len(self.required_tools)} required tools"
                )
        elif self.call_order_prefix is not None:
            raise ValueError(f"task {self.task_id!r} sets call_order_prefix without call_order: prefix")
        if self.src != f"{self.pack_id}:{self.template_id}":
            raise ValueError(f"task {self.task_id!r} src does not identify its pack and template")
        return self

    @classmethod
    def from_benchmark_row(cls, row: Mapping[str, Any]) -> CanonicalExportRow:
        """Project one parquet row, decoding the two encoded columns losslessly."""
        if not isinstance(row, Mapping):
            raise ValueError("a benchmark row must be a mapping")
        present = set(row)
        expected = set(BENCHMARK_ROW_FIELDS)
        if present != expected:
            missing = sorted(expected - present)
            extra = sorted(present - expected)
            raise ValueError(
                f"a benchmark row must carry exactly the published schema (missing={missing}, extra={extra})"
            )
        task_id = row["task_id"]
        label = task_id if isinstance(task_id, str) and task_id else "<unknown task>"
        raw_calls = list(row["expected_tool_calls"] or [])
        expected_calls = []
        for index, call in enumerate(raw_calls):
            if json_type_tag(call) != "object":
                raise ValueError(f"task {label!r} expected_tool_calls[{index}] must be a mapping")
            fields = dict(call)
            encoded_arguments = fields.pop("arguments", None)
            expected_calls.append(
                {
                    **fields,
                    "arguments": decode_lossless_arguments(
                        encoded_arguments,
                        label=f"task {label!r} expected_tool_calls[{index}].arguments",
                    ),
                }
            )
        encoded_tools = row["tools"]
        if not isinstance(encoded_tools, str):
            raise ValueError(f"task {label!r} tools must be canonical JSON text")
        payload = {
            **{name: row[name] for name in BENCHMARK_ROW_FIELDS},
            # Decoded through the column's own codec rather than a local reader, so
            # a change to how tools are stored cannot bypass this projection.
            "tools": require_canonical(
                decode_tools(encoded_tools),
                encoded_tools,
                label=f"task {label!r} tools",
            ),
            "expected_tool_calls": expected_calls,
            "metadata": decode_canonical_json(row["metadata"], label=f"task {label!r} metadata"),
        }
        return cls.model_validate(payload)

    def truth_payload(self) -> dict[str, Any]:
        """Project the fields no export may alter, in one comparable shape."""
        dumped = self.model_dump(mode="json")
        return {field: dumped[field] for field in EXPORT_TRUTH_FIELDS}


class BfclJsonMetadata(BaseModel):
    """Task descriptors and provenance a harness may filter or report on.

    None of this is scored, which is why it lives beside the truth fields rather
    than among them; keeping it in the record anyway is what lets a consumer slice
    results by turn policy or pack version without rejoining the parquet.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    template_id: StrictStr
    variant_index: NonNegativeInt
    intent: StrictStr | None = None
    category: StrictStr | None = None
    difficulty: StrictStr | None = None
    required_tools: tuple[StrictStr, ...] = ()
    required_tools_fingerprint: StrictStr
    tools_present: tuple[StrictStr, ...] = ()
    turn_policy: StrictStr
    is_multi_turn: StrictBool
    num_tool_calls: NonNegativeInt
    system_prompt_id: StrictStr
    tier: StrictStr
    gold_eligible: StrictBool
    validated_by: tuple[ValidationEvidence, ...]
    pack_id: StrictStr
    pack_version: StrictStr
    seed: NonNegativeInt
    paraphrase_model: StrictStr | None = None
    paraphrase_model_canonical: StrictStr | None = None
    held_out_hit: StrictBool | None = None
    fixture_refs: tuple[StrictStr, ...] = ()
    src: StrictStr
    surface: dict[str, Any]

    @field_validator("surface")
    @classmethod
    def validate_surface(cls, value: Any) -> Any:
        return _strict_json_object(value, label="metadata.surface")


class BfclJsonRecord(BaseModel):
    """One lossless carrier for a BFCL V4 question/ground-truth record pair.

    BFCL stores questions and possible answers in separate JSONL files joined by
    ``id``. :meth:`question_record` and :meth:`ground_truth_record` emit those
    upstream shapes. The fields retained here are the Nemotron extension needed
    to reconstruct exact parallel groups, ordering policy, and assertions.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = BFCL_JSON_SCHEMA_VERSION
    upstream_schema_version: Literal["BFCL_v4_multi_turn"] = BFCL_UPSTREAM_SCHEMA_VERSION
    id: StrictStr
    messages: tuple[ExportedMessage, ...]
    tools: tuple[dict[str, Any], ...]
    expected_tool_calls: tuple[ExportedToolCall, ...]
    success_assertions: tuple[StrictStr, ...] = ()
    call_order: CallOrderPolicy
    call_order_prefix: PositiveInt | None = None
    metadata: BfclJsonMetadata

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, value: Any) -> Any:
        return validated_tool_definitions(value)

    @classmethod
    def from_canonical(cls, row: CanonicalExportRow) -> BfclJsonRecord:
        return cls(
            id=row.task_id,
            messages=row.messages,
            tools=row.tools,
            expected_tool_calls=row.expected_tool_calls,
            success_assertions=row.success_assertions,
            call_order=row.call_order,
            call_order_prefix=row.call_order_prefix,
            metadata=BfclJsonMetadata(
                template_id=row.template_id,
                variant_index=row.variant_index,
                intent=row.intent,
                category=row.category,
                difficulty=row.difficulty,
                required_tools=row.required_tools,
                required_tools_fingerprint=row.required_tools_fingerprint,
                tools_present=row.tools_present,
                turn_policy=row.turn_policy,
                is_multi_turn=row.is_multi_turn,
                num_tool_calls=row.num_tool_calls,
                system_prompt_id=row.system_prompt_id,
                tier=row.tier,
                gold_eligible=row.gold_eligible,
                validated_by=row.validated_by,
                pack_id=row.pack_id,
                pack_version=row.pack_version,
                seed=row.seed,
                paraphrase_model=row.paraphrase_model,
                paraphrase_model_canonical=row.paraphrase_model_canonical,
                held_out_hit=row.held_out_hit,
                fixture_refs=row.fixture_refs,
                src=row.src,
                surface=row.metadata,
            ),
        )

    def truth_payload(self) -> dict[str, Any]:
        """Project the same shape :meth:`CanonicalExportRow.truth_payload` returns."""
        dumped = self.model_dump(mode="json")
        dumped["task_id"] = dumped.pop("id")
        return {field: dumped[field] for field in EXPORT_TRUTH_FIELDS}

    def question_record(self, call_groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Emit the BFCL V4 question record plus the lossless Nemotron extension.

        Upstream's ``question`` carries only what the model is shown: the system
        prompt and the user turns, never a tool result or a provenance field. The
        recorded oracle results and every descriptor stay under ``x-nemotron``, so
        a harness that renders ``question`` cannot leak them into the prompt.

        ``call_groups`` comes from the canonical projection. Upstream's per-turn
        answer list is flat, so parallel calls and sequential calls look alike
        there; carrying the grouping explicitly is what lets a consumer tell a task
        that expects two calls at once from one that expects them in order.
        """
        self._validate_call_groups(call_groups)
        rounds: list[list[dict[str, str]]] = []
        system_messages = [
            {"role": "system", "content": message.content}
            for message in self.messages
            if message.role == "system" and message.content is not None
        ]
        for message in self.messages:
            if message.role != "user" or message.content is None:
                continue
            turn = [{"role": "user", "content": message.content}]
            if not rounds and system_messages:
                turn = [*system_messages, *turn]
            rounds.append(turn)
        functions = [thaw_json(tool["function"]) for tool in self.tools]
        dumped = self.model_dump(mode="json")
        return {
            "id": self.id,
            "question": rounds,
            "function": functions,
            "x-nemotron": {
                "schema_version": self.schema_version,
                "upstream_schema_version": self.upstream_schema_version,
                "messages": dumped["messages"],
                "tools": dumped["tools"],
                "expected_tool_calls": dumped["expected_tool_calls"],
                "success_assertions": list(self.success_assertions),
                "call_order": self.call_order,
                "call_order_prefix": self.call_order_prefix,
                "call_groups": [dict(group) for group in call_groups],
                "metadata": self.metadata.model_dump(mode="json"),
            },
        }

    def _validate_call_groups(self, call_groups: Sequence[Mapping[str, Any]]) -> None:
        """Refuse a grouping that does not partition this record's expected calls."""
        covered: list[int] = []
        for group in call_groups:
            if set(group) != {"turn_index", "call_group", "user_turn_index", "is_parallel", "calls"}:
                raise ValueError(f"task {self.id!r} call group carries unexpected keys {sorted(group)}")
            indexes = list(group["calls"])
            if not indexes:
                raise ValueError(f"task {self.id!r} declares a call group that issues nothing")
            if group["is_parallel"] is not (len(indexes) > 1):
                raise ValueError(f"task {self.id!r} call group is parallel exactly when it issues several calls")
            covered.extend(indexes)
        if covered != list(range(len(self.expected_tool_calls))):
            raise ValueError(
                f"task {self.id!r} call groups must partition its {len(self.expected_tool_calls)} "
                "expected call(s) in trace order"
            )

    def ground_truth_record(
        self,
        calls_by_user_turn: Sequence[Sequence[ExportedToolCall]],
    ) -> dict[str, Any]:
        """Emit the BFCL V4 multi-turn possible-answer record shape.

        BFCL's executable multi-turn corpus stores Python call strings per user
        turn. Rendering from the exact typed arguments preserves ``None``, booleans,
        nested containers, and string-vs-number distinctions.

        The per-turn grouping is supplied by the canonical projection rather than
        re-derived here: a writer that walked the messages itself would be a second
        reading of the same trace, and two readings can disagree about which turn a
        call answers.
        """
        turn_count = sum(message.role == "user" for message in self.messages)
        if len(calls_by_user_turn) != turn_count:
            raise ValueError(
                f"task {self.id!r} has {turn_count} user turn(s) but was given {len(calls_by_user_turn)} turn group(s)"
            )
        grouped = [call for turn in calls_by_user_turn for call in turn]
        if list(grouped) != list(self.expected_tool_calls):
            raise ValueError(f"task {self.id!r} turn grouping does not account for exactly its expected calls")
        for call in grouped:
            function_parts = call.function_name.split(".")
            if any(not part.isidentifier() or keyword.iskeyword(part) for part in function_parts):
                raise ValueError(
                    f"task {self.id!r} function {call.function_name!r} cannot be represented "
                    "by BFCL's Python call-string ground truth"
                )
            incompatible_arguments = sorted(
                name for name in call.arguments if not name.isidentifier() or keyword.iskeyword(name)
            )
            if incompatible_arguments:
                raise ValueError(
                    f"task {self.id!r} function {call.function_name!r} has argument name(s) "
                    f"{incompatible_arguments} that BFCL's Python call-string ground truth cannot represent"
                )
        return {
            "id": self.id,
            "ground_truth": [
                [
                    f"{call.function_name}("
                    + ", ".join(f"{name}={thaw_json(value)!r}" for name, value in call.arguments.items())
                    + ")"
                    for call in turn
                ]
                for turn in calls_by_user_turn
            ],
        }


class NemoEvaluatorReplayStep(BaseModel):
    """One user turn and the gold material an adapter may reveal incrementally."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_turn_index: NonNegativeInt
    user_message: ExportedMessage
    expected_call_indexes: tuple[NonNegativeInt, ...] = ()
    reference_assistant_messages: tuple[ExportedMessage, ...] = ()
    tool_results: tuple[ExportedMessage, ...] = ()

    @model_validator(mode="after")
    def validate_step(self) -> NemoEvaluatorReplayStep:
        if self.user_message.role != "user":
            raise ValueError("an evaluator replay step must begin with one user message")
        if any(message.role != "assistant" for message in self.reference_assistant_messages):
            raise ValueError("reference_assistant_messages may only carry assistant messages")
        if any(message.role != "tool" for message in self.tool_results):
            raise ValueError("tool_results may only carry tool messages")
        if list(self.expected_call_indexes) != sorted(set(self.expected_call_indexes)):
            raise ValueError("expected_call_indexes must be sorted and distinct")
        return self


class NemoEvaluatorRecord(BaseModel):
    """One dataset row of the ``nemo_evaluator_bundle`` export.

    This is the input contract for the native evaluator adapter, not a launcher task
    registration. It separates answer-free seed messages from the gold reference
    trace and leaves descriptor-level concerns to :class:`NemoEvaluatorBundle`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = NEMO_EVALUATOR_SCHEMA_VERSION
    task_id: StrictStr
    # Only this field is model input. Gold assistant actions live under an
    # explicitly named reference field so a generic chat adapter cannot hand the
    # model its answer merely by forwarding ``messages``.
    seed_messages: tuple[ExportedMessage, ...]
    reference_trace: tuple[ExportedMessage, ...]
    replay_steps: tuple[NemoEvaluatorReplayStep, ...]
    tools: tuple[dict[str, Any], ...]
    expected_tool_calls: tuple[ExportedToolCall, ...]
    success_assertions: tuple[StrictStr, ...] = ()
    call_order: CallOrderPolicy
    call_order_prefix: PositiveInt | None = None
    turn_policy: StrictStr
    category: StrictStr | None = None
    difficulty: StrictStr | None = None

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, value: Any) -> Any:
        return validated_tool_definitions(value)

    @model_validator(mode="after")
    def validate_replay_contract(self) -> NemoEvaluatorRecord:
        roles = [message.role for message in self.seed_messages]
        if not roles or roles[-1] != "user" or any(role != "system" for role in roles[:-1]):
            raise ValueError("seed_messages must contain only leading system messages and the first user turn")
        user_messages = tuple(message for message in self.reference_trace if message.role == "user")
        if tuple(step.user_message for step in self.replay_steps) != user_messages:
            raise ValueError("replay_steps must account for every user turn in reference_trace, in order")
        if tuple(step.user_turn_index for step in self.replay_steps) != tuple(range(len(self.replay_steps))):
            raise ValueError("replay_steps must be numbered 0..n-1")
        first_user_position = next(
            (index for index, message in enumerate(self.reference_trace) if message.role == "user"),
            None,
        )
        if first_user_position is None or self.seed_messages != self.reference_trace[: first_user_position + 1]:
            raise ValueError("seed_messages must be exactly the safe prefix through the first user turn")
        covered = tuple(index for step in self.replay_steps for index in step.expected_call_indexes)
        if covered != tuple(range(len(self.expected_tool_calls))):
            raise ValueError("replay_steps must partition expected_tool_calls in trace order")
        return self

    @classmethod
    def from_canonical(cls, row: CanonicalExportRow) -> NemoEvaluatorRecord:
        first_user_position = next(
            (index for index, message in enumerate(row.messages) if message.role == "user"),
            None,
        )
        if first_user_position is None:
            raise ValueError(f"task {row.task_id!r} has no user turn to seed")
        if any(message.role != "system" for message in row.messages[:first_user_position]):
            raise ValueError(f"task {row.task_id!r} carries assistant or tool output before its first user turn")

        user_positions = [index for index, message in enumerate(row.messages) if message.role == "user"]
        replay_steps: list[NemoEvaluatorReplayStep] = []
        call_index = 0
        for user_turn_index, start in enumerate(user_positions):
            stop = (
                user_positions[user_turn_index + 1] if user_turn_index + 1 < len(user_positions) else len(row.messages)
            )
            segment = row.messages[start + 1 : stop]
            assistants = tuple(message for message in segment if message.role == "assistant")
            tool_results = tuple(message for message in segment if message.role == "tool")
            call_count = sum(len(message.tool_calls) for message in assistants)
            indexes = tuple(range(call_index, call_index + call_count))
            call_index += call_count
            replay_steps.append(
                NemoEvaluatorReplayStep(
                    user_turn_index=user_turn_index,
                    user_message=row.messages[start],
                    expected_call_indexes=indexes,
                    reference_assistant_messages=assistants,
                    tool_results=tool_results,
                )
            )
        return cls(
            task_id=row.task_id,
            seed_messages=row.messages[: first_user_position + 1],
            reference_trace=row.messages,
            replay_steps=tuple(replay_steps),
            tools=row.tools,
            expected_tool_calls=row.expected_tool_calls,
            success_assertions=row.success_assertions,
            call_order=row.call_order,
            call_order_prefix=row.call_order_prefix,
            turn_policy=row.turn_policy,
            category=row.category,
            difficulty=row.difficulty,
        )

    def truth_payload(self) -> dict[str, Any]:
        """Project the same shape :meth:`CanonicalExportRow.truth_payload` returns."""
        dumped = self.model_dump(mode="json")
        dumped["messages"] = dumped.pop("reference_trace")
        return {field: dumped[field] for field in EXPORT_TRUTH_FIELDS}


class NemoEvaluatorScoring(BaseModel):
    """What a scorer must compare, declared rather than inferred."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = NEMO_EVALUATOR_SCHEMA_VERSION
    metrics: tuple[ExportScoringMetric, ...]
    # Arguments are compared as canonical JSON, so ``{"a":1,"b":2}`` matches
    # ``{"b":2,"a":1}`` while ``1`` never matches ``"1"``.
    argument_match: Literal["canonical_json_exact"] = "canonical_json_exact"
    call_order_policies: tuple[CallOrderPolicy, ...]

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("an evaluator bundle must declare at least one scoring metric")
        if len(set(value)) != len(value):
            raise ValueError("evaluator scoring metrics must be unique")
        return tuple(metric for metric in EXPORT_SCORING_METRICS if metric in set(value))

    @field_validator("call_order_policies")
    @classmethod
    def validate_call_order_policies(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("evaluator scoring must declare at least one call_order policy")
        if len(set(value)) != len(value):
            raise ValueError("evaluator call_order policies must be unique")
        return tuple(sorted(value))


class NemoEvaluatorSource(BaseModel):
    """Back-reference from the bundle to the benchmark it was derived from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = NEMO_EVALUATOR_SCHEMA_VERSION
    benchmark_file: StrictStr
    benchmark_content_hash: ContentHash
    pack_id: StrictStr
    pack_version: StrictStr
    expt_name: StrictStr

    @field_validator("benchmark_file")
    @classmethod
    def validate_benchmark_file(cls, value: str) -> str:
        return relative_export_path(value, label="benchmark_file")


class NemoEvaluatorBundle(BaseModel):
    """The descriptor published beside the evaluator dataset file.

    The native adapter binds a task id to this dataset, so the bundle names both
    and pins the dataset's hash. This descriptor deliberately does not claim to be
    a NeMo Evaluator Launcher run config: that also needs a registered environment,
    solver/resource service, and candidate endpoint.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = NEMO_EVALUATOR_SCHEMA_VERSION
    task_name: StrictStr
    dataset_file: StrictStr
    dataset_schema_file: StrictStr
    metadata_file: StrictStr
    evaluator_config_file: StrictStr
    system_prompt_file: StrictStr
    record_count: PositiveInt
    dataset_content_hash: ContentHash
    scoring: NemoEvaluatorScoring
    source: NemoEvaluatorSource

    @field_validator("task_name")
    @classmethod
    def validate_task_name(cls, value: str) -> str:
        # Launcher task ids travel through shells, filenames and config keys.
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", value):
            raise ValueError(
                f"evaluator task_name {value!r} must be lowercase and limited to letters, digits, '_', '.' and '-'"
            )
        return value

    @field_validator(
        "dataset_file",
        "dataset_schema_file",
        "metadata_file",
        "evaluator_config_file",
        "system_prompt_file",
    )
    @classmethod
    def validate_bundle_path(cls, value: str, info: Any) -> str:
        return relative_export_path(value, label=info.field_name)

    @model_validator(mode="after")
    def validate_bundle(self) -> NemoEvaluatorBundle:
        paths = (
            self.dataset_file,
            self.dataset_schema_file,
            self.metadata_file,
            self.evaluator_config_file,
            self.system_prompt_file,
        )
        if len(set(paths)) != len(paths):
            raise ValueError("NeMo Evaluator bundle artifacts must use distinct paths")
        return self


def export_content_hash(contents: Mapping[str, bytes]) -> str:
    """Define ``content_hash`` for a format that writes more than one file.

    Names are hashed with bytes, so renaming a file, adding one, or reordering
    the pair changes the digest. Hashing only the concatenated bytes would let a
    question file and an answer file swap places unnoticed, and a harness reading
    the swapped pair would prompt the model with the answers.

    Taking bytes rather than paths lets a writer digest what it is about to write,
    so a descriptor can be validated before any file exists.
    """
    normalized: dict[str, bytes] = {}
    for original, payload in contents.items():
        path = relative_export_path(original, label="export file")
        if path in normalized:
            raise ValueError(
                f"an export tree hash requires distinct files after path normalization; {path!r} is repeated"
            )
        if not isinstance(payload, bytes):
            raise TypeError(f"export file {path!r} content must be bytes")
        normalized[path] = payload
    if not normalized:
        raise ValueError("an export that wrote no file has no content to hash")
    digest = hashlib.sha256()
    for path in sorted(normalized):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(normalized[path]).digest())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def export_tree_hash(root: Path, relative_paths: Sequence[str]) -> str:
    """Digest files already on disk, under the same definition."""
    ordered = [relative_export_path(path, label="export file") for path in relative_paths]
    if len(set(ordered)) != len(ordered):
        raise ValueError("an export tree hash requires distinct files")
    return export_content_hash({path: (root / path).read_bytes() for path in ordered})


def relative_export_path(value: str, *, label: str) -> str:
    """Require a bundle-relative POSIX path, so a bundle stays movable."""
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} must be non-empty")
    if normalized.startswith("/") or normalized.startswith("\\") or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError(f"{label} must be relative to the export directory, got {value!r}")
    if "\\" in normalized or ".." in normalized.split("/"):
        raise ValueError(f"{label} must be a POSIX path that stays inside the export directory, got {value!r}")
    return normalized


class ExportRowFailure(BaseModel):
    """One reason an export does not match the published benchmark."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: StrictStr
    reason: ExportFailureReason
    field: StrictStr | None = None

    @model_validator(mode="after")
    def validate_failure(self) -> ExportRowFailure:
        if not self.task_id.strip():
            raise ValueError("an export failure requires the task it concerns")
        if self.reason == "truth_field_changed":
            if self.field not in EXPORT_TRUTH_FIELDS:
                raise ValueError(f"a truth_field_changed failure requires one of {list(EXPORT_TRUTH_FIELDS)}")
        elif self.field is not None:
            raise ValueError(f"a {self.reason} failure concerns the whole row, not a field")
        return self


class ExportFormatReport(BaseModel):
    """The verdict for one written format."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = EXPORT_CONTRACT_VERSION
    format: ExportFormatName
    format_schema_version: StrictStr
    path: StrictStr
    rows: NonNegativeInt
    content_hash: ContentHash
    equivalent: StrictBool
    failures: tuple[ExportRowFailure, ...] = ()

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return relative_export_path(value, label="path")

    @model_validator(mode="after")
    def validate_report(self) -> ExportFormatReport:
        if self.format_schema_version != EXPORT_FORMAT_SCHEMA_VERSIONS[self.format]:
            raise ValueError(
                f"{self.format} is written at schema {EXPORT_FORMAT_SCHEMA_VERSIONS[self.format]!r}, "
                f"but the report claims {self.format_schema_version!r}"
            )
        if self.equivalent != (not self.failures):
            raise ValueError(
                "an export is equivalent exactly when it recorded no failure; a partial match is a failed export"
            )
        return self


class ExportValidationReport(BaseModel):
    """Stage 12's export evidence, written whether or not publication proceeds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = EXPORT_CONTRACT_VERSION
    benchmark_rows: NonNegativeInt
    benchmark_content_hash: ContentHash
    formats: tuple[ExportFormatReport, ...] = ()
    status: Literal["passed", "failed"]

    @model_validator(mode="after")
    def validate_report(self) -> ExportValidationReport:
        names = [report.format for report in self.formats]
        if len(set(names)) != len(names):
            raise ValueError("an export report may carry at most one entry per format")
        if names != sorted(names):
            raise ValueError("export format reports must be ordered by format name")
        mismatched = [
            report.format for report in self.formats if report.equivalent and report.rows != self.benchmark_rows
        ]
        if mismatched:
            raise ValueError(
                "an equivalent export must contain every benchmark row; row counts disagree for "
                + ", ".join(mismatched)
            )
        expected = "passed" if all(report.equivalent for report in self.formats) else "failed"
        if self.status != expected:
            raise ValueError(f"export status must be {expected!r} for the recorded format results")
        return self


def export_manifest_section(
    *,
    enabled: Mapping[str, bool],
    report: ExportValidationReport | None,
    validation_report_path: str | None = None,
    validation_report_content_hash: str | None = None,
) -> dict[str, Any]:
    """Describe compatibility exports for the run manifest.

    A disabled format reports ``enabled: false`` with no result rather than a
    passing one, because nothing was written and nothing was compared. An enabled
    format without a report is a contradiction and stops the run: the manifest
    would otherwise attest to an export that never happened.
    """
    unknown = sorted(set(enabled) - set(EXPORT_FORMATS))
    if unknown:
        raise ValueError(f"unknown export format(s) {unknown}; expected {list(EXPORT_FORMATS)}")
    if any(type(value) is not bool for value in enabled.values()):
        raise ValueError("export enabled state must use booleans")
    if report is None:
        if validation_report_path is not None or validation_report_content_hash is not None:
            raise ValueError("an absent export report cannot carry validation-report lineage")
    else:
        if validation_report_path is None or validation_report_content_hash is None:
            raise ValueError("a written export report requires its path and content hash in manifest lineage")
        relative_export_path(validation_report_path, label="validation_report_path")
        if not isinstance(validation_report_content_hash, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", validation_report_content_hash
        ):
            raise ValueError("validation_report_content_hash must be a sha256 digest")
    reports = {item.format: item for item in (report.formats if report is not None else ())}
    if unreported := sorted(name for name in EXPORT_FORMATS if enabled.get(name) and name not in reports):
        raise ValueError(f"export format(s) {unreported} are enabled but carry no export report")
    if unexpected := sorted(name for name in reports if not enabled.get(name)):
        raise ValueError(f"export format(s) {unexpected} were written without being enabled")
    if report is not None and not any(enabled.values()):
        raise ValueError("an export report cannot exist when every export format is disabled")
    formats: dict[str, Any] = {}
    for name in EXPORT_FORMATS:
        item = reports.get(name)
        if item is None:
            formats[name] = {"enabled": False}
            continue
        formats[name] = {
            "enabled": True,
            "schema_version": item.format_schema_version,
            "path": item.path,
            "rows": item.rows,
            "content_hash": item.content_hash,
            "equivalent": item.equivalent,
        }
    return {
        "schema_version": EXPORT_CONTRACT_VERSION,
        "evaluated": report is not None,
        "status": report.status if report is not None else None,
        "benchmark_rows": report.benchmark_rows if report is not None else 0,
        "benchmark_content_hash": report.benchmark_content_hash if report is not None else None,
        "validation_report": (
            {
                "path": validation_report_path,
                "content_hash": validation_report_content_hash,
            }
            if report is not None
            else None
        ),
        "formats": formats,
    }


def validate_export_equivalence(
    canonical: Sequence[CanonicalExportRow],
    exported: Sequence[BfclJsonRecord | NemoEvaluatorRecord],
) -> list[ExportRowFailure]:
    """Compare one written format against the canonical projection, row by row.

    Order is part of the comparison: a format that reorders rows breaks the
    ``selection_rank`` publication order Stage 11 fixed, and a consumer sampling
    the first N records would get a different slice than the parquet's.
    """
    canonical_ids = [row.task_id for row in canonical]
    exported_ids = [record.id if isinstance(record, BfclJsonRecord) else record.task_id for record in exported]
    published = set(canonical_ids)
    failures: list[ExportRowFailure] = []
    seen: set[str] = set()
    for task_id in exported_ids:
        if task_id in seen:
            failures.append(ExportRowFailure(task_id=task_id, reason="duplicate_row"))
        elif task_id not in published:
            failures.append(ExportRowFailure(task_id=task_id, reason="unexpected_row"))
        seen.add(task_id)
    failures.extend(
        ExportRowFailure(task_id=task_id, reason="missing_row") for task_id in canonical_ids if task_id not in seen
    )
    if failures:
        # Positional comparison is meaningless once the row sets differ; reporting
        # every field as changed would bury the one problem an author must fix.
        return _deduplicated(failures)
    if canonical_ids != exported_ids:
        return _deduplicated(
            [
                ExportRowFailure(task_id=expected_id, reason="row_order_changed")
                for expected_id, actual_id in zip(canonical_ids, exported_ids, strict=True)
                if expected_id != actual_id
            ]
        )
    for row, record in zip(canonical, exported, strict=True):
        expected = row.truth_payload()
        actual = record.truth_payload()
        failures.extend(
            ExportRowFailure(task_id=row.task_id, reason="truth_field_changed", field=field)
            for field in EXPORT_TRUTH_FIELDS
            if not json_equal(expected[field], actual[field])
        )
    return _deduplicated(failures)


def _deduplicated(failures: Sequence[ExportRowFailure]) -> list[ExportRowFailure]:
    """Order failures for a stable report without repeating a task and reason."""
    unique = {(failure.task_id, failure.reason, failure.field): failure for failure in failures}
    return [unique[key] for key in sorted(unique, key=lambda key: (key[0], key[1], key[2] or ""))]
