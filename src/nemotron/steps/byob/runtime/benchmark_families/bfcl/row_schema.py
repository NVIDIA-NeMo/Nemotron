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

"""Explicit Arrow schema for benchmark rows, plus the codecs consumers need.

Two columns cannot be stored as inferred Arrow structs. Tool arguments and JSON
Schema property maps have per-tool keys, so inference unions them and pads every
row with nulls — which turns "argument absent" into "argument is null" and
advertises parameters a tool does not accept. ``arguments`` is therefore an Arrow
map of canonical-JSON values (lossless, preserves exactly the keys that were
bound) and ``tools`` is canonical JSON text.
"""

from __future__ import annotations

import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize strict JSON; NaN and infinities are invalid benchmark data."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def encode_arguments(arguments: dict[str, Any]) -> dict[str, str]:
    """Encode a call's arguments object for the Arrow map column."""
    return {str(name): canonical_json(value) for name, value in arguments.items()}


def decode_arguments(encoded: Any) -> dict[str, Any]:
    """Decode an ``arguments`` cell back to the JSON object stage 7 derived."""
    items = encoded.items() if hasattr(encoded, "items") else list(encoded or [])
    return {str(name): json.loads(value) for name, value in items}


def decode_tools(encoded: str) -> list[dict[str, Any]]:
    """Decode the model-facing tool definitions column."""
    return json.loads(encoded)


def benchmark_schema() -> Any:
    """Return the pyarrow schema every BFCL benchmark parquet is written with."""
    import pyarrow as pa

    function_type = pa.struct([("name", pa.string()), ("arguments", pa.string())])
    tool_call_type = pa.struct(
        [("id", pa.string()), ("type", pa.string()), ("function", function_type)]
    )
    message_type = pa.struct(
        [
            ("role", pa.string()),
            ("content", pa.string()),
            ("tool_calls", pa.list_(tool_call_type)),
            ("tool_call_id", pa.string()),
        ]
    )
    expected_call_type = pa.struct(
        [
            ("turn_index", pa.int32()),
            ("call_group", pa.int32()),
            ("position_in_group", pa.int32()),
            ("function_name", pa.string()),
            ("arguments", pa.map_(pa.string(), pa.string())),
        ]
    )
    string_list = pa.list_(pa.string())
    return pa.schema(
        [
            ("task_id", pa.string()),
            ("template_id", pa.string()),
            ("variant_index", pa.int32()),
            ("messages", pa.list_(message_type)),
            ("tools", pa.string()),
            ("expected_tool_calls", pa.list_(expected_call_type)),
            ("success_assertions", string_list),
            ("fixture_refs", string_list),
            ("intent", pa.string()),
            ("category", pa.string()),
            ("difficulty", pa.string()),
            ("required_tools", string_list),
            ("required_tools_fingerprint", pa.string()),
            ("tools_present", string_list),
            ("turn_policy", pa.string()),
            ("is_multi_turn", pa.bool_()),
            ("num_tool_calls", pa.int32()),
            ("call_order", pa.string()),
            ("call_order_prefix", pa.int32()),
            ("system_prompt_id", pa.string()),
            ("tier", pa.string()),
            ("gold_eligible", pa.bool_()),
            ("validated_by", string_list),
            ("pack_id", pa.string()),
            ("pack_version", pa.string()),
            ("seed", pa.uint64()),
            ("paraphrase_model", pa.string()),
            ("paraphrase_model_canonical", pa.string()),
            ("held_out_hit", pa.bool_()),
            ("src", pa.string()),
            ("metadata", pa.string()),
        ]
    )
