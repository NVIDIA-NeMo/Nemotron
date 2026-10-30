# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tool-call schema verification (JSON-schema validation of LLM tool calls)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple


class ToolCallVerifier:
    """Validates LLM-generated tool calls against their tool JSON schemas."""

    def verify(self, tool_calls: List[Dict[str, Any]], tools: List[Dict[str, Any]]):
        if not tool_calls:
            return True, [], [], [], []
        all_success, all_error, err_msgs, correct = [], [], [], []
        for tc in tool_calls:
            ok, err = self.verify_single(tc, tools)
            all_success.append(ok)
            all_error.append(err)
            if not ok:
                err_msgs.append({
                    "role": "tool",
                    "content": (f"Error in calling tool `{tc['function']['name']}` "
                                f"with arguments `{tc['function']['arguments']}`: {err}"),
                    "tool_call_id": tc["id"],
                })
            else:
                correct.append(tc)
        return all(all_success), err_msgs, all_success, all_error, correct

    def verify_single(self, tool_call: Dict[str, Any], tools: List[Dict[str, Any]]) -> Tuple[bool, str | None]:
        try:
            name = tool_call.get("function", {}).get("name")
            args = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
            if not name:
                return False, f"Invalid tool call: `{json.dumps(tool_call)}`"
            target = None
            for td in tools:
                spec = td["tool"] if isinstance(td, dict) and "tool" in td else td
                fn = spec.get("function", {})
                if fn.get("name") == name:
                    target = fn
                    break
            if not target:
                return False, f"Tool `{name}` not found in available tools."
            params = target.get("parameters", {})
            props = params.get("properties", {})
            for req in params.get("required", []):
                if req not in args:
                    return False, f"Required parameter `{req}` missing."
            for arg, val in args.items():
                if arg not in props:
                    return False, f"Argument `{arg}` not defined in schema."
                ok, msg = self._check_type(val, props[arg])
                if not ok:
                    return False, f"Invalid argument `{arg}`: {msg}"
                if "enum" in props[arg] and val not in props[arg]["enum"]:
                    return False, f"Value `{val}` for `{arg}` not in {props[arg]['enum']}"
            return True, None
        except Exception as e:
            return False, f"Unexpected error: {e}"

    def _check_type(self, value: Any, spec: Dict[str, Any]) -> Tuple[bool, str | None]:
        t = spec.get("type")
        checks = {
            "string": lambda v: isinstance(v, str),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
        }
        if t in checks:
            return (True, None) if checks[t](value) else (False, f"expected {t}, got {type(value).__name__}")
        if t == "array":
            if not isinstance(value, list):
                return False, f"expected array, got {type(value).__name__}"
            item_spec = spec.get("items")
            if item_spec:
                for it in value:
                    ok, m = self._check_type(it, item_spec)
                    if not ok:
                        return False, m
            return True, None
        if t == "object":
            if not isinstance(value, dict):
                return False, f"expected object, got {type(value).__name__}"
            return True, None
        return True, None
