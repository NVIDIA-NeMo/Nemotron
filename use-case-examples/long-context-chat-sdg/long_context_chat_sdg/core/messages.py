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

"""Message / tool formatting helpers for prompts."""

from __future__ import annotations

import json
from typing import Any, Dict, List


def format_tools_for_prompt(tools: List[Dict[str, Any]]) -> str:
    lines = []
    for td in tools:
        tool = td.get("tool", {}) if "tool" in td else td
        fn = tool.get("function", {})
        lines.append(f"- {fn.get('name', 'Unknown')}: {fn.get('description', 'No description')}")
    return "\n".join(lines)


def format_history_compact(messages: List[Dict[str, Any]], *, max_chars: int = 6000,
                           tool_snippet: int = 200) -> str:
    """A BOUNDED, system-stripped rendering for the user-simulator prompt.

    The user model only needs enough recent context to write a coherent next turn;
    it must NOT receive the full multi-hop trace (that blows small context windows).
    Tool outputs are snipped and the whole thing is capped to ``max_chars`` (tail-kept).
    """
    lines: List[str] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            lines.append(f"User: {m.get('content', '')}")
        elif role == "assistant":
            c = m.get("content", "") or ""
            tcs = m.get("tool_calls") or []
            if tcs:
                names = ", ".join(tc.get("function", {}).get("name", "") for tc in tcs)
                lines.append(f"Assistant: (used tools: {names}) {c}".rstrip())
            else:
                lines.append(f"Assistant: {c}")
        elif role == "tool":
            snip = (m.get("content", "") or "")[:tool_snippet].replace("\n", " ")
            lines.append(f"Tool: {snip}…")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = "…(earlier turns omitted)…\n" + text[-max_chars:]
    return text


def format_conversation_history_for_prompt(messages: List[Dict[str, Any]]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            lines.append(f"User: {msg.get('content', '')}")
        elif role == "assistant":
            text = f"Assistant: {msg.get('content', '')}"
            tcs = msg.get("tool_calls") or []
            if tcs:
                rendered = []
                for tc in tcs:
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        args = fn.get("arguments", "")
                    rendered.append({"name": fn.get("name"), "arguments": args})
                text += f" | Tool-Calls: {rendered}"
            lines.append(text)
        elif role == "tool":
            lines.append(f"Tool: {msg.get('content', '')}")
        elif role == "system" and msg.get("content"):
            lines.append(f"System: {msg.get('content', '')}")
    return "\n".join(lines)
