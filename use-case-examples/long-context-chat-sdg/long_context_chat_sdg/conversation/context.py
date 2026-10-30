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

"""Context compression: preserve head + tail verbatim, compress only the middle.

The trajectory is assembled turn-by-turn; this bounds what each LLM call must
*read* without dropping any message (tool_call/tool pairs stay intact). The
stored training trajectory keeps full chunks — only the generation view shrinks.

  - HEAD  = leading system prompt(s)                    -> kept verbatim
  - TAIL  = everything from the last user turn to end    -> kept verbatim
  - MIDDLE= between them                                 -> compressed once over budget,
            keeping the last ``window_k`` tool responses raw
A running scratchpad (distilled findings) is inserted after the head so the
assistant reasons over notes instead of re-reading every raw chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScratchpadEntry:
    hop: int
    query: str
    finding: str
    source_ids: List[str] = field(default_factory=list)


@dataclass
class ResearchMemory:
    plan: str = ""
    entries: List[ScratchpadEntry] = field(default_factory=list)

    def add(self, hop: int, query: str, finding: str, source_ids: List[str]) -> None:
        self.entries.append(ScratchpadEntry(hop, query, finding, source_ids))

    def render(self) -> str:
        if not self.entries:
            return "(no findings yet)"
        return "\n".join(
            f"- (hop {e.hop}) {e.finding}" + (f" [{', '.join(e.source_ids)}]" if e.source_ids else "")
            for e in self.entries)


def _estimate_tokens(msgs: List[Dict[str, Any]]) -> int:
    # count only what is actually SENT to the model: message content. Past
    # reasoning_content is NOT replayed to the model (see llm._to_openai_message),
    # so it does not count toward the context budget.
    return sum(len(m.get("content", "") or "") for m in msgs) // 4


def _compact_tool_message(msg: Dict[str, Any], mode: str, snippet_chars: int = 160) -> Dict[str, Any]:
    content = msg.get("content", "") or ""
    cid = msg.get("tool_call_id", "?")
    if mode == "drop":
        content = f"[tool result {cid}: omitted]"
    elif mode == "summary":
        content = content[:snippet_chars] + "…" if len(content) > snippet_chars else content
    else:  # reference: id + short snippet
        snip = content[:snippet_chars].replace("\n", " ")
        content = f"[tool result {cid}: {snip}…]" if len(content) > snippet_chars else f"[tool result {cid}: {snip}]"
    out = dict(msg)
    out["content"] = content
    return out


def _shrink_further(msg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(msg)
    if out.get("role") == "assistant" and out.get("reasoning_content"):
        out["reasoning_content"] = ""
    c = out.get("content", "") or ""
    if len(c) > 80:
        out["content"] = c[:80].replace("\n", " ") + "…"
    return out


def _compress_span(middle: List[Dict[str, Any]], window_k: int, mode: str, budget: int) -> List[Dict[str, Any]]:
    if _estimate_tokens(middle) <= budget:
        return middle
    tool_positions = [i for i, m in enumerate(middle) if m.get("role") == "tool"]
    keep_raw = set(tool_positions[-window_k:]) if window_k > 0 else set()
    out = [_compact_tool_message(m, mode) if (m.get("role") == "tool" and i not in keep_raw) else m
           for i, m in enumerate(middle)]
    if _estimate_tokens(out) <= budget:
        return out
    return [m if i in keep_raw else _shrink_further(m) for i, m in enumerate(out)]


# hard ceiling on the assembled view, independent of the per-turn boundary — a
# single deep turn keeps its tail RAW, which could otherwise blow the context
# window. Kept above compression_token_limit so it never clamps the middle budget.
HARD_VIEW_TOKENS = 50000


def _global_cap(view: List[Dict[str, Any]], window_k: int, mode: str, cap: int) -> List[Dict[str, Any]]:
    """Last-resort: compact tool messages ANYWHERE (except the last window_k) so the
    total never exceeds ``cap`` tokens, no matter how deep the current turn is."""
    if _estimate_tokens(view) <= cap:
        return view
    tool_pos = [i for i, m in enumerate(view) if m.get("role") == "tool"]
    keep_raw = set(tool_pos[-window_k:]) if window_k > 0 else set()
    out = [_compact_tool_message(m, mode) if (m.get("role") == "tool" and i not in keep_raw) else m
           for i, m in enumerate(view)]
    if _estimate_tokens(out) <= cap:
        return out
    return [m if i in keep_raw else _shrink_further(m) for i, m in enumerate(out)]


def build_assistant_view(messages: List[Dict[str, Any]], memory: Optional[ResearchMemory], *,
                         window_k: int = 2, compaction_mode: str = "reference",
                         compression_token_limit: int = 2000, use_scratchpad: bool = True) -> List[Dict[str, Any]]:
    n = len(messages)
    head_end = 0
    while head_end < n and messages[head_end].get("role") == "system":
        head_end += 1
    tail_start = n
    for i in range(n - 1, head_end - 1, -1):
        if messages[i].get("role") == "user":
            tail_start = i
            break
    head = list(messages[:head_end])
    middle = _compress_span(list(messages[head_end:tail_start]), window_k, compaction_mode, compression_token_limit)
    view = head + middle + list(messages[tail_start:])
    if use_scratchpad and memory is not None:
        note = "Research plan:\n" + (memory.plan or "(none)") + "\n\nFindings so far:\n" + memory.render()
        # merge into the leading system message (many endpoints reject a 2nd/mid
        # system message — "system must be at the beginning"); prepend one only if
        # there is no head system to merge into.
        if head_end > 0:
            merged = dict(view[head_end - 1])
            merged["content"] = (merged.get("content", "") or "") + "\n\n--- research scratchpad ---\n" + note
            view[head_end - 1] = merged
        else:
            view.insert(0, {"role": "system", "content": note})
    return _global_cap(view, window_k, compaction_mode, HARD_VIEW_TOKENS)


def build_judge_view(messages: List[Dict[str, Any]], *, window_k: int = 2,
                     compaction_mode: str = "reference", compression_token_limit: int = 2000) -> List[Dict[str, Any]]:
    return build_assistant_view(messages, None, window_k=window_k, compaction_mode=compaction_mode,
                                compression_token_limit=compression_token_limit, use_scratchpad=False)
