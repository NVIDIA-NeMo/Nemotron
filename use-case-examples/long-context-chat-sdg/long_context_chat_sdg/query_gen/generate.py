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

"""Generate one end-user question per GenUnit via an LLM.

Reads the unit's chunk text (bounded), applies the kind directive, and asks the
model for a single natural question grounded in the passages. Returns a seed dict
carrying provenance (kind + source chunk ids) for downstream validation/eval.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from .prompts import KIND_DIRECTIVES, QUERY_GEN_SYSTEM, QUERY_GEN_TURN
from .sampler import GenUnit


def _passages(unit: GenUnit, max_chars: int) -> str:
    # separated by a divider, NOT numbered — numbering tempts the model to write
    # "as described in Passage 1", which leaks the scaffolding into the question.
    parts = [" ".join((ch.text or "").split())[:max_chars] for ch in unit.chunks]
    return "\n\n———\n\n".join(parts)


def _parse(text: str) -> Optional[str]:
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        q = json.loads(text[a:b + 1]).get("query", "")
    except json.JSONDecodeError:
        return None
    q = str(q).strip()
    return q or None


def generate_query(unit: GenUnit, caller: Callable[[str, str], str], *,
                   max_chars_per_chunk: int = 1600) -> Optional[Dict[str, Any]]:
    """Return a seed dict {query, kind, source_ids, cluster_id} or None on failure.

    ``caller(system, user) -> str`` is a direct OpenAI-style caller
    (see core.caller.make_openai_caller).
    """
    directive = KIND_DIRECTIVES.get(unit.kind, KIND_DIRECTIVES["factual"])
    prompt = QUERY_GEN_TURN.format(passages=_passages(unit, max_chars_per_chunk), directive=directive)
    text = caller(QUERY_GEN_SYSTEM, prompt) or ""
    query = _parse(text)
    if not query:
        return None
    return {"query": query, "kind": unit.kind,
            "source_ids": [c.id for c in unit.chunks],
            "cluster_id": unit.cluster_id}
