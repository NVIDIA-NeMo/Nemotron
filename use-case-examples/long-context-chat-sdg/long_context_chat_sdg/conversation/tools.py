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

"""ToolEnvironment — routes each tool call to the retrieval service or an LLM sim.

- Retrieval tools (``cfg.retrieval_tools``) call the HTTP client, which oversamples
  and randomly subsamples to top_k. The retrieval log is kept for audit.
- Every other (user-defined) tool is simulated by an LLM producing a realistic
  JSON response — so the customer's own tool list works with no backend.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, List, Tuple

from ..core.llm import call_llm
from .prompts import (
    AUX_TOOL_SIM_SYSTEM,
    AUX_TOOL_SIM_TURN,
    SIMULATED_RETRIEVAL_SYSTEM,
    SIMULATED_RETRIEVAL_TURN,
)

AUX_MODEL = "aux_response_model"


def _fn(td: Dict[str, Any]) -> Dict[str, Any]:
    return (td["tool"] if "tool" in td else td).get("function", {})


def sample_tools(all_tools: List[Dict[str, Any]], max_tools: int, must_include: List[str],
                 rng: random.Random) -> List[Dict[str, Any]]:
    """Always keep the retrieval tools; shuffle the rest; cap to ``max_tools``."""
    keep = [t for t in all_tools if _fn(t).get("name") in must_include]
    rest = [t for t in all_tools if _fn(t).get("name") not in must_include]
    rng.shuffle(rest)
    room = max(0, max_tools - len(keep))
    return keep + rest[:room]


class ToolEnvironment:
    def __init__(self, cfg, client, tools: List[Dict[str, Any]]):
        self.cfg = cfg
        self.client = client
        self.tools = tools
        self.retrieval_log: List[Dict[str, Any]] = []
        # progress signal only (NOT used to filter retrieval): ids seen so far, and
        # whether the current hop's retrieval brought anything new.
        self.seen_ids: set = set()
        self.retrieved_this_hop = False
        self.new_this_hop = 0
        # compression audit: how often/how much the per-step view was compressed.
        self.comp_steps = 0        # assistant steps where compression reduced the view
        self.comp_raw_max = 0      # largest uncompressed view (tokens)
        self.comp_view_max = 0     # that view's compressed size (tokens)

    def note_view(self, raw_tokens: int, view_tokens: int) -> None:
        if view_tokens < raw_tokens:
            self.comp_steps += 1
        if raw_tokens > self.comp_raw_max:
            self.comp_raw_max, self.comp_view_max = raw_tokens, view_tokens

    def is_retrieval(self, name: str) -> bool:
        return name in self.cfg.retrieval_tools

    def _first(self, args: Dict[str, Any], names: List[str], default: Any) -> Any:
        for n in names:
            if args.get(n) not in (None, ""):
                return args[n]
        return default

    def respond(self, tool_call: Dict[str, Any], models: Dict[str, Any], user_query: str,
                rng: random.Random) -> Tuple[str, bool]:
        """Return (tool_response_content, was_retrieval)."""
        fn = tool_call.get("function", {})
        name = fn.get("name", "")
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            args = {}
        if self.is_retrieval(name):
            if self.client is not None:
                return self._retrieve(args, user_query, rng), True
            if self.cfg.retrieval_mode == "simulated":
                return self._simulate_retrieval(args, models, user_query), True
            raise RuntimeError("retrieval tool called without an HTTP client")
        return self._simulate(name, args, models, user_query), False

    def _retrieve(self, args: Dict[str, Any], user_query: str, rng: random.Random) -> str:
        query = str(self._first(args, self.cfg.query_arg_names, user_query))
        # top_k is governed by the CONFIG KNOB, not the model: the client requests
        # k*oversample_factor from the service and randomly keeps k. Any top_k the
        # model puts in the tool call is ignored on purpose (kept constant).
        k = max(1, int(self.cfg.top_k))
        chunks = self.client.retrieve(query, k, rng=rng)
        new = [c.id for c in chunks if c.id not in self.seen_ids]  # progress signal only
        self.seen_ids.update(c.id for c in chunks)
        self.retrieved_this_hop = True
        self.new_this_hop = len(new)
        self.retrieval_log.append({"query": query, "ids": [c.id for c in chunks], "new": len(new)})
        return json.dumps({"results": [c.to_payload() for c in chunks]}, ensure_ascii=False)

    def _simulate_retrieval(self, args: Dict[str, Any], models: Dict[str, Any],
                            user_query: str) -> str:
        """Generate demo evidence and normalize it to the production retrieval shape."""
        query = str(self._first(args, self.cfg.query_arg_names, user_query))
        response = call_llm(models, AUX_MODEL, [
            {"role": "system", "content": SIMULATED_RETRIEVAL_SYSTEM},
            {"role": "user", "content": SIMULATED_RETRIEVAL_TURN.format(
                query=query, user_query=user_query, top_k=max(1, int(self.cfg.top_k)))},
        ])
        raw = (response.get("content") if isinstance(response, dict) else str(response)) or "{}"
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            payload = {}
        items = payload.get("results", []) if isinstance(payload, dict) else []
        if not isinstance(items, list):
            items = []

        results = []
        for index, item in enumerate(items[:max(1, int(self.cfg.top_k))]):
            if not isinstance(item, dict) or not str(item.get("text", "")).strip():
                continue
            text = str(item["text"]).strip()
            chunk_id = "h" + hashlib.sha1(
                f"simulated|{query}|{index}|{text}".encode("utf-8")
            ).hexdigest()[:12]
            try:
                score = round(float(item.get("score", 1.0)), 4)
            except (TypeError, ValueError):
                score = 1.0
            result = {"id": chunk_id, "text": text, "score": score, "simulated": True}
            if item.get("doc_id"):
                result["doc_id"] = str(item["doc_id"])
            results.append(result)

        new = [item["id"] for item in results if item["id"] not in self.seen_ids]
        self.seen_ids.update(item["id"] for item in results)
        self.retrieved_this_hop = True
        self.new_this_hop = len(new)
        self.retrieval_log.append({"query": query, "ids": [item["id"] for item in results],
                                   "new": len(new), "mode": "simulated"})
        return json.dumps({"results": results, "simulated": True}, ensure_ascii=False)

    def _simulate(self, name: str, args: Dict[str, Any], models: Dict[str, Any], user_query: str) -> str:
        spec = next((t for t in self.tools if _fn(t).get("name") == name), {})
        tool_json = json.dumps(_fn(spec) or {"name": name}, ensure_ascii=False)
        resp = call_llm(models, AUX_MODEL, [
            {"role": "system", "content": AUX_TOOL_SIM_SYSTEM},
            {"role": "user", "content": AUX_TOOL_SIM_TURN.format(
                tool=tool_json, arguments=json.dumps(args, ensure_ascii=False), user_query=user_query)}])
        return (resp.get("content") if isinstance(resp, dict) else str(resp)) or "{}"
