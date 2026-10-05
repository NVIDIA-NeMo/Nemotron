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

"""ConversationSimulatorGenerator — the multi-turn retrieval-conversation engine.

One row -> one grounded multi-turn / multi-step tool-calling trajectory:

    sample shape -> plan turns -> [gate] -> for each turn:
        (follow-up) -> autonomous TOOL LOOP (HTTP retrieval + LLM-sim tools,
        context-compressed, depth-floored) -> grounded answer
    -> inline trajectory judge

Requires the Data Designer runtime; the rest of the package stays importable
without DD (this module is only imported by the plugin).
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, List, Optional, Tuple

from data_designer.engine.column_generators.generators.base import (
    ColumnGeneratorCellByCell, ColumnGeneratorWithModelRegistry,
)

from ..core.llm import call_llm, call_llm_with_majority_vote, configure_direct_clients
from ..core.messages import format_history_compact, format_tools_for_prompt
from ..core.persona import format_persona_for_prompt
from ..retrieval.client import HttpRetrievalClient
from .config import ConversationSimulatorConfig
from .context import ResearchMemory, build_assistant_view, _estimate_tokens
from .judges import run_inline_judge
from .planner import plan_conversation
from .prompts import (
    ASSISTANT_SYSTEM_PROMPT, KIND_DIRECTIVES, QUERY_GATE_JUDGE_PROMPT, TRAJECTORY_JUDGE_PROMPT,
    USER_AGENT_SYSTEM_PROMPT, USER_CLARIFY_DIRECTIVE, USER_FOLLOWUP_DIRECTIVE, USER_OPENING_DIRECTIVE,
)
from .tools import ToolEnvironment, sample_tools
from .verifiers import ToolCallVerifier

MODEL_USER, MODEL_ASSISTANT, MODEL_JUDGE, MODEL_AUX = (
    "user_model", "assistant_model", "judge_model", "aux_response_model")
MODEL_ALIASES = [MODEL_USER, MODEL_ASSISTANT, MODEL_JUDGE, MODEL_AUX]


class ConversationSimulatorGenerator(
    ColumnGeneratorCellByCell[ConversationSimulatorConfig],
    ColumnGeneratorWithModelRegistry[ConversationSimulatorConfig],
):
    # ── async entry point: DD runs the client in async mode ──────────────────
    async def agenerate(self, data: dict) -> dict:
        import asyncio
        from ..core.llm import DD_EVENT_LOOP
        DD_EVENT_LOOP.set(asyncio.get_running_loop())
        return await asyncio.to_thread(self.generate, dict(data))

    # ── sync loop (runs in a worker thread under agenerate) ──────────────────
    def generate(self, data: dict) -> dict:
        cfg = self.config
        configure_direct_clients(cfg.model_clients)
        models = {a: self.get_model(a) for a in MODEL_ALIASES}
        self._verifier = ToolCallVerifier()

        query = str(data.get(cfg.query_column, "") or "").strip()
        kind = str(data.get(cfg.kind_column, "") or "").strip()  # optional; planner samples if absent
        cluster_id = str(data.get(cfg.cluster_id_column, "") or "")
        persona = _as_obj(data.get(cfg.persona_column) or "{}")
        all_tools = _as_obj(data.get(cfg.tools_column) or "[]")

        rng = random.Random(int(hashlib.sha256(f"{cluster_id}|{query}".encode()).hexdigest()[:12], 16))
        tools = sample_tools(all_tools, cfg.max_tools, cfg.retrieval_tools, rng)
        tools_str = format_tools_for_prompt(tools)
        client = self._make_client(cfg)
        env = ToolEnvironment(cfg, client, tools)

        plan = plan_conversation(cfg.conversation_plan, rng, seed_kind=kind,
                                 min_turns=cfg.min_turns, max_turns=cfg.max_turns,
                                 min_hops=cfg.min_hops, max_steps=cfg.max_steps)
        persona_str = format_persona_for_prompt(persona) if persona else ""

        # opening user turn: re-voice the seed query in the persona (optional)
        opening = query
        if cfg.persona_voice and persona_str and query:
            opening = self._voice_query(models, persona_str, tools_str, query) or query

        data["cluster_id"] = cluster_id
        data["retrieval_mode"] = cfg.retrieval_mode
        data["conversation_plan"] = json.dumps(plan.kinds(), ensure_ascii=False)
        data["user_query"] = opening

        # optional gate: skip weak seed queries before spending the loop
        if cfg.gate_query and opening:
            verdict = run_inline_judge(models, MODEL_JUDGE, QUERY_GATE_JUDGE_PROMPT.format(query=opening))
            if verdict["parsed"] and not verdict["success"]:
                return self._skipped(data, cfg, "query_gate_failed", verdict["explanation"])

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": ASSISTANT_SYSTEM_PROMPT.format(tools=format_tools_for_prompt(tools))},
            {"role": "user", "content": opening},
        ]
        memory = ResearchMemory(plan="Answer the user by researching with the tools; ground every claim.")

        total_hops = 0
        n_turns = min(plan.n_turns, cfg.max_turns)
        for turn_idx in range(n_turns):
            if turn_idx > 0:
                followup = self._gen_followup(models, persona_str, tools_str, messages, plan.turns[turn_idx].kind)
                if not followup:
                    break
                messages.append({"role": "user", "content": followup})
            user_idx = len(messages) - 1  # the user message that opened this turn
            spec = plan.turns[turn_idx]
            hops, produced = self._tool_loop(cfg, models, env, tools, messages, memory, spec, rng,
                                             opening, persona_str, tools_str)
            total_hops += hops
            if not produced:
                # this turn produced no fresh grounded answer -> DROP it and stop, so the
                # trajectory ends on the last turn that actually answered (stays coherent;
                # never reuse a previous turn's answer).
                del messages[user_idx:]
                break

        return self._finalize(data, cfg, models, messages, env, total_hops, tools_str)

    def _view(self, cfg, messages, memory, env=None):
        view = build_assistant_view(messages, memory, window_k=cfg.context_window_k,
                                    compaction_mode=cfg.compaction_mode,
                                    compression_token_limit=cfg.compression_token_limit,
                                    use_scratchpad=cfg.use_scratchpad)
        if env is not None:  # audit: did compression reduce this step's view, and by how much?
            env.note_view(_estimate_tokens(messages), _estimate_tokens(view))
        return view

    # ── the multi-step tool loop for ONE user turn ───────────────────────────
    def _tool_loop(self, cfg, models, env, tools, messages, memory, spec, rng, user_query,
                   persona_str, tools_str):
        """Run ONE user turn. Returns (hops, produced_answer). Appends a final
        answer only if it is non-empty; NEVER reuses a previous turn's answer. If
        the turn can't produce a fresh grounded answer, returns produced=False and
        the caller drops the turn (keeps the conversation coherent)."""
        # conversational turn: answer from the conversation so far, NO retrieval
        if getattr(spec, "no_tool", False):
            resp = call_llm_with_majority_vote(models, MODEL_ASSISTANT,
                                               self._view(cfg, messages, memory, env), tools,
                                               n=cfg.majority_vote_n, tool_choice="none")
            content = (resp.get("content") or "").strip()
            if content:
                messages.append(_assistant_msg(resp))
                return 0, True
            return 0, False

        hops, asked_clarify, stall = 0, False, 0
        for _ in range(spec.max_steps):
            # don't force a tool on a clarify turn until AFTER the assistant has asked its question
            force = (cfg.force_first_tool and hops < spec.min_hops
                     and not (spec.clarify and cfg.allow_clarify and not asked_clarify))
            resp = call_llm_with_majority_vote(models, MODEL_ASSISTANT, self._view(cfg, messages, memory, env),
                                               tools, n=cfg.majority_vote_n,
                                               tool_choice="required" if force else "auto")
            tool_calls = _sanitize_tool_calls((resp.get("tool_calls") or [])[: cfg.max_tool_calls_per_turn], tools)

            if tool_calls:
                messages.append(_assistant_msg(resp, tool_calls))
                env.retrieved_this_hop, env.new_this_hop = False, 0
                self._execute(env, models, tools, messages, memory, tool_calls, hops, rng, user_query)
                hops += 1
                # progress stall: a retrieval that returns only already-seen chunks
                # means the corpus has nothing more for this line — stop searching.
                if hops >= spec.min_hops and env.retrieved_this_hop and env.new_this_hop == 0:
                    stall += 1
                    if stall >= 2:
                        break
                else:
                    stall = 0
                continue

            content = resp.get("content", "") or ""
            # a trailing "?" on a clarify-turn -> let the user answer, then keep going.
            if spec.clarify and cfg.allow_clarify and not asked_clarify and content.strip().endswith("?"):
                messages.append(_assistant_msg(resp))
                messages.append({"role": "user",
                                 "content": self._clarify(models, persona_str, tools_str, messages, user_query)})
                asked_clarify = True
                continue
            if content.strip():                       # a real grounded answer -> done
                messages.append(_assistant_msg(resp))
                return hops, True
            break                                     # empty answer -> try one clean close-out below

        # close out: one tool-free call for a real answer. If STILL empty, the turn
        # failed — return produced=False (no stale salvage, no empty message appended).
        final = call_llm(models, MODEL_ASSISTANT, self._view(cfg, messages, memory, env))
        content = (final.get("content") if isinstance(final, dict) else str(final)) or ""
        if content.strip():
            messages.append(_assistant_msg(final if isinstance(final, dict) else {"content": content}))
            return hops, True
        return hops, False

    # ── execute tool calls: verify -> retrieve/simulate -> record ────────────
    def _execute(self, env, models, tools, messages, memory, tool_calls, hop, rng, user_query) -> None:
        for tc in tool_calls:
            ok, err = self._verifier.verify_single(tc, tools)
            if not ok:
                messages.append({"role": "tool", "content": f"Error: {err}", "tool_call_id": tc.get("id", "")})
                continue
            content, was_retrieval = env.respond(tc, models, user_query, rng)
            messages.append({"role": "tool", "content": content, "tool_call_id": tc.get("id", "")})
            if was_retrieval:
                self._remember(memory, content, hop)

    def _remember(self, memory, content: str, hop: int) -> None:
        try:
            results = json.loads(content).get("results", [])
        except (json.JSONDecodeError, TypeError):
            return
        ids = [str(r.get("id", "")) for r in results if r.get("id")]
        snippet = (results[0].get("text", "")[:180] if results else "").replace("\n", " ")
        if ids:
            memory.add(hop, query="", finding=snippet or "(retrieved evidence)", source_ids=ids)

    # ── user-simulator helpers (one rich system prompt + a per-turn directive) ─
    def _user_system(self, persona_str, tools_str) -> Dict[str, Any]:
        return {"role": "system",
                "content": USER_AGENT_SYSTEM_PROMPT.format(persona=persona_str or "(a curious, everyday person)",
                                                           tools=tools_str)}

    def _voice_query(self, models, persona_str, tools_str, query) -> str:
        r = call_llm(models, MODEL_USER, [self._user_system(persona_str, tools_str),
                     {"role": "user", "content": USER_OPENING_DIRECTIVE.format(need=query)}])
        return _content(r).strip()

    def _gen_followup(self, models, persona_str, tools_str, messages, kind) -> str:
        directive = KIND_DIRECTIVES.get(kind, KIND_DIRECTIVES["related"])
        r = call_llm(models, MODEL_USER, [self._user_system(persona_str, tools_str),
                     {"role": "user", "content": USER_FOLLOWUP_DIRECTIVE.format(
                         instruction=directive, conversation=format_history_compact(messages))}])
        return _content(r).strip()

    def _clarify(self, models, persona_str, tools_str, messages, user_query) -> str:
        question = messages[-1].get("content", "") if messages else ""
        r = call_llm(models, MODEL_USER, [self._user_system(persona_str, tools_str),
                     {"role": "user", "content": USER_CLARIFY_DIRECTIVE.format(
                         conversation=format_history_compact(messages), question=question)}])
        return _content(r).strip() or "Please proceed with your best interpretation."

    # ── retrieval client ─────────────────────────────────────────────────────
    def _make_client(self, cfg) -> Optional[HttpRetrievalClient]:
        if cfg.retrieval_mode == "simulated":
            return None
        if not cfg.retrieval_endpoint:
            raise ValueError("HTTP retrieval mode requires retrieval_endpoint")
        return HttpRetrievalClient(cfg.retrieval_endpoint, oversample_factor=cfg.oversample_factor,
                                   timeout=cfg.retrieval_timeout, field_map=cfg.retrieval_field_map,
                                   headers=cfg.retrieval_headers or None,
                                   max_retries=getattr(cfg, "retrieval_max_retries", 2))

    # ── finalize: judge + write side-effect columns ──────────────────────────
    def _finalize(self, data, cfg, models, messages, env, hops, tools_str) -> dict:
        # if not even the opening turn produced an answer, there's nothing to keep
        has_answer = any(m.get("role") == "assistant" and not m.get("tool_calls") and (m.get("content") or "").strip()
                         for m in messages)
        if not has_answer:
            return self._skipped(data, cfg, "no_grounded_answer", "no turn produced a grounded answer")
        last = messages[-1] if messages else {}
        ends_on_answer = last.get("role") == "assistant" and not last.get("tool_calls") and bool(last.get("content"))
        status, judgment = ends_on_answer, {}
        if cfg.inline_judge and ends_on_answer:
            verdict = run_inline_judge(models, MODEL_JUDGE, TRAJECTORY_JUDGE_PROMPT.format(
                tools=tools_str,
                conversation=format_history_compact(messages, max_chars=16000, tool_snippet=400)))
            judgment = {"explanation": verdict["explanation"], "rating": verdict["rating"]}
            status = verdict["success"]

        data["conversation_messages"] = json.dumps(messages, ensure_ascii=False, default=str)
        data["conversation_messages_raw"] = data["conversation_messages"]
        data["conversation_status"] = bool(status)
        data["hops_taken"] = int(hops)
        data["retrieval_log"] = json.dumps(env.retrieval_log, ensure_ascii=False)
        data["trajectory_judgment"] = json.dumps(judgment, ensure_ascii=False)
        data["conversation_metadata"] = json.dumps(
            {"n_retrievals": len(env.retrieval_log), "ends_on_answer": ends_on_answer,
             "retrieval_mode": cfg.retrieval_mode}, ensure_ascii=False)
        data["compression"] = env.comp_steps   # count of steps where the view was compressed
        if getattr(cfg, "name", None) and cfg.name != "conversation_messages":
            data[cfg.name] = data["conversation_messages"]
        return data

    def _skipped(self, data, cfg, reason, detail) -> dict:
        data["conversation_messages"] = json.dumps([], ensure_ascii=False)
        data["conversation_messages_raw"] = data["conversation_messages"]
        data["conversation_status"] = False
        data["hops_taken"] = 0
        data["retrieval_log"] = "[]"
        data["trajectory_judgment"] = json.dumps({"rating": "failure", "explanation": detail}, ensure_ascii=False)
        data["conversation_metadata"] = json.dumps({"skipped": True, "reason": reason}, ensure_ascii=False)
        data["retrieval_mode"] = cfg.retrieval_mode
        data["compression"] = 0
        data.setdefault("conversation_plan", "[]")
        data.setdefault("user_query", "")
        if getattr(cfg, "name", None) and cfg.name != "conversation_messages":
            data[cfg.name] = data["conversation_messages"]
        return data


def _sanitize_tool_calls(tool_calls: list, tools: List[Dict[str, Any]]) -> list:
    """Drop arguments not defined in each tool's schema (e.g. a stray top_k the model
    added) so the RECORDED call matches the schema exactly. Generic across any
    toolset; keeps trajectories free of schema-error noise while still requiring the
    real params (missing/wrong-type args are still caught by the verifier)."""
    allowed_by_name: Dict[str, set] = {}
    for td in tools:
        fn = (td["tool"] if isinstance(td, dict) and "tool" in td else td).get("function", {})
        allowed_by_name[fn.get("name")] = set((fn.get("parameters", {}) or {}).get("properties", {}) or {})
    cleaned = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        allowed = allowed_by_name.get(fn.get("name"))
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            # model emitted malformed JSON args -> coerce to valid empty JSON so the
            # recorded call stays API-valid (the endpoint 400s on invalid-JSON replay).
            cleaned.append({**tc, "function": {**fn, "arguments": "{}"}})
            continue
        if allowed is not None and isinstance(args, dict):
            filtered = {k: v for k, v in args.items() if k in allowed}
            if filtered != args:
                tc = {**tc, "function": {**fn, "arguments": json.dumps(filtered, ensure_ascii=False)}}
        cleaned.append(tc)
    return cleaned


def _assistant_msg(resp: Dict[str, Any], tool_calls: Optional[list] = None) -> Dict[str, Any]:
    """Build an assistant message that ALWAYS preserves the reasoning trace."""
    msg: Dict[str, Any] = {"role": "assistant", "content": resp.get("content", "") or "",
                           "reasoning_content": resp.get("reasoning_content", "") or ""}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _as_obj(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except (json.JSONDecodeError, TypeError):
        return {}


def _content(resp: Any) -> str:
    if isinstance(resp, dict):
        return resp.get("content", "") or ""
    return str(resp or "")
