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

import json
import random
from types import SimpleNamespace

import pytest

from long_context_chat_sdg.conversation import tools as tools_module
from long_context_chat_sdg.conversation.tools import ToolEnvironment
from long_context_chat_sdg.conversation.verifiers import ToolCallVerifier
from pipeline import validate_retrieval_config


SEARCH_TOOL = {
    "function": {
        "name": "search",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
}


def _cfg(mode="simulated"):
    return SimpleNamespace(
        retrieval_mode=mode,
        retrieval_tools=["search"],
        query_arg_names=["query"],
        top_k=2,
    )


def test_http_generation_requires_endpoint():
    with pytest.raises(SystemExit, match="RETRIEVAL_ENDPOINT"):
        validate_retrieval_config({"retrieval": {"mode": "http", "endpoint": ""}})
    assert validate_retrieval_config(
        {"retrieval": {"mode": "http", "endpoint": "http://retriever.example/search"}}
    ) == "http"


def test_simulated_mode_is_explicit_and_does_not_need_endpoint():
    assert validate_retrieval_config({"retrieval": {"mode": "simulated"}}) == "simulated"
    with pytest.raises(SystemExit, match="must be 'http' or 'simulated'"):
        validate_retrieval_config({"retrieval": {"mode": "automatic"}})


def test_simulated_retrieval_is_normalized_and_provenanced(monkeypatch):
    monkeypatch.setattr(tools_module, "call_llm", lambda *_args, **_kwargs: {
        "content": json.dumps({"results": [
            {"text": "First synthetic passage.", "score": 0.9, "doc_id": "demo-a"},
            {"text": "Second synthetic passage.", "score": "0.8"},
        ]})
    })
    env = ToolEnvironment(_cfg(), client=None, tools=[SEARCH_TOOL])
    call = {"function": {"name": "search", "arguments": json.dumps({"query": "topic"})}}

    content, was_retrieval = env.respond(call, models={}, user_query="underlying", rng=random.Random(7))
    payload = json.loads(content)

    assert was_retrieval is True
    assert payload["simulated"] is True
    assert len(payload["results"]) == 2
    assert all(item["simulated"] is True for item in payload["results"])
    assert all(item["id"].startswith("h") and len(item["id"]) == 13 for item in payload["results"])
    assert env.retrieval_log == [{
        "query": "topic",
        "ids": [item["id"] for item in payload["results"]],
        "new": 2,
        "mode": "simulated",
    }]


def test_http_mode_never_silently_simulates(monkeypatch):
    monkeypatch.setattr(tools_module, "call_llm", lambda *_args, **_kwargs: pytest.fail(
        "HTTP retrieval must not fall back to the LLM simulator"
    ))
    env = ToolEnvironment(_cfg("http"), client=None, tools=[SEARCH_TOOL])
    call = {"function": {"name": "search", "arguments": '{"query": "topic"}'}}
    with pytest.raises(RuntimeError, match="without an HTTP client"):
        env.respond(call, models={}, user_query="underlying", rng=random.Random(7))


def test_simulated_trajectory_can_be_exported_with_provenance():
    pytest.importorskip("data_designer")
    from evaluate import _for_sft, _objective, _summary

    chunk_id = "h123456789abc"
    call = {"id": "call-1", "function": {"name": "search", "arguments": '{"query": "topic"}'}}
    row = {
        "retrieval_mode": "simulated",
        "tools": [SEARCH_TOOL],
        "messages": [
            {"role": "user", "content": "Explain the topic."},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call-1", "content": json.dumps({
                "results": [{"id": chunk_id, "text": "Synthetic supporting evidence.",
                             "simulated": True}],
                "simulated": True,
            })},
            {"role": "assistant", "content": f"The answer follows from [{chunk_id}]."},
        ],
    }
    objective = _objective(row, ToolCallVerifier())
    assert objective["objective_ok"] is True
    assert _for_sft(row, keep_reasoning=True)["retrieval_mode"] == "simulated"

    scored = [{**row, "eval": {**objective, "grounding_overlap": 0.1}}]
    summary = _summary(scored, [row], SimpleNamespace(judge=False))
    assert summary["retrieval_mode_counts"] == {"simulated": 1}
    assert summary["kept_retrieval_mode_counts"] == {"simulated": 1}


def test_citation_integrity_uses_external_retriever_ids():
    pytest.importorskip("data_designer")
    from evaluate import _citation_integrity

    messages = [
        {"role": "tool", "content": json.dumps({
            "results": [{"id": "doc-17/chunk_3", "text": "evidence"}],
        })},
        {"role": "assistant", "content": "Supported [doc-17/chunk_3], not [invented-9]."},
    ]
    result = _citation_integrity({"messages": messages})
    assert result["cited_ids"] == 2
    assert result["fabricated"] == ["invented-9"]
    assert result["citation_ok"] is False


def test_empty_retrieval_evidence_fails_objective_and_is_reported():
    pytest.importorskip("data_designer")
    from evaluate import _objective, _summary

    call = {"id": "call-empty", "function": {"name": "search", "arguments": '{"query":"topic"}'}}
    row = {
        "retrieval_mode": "http",
        "retrieval_tools": ["search"],
        "tools": [SEARCH_TOOL],
        "messages": [
            {"role": "user", "content": "Explain."},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "call-empty", "content": '{"results":[]}'},
            {"role": "assistant", "content": "An unsupported answer."},
        ],
    }
    objective = _objective(row, ToolCallVerifier())
    assert objective["objective_ok"] is False
    assert objective["retrievals"] == 0
    assert objective["empty_retrievals"] == 1
    scored = [{**row, "eval": {**objective, "grounding_overlap": 0.0}}]
    summary = _summary(scored, [], SimpleNamespace(judge=False, strip_reasoning=False))
    assert summary["empty_evidence_rows"] == 1
    assert summary["retrieval_results_total"] == 0


def test_results_from_non_retrieval_tool_do_not_satisfy_grounding_gate():
    pytest.importorskip("data_designer")
    from evaluate import _objective

    memory_tool = {
        "function": {"name": "memory_read", "parameters": {"type": "object", "properties": {}}}
    }
    call = {"id": "memory-call", "function": {"name": "memory_read", "arguments": "{}"}}
    row = {
        "retrieval_tools": ["search"],
        "tools": [SEARCH_TOOL, memory_tool],
        "messages": [
            {"role": "user", "content": "Explain."},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "memory-call",
             "content": '{"results":[{"id":"not-evidence","text":"memory"}]}'},
            {"role": "assistant", "content": "Answer."},
        ],
    }
    objective = _objective(row, ToolCallVerifier())
    assert objective["retrievals"] == 0
    assert objective["objective_ok"] is False


def test_eval_cache_separates_modes_models_and_inputs_but_not_thresholds():
    pytest.importorskip("data_designer")
    from evaluate import _eval_cache_key

    rows = [{"messages": [], "tools": []}]
    objective = _eval_cache_key(rows, judge=False)
    judge_a = _eval_cache_key(rows, judge=True, model="a", endpoint="http://judge")
    assert objective != judge_a
    assert judge_a == _eval_cache_key(rows, judge=True, model="a", endpoint="http://judge")
    assert judge_a != _eval_cache_key(rows, judge=True, model="b", endpoint="http://judge")
    assert judge_a != _eval_cache_key(rows + [{"messages": [1]}], judge=True,
                                      model="a", endpoint="http://judge")


def test_summary_records_reasoning_policy_and_honest_judge_state():
    pytest.importorskip("data_designer")
    from evaluate import _summary

    row = {"messages": [], "retrieval_mode": "http", "eval": {
        "objective_ok": True, "citation_ok": True, "reasoning_citation_ok": False,
        "grounding_overlap": 0.1, "rubric": None,
    }}
    summary = _summary([row], [], SimpleNamespace(judge=True, strip_reasoning=True))
    assert summary["judge_requested"] is True
    assert summary["judged"] is False
    assert summary["judged_rows"] == 0
    assert summary["strip_reasoning"] is True
    assert summary["reasoning_citation_gate_applied"] is False
