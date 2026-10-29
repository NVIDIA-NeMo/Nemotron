"""Replay a published conversation against a candidate deterministically."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval import (
    CandidateApi,
    CandidateAuthenticationError,
    CandidateEligibility,
    CandidateInference,
    CandidateModelIdentity,
    CanonicalCallMatchGate,
    CommonEvaluationTaskSet,
    ConversationAuthorizationError,
    ConversationScript,
    ConversationScriptError,
    ConversationTransitionError,
    EligibleEvalPlan,
    EvalCandidate,
    EvalFileRef,
    EvalLimits,
    EvalScoringConfig,
    ModelFacingConversation,
    TurnMatch,
    build_conversation_script,
    candidate_identity_claim,
    conversation_driver,
    run_candidate_episode,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_cache import CandidateIOCache
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_client import (
    NativeFunctionCallingClient,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import CanonicalExportRow
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_projection import (
    CanonicalExportProjection,
    ProjectionSource,
    conversation_plan,
    derive_provenance,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

SOURCE_IDENTITY = "sha256:" + "1" * 64
OTHER_IDENTITY = "sha256:" + "2" * 64
BENCHMARK_HASH = "sha256:" + "b" * 64

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_balance",
            "description": "Read the balance of one account.",
            "parameters": {
                "type": "object",
                "$defs": {
                    "display_options": {
                        "type": "object",
                        "properties": {"format": {"type": "string", "default": "short"}},
                    }
                },
                "properties": {
                    "account_id": {"type": "string"},
                    "currency": {"type": "string", "default": "VND"},
                    "options": {"$ref": "#/$defs/display_options"},
                },
                "required": ["account_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_cards",
            "description": "List the cards of one account.",
            "parameters": {
                "type": "object",
                "properties": {"account_id": {"type": "string"}},
                "required": ["account_id"],
                "additionalProperties": False,
            },
        },
    },
]

METADATA_KEYS = {
    "base_task_id": "b1",
    "expt_name": "w55",
    "language": "vi",
    "profile_hash": "ph",
    "surface_source": "oracle",
}


# --------------------------------------------------------------------------------------
# Row builders. Each returns a published row plus the plan stage 12 derives from it, so
# every test drives the same projection a runner would.
# --------------------------------------------------------------------------------------


def _row(
    *,
    task_id: str = "t__1",
    messages: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    turn_policy: str = "single_turn",
    multi_turn: bool = False,
    call_order: str = "strict",
    call_order_prefix: int | None = None,
) -> CanonicalExportRow:
    required = [call["function_name"] for call in expected]
    return CanonicalExportRow(
        task_id=task_id,
        template_id="t1",
        variant_index=0,
        messages=messages,
        tools=TOOLS,
        expected_tool_calls=expected,
        success_assertions=(),
        fixture_refs=(),
        intent="check_balance",
        category="banking",
        difficulty="easy",
        required_tools=tuple(required),
        required_tools_fingerprint=canonical_json(sorted(required)),
        tools_present=("get_balance", "list_cards"),
        turn_policy=turn_policy,
        is_multi_turn=multi_turn,
        num_tool_calls=len(expected),
        call_order=call_order,
        call_order_prefix=call_order_prefix,
        system_prompt_id="sp1",
        tier="gold",
        gold_eligible=True,
        validated_by=("schema", "replay"),
        pack_id="banking_vn",
        pack_version="1.0.0",
        seed=7,
        paraphrase_model=None,
        paraphrase_model_canonical=None,
        held_out_hit=None,
        src="banking_vn:t1",
        metadata=dict(METADATA_KEYS),
    )


def _assistant_calls(calls: list[tuple[str, str, dict[str, Any]]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": canonical_json(arguments)},
            }
            for call_id, name, arguments in calls
        ],
    }


def _single_turn_row() -> CanonicalExportRow:
    """One user request, one call, one spoken answer."""
    return _row(
        messages=[
            {"role": "system", "content": "You are a bank assistant."},
            {"role": "user", "content": "Balance of account 1?"},
            _assistant_calls([("call_0", "get_balance", {"account_id": "1"})]),
            {"role": "tool", "tool_call_id": "call_0", "content": canonical_json({"balance": 500})},
            {"role": "assistant", "content": "Account 1 holds 500."},
        ],
        expected=[
            {
                "turn_index": 0,
                "call_group": 0,
                "position_in_group": 0,
                "function_name": "get_balance",
                "arguments": {"account_id": "1"},
            }
        ],
    )


def _missing_slot_row() -> CanonicalExportRow:
    """The model must ask for the account before it may call anything."""
    return _row(
        task_id="t__2",
        turn_policy="missing_slot",
        multi_turn=True,
        messages=[
            {"role": "system", "content": "You are a bank assistant."},
            {"role": "user", "content": "What is my balance?"},
            {"role": "assistant", "content": "Which account should I check?"},
            {"role": "user", "content": "Account 1."},
            _assistant_calls([("call_0", "get_balance", {"account_id": "1"})]),
            {"role": "tool", "tool_call_id": "call_0", "content": canonical_json({"balance": 500})},
            {"role": "assistant", "content": "Account 1 holds 500."},
        ],
        expected=[
            {
                "turn_index": 1,
                "call_group": 0,
                "position_in_group": 0,
                "function_name": "get_balance",
                "arguments": {"account_id": "1"},
            }
        ],
    )


def _parallel_row(
    *,
    call_order: str = "strict",
    call_order_prefix: int | None = None,
) -> CanonicalExportRow:
    """Two calls the model is expected to issue in one turn."""
    return _row(
        task_id="t__3",
        call_order=call_order,
        messages=[
            {"role": "system", "content": "You are a bank assistant."},
            {"role": "user", "content": "Balance and cards for account 1?"},
            _assistant_calls(
                [
                    ("call_0", "get_balance", {"account_id": "1"}),
                    ("call_1", "list_cards", {"account_id": "1"}),
                ]
            ),
            {"role": "tool", "tool_call_id": "call_0", "content": canonical_json({"balance": 500})},
            {"role": "tool", "tool_call_id": "call_1", "content": canonical_json({"cards": ["v1"]})},
            {"role": "assistant", "content": "Balance 500, one card."},
        ],
        expected=[
            {
                "turn_index": 0,
                "call_group": 0,
                "position_in_group": 0,
                "function_name": "get_balance",
                "arguments": {"account_id": "1"},
            },
            {
                "turn_index": 0,
                "call_group": 0,
                "position_in_group": 1,
                "function_name": "list_cards",
                "arguments": {"account_id": "1"},
            },
        ],
        call_order_prefix=call_order_prefix,
    )


def _prefix_row() -> CanonicalExportRow:
    """The first call is ordered; the final two may be permuted."""
    return _row(
        task_id="t__4",
        call_order="prefix",
        call_order_prefix=1,
        messages=[
            {"role": "system", "content": "You are a bank assistant."},
            {"role": "user", "content": "Check account 1, then its cards and account 2."},
            _assistant_calls(
                [
                    ("call_0", "get_balance", {"account_id": "1"}),
                    ("call_1", "list_cards", {"account_id": "1"}),
                    ("call_2", "get_balance", {"account_id": "2"}),
                ]
            ),
            {"role": "tool", "tool_call_id": "call_0", "content": canonical_json({"balance": 500})},
            {"role": "tool", "tool_call_id": "call_1", "content": canonical_json({"cards": ["v1"]})},
            {"role": "tool", "tool_call_id": "call_2", "content": canonical_json({"balance": 700})},
            {"role": "assistant", "content": "Done."},
        ],
        expected=[
            {
                "turn_index": 0,
                "call_group": 0,
                "position_in_group": 0,
                "function_name": "get_balance",
                "arguments": {"account_id": "1"},
            },
            {
                "turn_index": 0,
                "call_group": 0,
                "position_in_group": 1,
                "function_name": "list_cards",
                "arguments": {"account_id": "1"},
            },
            {
                "turn_index": 0,
                "call_group": 0,
                "position_in_group": 2,
                "function_name": "get_balance",
                "arguments": {"account_id": "2"},
            },
        ],
    )


def _nested_default_row() -> CanonicalExportRow:
    return _row(
        task_id="t__5",
        messages=[
            {"role": "system", "content": "You are a bank assistant."},
            {"role": "user", "content": "Show account 1 with default options."},
            _assistant_calls(
                [("call_0", "get_balance", {"account_id": "1", "options": {}})]
            ),
            {"role": "tool", "tool_call_id": "call_0", "content": canonical_json({"balance": 500})},
            {"role": "assistant", "content": "Account 1 holds 500."},
        ],
        expected=[
            {
                "turn_index": 0,
                "call_group": 0,
                "position_in_group": 0,
                "function_name": "get_balance",
                "arguments": {"account_id": "1", "options": {}},
            }
        ],
    )


def _projection(
    rows: tuple[CanonicalExportRow, ...],
    *,
    content_hash: str = BENCHMARK_HASH,
) -> CanonicalExportProjection:
    return CanonicalExportProjection(
        source=ProjectionSource(
            file="benchmark.parquet",
            content_hash=content_hash,
            rows=len(rows),
        ),
        provenance=derive_provenance(rows),
        rows=rows,
        plans=tuple(conversation_plan(row) for row in rows),
    )


def _source(
    rows: tuple[CanonicalExportRow, ...],
    *,
    identity: str = SOURCE_IDENTITY,
    content_hash: str = BENCHMARK_HASH,
) -> Any:
    return SimpleNamespace(
        evaluation_benchmark=SimpleNamespace(content_hash=content_hash, rows=len(rows)),
        task_ids=tuple(row.task_id for row in rows),
        verification_identity=identity,
    )


def _script(row: CanonicalExportRow, *, identity: str = SOURCE_IDENTITY) -> ConversationScript:
    rows = (row,)
    return build_conversation_script(
        _projection(rows),
        row.task_id,
        source=_source(rows, identity=identity),
    )


# --------------------------------------------------------------------------------------
# Authorization and runtime fixtures.
# --------------------------------------------------------------------------------------


def _candidate(alias: str = "candidate_a", *, revision: str = "a" * 40) -> EvalCandidate:
    return EvalCandidate(
        alias=alias,
        model="candidate-route",
        provider="nvidia",
        provider_api_version="v1",
        api=CandidateApi(base_url="https://candidate.example.com/v1", api_key_env="CANDIDATE_API_KEY"),
        model_identity=CandidateModelIdentity(source="huggingface", model="org/candidate", revision=revision),
        inference=CandidateInference(
            temperature=0.0,
            top_p=1.0,
            max_tokens=512,
            seed=42,
            tool_choice="auto",
            provider_extensions={},
        ),
    )


def _limits(*, max_turns: int = 6, episode_timeout_s: float = 30.0) -> EvalLimits:
    return EvalLimits(
        max_turns=max_turns,
        tool_timeout_s=1.0,
        candidate_timeout_s=5.0,
        episode_timeout_s=episode_timeout_s,
        max_parallel_tasks=2,
        max_retries=0,
    )


def _plan(
    *,
    task_ids: tuple[str, ...] = ("t__1", "t__2", "t__3", "t__4", "t__5"),
    identity: str = SOURCE_IDENTITY,
    scoring: EvalScoringConfig | None = None,
) -> EligibleEvalPlan:
    candidate = _candidate()
    policy = scoring or _scoring()
    return EligibleEvalPlan(
        eval_config_hash="sha256:" + "3" * 64,
        scoring_policy_hash=policy.scoring_policy_hash,
        source_verification_identity=identity,
        source_run_id="run-1",
        source_task_ids_hash="sha256:" + "4" * 64,
        enforce=True,
        on_violation="fail_run",
        comparison_set="common_intersection",
        candidates=(
            CandidateEligibility(
                alias=candidate.alias,
                identity=candidate_identity_claim(candidate),
                canonical_model_identity=candidate.canonical_model_identity,
                eligible_task_ids=task_ids,
            ),
        ),
        common=CommonEvaluationTaskSet(
            comparison_set="common_intersection",
            task_ids=task_ids,
            candidate_aliases=(candidate.alias,),
        ),
        publication_allowed=True,
    )


def _scoring(**overrides: Any) -> EvalScoringConfig:
    fields: dict[str, Any] = {
        "contract": EvalFileRef(path="/refs/bfcl-eval-scoring-contract.md", content_hash="sha256:" + "e" * 64),
        "argument_matching": "schema_then_canonical",
        "insert_declared_defaults": True,
        "respect_call_order": True,
        "respect_call_group": True,
        "allow_llm_repair": False,
        "task_success": "all_applicable_gates",
    }
    fields.update(overrides)
    return EvalScoringConfig(**fields)


class _Provider:
    """Answers each assistant turn from a pre-planned list of provider bodies."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.prompts: list[list[dict[str, Any]]] = []

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            self.prompts.append(json.loads(request.content)["messages"])
            reply = self.replies.pop(0)
            if isinstance(reply, int):
                return httpx.Response(reply, json={"error": "no"})
            return httpx.Response(200, json=reply)

        return httpx.MockTransport(handle)


def _says(text: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-text",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }


def _calls(calls: list[tuple[str, str, Any]]) -> dict[str, Any]:
    return {
        "id": "chatcmpl-calls",
        "object": "chat.completion",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
                        for call_id, name, arguments in calls
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def _drive(
    script: ConversationScript,
    replies: list[Any],
    tmp_path: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
    limits: EvalLimits | None = None,
    plan: EligibleEvalPlan | None = None,
    scoring: EvalScoringConfig | None = None,
    gate: Any = None,
    cache: CandidateIOCache | None = None,
    candidate: EvalCandidate | None = None,
) -> tuple[Any, _Provider]:
    monkeypatch.setenv("CANDIDATE_API_KEY", "secret-key")
    selected_candidate = candidate or _candidate()
    bounds = limits or _limits()
    provider = _Provider(replies)
    io_cache = cache if cache is not None else CandidateIOCache(tmp_path / "cache")

    async def execute() -> Any:
        client = NativeFunctionCallingClient(
            selected_candidate,
            bounds,
            io_cache,
            transport=provider.transport(),
        )
        try:
            return await run_candidate_episode(
                candidate=selected_candidate,
                limits=bounds,
                client=client,
                script=script,
                plan=plan or _plan(scoring=scoring),
                gate=gate or CanonicalCallMatchGate(scoring or _scoring()),
            )
        finally:
            await client.aclose()

    return asyncio.run(execute()), provider


# --------------------------------------------------------------------------------------
# Projection: what a published row says about how its conversation pauses.
# --------------------------------------------------------------------------------------


def test_a_single_turn_row_becomes_a_call_turn_and_a_spoken_turn() -> None:
    script = _script(_single_turn_row())

    assert [message.role for message in script.seed_messages] == ["system", "user"]
    assert [turn.expects_tool_calls for turn in script.turns] == [True, False]
    assert script.turns[0].calls[0].function_name == "get_balance"
    assert json.loads(script.turns[0].calls[0].recorded_result) == {"balance": 500}
    assert script.turns[-1].is_terminal
    assert script.user_turns == 1


def test_the_second_user_request_hangs_off_the_turn_that_must_earn_it() -> None:
    script = _script(_missing_slot_row())

    assert script.user_turns == 2
    # The model must ask for the account first; only that turn releases "Account 1."
    assert not script.turns[0].expects_tool_calls
    assert script.turns[0].releases_user_message is not None
    assert script.turns[0].releases_user_message.content == "Account 1."
    assert script.turns[1].expects_tool_calls
    assert script.turns[1].releases_user_message is None


def test_a_parallel_group_stays_one_turn_with_both_recorded_results() -> None:
    script = _script(_parallel_row())

    assert len(script.turns) == 2
    assert [call.function_name for call in script.turns[0].calls] == ["get_balance", "list_cards"]
    assert script.turns[0].call_group == 0


def test_the_seed_never_carries_an_assistant_or_tool_message() -> None:
    for row in (_single_turn_row(), _missing_slot_row(), _parallel_row()):
        script = _script(row)
        assert all(message.role in {"system", "user"} for message in script.seed_messages)
        assert sum(message.role == "user" for message in script.seed_messages) == 1


def test_a_row_whose_results_answer_the_wrong_call_is_not_replayable() -> None:
    row = _single_turn_row()
    broken = list(row.messages)
    broken[3] = broken[3].model_copy(update={"tool_call_id": "call_9"})
    broken_row = row.model_copy(update={"messages": tuple(broken)})
    rows = (broken_row,)

    with pytest.raises(ConversationScriptError, match="answers a different call"):
        build_conversation_script(
            _projection(rows),
            row.task_id,
            source=_source(rows),
        )


def test_a_task_absent_from_the_bound_projection_is_refused() -> None:
    rows = (_single_turn_row(),)

    with pytest.raises(ConversationScriptError, match="not present"):
        build_conversation_script(_projection(rows), "not_here", source=_source(rows))


def test_a_projection_cannot_be_stamped_with_an_unrelated_verified_source() -> None:
    rows = (_single_turn_row(),)

    with pytest.raises(ConversationAuthorizationError, match="complete benchmark"):
        build_conversation_script(
            _projection(rows, content_hash=BENCHMARK_HASH),
            rows[0].task_id,
            source=_source(rows, content_hash="sha256:" + "c" * 64),
        )


def test_the_script_hash_is_a_function_of_the_conversation_alone() -> None:
    first = _script(_single_turn_row())
    second = _script(_single_turn_row())
    elsewhere = _script(_single_turn_row(), identity=OTHER_IDENTITY)

    assert first.script_hash == second.script_hash
    assert first.script_hash != elsewhere.script_hash


# --------------------------------------------------------------------------------------
# The firewall: what may reach a prompt.
# --------------------------------------------------------------------------------------


def test_a_conversation_offers_no_way_to_append_a_gold_message() -> None:
    conversation = ModelFacingConversation(_script(_single_turn_row()))

    assert not hasattr(conversation, "append")
    assert conversation.provenance == ("seed", "seed")


def test_a_result_with_no_call_to_answer_is_refused() -> None:
    conversation = ModelFacingConversation(_script(_single_turn_row()))

    with pytest.raises(ConversationTransitionError, match="no call of the candidate's"):
        conversation.append_tool_results([("", '{"balance":500}')])


# --------------------------------------------------------------------------------------
# Driving: the happy paths.
# --------------------------------------------------------------------------------------


def test_a_model_that_matches_the_trace_reaches_the_end_of_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script(_single_turn_row())
    episode, provider = _drive(
        script,
        [
            _calls([("c1", "get_balance", '{"account_id":"1"}')]),
            _says("Account 1 holds 500."),
        ],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "completed"
    assert episode.succeeded
    assert episode.assistant_turns == 2
    assert episode.released_tool_results == 1
    assert episode.released_user_turns == 0
    # The second prompt shows the released result addressed to the candidate's own id.
    assert provider.prompts[1][-1] == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": canonical_json({"balance": 500}),
    }


def test_the_next_user_request_appears_only_after_the_turn_that_earns_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script(_missing_slot_row())
    episode, provider = _drive(
        script,
        [
            _says("Which account should I check?"),
            _calls([("c1", "get_balance", '{"account_id":"1"}')]),
            _says("Account 1 holds 500."),
        ],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "completed"
    assert episode.released_user_turns == 1
    # Turn 0 is asked without ever seeing the account number it has to request.
    assert [message["content"] for message in provider.prompts[0] if message["role"] == "user"] == [
        "What is my balance?"
    ]
    assert provider.prompts[1][-1]["content"] == "Account 1."


def test_a_model_that_calls_before_asking_never_sees_the_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script(_missing_slot_row())
    episode, provider = _drive(
        script,
        [_calls([("c1", "get_balance", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "candidate_mismatch"
    assert "answers this request in words" in episode.detail
    assert episode.released_tool_results == 0
    assert episode.released_user_turns == 0
    assert len(provider.prompts) == 1


def test_arbitrary_text_does_not_unlock_a_hidden_user_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode, provider = _drive(
        _script(_missing_slot_row()),
        [_says("I refuse to ask for the account.")],
        tmp_path,
        monkeypatch=monkeypatch,
        scoring=_scoring(intermediate_text_matching="verbatim"),
    )

    assert episode.status == "candidate_mismatch"
    assert "scripted intermediate assistant text" in episode.detail
    assert episode.released_user_turns == 0
    assert len(provider.prompts) == 1


def test_structural_matching_releases_the_next_request_for_a_paraphrase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode, provider = _drive(
        _script(_missing_slot_row()),
        [
            _says("Which one?"),
            _calls([("c1", "get_balance", '{"account_id":"1"}')]),
            _says("Account 1 holds 500."),
        ],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "completed"
    assert episode.released_user_turns == 1
    assert len(provider.prompts) == 3


def test_an_empty_intermediate_turn_never_unlocks_the_next_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Structural matching asks what the turn did, and a turn that said nothing
    # did not ask the user anything, so it has earned no further request.
    episode, provider = _drive(
        _script(_missing_slot_row()),
        [_says("")],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "candidate_mismatch"
    assert "no non-empty textual content" in episode.detail
    assert episode.released_user_turns == 0
    assert len(provider.prompts) == 1


def test_the_prompt_a_candidate_sees_is_only_ever_earned_material(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script(_missing_slot_row())
    _, provider = _drive(
        script,
        [
            _says("Which account should I check?"),
            _calls([("c1", "get_balance", '{"account_id":"1"}')]),
            _says("Account 1 holds 500."),
        ],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    for prompt in provider.prompts:
        for message in prompt:
            if message["role"] != "assistant":
                continue
            # Every assistant message in a prompt is one the candidate itself produced,
            # so no gold id from the published row can appear in it.
            for call in message.get("tool_calls") or ():
                assert not call["id"].startswith("call_")


# --------------------------------------------------------------------------------------
# Driving: the ways an episode stops.
# --------------------------------------------------------------------------------------


def test_wrong_arguments_stop_the_episode_and_say_which_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode, _ = _drive(
        _script(_single_turn_row()),
        [_calls([("c1", "get_balance", '{"account_id":"999"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "candidate_mismatch"
    assert "differing account_id" in episode.detail
    assert episode.released_tool_results == 0


def test_calling_the_wrong_tool_stops_the_episode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    episode, _ = _drive(
        _script(_single_turn_row()),
        [_calls([("c1", "list_cards", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "candidate_mismatch"
    assert "the trace calls get_balance" in episode.detail


def test_unparseable_arguments_are_a_mismatch_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode, _ = _drive(
        _script(_single_turn_row()),
        [_calls([("c1", "get_balance", "{not json")])],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "candidate_mismatch"
    assert "invalid_json arguments" in episode.detail


def test_a_tool_call_with_no_id_cannot_be_answered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    episode, _ = _drive(
        _script(_single_turn_row()),
        [_calls([("", "get_balance", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "unusable_tool_call_ids"
    assert episode.released_tool_results == 0


def test_duplicate_tool_call_ids_cannot_be_answered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode, _ = _drive(
        _script(_parallel_row()),
        [
            _calls(
                [
                    ("same", "get_balance", '{"account_id":"1"}'),
                    ("same", "list_cards", '{"account_id":"1"}'),
                ]
            )
        ],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "unusable_tool_call_ids"
    assert "duplicate" in episode.detail
    assert episode.released_tool_results == 0


def test_a_missing_tool_call_type_is_not_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = _calls([("c1", "get_balance", '{"account_id":"1"}')])
    del reply["choices"][0]["message"]["tool_calls"][0]["type"]
    episode, _ = _drive(
        _script(_single_turn_row()),
        [reply],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "candidate_mismatch"
    assert "only 'function'" in episode.detail


def test_a_provider_failure_ends_the_episode_as_a_call_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode, _ = _drive(_script(_single_turn_row()), [400], tmp_path, monkeypatch=monkeypatch)

    assert episode.status == "candidate_call_failed"
    assert episode.observed[0].call_status == "provider_rejected"
    assert not episode.observed[0].advanced


def test_a_rejected_credential_is_never_an_episode_the_driver_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused key is not a turn the candidate lost.

    A provider that rejects the request (400 above) is evidence about this turn,
    so the episode ends and carries it. A provider that refuses the credential is
    evidence about the run's configuration, and every remaining task would meet
    the same refusal. Returning an episode would spend the whole task set on it
    and aggregate the refusals into metrics that read like a score, so the driver
    must let the failure out instead.
    """
    with pytest.raises(CandidateAuthenticationError):
        _drive(_script(_single_turn_row()), [401], tmp_path, monkeypatch=monkeypatch)


def test_a_response_that_is_not_a_completion_is_the_models_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode, _ = _drive(
        _script(_single_turn_row()),
        [{"id": "x", "object": "chat.completion", "choices": []}],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "malformed_response"


def test_a_trace_longer_than_the_turn_budget_stops_at_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode, provider = _drive(
        _script(_single_turn_row()),
        [_calls([("c1", "get_balance", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
        limits=_limits(max_turns=1),
    )

    assert episode.status == "max_turns_exceeded"
    assert "limits.max_turns is 1" in episode.detail
    assert len(provider.prompts) == 1
    # The one turn it did take was fine; the episode failed on budget, not on the model.
    assert episode.observed[0].advanced


def test_an_episode_out_of_time_records_the_turn_it_could_not_ask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Spent:
        """Real time for the budget and the first turn, then far past the deadline."""

        def __init__(self) -> None:
            self.reads = 0

        def monotonic(self) -> float:
            self.reads += 1
            return time.monotonic() + (0.0 if self.reads <= 2 else 1e6)

    monkeypatch.setattr(conversation_driver, "time", _Spent())
    episode, provider = _drive(
        _script(_single_turn_row()),
        [_calls([("c1", "get_balance", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "episode_timeout"
    assert episode.assistant_turns == 1
    assert episode.observed[0].advanced
    # The turn it could not afford was never sent.
    assert len(provider.prompts) == 1


def test_timeout_before_the_first_send_fabricates_no_observed_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _SpentImmediately:
        def __init__(self) -> None:
            self.reads = 0

        def monotonic(self) -> float:
            self.reads += 1
            return time.monotonic() + (0.0 if self.reads == 1 else 1e6)

    monkeypatch.setattr(conversation_driver, "time", _SpentImmediately())
    episode, provider = _drive(
        _script(_single_turn_row()),
        [],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "episode_timeout"
    assert episode.assistant_turns == 0
    assert episode.observed == ()
    assert provider.prompts == []


# --------------------------------------------------------------------------------------
# Matching policy.
# --------------------------------------------------------------------------------------


def test_a_declared_default_spelled_out_is_neither_rewarded_nor_punished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode, _ = _drive(
        _script(_single_turn_row()),
        [
            _calls([("c1", "get_balance", '{"account_id":"1","currency":"VND"}')]),
            _says("Account 1 holds 500."),
        ],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "completed"


def test_without_default_insertion_the_same_call_no_longer_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode, _ = _drive(
        _script(_single_turn_row()),
        [_calls([("c1", "get_balance", '{"account_id":"1","currency":"VND"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
        scoring=_scoring(insert_declared_defaults=False),
    )

    assert episode.status == "candidate_mismatch"
    assert "unexpected currency" in episode.detail


def test_nested_defaults_behind_a_local_ref_are_inserted_recursively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode, _ = _drive(
        _script(_nested_default_row()),
        [
            _calls(
                [
                    (
                        "c1",
                        "get_balance",
                        '{"account_id":"1","options":{"format":"short"}}',
                    )
                ]
            ),
            _says("Account 1 holds 500."),
        ],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "completed"


def test_a_strict_row_requires_the_traces_order_inside_one_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    swapped = _calls(
        [
            ("c1", "list_cards", '{"account_id":"1"}'),
            ("c2", "get_balance", '{"account_id":"1"}'),
        ]
    )
    episode, _ = _drive(_script(_parallel_row()), [swapped], tmp_path, monkeypatch=monkeypatch)

    assert episode.status == "candidate_mismatch"
    assert "different order" in episode.detail


def test_an_unordered_row_accepts_the_same_calls_in_either_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _parallel_row(call_order="any")
    swapped = _calls(
        [
            ("c1", "list_cards", '{"account_id":"1"}'),
            ("c2", "get_balance", '{"account_id":"1"}'),
        ]
    )
    episode, provider = _drive(
        _script(row),
        [swapped, _says("Balance 500, one card.")],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "completed"
    assert episode.released_tool_results == 2
    # Each recorded result went to the call it actually answers, not to the one in
    # the same position.
    released = {
        message["tool_call_id"]: message["content"]
        for message in provider.prompts[1]
        if message["role"] == "tool"
    }
    assert json.loads(released["c1"]) == {"cards": ["v1"]}
    assert json.loads(released["c2"]) == {"balance": 500}


def test_prefix_order_only_locks_the_declared_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode, provider = _drive(
        _script(_prefix_row()),
        [
            _calls(
                [
                    ("c1", "get_balance", '{"account_id":"1"}'),
                    ("c2", "get_balance", '{"account_id":"2"}'),
                    ("c3", "list_cards", '{"account_id":"1"}'),
                ]
            ),
            _says("Done."),
        ],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "completed"
    released = {
        message["tool_call_id"]: json.loads(message["content"])
        for message in provider.prompts[1]
        if message["role"] == "tool"
    }
    assert released["c2"] == {"balance": 700}
    assert released["c3"] == {"cards": ["v1"]}


def test_prefix_order_still_rejects_the_wrong_first_required_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode, _ = _drive(
        _script(_prefix_row()),
        [
            _calls(
                [
                    ("c1", "list_cards", '{"account_id":"1"}'),
                    ("c2", "get_balance", '{"account_id":"2"}'),
                    ("c3", "get_balance", '{"account_id":"1"}'),
                ]
            )
        ],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "candidate_mismatch"
    assert "required-tool prefix" in episode.detail


def test_a_turn_with_the_wrong_number_of_calls_has_no_faithful_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode, _ = _drive(
        _script(_parallel_row()),
        [_calls([("c1", "get_balance", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "candidate_mismatch"
    assert "holds 2 call(s)" in episode.detail


def test_speaking_when_the_trace_speaks_requires_saying_something(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = {
        "id": "chatcmpl-empty",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": None}, "finish_reason": "stop"}],
    }
    episode, _ = _drive(
        _script(_single_turn_row()),
        [_calls([("c1", "get_balance", '{"account_id":"1"}')]), empty],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "candidate_mismatch"
    assert "neither content nor a tool call" in episode.detail


# --------------------------------------------------------------------------------------
# Authorization and determinism.
# --------------------------------------------------------------------------------------


def test_a_task_the_gate_excluded_is_never_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConversationAuthorizationError, match="did not authorize"):
        _drive(
            _script(_single_turn_row()),
            [_says("hi")],
            tmp_path,
            monkeypatch=monkeypatch,
            plan=_plan(task_ids=("t__2",)),
        )


def test_a_script_from_another_publication_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConversationAuthorizationError, match="different verified benchmark"):
        _drive(
            _script(_single_turn_row(), identity=OTHER_IDENTITY),
            [_says("hi")],
            tmp_path,
            monkeypatch=monkeypatch,
        )


def test_a_different_model_cannot_reuse_an_authorized_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ConversationAuthorizationError, match="changed after"):
        _drive(
            _script(_single_turn_row()),
            [_says("hi")],
            tmp_path,
            monkeypatch=monkeypatch,
            candidate=_candidate(revision="b" * 40),
        )


def test_two_runs_of_the_same_episode_agree_on_every_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script(_single_turn_row())
    replies = [
        _calls([("c1", "get_balance", '{"account_id":"1"}')]),
        _says("Account 1 holds 500."),
    ]
    first, _ = _drive(script, list(replies), tmp_path / "a", monkeypatch=monkeypatch)
    second, _ = _drive(script, list(replies), tmp_path / "b", monkeypatch=monkeypatch)

    assert first.episode_hash == second.episode_hash
    assert not first.replayed and not second.replayed


def test_a_second_run_against_a_warm_cache_is_a_replay_of_the_same_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script(_single_turn_row())
    replies = [
        _calls([("c1", "get_balance", '{"account_id":"1"}')]),
        _says("Account 1 holds 500."),
    ]
    cache = CandidateIOCache(tmp_path / "shared")
    paid, _ = _drive(script, list(replies), tmp_path, monkeypatch=monkeypatch, cache=cache)
    # No replies left: every turn must come back out of the cache or the run fails.
    replayed, provider = _drive(script, [], tmp_path, monkeypatch=monkeypatch, cache=cache)

    assert provider.prompts == []
    assert replayed.episode_hash == paid.episode_hash
    assert replayed.replayed and not paid.replayed


def test_the_episode_record_names_the_plan_the_task_was_authorized_under(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan()
    episode, _ = _drive(
        _script(_single_turn_row()),
        [
            _calls([("c1", "get_balance", '{"account_id":"1"}')]),
            _says("Account 1 holds 500."),
        ],
        tmp_path,
        monkeypatch=monkeypatch,
        plan=plan,
    )

    document = episode.as_document()
    assert document["plan_identity"] == plan.plan_identity
    assert document["source_verification_identity"] == SOURCE_IDENTITY
    assert document["episode_hash"] == episode.episode_hash
    assert [event["kind"] for event in document["events"]][0] == "seed"


def test_an_injected_gate_is_what_decides_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _RefuseEverything:
        def evaluate(self, turn: Any, response: Any, *, script: Any) -> TurnMatch:
            return TurnMatch(advanced=False, detail="this gate approves of nothing")

    episode, _ = _drive(
        _script(_single_turn_row()),
        [_calls([("c1", "get_balance", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
        gate=_RefuseEverything(),
    )

    assert episode.status == "candidate_mismatch"
    assert "approves of nothing" in episode.detail
