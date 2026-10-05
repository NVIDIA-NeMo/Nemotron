"""Turn a recorded episode into a gate-by-gate task score.

Every score in this file is taken over evidence a driver actually produced: the
tests drive an episode against a mock provider and then score it, so a claim
about a gate is a claim about the pipeline rather than about a hand-built object.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from nemotron.steps.byob.runtime.benchmark_families.bfcl import eval as eval_surface
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval import (
    EXPORT_METRIC_BY_GATE,
    REASON_CODE_NAMESPACES,
    SCORING_GATES,
    CandidateApi,
    CandidateEligibility,
    CandidateEpisode,
    CandidateInference,
    CandidateModelIdentity,
    CanonicalCallMatchGate,
    CommonEvaluationTaskSet,
    ConversationScript,
    EligibleEvalPlan,
    EvalCandidate,
    EvalFileRef,
    EvalLimits,
    EvalScoringConfig,
    GateResult,
    TraceEvidenceError,
    TraceScoringPolicyError,
    TraceTaskScore,
    build_conversation_script,
    candidate_identity_claim,
    parse_observed_trace,
    run_candidate_episode,
    score_trace_episode,
    trace_failure_records,
    trace_task_result,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.call_comparison import (
    compare_arguments,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_cache import CandidateIOCache
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_client import (
    NativeFunctionCallingClient,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    EXPORT_SCORING_METRICS,
    CanonicalExportRow,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_projection import (
    CanonicalExportProjection,
    ProjectionSource,
    conversation_plan,
    derive_provenance,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

SOURCE_IDENTITY = "sha256:" + "1" * 64
BENCHMARK_HASH = "sha256:" + "b" * 64
CONTRACT_HASH = "sha256:" + "e" * 64
OTHER_CONTRACT_HASH = "sha256:" + "f" * 64
EVAL_CONFIG_HASH = "sha256:" + "3" * 64

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_balance",
            "description": "Read the balance of one account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string"},
                    "currency": {"type": "string", "default": "VND"},
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
    {
        "type": "function",
        "function": {
            "name": "close_account",
            "description": "Close one account, with a confirmation the caller must pass.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string"},
                    # Defaulted *and* required: filling the default in before
                    # validating would launder a missing argument into a pass.
                    "confirm": {"type": "boolean", "default": False},
                },
                "required": ["account_id", "confirm"],
                "additionalProperties": False,
            },
        },
    },
]

METADATA_KEYS = {
    "base_task_id": "b1",
    "expt_name": "scoring",
    "language": "vi",
    "profile_hash": "ph",
    "surface_source": "oracle",
}

TASK_IDS = ("t__1", "t__2", "t__3", "t__4", "t__5", "t__6")


# --------------------------------------------------------------------------------------
# Published rows, and the projection a runner would build a script from.
# --------------------------------------------------------------------------------------


def _row(
    *,
    task_id: str,
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
        tools_present=("get_balance", "list_cards", "close_account"),
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
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": canonical_json(arguments)}}
            for call_id, name, arguments in calls
        ],
    }


def _gold(turn: int, position: int, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_index": turn,
        "call_group": 0,
        "position_in_group": position,
        "function_name": name,
        "arguments": arguments,
    }


def _single_turn_row() -> CanonicalExportRow:
    """One request, one call, one spoken answer."""
    return _row(
        task_id="t__1",
        messages=[
            {"role": "system", "content": "You are a bank assistant."},
            {"role": "user", "content": "Balance of account 1?"},
            _assistant_calls([("call_0", "get_balance", {"account_id": "1"})]),
            {"role": "tool", "tool_call_id": "call_0", "content": canonical_json({"balance": 500})},
            {"role": "assistant", "content": "Account 1 holds 500."},
        ],
        expected=[_gold(0, 0, "get_balance", {"account_id": "1"})],
    )


def _missing_slot_row() -> CanonicalExportRow:
    """The model must ask which account before it may call anything."""
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
        expected=[_gold(1, 0, "get_balance", {"account_id": "1"})],
    )


def _parallel_row(*, call_order: str = "strict") -> CanonicalExportRow:
    """Two calls the trace issues in one turn, in a declared order."""
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
            _gold(0, 0, "get_balance", {"account_id": "1"}),
            _gold(0, 1, "list_cards", {"account_id": "1"}),
        ],
    )


def _prefix_row() -> CanonicalExportRow:
    """The first required tool is ordered; what follows may be permuted."""
    return _row(
        task_id="t__4",
        call_order="prefix",
        call_order_prefix=1,
        messages=[
            {"role": "system", "content": "You are a bank assistant."},
            {"role": "user", "content": "Check account 1, then its cards."},
            _assistant_calls(
                [
                    ("call_0", "get_balance", {"account_id": "1"}),
                    ("call_1", "list_cards", {"account_id": "1"}),
                ]
            ),
            {"role": "tool", "tool_call_id": "call_0", "content": canonical_json({"balance": 500})},
            {"role": "tool", "tool_call_id": "call_1", "content": canonical_json({"cards": ["v1"]})},
            {"role": "assistant", "content": "Done."},
        ],
        expected=[
            _gold(0, 0, "get_balance", {"account_id": "1"}),
            _gold(0, 1, "list_cards", {"account_id": "1"}),
        ],
    )


def _irrelevant_row() -> CanonicalExportRow:
    """A request no declared tool can serve, so the trace answers in words."""
    return _row(
        task_id="t__5",
        turn_policy="irrelevant",
        messages=[
            {"role": "system", "content": "You are a bank assistant."},
            {"role": "user", "content": "What is the weather in Hanoi?"},
            {"role": "assistant", "content": "I can only help with banking."},
        ],
        expected=[],
    )


def _required_default_row() -> CanonicalExportRow:
    """The gold call spells out a flag the schema both defaults and requires."""
    return _row(
        task_id="t__6",
        messages=[
            {"role": "system", "content": "You are a bank assistant."},
            {"role": "user", "content": "Close account 1."},
            _assistant_calls([("call_0", "close_account", {"account_id": "1", "confirm": True})]),
            {"role": "tool", "tool_call_id": "call_0", "content": canonical_json({"closed": True})},
            {"role": "assistant", "content": "Account 1 is closed."},
        ],
        expected=[_gold(0, 0, "close_account", {"account_id": "1", "confirm": True})],
    )


def _projection(rows: tuple[CanonicalExportRow, ...]) -> CanonicalExportProjection:
    return CanonicalExportProjection(
        source=ProjectionSource(file="benchmark.parquet", content_hash=BENCHMARK_HASH, rows=len(rows)),
        provenance=derive_provenance(rows),
        rows=rows,
        plans=tuple(conversation_plan(row) for row in rows),
    )


def _source(rows: tuple[CanonicalExportRow, ...]) -> Any:
    return SimpleNamespace(
        evaluation_benchmark=SimpleNamespace(content_hash=BENCHMARK_HASH, rows=len(rows)),
        task_ids=tuple(row.task_id for row in rows),
        verification_identity=SOURCE_IDENTITY,
    )


def _script(row: CanonicalExportRow) -> ConversationScript:
    rows = (row,)
    return build_conversation_script(_projection(rows), row.task_id, source=_source(rows))


# --------------------------------------------------------------------------------------
# Driving an episode, so every score is taken over evidence the driver produced.
# --------------------------------------------------------------------------------------


def _candidate() -> EvalCandidate:
    return EvalCandidate(
        alias="candidate_a",
        model="candidate-route",
        provider="nvidia",
        provider_api_version="v1",
        api=CandidateApi(base_url="https://candidate.example.com/v1", api_key_env="CANDIDATE_API_KEY"),
        model_identity=CandidateModelIdentity(source="huggingface", model="org/candidate", revision="a" * 40),
        inference=CandidateInference(
            temperature=0.0,
            top_p=1.0,
            max_tokens=512,
            seed=42,
            tool_choice="auto",
            provider_extensions={},
        ),
    )


def _limits(*, max_turns: int = 6) -> EvalLimits:
    return EvalLimits(
        max_turns=max_turns,
        tool_timeout_s=1.0,
        candidate_timeout_s=5.0,
        episode_timeout_s=30.0,
        max_parallel_tasks=2,
        max_retries=0,
    )


def _plan(*, scoring: EvalScoringConfig | None = None) -> EligibleEvalPlan:
    candidate = _candidate()
    policy = scoring or _scoring()
    return EligibleEvalPlan(
        eval_config_hash=EVAL_CONFIG_HASH,
        scoring_policy_hash=policy.scoring_policy_hash,
        source_verification_identity=SOURCE_IDENTITY,
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
                eligible_task_ids=TASK_IDS,
            ),
        ),
        common=CommonEvaluationTaskSet(
            comparison_set="common_intersection",
            task_ids=TASK_IDS,
            candidate_aliases=(candidate.alias,),
        ),
        publication_allowed=True,
    )


def _scoring(**overrides: Any) -> EvalScoringConfig:
    fields: dict[str, Any] = {
        "contract": EvalFileRef(path="/refs/bfcl-eval-scoring-contract.md", content_hash=CONTRACT_HASH),
        "argument_matching": "schema_then_canonical",
        "insert_declared_defaults": True,
        "respect_call_order": True,
        "respect_call_group": True,
        "allow_llm_repair": False,
        "task_success": "all_applicable_gates",
    }
    fields.update(overrides)
    return EvalScoringConfig(**fields)


def _says(text: Any, *, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": "chatcmpl-text",
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}],
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


def _episode(
    script: ConversationScript,
    replies: list[Any],
    tmp_path: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
    scoring: EvalScoringConfig | None = None,
    limits: EvalLimits | None = None,
) -> CandidateEpisode:
    monkeypatch.setenv("CANDIDATE_API_KEY", "secret-key")
    candidate = _candidate()
    bounds = limits or _limits()
    policy = scoring or _scoring()
    pending = list(replies)

    def handle(request: httpx.Request) -> httpx.Response:
        reply = pending.pop(0)
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error": "no"})
        return httpx.Response(200, json=reply)

    async def execute() -> CandidateEpisode:
        client = NativeFunctionCallingClient(
            candidate,
            bounds,
            CandidateIOCache(tmp_path / "cache"),
            transport=httpx.MockTransport(handle),
        )
        try:
            return await run_candidate_episode(
                candidate=candidate,
                limits=bounds,
                client=client,
                script=script,
                plan=_plan(scoring=policy),
                gate=CanonicalCallMatchGate(policy),
            )
        finally:
            await client.aclose()

    return asyncio.run(execute())


def _score(
    row: CanonicalExportRow,
    replies: list[Any],
    tmp_path: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
    scoring: EvalScoringConfig | None = None,
    limits: EvalLimits | None = None,
) -> tuple[TraceTaskScore, CandidateEpisode]:
    policy = scoring or _scoring()
    script = _script(row)
    episode = _episode(script, replies, tmp_path, monkeypatch=monkeypatch, scoring=policy, limits=limits)
    return (
        score_trace_episode(
            episode=episode,
            script=script,
            scoring=policy,
            plan=_plan(scoring=policy),
        ),
        episode,
    )


def _outcomes(score: TraceTaskScore) -> dict[str, str]:
    return {gate.gate: gate.outcome for gate in score.gates}


def test_public_eval_surface_exposes_only_the_authorized_trace_scorer() -> None:
    assert eval_surface.score_trace_episode is score_trace_episode
    assert not hasattr(eval_surface, "score_normalized_trace")
    assert not hasattr(eval_surface, "NormalizedTraceScore")


# --------------------------------------------------------------------------------------
# The parser: the episode and the script must be two halves of one replay.
# --------------------------------------------------------------------------------------


def test_an_episode_recorded_for_another_task_is_not_scored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    episode = _episode(
        _script(_single_turn_row()),
        [_calls([("c1", "get_balance", '{"account_id":"1"}')]), _says("Account 1 holds 500.")],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(TraceEvidenceError, match="different task"):
        parse_observed_trace(episode, _script(_parallel_row()))


def test_an_episode_of_a_different_conversation_is_not_scored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same task re-projected into a different trace is a different question."""
    row = _single_turn_row()
    episode = _episode(
        _script(row),
        [_calls([("c1", "get_balance", '{"account_id":"1"}')]), _says("Account 1 holds 500.")],
        tmp_path,
        monkeypatch=monkeypatch,
    )
    reworded = list(row.messages)
    reworded[1] = reworded[1].model_copy(update={"content": "Tell me the balance of account 1."})
    elsewhere = _script(row.model_copy(update={"messages": tuple(reworded)}))

    with pytest.raises(TraceEvidenceError, match="different conversation"):
        parse_observed_trace(episode, elsewhere)


def test_an_episode_cannot_be_restamped_with_another_source_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script(_single_turn_row())
    episode = _episode(
        script,
        [_calls([("c1", "get_balance", '{"account_id":"1"}')]), _says("Account 1 holds 500.")],
        tmp_path,
        monkeypatch=monkeypatch,
    )
    restamped = episode.model_copy(update={"source_verification_identity": "sha256:" + "9" * 64})

    with pytest.raises(TraceEvidenceError, match="different verified benchmark"):
        parse_observed_trace(restamped, script)


def test_completed_evidence_cannot_hide_scripted_turns_it_never_observed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script(_single_turn_row())
    episode = _episode(
        script,
        [_calls([("c1", "list_cards", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(TraceEvidenceError, match="claims completion"):
        parse_observed_trace(
            episode.model_copy(update={"status": "completed"}),
            script,
        )


def test_the_parser_flattens_each_call_to_where_it_appeared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = _script(_parallel_row())
    episode = _episode(
        script,
        [
            _calls(
                [
                    ("c1", "get_balance", '{"account_id":"1"}'),
                    ("c2", "list_cards", '{"account_id":"1"}'),
                ]
            ),
            _says("Balance 500, one card."),
        ],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    trace = parse_observed_trace(episode, script)

    assert [turn.kind for turn in trace.turns] == ["tool_calls", "text"]
    assert [(call.turn_index, call.position_in_turn, call.function_name) for call in trace.turns[0].calls] == [
        (0, 0, "get_balance"),
        (0, 1, "list_cards"),
    ]
    assert all(call.arguments_status == "valid_object" for call in trace.turns[0].calls)
    assert trace.observed_calls == 2
    assert trace.reached_the_end
    assert trace.unsent_turn_indexes == ()
    assert not trace.non_candidate_stop


def test_a_turn_the_episode_never_reached_is_listed_rather_than_invented(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script(_single_turn_row())
    episode = _episode(
        script,
        [_calls([("c1", "list_cards", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    trace = parse_observed_trace(episode, script)

    assert len(trace.turns) == 1
    assert trace.scripted_turns == 2
    assert trace.unsent_turn_indexes == (1,)
    assert not trace.reached_the_end


def test_arguments_that_never_parsed_stay_unparsed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = _script(_single_turn_row())
    episode = _episode(
        script,
        [_calls([("c1", "get_balance", "{not json")])],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    call = parse_observed_trace(episode, script).turns[0].calls[0]

    assert call.arguments_status == "invalid_json"
    assert call.parsed_arguments is None


# --------------------------------------------------------------------------------------
# Policy: a mode this scorer cannot honour is refused, not approximated.
# --------------------------------------------------------------------------------------


def test_a_policy_that_asks_for_repair_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = _script(_single_turn_row())
    episode = _episode(
        script,
        [_calls([("c1", "get_balance", '{"account_id":"1"}')]), _says("Account 1 holds 500.")],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(TraceScoringPolicyError, match="repaired"):
        score_trace_episode(
            episode=episode,
            script=script,
            scoring=_scoring(allow_llm_repair=True),
            plan=_plan(),
        )


def test_a_policy_that_asks_for_pack_assertions_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = _script(_single_turn_row())
    episode = _episode(
        script,
        [_calls([("c1", "get_balance", '{"account_id":"1"}')]), _says("Account 1 holds 500.")],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(TraceScoringPolicyError, match="assertions"):
        score_trace_episode(
            episode=episode,
            script=script,
            scoring=_scoring(task_success="assertions_only"),
            plan=_plan(),
        )


# --------------------------------------------------------------------------------------
# A trace answered exactly.
# --------------------------------------------------------------------------------------


def test_an_episode_that_answers_the_trace_passes_every_applicable_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, episode = _score(
        _single_turn_row(),
        [_calls([("c1", "get_balance", '{"account_id":"1"}')]), _says("Account 1 holds 500.")],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "completed"
    assert score.task_success
    assert score.failed_gates == ()
    assert _outcomes(score) == {
        "tool_selection": "passed",
        "arguments": "passed",
        "schema_valid": "passed",
        "call_grouping": "passed",
        # One call has no order to respect.
        "call_ordering": "not_applicable",
        "text_turn": "passed",
        "trace_completion": "passed",
    }
    assert (score.expected_calls, score.observed_calls, score.matched_calls) == (1, 1, 1)
    assert score.scope == "trace"


def test_a_score_reports_every_gate_the_contract_defines_in_contract_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, _ = _score(
        _single_turn_row(),
        [_calls([("c1", "get_balance", '{"account_id":"1"}')]), _says("Account 1 holds 500.")],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert tuple(gate.gate for gate in score.gates) == SCORING_GATES
    assert all(gate.reason_code.startswith(f"{gate.gate}.") for gate in score.gates)


def test_a_candidate_supplied_default_absent_from_gold_is_equivalent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, _ = _score(
        _single_turn_row(),
        [
            _calls([("c1", "get_balance", '{"account_id":"1","currency":"VND"}')]),
            _says("Account 1 holds 500."),
        ],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert score.task_success
    assert score.gate("arguments").outcome == "passed"


def test_defaults_fill_candidate_omissions_but_never_erase_candidate_claims() -> None:
    scoring = _scoring()

    omitted = compare_arguments(
        {"account_id": "1", "currency": "VND"},
        {"account_id": "1"},
        function_name="get_balance",
        tools=TOOLS,
        scoring=scoring,
    )
    unearned_confirmation = compare_arguments(
        {"account_id": "1"},
        {"account_id": "1", "confirm": True},
        function_name="close_account",
        tools=TOOLS,
        scoring=scoring,
    )

    assert omitted.empty
    assert unearned_confirmation.differing == ("confirm",)


def test_a_nondefault_argument_differs_from_the_gold_omission_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Candidate-supplied fields are claims made by the candidate. Ignoring one
    # merely because the schema declares a default can hide consequential values
    # such as an unearned confirmation flag.
    score, _ = _score(
        _single_turn_row(),
        [
            _calls([("c1", "get_balance", '{"account_id":"1","currency":"USD"}')]),
            _says("Account 1 holds 500."),
        ],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert score.gate("arguments").outcome == "failed"
    assert not score.task_success


def test_a_multi_turn_trace_scores_the_text_turn_that_earned_the_next_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, episode = _score(
        _missing_slot_row(),
        [
            _says("Which account should I check?"),
            _calls([("c1", "get_balance", '{"account_id":"1"}')]),
            _says("Account 1 holds 500."),
        ],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "completed"
    assert score.task_success
    assert [turn.kind for turn in score.turns] == ["text", "tool_calls", "text"]
    assert score.turns[0].text_matched
    assert score.turns[1].calls[0].gold_call_index == 0


# --------------------------------------------------------------------------------------
# Selection and arguments: coverage of the gold trace.
# --------------------------------------------------------------------------------------


def test_the_wrong_tool_fails_selection_and_names_the_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    score, episode = _score(
        _single_turn_row(),
        [_calls([("c1", "list_cards", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "candidate_mismatch"
    assert not score.task_success
    selection = score.gate("tool_selection")
    assert selection.outcome == "failed"
    assert selection.turn_index == 0
    assert "the trace calls get_balance" in selection.detail
    assert score.turns[0].calls[0].gold_call_index is None
    assert score.matched_calls == 0


def test_a_wrong_argument_fails_arguments_but_not_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    score, _ = _score(
        _single_turn_row(),
        [_calls([("c1", "get_balance", '{"account_id":"2"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert score.gate("tool_selection").outcome == "passed"
    arguments = score.gate("arguments")
    assert arguments.outcome == "failed"
    assert arguments.turn_index == 0
    assert "differing account_id" in arguments.detail
    call = score.turns[0].calls[0]
    assert call.name_matched and not call.arguments_matched
    assert call.diff is not None
    assert call.diff.differing == ("account_id",)


def test_a_gold_call_the_episode_never_reached_fails_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The trace's second turn is never asked, so its call is unmatched, not excused."""
    score, episode = _score(
        _missing_slot_row(),
        [_says("Which account, exactly?")],
        tmp_path,
        monkeypatch=monkeypatch,
        scoring=_scoring(intermediate_text_matching="verbatim"),
    )

    assert episode.status == "candidate_mismatch"
    assert score.gate("text_turn").outcome == "failed"
    assert score.gate("tool_selection").outcome == "failed"
    # The unrequested gold call is attributed to the scripted turn that asks for it.
    assert score.gate("tool_selection").turn_index == 1
    assert "never requested" in score.gate("tool_selection").detail


def test_unparseable_arguments_are_a_mismatch_rather_than_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, _ = _score(
        _single_turn_row(),
        [_calls([("c1", "get_balance", "{not json")])],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    call = score.turns[0].calls[0]
    assert call.arguments_status == "invalid_json"
    assert call.name_matched and not call.arguments_matched
    assert score.gate("arguments").outcome == "failed"
    assert score.gate("tool_selection").outcome == "passed"


# --------------------------------------------------------------------------------------
# The schema gate: default insertion cannot launder a missing required argument.
# --------------------------------------------------------------------------------------


def test_a_required_argument_left_to_its_default_fails_before_result_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, episode = _score(
        _required_default_row(),
        [_calls([("c1", "close_account", '{"account_id":"1"}')]), _says("Account 1 is closed.")],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    # Defaults make equivalent optional spellings compare equally, but cannot
    # satisfy a parameter the schema requires the caller to send.
    assert episode.status == "candidate_mismatch"
    assert score.gate("arguments").outcome == "failed"
    schema = score.gate("schema_valid")
    assert schema.outcome == "failed"
    assert schema.turn_index == 0
    assert "missing_required_argument" in schema.detail
    assert not score.task_success


def test_an_argument_of_the_wrong_type_fails_the_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    score, _ = _score(
        _single_turn_row(),
        [_calls([("c1", "get_balance", '{"account_id":1}')])],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert score.gate("schema_valid").outcome == "failed"
    assert "account_id" in str(score.turns[0].calls[0].schema_failures)


def test_a_canonical_only_policy_reports_no_schema_verdict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    score, _ = _score(
        _required_default_row(),
        [_calls([("c1", "close_account", '{"account_id":"1","confirm":true}')]), _says("Account 1 is closed.")],
        tmp_path,
        monkeypatch=monkeypatch,
        scoring=_scoring(argument_matching="canonical_only", insert_declared_defaults=False),
    )

    assert score.gate("schema_valid").outcome == "not_applicable"
    assert score.turns[0].calls[0].schema_failures == ()
    assert score.task_success


# --------------------------------------------------------------------------------------
# Grouping and ordering: consistency of the turns that were asked.
# --------------------------------------------------------------------------------------


def test_calls_made_out_of_order_fail_ordering_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A permuted group selected the right tools, so only the order gate fails."""
    score, episode = _score(
        _parallel_row(),
        [
            _calls(
                [
                    ("c1", "list_cards", '{"account_id":"1"}'),
                    ("c2", "get_balance", '{"account_id":"1"}'),
                ]
            )
        ],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "candidate_mismatch"
    assert score.gate("tool_selection").outcome == "passed"
    assert score.gate("arguments").outcome == "passed"
    assert score.gate("call_grouping").outcome == "passed"
    ordering = score.gate("call_ordering")
    assert ordering.outcome == "failed"
    assert ordering.turn_index == 0
    assert "different order" in ordering.detail
    assert episode.observed[0].detail == ordering.detail
    assert score.matched_calls == 2
    # The episode still failed: an out-of-order turn earns no recorded result.
    assert score.gate("trace_completion").outcome == "failed"
    assert not score.task_success


def test_a_row_that_declares_any_order_has_no_ordering_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    score, episode = _score(
        _parallel_row(call_order="any"),
        [
            _calls(
                [
                    ("c1", "list_cards", '{"account_id":"1"}'),
                    ("c2", "get_balance", '{"account_id":"1"}'),
                ]
            ),
            _says("Balance 500, one card."),
        ],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "completed"
    assert score.gate("call_ordering").outcome == "not_applicable"
    assert "call_order: any" in score.gate("call_ordering").detail
    assert score.task_success


def test_a_policy_that_does_not_order_calls_has_no_ordering_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, episode = _score(
        _parallel_row(),
        [
            _calls(
                [
                    ("c1", "list_cards", '{"account_id":"1"}'),
                    ("c2", "get_balance", '{"account_id":"1"}'),
                ]
            ),
            _says("Balance 500, one card."),
        ],
        tmp_path,
        monkeypatch=monkeypatch,
        scoring=_scoring(respect_call_order=False),
    )

    assert episode.status == "completed"
    assert score.gate("call_ordering").outcome == "not_applicable"
    assert "respect_call_order" in score.gate("call_ordering").detail
    assert score.task_success


def test_a_prefix_row_orders_only_its_declared_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ordered, _ = _score(
        _prefix_row(),
        [
            _calls(
                [
                    ("c1", "get_balance", '{"account_id":"1"}'),
                    ("c2", "list_cards", '{"account_id":"1"}'),
                ]
            ),
            _says("Done."),
        ],
        tmp_path,
        monkeypatch=monkeypatch,
    )
    violated, _ = _score(
        _prefix_row(),
        [
            _calls(
                [
                    ("c1", "list_cards", '{"account_id":"1"}'),
                    ("c2", "get_balance", '{"account_id":"1"}'),
                ]
            )
        ],
        tmp_path / "second",
        monkeypatch=monkeypatch,
    )

    assert ordered.task_success
    assert ordered.gate("call_ordering").outcome == "passed"
    assert violated.gate("call_ordering").outcome == "failed"
    assert "required-tool prefix" in violated.gate("call_ordering").detail


def test_a_turn_that_issues_the_wrong_number_of_calls_fails_grouping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, episode = _score(
        _parallel_row(),
        [_calls([("c1", "get_balance", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "candidate_mismatch"
    grouping = score.gate("call_grouping")
    assert grouping.outcome == "failed"
    assert grouping.turn_index == 0
    assert "2 call(s); the candidate issued 1" in grouping.detail
    assert score.turns[0].group_size_matched is False


def test_relaxing_the_group_policy_drops_the_grouping_gate_but_not_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay still needs one result per gold call, so the episode stops either way."""
    score, episode = _score(
        _parallel_row(),
        [_calls([("c1", "get_balance", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
        scoring=_scoring(respect_call_group=False),
    )

    assert episode.status == "candidate_mismatch"
    assert score.gate("call_grouping").outcome == "not_applicable"
    assert "respect_call_group" in score.gate("call_grouping").detail
    # The call the candidate never made is still an unmatched gold call.
    assert score.gate("tool_selection").outcome == "failed"
    assert score.matched_calls == 1


# --------------------------------------------------------------------------------------
# Text turns, including a trace whose whole answer is words.
# --------------------------------------------------------------------------------------


def test_a_trace_that_asks_for_no_call_is_answered_in_words(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    score, episode = _score(
        _irrelevant_row(),
        [_says("I cannot help with the weather, only with banking.")],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "completed"
    assert score.task_success
    assert _outcomes(score) == {
        "tool_selection": "passed",
        "arguments": "not_applicable",
        "schema_valid": "not_applicable",
        "call_grouping": "not_applicable",
        "call_ordering": "not_applicable",
        "text_turn": "passed",
        "trace_completion": "passed",
    }
    assert score.expected_calls == 0


def test_calling_a_tool_where_the_trace_speaks_is_a_selection_and_text_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, episode = _score(
        _irrelevant_row(),
        [_calls([("c1", "get_balance", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "candidate_mismatch"
    assert score.gate("tool_selection").outcome == "failed"
    assert score.gate("text_turn").outcome == "failed"
    # The call is recorded as evidence, paired with nothing.
    call = score.turns[0].calls[0]
    assert call.predicted_function_name == "get_balance"
    assert call.gold_call_index is None
    assert "answers this request in words" in call.detail
    # A schema-valid call to a tool the trace never asks for still satisfies its schema.
    assert score.gate("schema_valid").outcome == "passed"


def test_verbatim_matching_holds_an_intermediate_turn_to_the_recorded_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, _ = _score(
        _missing_slot_row(),
        [_says("Sure, which one?")],
        tmp_path,
        monkeypatch=monkeypatch,
        scoring=_scoring(intermediate_text_matching="verbatim"),
    )

    text = score.gate("text_turn")
    assert text.outcome == "failed"
    assert text.turn_index == 0
    assert "scripted intermediate assistant text" in text.detail


def test_structural_matching_credits_an_intermediate_turn_worded_differently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The trace asks "Which account should I check?". Asking the same thing in
    # other words is the behavior the trace recorded; the recorded sentence is
    # one phrasing of it, and holding a model to the phrasing scores translation.
    score, _ = _score(
        _missing_slot_row(),
        [
            _says("Sure, which one?"),
            _calls([("c1", "get_balance", '{"account_id":"1"}')]),
            _says("Account 1 holds 500."),
        ],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert score.gate("text_turn").outcome == "passed"
    assert score.task_success


def test_structural_matching_still_refuses_an_empty_intermediate_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, _ = _score(
        _missing_slot_row(),
        [_says("   ")],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    text = score.gate("text_turn")
    assert text.outcome == "failed"
    assert text.turn_index == 0
    assert "no non-empty textual content" in text.detail


def test_a_terminal_turn_may_word_its_answer_freely(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    score, _ = _score(
        _single_turn_row(),
        [
            _calls([("c1", "get_balance", '{"account_id":"1"}')]),
            _says("The balance you asked about is five hundred."),
        ],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert score.task_success
    assert score.turns[-1].text_matched


@pytest.mark.parametrize(
    "empty_content",
    ["", "   ", [], {}, {"type": "text", "text": "  "}],
)
def test_a_terminal_turn_requires_non_empty_textual_content(
    empty_content: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score, episode = _score(
        _single_turn_row(),
        [_calls([("c1", "get_balance", '{"account_id":"1"}')]), _says(empty_content)],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "candidate_mismatch"
    assert score.turns[-1].text_matched is False
    assert not score.task_success


def test_structured_terminal_text_is_accepted_when_it_contains_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, _ = _score(
        _single_turn_row(),
        [
            _calls([("c1", "get_balance", '{"account_id":"1"}')]),
            _says([{"type": "text", "text": "Account 1 holds 500."}]),
        ],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert score.turns[-1].text_matched
    assert score.task_success


def test_an_incomplete_provider_finish_cannot_be_scored_as_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script(_single_turn_row())
    episode = _episode(
        script,
        [
            _calls([("c1", "get_balance", '{"account_id":"1"}')]),
            _says("Account 1 holds", finish_reason="length"),
        ],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "candidate_mismatch"
    assert "finish_reason" in episode.detail

    # Imported evidence cannot bypass the check by merely claiming completion:
    # the parser retained the provider's explicit incomplete finish reason.
    restamped = episode.model_copy(update={"status": "completed"})
    score = score_trace_episode(
        episode=restamped,
        script=script,
        scoring=_scoring(),
        plan=_plan(),
    )
    completion = score.gate("trace_completion")
    assert completion.outcome == "failed"
    assert completion.reason_code == "trace_completion.incomplete_finish_reason"
    assert not score.task_success


# --------------------------------------------------------------------------------------
# Completion: an episode that stopped for a non-candidate reason still failed.
# --------------------------------------------------------------------------------------


def test_an_unreachable_candidate_fails_the_task_and_is_marked_as_such(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, episode = _score(_single_turn_row(), [500], tmp_path, monkeypatch=monkeypatch)

    assert episode.status == "candidate_call_failed"
    assert score.non_candidate_stop
    assert not score.task_success
    assert score.gate("trace_completion").outcome == "failed"


def test_a_turn_budget_below_the_trace_fails_completion_without_blaming_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, episode = _score(
        _single_turn_row(),
        [_calls([("c1", "get_balance", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
        limits=_limits(max_turns=1),
    )

    assert episode.status == "max_turns_exceeded"
    assert score.non_candidate_stop
    assert score.gate("trace_completion").outcome == "failed"
    assert score.gate("trace_completion").turn_index == 1
    # The one turn it did take was right.
    assert score.gate("tool_selection").outcome == "passed"
    assert score.gate("arguments").outcome == "passed"
    assert not score.task_success


def test_a_candidate_mismatch_is_not_a_non_candidate_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    score, episode = _score(
        _single_turn_row(),
        [_calls([("c1", "list_cards", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "candidate_mismatch"
    assert not score.non_candidate_stop


# --------------------------------------------------------------------------------------
# Attribution: a failed gate says whether the candidate or the run is answerable.
# --------------------------------------------------------------------------------------


def test_an_endpoint_that_never_answered_is_the_runs_failure_not_the_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing the candidate did can be read off a turn it was never sent."""
    score, episode = _score(_single_turn_row(), [500], tmp_path, monkeypatch=monkeypatch)

    assert episode.status == "candidate_call_failed"
    assert score.gate("trace_completion").failure_class == "infrastructure"
    # Coverage still fails — the gold call was never requested — but the reason it
    # was never requested is the endpoint, so the failure is not the model's.
    assert score.gate("tool_selection").outcome == "failed"
    assert score.gate("tool_selection").failure_class == "infrastructure"
    assert not score.task_success


def test_a_turn_budget_below_the_trace_blames_the_run_for_what_it_cut_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, episode = _score(
        _missing_slot_row(),
        [_says("Which account should I check?")],
        tmp_path,
        monkeypatch=monkeypatch,
        limits=_limits(max_turns=1),
    )

    assert episode.status == "max_turns_exceeded"
    # The one turn it was asked, it answered as the trace does.
    assert score.gate("text_turn").outcome == "passed"
    for gate in score.gates:
        if gate.outcome == "failed":
            assert gate.failure_class == "infrastructure", gate.gate


def test_an_infrastructure_stop_preserves_a_model_failure_on_an_answered_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attribution is per gate; one score can contain both responsible parties."""
    script = _script(_missing_slot_row())
    verbatim = _scoring(intermediate_text_matching="verbatim")
    episode = _episode(
        script,
        [_says("I will not ask which account.")],
        tmp_path,
        monkeypatch=monkeypatch,
        scoring=verbatim,
    )
    score = score_trace_episode(
        episode=episode.model_copy(update={"status": "max_turns_exceeded"}),
        script=script,
        scoring=verbatim,
        plan=_plan(scoring=verbatim),
    )

    # The model answered the first text turn incorrectly.
    assert score.gate("text_turn").failure_class == "candidate"
    # The run, not the model, owns calls and completion in turns it cut off.
    assert score.gate("tool_selection").failure_class == "infrastructure"
    assert score.gate("arguments").failure_class == "infrastructure"
    assert score.gate("trace_completion").failure_class == "infrastructure"
    assert score.non_candidate_stop


def test_a_model_that_stopped_the_episode_keeps_its_own_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Turns lost to the candidate's own mistake are still the candidate's."""
    score, episode = _score(
        _missing_slot_row(),
        [_calls([("c1", "get_balance", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "candidate_mismatch"
    assert not score.non_candidate_stop
    assert score.failed_gates
    for gate in score.gates:
        if gate.outcome == "failed":
            assert gate.failure_class == "candidate", gate.gate


@pytest.mark.parametrize("status", ["malformed_response", "unusable_tool_call_ids"])
def test_candidate_protocol_terminals_keep_candidate_attribution(
    status: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _script(_single_turn_row())
    episode = _episode(
        script,
        [_calls([("c1", "list_cards", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
    )
    score = score_trace_episode(
        episode=episode.model_copy(update={"status": status}),
        script=script,
        scoring=_scoring(),
        plan=_plan(),
    )

    assert not score.non_candidate_stop
    assert {
        gate.failure_class for gate in score.gates if gate.outcome == "failed"
    } == {"candidate"}


def test_episode_timeout_keeps_infrastructure_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script(_single_turn_row())
    episode = _episode(script, [500], tmp_path, monkeypatch=monkeypatch)
    score = score_trace_episode(
        episode=episode.model_copy(update={"status": "episode_timeout"}),
        script=script,
        scoring=_scoring(),
        plan=_plan(),
    )

    assert score.non_candidate_stop
    assert {
        gate.failure_class for gate in score.gates if gate.outcome == "failed"
    } == {"infrastructure"}


def test_a_truncated_answer_in_a_finished_episode_is_the_models_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script(_single_turn_row())
    episode = _episode(
        script,
        [
            _calls([("c1", "get_balance", '{"account_id":"1"}')]),
            _says("Account 1 holds", finish_reason="length"),
        ],
        tmp_path,
        monkeypatch=monkeypatch,
    )
    score = score_trace_episode(
        episode=episode.model_copy(update={"status": "completed"}),
        script=script,
        scoring=_scoring(),
        plan=_plan(),
    )

    completion = score.gate("trace_completion")
    assert completion.reason_code == "trace_completion.incomplete_finish_reason"
    assert completion.failure_class == "candidate"
    assert not completion.blames_the_run


def test_a_passing_gate_is_never_attributed_and_a_failing_one_always_is() -> None:
    with pytest.raises(ValueError, match="whose failure it is"):
        GateResult(
            gate="tool_selection",
            outcome="failed",
            reason_code="tool_selection.mismatch",
            detail="the wrong tool was called",
        )
    with pytest.raises(ValueError, match="only a failed gate is attributed"):
        GateResult(
            gate="tool_selection",
            outcome="passed",
            failure_class="candidate",
            reason_code="tool_selection.matched",
            detail="every gold call was requested",
        )


@pytest.mark.parametrize("reason_code", ["", "arguments.mismatch"])
def test_a_gate_refuses_an_empty_or_foreign_reason_code(reason_code: str) -> None:
    with pytest.raises(ValueError, match="reason code"):
        GateResult(
            gate="tool_selection",
            outcome="failed",
            failure_class="candidate",
            reason_code=reason_code,
            detail="the wrong tool was called",
        )


def test_non_candidate_stop_must_be_derived_from_the_episode_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, _ = _score(_single_turn_row(), [500], tmp_path, monkeypatch=monkeypatch)
    payload = score.model_dump()
    payload["non_candidate_stop"] = False

    with pytest.raises(ValueError, match="episode status taxonomy"):
        TraceTaskScore.model_validate(payload)


def test_trace_score_1_0_requires_rescoring_under_the_1_1_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, _ = _score(
        _single_turn_row(),
        [_calls([("c1", "list_cards", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
    )
    payload = score.model_dump()
    payload["schema_version"] = "1.0"

    with pytest.raises(ValueError, match="schema_version"):
        TraceTaskScore.model_validate(payload)


def test_a_1_1_failed_gate_cannot_omit_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, _ = _score(
        _single_turn_row(),
        [_calls([("c1", "list_cards", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
    )
    payload = score.model_dump()
    failed = next(gate for gate in payload["gates"] if gate["outcome"] == "failed")
    del failed["failure_class"]

    with pytest.raises(ValueError, match="whose failure it is"):
        TraceTaskScore.model_validate(payload)


def test_a_score_that_blames_the_run_without_a_non_candidate_stop_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, _ = _score(
        _single_turn_row(),
        [_calls([("c1", "list_cards", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
    )
    payload = score.model_dump()
    for gate in payload["gates"]:
        if gate["outcome"] == "failed":
            gate["failure_class"] = "infrastructure"

    with pytest.raises(ValueError, match="only when the episode stopped"):
        TraceTaskScore.model_validate(payload)


def test_a_non_candidate_stop_that_does_not_blame_the_run_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, _ = _score(_single_turn_row(), [500], tmp_path, monkeypatch=monkeypatch)
    payload = score.model_dump()
    for gate in payload["gates"]:
        if gate["outcome"] == "failed":
            gate["failure_class"] = "candidate"

    with pytest.raises(ValueError, match="answerable"):
        TraceTaskScore.model_validate(payload)


def test_attribution_is_part_of_the_score_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two reports that disagree about whose failure it was are two scores."""
    score, _ = _score(_single_turn_row(), [500], tmp_path, monkeypatch=monkeypatch)
    completion = score.gate("trace_completion")
    reblamed = completion.model_copy(update={"failure_class": "candidate"})
    others = tuple(gate for gate in score.gates if gate.gate != "trace_completion")

    assert score.model_copy(update={"gates": (*others, reblamed)}).score_hash != score.score_hash


# --------------------------------------------------------------------------------------
# Taxonomy: a trace failure reads in the same vocabulary as an executable one.
# --------------------------------------------------------------------------------------


def test_a_task_that_passed_every_gate_records_no_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, _ = _score(
        _single_turn_row(),
        [_calls([("c1", "get_balance", '{"account_id":"1"}')]), _says("Account 1 holds 500.")],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert score.task_success
    assert score.failure_records() == ()


def test_a_broken_run_projects_its_terminal_and_its_gates_onto_the_taxonomy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, _ = _score(_single_turn_row(), [500], tmp_path, monkeypatch=monkeypatch)

    records = [record.as_document() for record in trace_failure_records(score)]

    # The episode record carries what no gate can: every unfinished episode fails
    # the same completion gate, and only the terminal says what ended this one.
    assert records[0] == {
        "layer": "episode",
        "code": "episode.candidate_call_failed",
        "attribution": "infrastructure",
        "subject": "episode",
    }
    assert {record["attribution"] for record in records} == {"infrastructure"}
    assert {record["subject"] for record in records[1:]} == set(score.failed_gates)
    assert records == [record.as_document() for record in score.failure_records()]


def test_a_wrong_model_projects_candidate_failures_and_no_terminal_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, episode = _score(
        _single_turn_row(),
        [_calls([("c1", "list_cards", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    records = [record.as_document() for record in score.failure_records()]

    assert episode.status == "candidate_mismatch"
    assert records[0]["code"] == "episode.candidate_mismatch"
    assert {record["attribution"] for record in records} == {"candidate"}
    assert [record["layer"] for record in records[1:]] == ["gate"] * len(score.failed_gates)


def test_a_trace_score_projects_onto_shared_task_result_columns_without_oracle_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, _ = _score(
        _single_turn_row(),
        [_calls([("c1", "list_cards", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    result = trace_task_result(score)

    assert result["mode"] == "trace"
    assert result["tool_name_correct"] is False
    assert result["task_success"] is False
    assert result["failure_records"] == [
        record.as_document() for record in score.failure_records()
    ]
    assert result["milestones_correct"] is None
    assert result["execution_success"] is None
    assert result["assertions_passed"] is None
    assert result["final_answer_passed"] is None


def test_every_reason_code_a_trace_gate_emits_is_a_registered_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gate record is only readable if its namespace is declared taxonomy-wide."""
    scores = (
        _score(
            _single_turn_row(),
            [_calls([("c1", "list_cards", '{"account_id":"1"}')])],
            tmp_path / "candidate",
            monkeypatch=monkeypatch,
        )[0],
        _score(
            _single_turn_row(),
            [500],
            tmp_path / "infrastructure",
            monkeypatch=monkeypatch,
        )[0],
        _score(
            _single_turn_row(),
            [
                _calls([("c1", "get_balance", '{"account_id":"1"}')]),
                _says("Account 1 holds", finish_reason="length"),
            ],
            tmp_path / "truncated",
            monkeypatch=monkeypatch,
        )[0],
    )

    emitted = {
        gate.reason_code.partition(".")[0]
        for score in scores
        for gate in score.gates
    }
    assert emitted <= REASON_CODE_NAMESPACES
    assert emitted == set(SCORING_GATES)


# --------------------------------------------------------------------------------------
# Identity: the same evidence, scored under the same rules, is the same score.
# --------------------------------------------------------------------------------------


def test_scoring_the_same_evidence_twice_reproduces_the_score_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script(_single_turn_row())
    episode = _episode(
        script,
        [_calls([("c1", "get_balance", '{"account_id":"1"}')]), _says("Account 1 holds 500.")],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    first = score_trace_episode(
        episode=episode,
        script=script,
        scoring=_scoring(),
        plan=_plan(),
    )
    second = score_trace_episode(
        episode=episode,
        script=script,
        scoring=_scoring(),
        plan=_plan(),
    )

    assert first.score_hash == second.score_hash
    assert first.schema_version == "1.1"
    assert first.as_document()["score_hash"] == first.score_hash


def test_scoring_with_a_policy_other_than_the_authorized_one_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script(_single_turn_row())
    episode = _episode(
        script,
        [_calls([("c1", "get_balance", '{"account_id":"1"}')]), _says("Account 1 holds 500.")],
        tmp_path,
        monkeypatch=monkeypatch,
    )
    elsewhere = _scoring(
        contract=EvalFileRef(path="/refs/bfcl-eval-scoring-contract.md", content_hash=OTHER_CONTRACT_HASH)
    )

    with pytest.raises(TraceScoringPolicyError, match="not the scoring policy"):
        score_trace_episode(
            episode=episode,
            script=script,
            scoring=elsewhere,
            plan=_plan(),
        )


def test_a_score_cites_the_episode_and_the_policy_it_was_taken_under(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script(_single_turn_row())
    scoring = _scoring()
    episode = _episode(
        script,
        [_calls([("c1", "get_balance", '{"account_id":"1"}')]), _says("Account 1 holds 500.")],
        tmp_path,
        monkeypatch=monkeypatch,
        scoring=scoring,
    )

    score = score_trace_episode(
        episode=episode,
        script=script,
        scoring=scoring,
        plan=_plan(scoring=scoring),
    )

    assert score.script_hash == script.script_hash
    assert score.episode_hash == episode.episode_hash
    assert score.source_verification_identity == SOURCE_IDENTITY
    assert score.plan_identity == _plan().plan_identity
    assert score.eval_config_hash == EVAL_CONFIG_HASH
    assert score.scoring_contract_hash == CONTRACT_HASH
    assert score.as_document()["scoring_policy"]["argument_matching"] == "schema_then_canonical"


def test_a_replayed_episode_scores_identically_to_the_one_that_paid_for_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script(_single_turn_row())
    replies = [_calls([("c1", "get_balance", '{"account_id":"1"}')]), _says("Account 1 holds 500.")]
    monkeypatch.setenv("CANDIDATE_API_KEY", "secret-key")
    cache = CandidateIOCache(tmp_path / "cache")
    candidate = _candidate()
    bounds = _limits()
    pending = list(replies)

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=pending.pop(0))

    async def run(transport: httpx.MockTransport) -> CandidateEpisode:
        client = NativeFunctionCallingClient(candidate, bounds, cache, transport=transport)
        try:
            return await run_candidate_episode(
                candidate=candidate,
                limits=bounds,
                client=client,
                script=script,
                plan=_plan(),
                gate=CanonicalCallMatchGate(_scoring()),
            )
        finally:
            await client.aclose()

    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError("a replayed episode must not contact the provider")

    live = asyncio.run(run(httpx.MockTransport(handle)))
    replayed = asyncio.run(run(httpx.MockTransport(refuse)))

    assert replayed.replayed
    assert (
        score_trace_episode(
            episode=live,
            script=script,
            scoring=_scoring(),
            plan=_plan(),
        ).score_hash
        == score_trace_episode(
            episode=replayed,
            script=script,
            scoring=_scoring(),
            plan=_plan(),
        ).score_hash
    )


def test_an_episode_cannot_be_stamped_with_another_eval_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script(_single_turn_row())
    episode = _episode(
        script,
        [_calls([("c1", "get_balance", '{"account_id":"1"}')]), _says("Account 1 holds 500.")],
        tmp_path,
        monkeypatch=monkeypatch,
    )
    another_plan = _plan().model_copy(update={"eval_config_hash": "sha256:" + "8" * 64})

    with pytest.raises(TraceEvidenceError, match="authorization plan"):
        score_trace_episode(
            episode=episode,
            script=script,
            scoring=_scoring(),
            plan=another_plan,
        )


def test_diagnostic_wording_does_not_change_the_score_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, _ = _score(
        _single_turn_row(),
        [_calls([("c1", "list_cards", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
    )
    changed_call = score.turns[0].calls[0].model_copy(update={"detail": "rewritten call diagnostic"})
    changed_turn = score.turns[0].model_copy(update={"calls": (changed_call,), "detail": "rewritten turn diagnostic"})
    changed_gate = score.gates[0].model_copy(update={"detail": "rewritten gate diagnostic"})
    rewritten = score.model_copy(
        update={
            "turns": (changed_turn,),
            "gates": (changed_gate, *score.gates[1:]),
            "detail": "rewritten score diagnostic",
        }
    )

    assert rewritten.score_hash == score.score_hash


def test_a_constraint_named_detail_stays_part_of_the_score_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    score, _ = _score(
        _single_turn_row(),
        [_calls([("c1", "list_cards", '{"account_id":"1"}')])],
        tmp_path,
        monkeypatch=monkeypatch,
    )
    reported = score.turns[0].calls[0].model_copy(
        update={"schema_failures": ({"reason": "out_of_range", "detail": "upper bound 10"},)}
    )
    other = score.turns[0].calls[0].model_copy(
        update={"schema_failures": ({"reason": "out_of_range", "detail": "upper bound 20"},)}
    )

    assert score.model_copy(
        update={"turns": (score.turns[0].model_copy(update={"calls": (reported,)}),)}
    ).score_hash != score.model_copy(
        update={"turns": (score.turns[0].model_copy(update={"calls": (other,)}),)}
    ).score_hash


# --------------------------------------------------------------------------------------
# Parity: the release gate and the scorer read the same comparison.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("row_factory", "replies"),
    [
        (
            _single_turn_row,
            [_calls([("c1", "get_balance", '{"account_id":"1"}')]), _says("Account 1 holds 500.")],
        ),
        (
            _single_turn_row,
            [_calls([("c1", "list_cards", '{"account_id":"1"}')])],
        ),
        (
            _single_turn_row,
            [_calls([("c1", "get_balance", '{"account_id":"2"}')])],
        ),
        (
            _parallel_row,
            [
                _calls(
                    [
                        ("c1", "get_balance", '{"account_id":"1"}'),
                        ("c2", "list_cards", '{"account_id":"1"}'),
                    ]
                ),
                _says("Balance 500, one card."),
            ],
        ),
        (
            _parallel_row,
            [_calls([("c1", "list_cards", '{"account_id":"1"}'), ("c2", "get_balance", '{"account_id":"1"}')])],
        ),
        (
            _missing_slot_row,
            [
                _says("Which account should I check?"),
                _calls([("c1", "get_balance", '{"account_id":"1"}')]),
                _says("Account 1 holds 500."),
            ],
        ),
        (
            # A paraphrase of the recorded question: the driver releases the next
            # request, so the scorer must credit the turn it was released on.
            _missing_slot_row,
            [
                _says("Which one?"),
                _calls([("c1", "get_balance", '{"account_id":"1"}')]),
                _says("Account 1 holds 500."),
            ],
        ),
        (_irrelevant_row, [_says("Banking only, sorry.")]),
    ],
)
def test_a_turn_the_driver_released_is_never_a_turn_the_scorer_faults(
    row_factory: Any,
    replies: list[Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gate stricter than the scorer would fail a model on transport grounds."""
    score, episode = _score(row_factory(), replies, tmp_path, monkeypatch=monkeypatch)

    for observed, scored in zip(episode.observed, score.turns, strict=True):
        if not observed.advanced:
            continue
        if scored.kind == "text":
            assert scored.text_matched, scored.detail
            continue
        assert all(call.matched for call in scored.calls), scored.detail
        assert scored.group_size_matched, scored.detail
        assert scored.order_respected is not False, scored.order_detail


def test_the_driver_and_scorer_share_the_schema_verdict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Continuation checks the call comparison; the schema step is the scorer's alone."""
    score, episode = _score(
        _required_default_row(),
        [_calls([("c1", "close_account", '{"account_id":"1"}')]), _says("Account 1 is closed.")],
        tmp_path,
        monkeypatch=monkeypatch,
    )

    assert episode.status == "candidate_mismatch"
    assert score.gate("arguments").outcome == "failed"
    assert score.gate("schema_valid").outcome == "failed"


# --------------------------------------------------------------------------------------
# The published contract: every gate rolls up into a declared metric.
# --------------------------------------------------------------------------------------


def test_every_gate_maps_onto_a_metric_the_export_bundle_declares() -> None:
    assert set(EXPORT_METRIC_BY_GATE) == set(SCORING_GATES)
    assert set(EXPORT_METRIC_BY_GATE.values()) <= set(EXPORT_SCORING_METRICS)


def test_the_only_export_metric_without_a_trace_gate_is_the_one_replay_measures() -> None:
    """A trace score cannot claim results: nothing here executes a tool."""
    assert set(EXPORT_SCORING_METRICS) - set(EXPORT_METRIC_BY_GATE.values()) == {"results"}
