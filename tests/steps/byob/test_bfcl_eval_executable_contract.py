from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_contract import (
    EXECUTABLE_CONTRACT_VERSION,
    AssertionOutcome,
    ExecutableEpisode,
    ExecutableEvent,
    ExecutableTurn,
    ExecutedToolCall,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_contract import (
    VerifiedOracleSource,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

HASH = "sha256:" + "1" * 64
OTHER_HASH = "sha256:" + "2" * 64


def _hash(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


def _candidate_call(
    *,
    call_id: str | None = "call-1",
    function_name: str | None = "get_balance",
    arguments: dict[str, Any] | None = None,
    arguments_status: str = "valid_object",
) -> dict[str, Any]:
    parsed = {"account_id": "A-1"} if arguments is None else arguments
    return {
        "index": 0,
        "id": call_id,
        "type": "function",
        "function_name": function_name,
        "raw_arguments": canonical_json(parsed),
        "parsed_arguments": parsed if arguments_status == "valid_object" else None,
        "arguments_status": arguments_status,
    }


def _execution(
    *,
    arguments: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    status: str = "completed",
    state_commit: str = "not_applicable",
    released_to_model: bool = True,
    reason_code: str = "tool_execution.completed",
    detail: str = "tool returned a JSON result",
) -> ExecutedToolCall:
    payload = {"account_id": "A-1", "balance": 500} if result is None else result
    return ExecutedToolCall(
        execution_index=0,
        turn_index=0,
        position_in_turn=0,
        provider_call_index=0,
        call_id="call-1",
        type="function",
        function_name="get_balance",
        arguments_status="valid_object",
        parsed_arguments={"account_id": "A-1"} if arguments is None else arguments,
        schema_valid=True,
        status=status,
        state_commit=state_commit,
        result=payload,
        result_hash=_hash(payload),
        released_to_model=released_to_model,
        reason_code=reason_code,
        detail=detail,
    )


def _turn(*, arguments: dict[str, Any] | None = None, detail: str = "live result released") -> ExecutableTurn:
    return ExecutableTurn(
        turn_index=0,
        request_hash=HASH,
        call_status="completed",
        response_hash=OTHER_HASH,
        finish_reason="tool_calls",
        tool_calls=(_candidate_call(arguments=arguments),),
        tool_call_outcome_indexes=(0,),
        paired_call_indexes=(0,),
        advanced=True,
        reason_code="turn.advanced",
        detail=detail,
    )


def _assertion(*, detail: str = "state is correct") -> AssertionOutcome:
    return AssertionOutcome(
        assertion_index=0,
        name="assert_balance",
        category="state",
        status="passed",
        reason_code="assertion.passed",
        detail=detail,
    )


def _episode(
    *,
    execution: ExecutedToolCall | None = None,
    turn: ExecutableTurn | None = None,
    assertion: AssertionOutcome | None = None,
    event_detail: str = "episode reached terminal state",
    detail: str = "live conversation completed",
    replayed: bool = False,
) -> ExecutableEpisode:
    return ExecutableEpisode(
        candidate_alias="candidate-a",
        canonical_model_identity="hf:org/model@" + "a" * 40,
        task_id="task-1",
        plan_identity=HASH,
        eval_config_hash=OTHER_HASH,
        source_verification_identity="sha256:" + "3" * 64,
        oracle_verification_identity="sha256:" + "4" * 64,
        script_hash="sha256:" + "5" * 64,
        task_spec_hash="sha256:" + "7" * 64,
        status="completed",
        reason_code="episode.completed",
        detail=detail,
        assistant_turns=1,
        observed=(turn or _turn(),),
        executions=(execution or _execution(),),
        final_state_hash="sha256:" + "6" * 64,
        assertions=(assertion or _assertion(),),
        events=(
            ExecutableEvent(
                index=0,
                kind="terminal",
                turn_index=0,
                reason_code="episode.completed",
                detail=event_detail,
            ),
        ),
        replayed=replayed,
    )


def test_the_executable_evidence_contract_is_versioned_frozen_and_self_describing() -> None:
    episode = _episode()

    assert episode.schema_version == EXECUTABLE_CONTRACT_VERSION == "1.2"
    assert episode.succeeded
    assert episode.executions[0].attempted
    assert episode.executions[0].has_result
    assert episode.assertions[0].passed
    assert episode.as_document()["episode_hash"] == episode.episode_hash
    with pytest.raises(ValidationError, match="frozen"):
        episode.status = "candidate_mismatch"  # type: ignore[misc]


def test_caller_owned_result_and_arguments_are_frozen() -> None:
    result = {"items": [{"value": 1}]}
    arguments = {"account_id": "A-1"}
    execution = ExecutedToolCall(
        **{
            **_execution(result=result).model_dump(mode="python"),
            "parsed_arguments": arguments,
        }
    )
    result["items"][0]["value"] = 99
    arguments["account_id"] = "changed"

    assert execution.semantic_payload()["result"] == {"items": [{"value": 1}]}
    assert execution.semantic_payload()["parsed_arguments"] == {"account_id": "A-1"}


def test_a_tool_result_hash_must_identify_the_canonical_result() -> None:
    with pytest.raises(ValidationError, match="result_hash"):
        ExecutedToolCall(
            **{
                **_execution().model_dump(mode="python"),
                "result_hash": OTHER_HASH,
            }
        )


def test_a_business_rejection_is_a_completed_structured_error() -> None:
    result = {"error": {"code": "INSUFFICIENT_FUNDS", "message": "declined"}}
    execution = _execution(
        result=result,
        status="business_rejection",
        state_commit="not_committed",
        reason_code="tool_execution.business_rejection",
    )

    assert execution.has_result
    assert execution.status == "business_rejection"


@pytest.mark.parametrize(
    "result",
    [
        {"error": {}},
        {"error": {"code": ""}},
        {"error": "declined"},
    ],
)
def test_a_business_rejection_requires_a_non_empty_error_code(result: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="error.code"):
        _execution(
            result=result,
            status="business_rejection",
            state_commit="not_committed",
        )


def test_a_structured_oracle_error_cannot_be_mislabeled_as_a_normal_result() -> None:
    result = {"error": {"code": "DECLINED"}}
    with pytest.raises(ValidationError, match="business rejection"):
        _execution(result=result, status="completed")


def test_an_invalid_candidate_call_is_retained_as_not_executed() -> None:
    call = _candidate_call(call_id=None, arguments_status="invalid_json")
    execution = ExecutedToolCall(
        execution_index=0,
        turn_index=0,
        position_in_turn=0,
        provider_call_index=0,
        call_id=None,
        type="function",
        function_name="get_balance",
        arguments_status="invalid_json",
        schema_valid=None,
        status="not_executed",
        state_commit="not_started",
        reason_code="tool_execution.invalid_arguments",
        detail="arguments did not parse",
    )
    turn = ExecutableTurn(
        turn_index=0,
        request_hash=HASH,
        call_status="completed",
        response_hash=OTHER_HASH,
        tool_calls=(call,),
        tool_call_outcome_indexes=(0,),
        advanced=False,
        reason_code="turn.invalid_arguments",
        detail="the call could not execute",
    )
    episode = ExecutableEpisode(
        candidate_alias="candidate-a",
        canonical_model_identity="hf:org/model@" + "a" * 40,
        task_id="task-1",
        plan_identity=HASH,
        eval_config_hash=OTHER_HASH,
        source_verification_identity="sha256:" + "3" * 64,
        oracle_verification_identity="sha256:" + "4" * 64,
        script_hash="sha256:" + "5" * 64,
        task_spec_hash="sha256:" + "7" * 64,
        status="candidate_mismatch",
        reason_code="episode.invalid_arguments",
        detail="candidate call was not executable",
        assistant_turns=1,
        observed=(turn,),
        executions=(execution,),
    )

    assert not episode.executions[0].attempted
    assert episode.executions[0].status == "not_executed"


def test_an_attempted_call_must_be_a_schema_valid_named_function_call() -> None:
    with pytest.raises(ValidationError, match="schema-valid named function call"):
        ExecutedToolCall(
            **{
                **_execution().model_dump(mode="python"),
                "schema_valid": False,
                "schema_failures": ({"reason": "missing_required_argument"},),
            }
        )


def test_every_provider_call_has_exactly_one_ordered_outcome() -> None:
    with pytest.raises(ValidationError, match="exactly one executable outcome"):
        ExecutableTurn(
            **{
                **_turn().model_dump(mode="python"),
                "tool_call_outcome_indexes": (),
            }
        )

    with pytest.raises(ValidationError, match="turns cite every tool outcome once"):
        ExecutableEpisode(
            **{
                **_episode().model_dump(mode="python"),
                "executions": (_execution(), _execution().model_copy(update={"execution_index": 1})),
            }
        )


def test_an_outcome_must_identify_the_exact_provider_call() -> None:
    wrong = _execution().model_copy(update={"call_id": "another-call"})
    with pytest.raises(ValidationError, match="provider call"):
        _episode(execution=wrong)


@pytest.mark.parametrize(
    ("sent", "recorded"),
    [
        ({"confirm": True}, {"confirm": 1}),
        ({"count": 1}, {"count": True}),
        ({"amount": 10}, {"amount": 10.0}),
        ({"account_id": "1"}, {"account_id": 1}),
    ],
)
def test_an_outcome_cannot_bind_to_a_call_by_coercing_its_arguments(
    sent: dict[str, Any], recorded: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError, match="provider call"):
        _episode(turn=_turn(arguments=sent), execution=_execution(arguments=recorded))


def test_only_a_nonterminal_advanced_turn_must_release_its_live_result() -> None:
    obtained = _execution(released_to_model=False)
    aborted = _turn().model_copy(update={"advanced": False})
    episode = ExecutableEpisode(
        **{
            **_episode().model_dump(mode="python"),
            "status": "episode_timeout",
            "reason_code": "episode.timeout",
            "observed": (aborted,),
            "executions": (obtained,),
            "final_state_hash": None,
            "assertions": (),
        }
    )

    assert episode.executions[0].has_result
    assert not episode.executions[0].released_to_model
    assert episode.released_tool_results == 0
    assert episode.results_released_in(0) == 0

    terminal = _episode(execution=obtained)
    assert terminal.status == "completed"
    assert terminal.released_tool_results == 0

    final_text = ExecutableTurn(
        turn_index=1,
        request_hash=HASH,
        call_status="completed",
        response_hash=OTHER_HASH,
        finish_reason="stop",
        assistant_content="done",
        advanced=True,
        reason_code="turn.terminal",
        detail="terminal answer",
    )
    base = _episode().model_dump(mode="python")
    with pytest.raises(ValidationError, match="nonterminal turn"):
        ExecutableEpisode(
            **{
                **base,
                "assistant_turns": 2,
                "observed": (_turn(), final_text),
                "executions": (obtained,),
                "events": (
                    ExecutableEvent(
                        index=0,
                        kind="terminal",
                        turn_index=1,
                        reason_code="episode.completed",
                    ),
                ),
            }
        )


def test_a_release_requires_a_result_to_release() -> None:
    with pytest.raises(ValidationError, match="nothing to release"):
        ExecutedToolCall(
            **{
                **_execution().model_dump(mode="python"),
                "status": "tool_error",
                "state_commit": "not_committed",
                "result": None,
                "result_hash": None,
                "released_to_model": True,
                "reason_code": "tool_execution.tool_error",
            }
        )


def test_a_mutation_commit_verdict_is_bound_to_state_snapshot_hashes() -> None:
    with pytest.raises(ValidationError, match="both state snapshots"):
        _execution(state_commit="committed")

    with pytest.raises(ValidationError, match="preserved its state hash"):
        ExecutedToolCall(
            **{
                **_execution().model_dump(mode="python"),
                "state_commit": "not_committed",
                "state_before_hash": HASH,
                "state_after_hash": OTHER_HASH,
            }
        )


def test_a_non_object_oracle_return_is_recorded_rather_than_discarded() -> None:
    malformed = ExecutedToolCall(
        **{
            **_execution().model_dump(mode="python"),
            "status": "malformed_result",
            "state_commit": "not_committed",
            "result": None,
            "result_hash": None,
            "released_to_model": False,
            "malformed_result": [1, 2, 3],
            "malformed_result_type": "array",
            "malformed_result_hash": _hash([1, 2, 3]),
            "reason_code": "tool_execution.malformed_result",
        }
    )
    episode = ExecutableEpisode(
        **{
            **_episode().model_dump(mode="python"),
            "status": "oracle_result_malformed",
            "reason_code": "episode.oracle_result_malformed",
            "observed": (_turn().model_copy(update={"advanced": False}),),
            "executions": (malformed,),
            "final_state_hash": None,
            "assertions": (),
        }
    )

    assert episode.executions[0].attempted
    assert not episode.executions[0].has_result
    assert episode.executions[0].malformed_result_hash == _hash([1, 2, 3])

    with pytest.raises(ValidationError, match="names the JSON shape"):
        ExecutedToolCall(
            **{
                **malformed.model_dump(mode="python"),
                "malformed_result_type": "str",
            }
        )
    with pytest.raises(ValidationError, match="canonical malformed result"):
        ExecutedToolCall(
            **{
                **malformed.model_dump(mode="python"),
                "malformed_result_hash": OTHER_HASH,
            }
        )
    with pytest.raises(ValidationError, match="belongs only to a malformed-result outcome"):
        ExecutedToolCall(
            **{
                **_execution().model_dump(mode="python"),
                "malformed_result": [1, 2, 3],
                "malformed_result_type": "array",
                "malformed_result_hash": _hash([1, 2, 3]),
            }
        )


@pytest.mark.parametrize(
    ("value", "kind"),
    [(None, "null"), (True, "bool"), (1, "int"), (1.5, "float"), ("bad", "str"), ([1], "array")],
)
def test_malformed_json_result_evidence_is_typed_and_self_verifying(
    value: Any, kind: str
) -> None:
    outcome = ExecutedToolCall(
        **{
            **_execution().model_dump(mode="python"),
            "status": "malformed_result",
            "state_commit": "not_committed",
            "result": None,
            "result_hash": None,
            "released_to_model": False,
            "malformed_result": value,
            "malformed_result_type": kind,
            "malformed_result_hash": _hash(value),
            "reason_code": "tool_execution.malformed_result",
        }
    )

    assert outcome.semantic_payload()["malformed_result"] == value


def test_an_object_cannot_pose_as_a_malformed_non_object_result() -> None:
    with pytest.raises(ValidationError, match="belongs in result"):
        ExecutedToolCall(
            **{
                **_execution().model_dump(mode="python"),
                "status": "malformed_result",
                "state_commit": "not_committed",
                "result": None,
                "result_hash": None,
                "released_to_model": False,
                "malformed_result": {"still": "an object"},
                "malformed_result_type": "array",
                "malformed_result_hash": _hash({"still": "an object"}),
                "reason_code": "tool_execution.malformed_result",
            }
        )


def test_terminal_execution_failures_cannot_be_restamped_as_completed() -> None:
    malformed = ExecutedToolCall(
        **{
            **_execution().model_dump(mode="python"),
            "status": "malformed_result",
            "state_commit": "not_committed",
            "result": None,
            "result_hash": None,
            "released_to_model": False,
            "malformed_result": [1],
            "malformed_result_type": "array",
            "malformed_result_hash": _hash([1]),
            "reason_code": "tool_execution.malformed_result",
        }
    )
    with pytest.raises(ValidationError, match="highest-priority tool outcome"):
        _episode(execution=malformed)

    unknown = ExecutedToolCall(
        **{
            **_execution().model_dump(mode="python"),
            "status": "unknown_commit_state",
            "state_commit": "unknown",
            "result": None,
            "result_hash": None,
            "released_to_model": False,
            "reason_code": "tool_execution.unknown_commit_state",
        }
    )
    with pytest.raises(ValidationError, match="highest-priority tool outcome"):
        _episode(execution=unknown)

    failed = ExecutedToolCall(
        **{
            **_execution().model_dump(mode="python"),
            "status": "tool_error",
            "state_commit": "not_committed",
            "result": None,
            "result_hash": None,
            "released_to_model": False,
            "reason_code": "tool_execution.tool_error",
        }
    )
    with pytest.raises(ValidationError, match="highest-priority tool outcome"):
        _episode(execution=failed)


def test_a_malformed_result_episode_identifies_the_offending_call() -> None:
    with pytest.raises(ValidationError, match="highest-priority tool outcome"):
        ExecutableEpisode(
            **{
                **_episode().model_dump(mode="python"),
                "status": "oracle_result_malformed",
                "reason_code": "episode.oracle_result_malformed",
                "observed": (_turn().model_copy(update={"advanced": False}),),
                "final_state_hash": None,
                "assertions": (),
            }
        )


@pytest.mark.parametrize(
    ("tool_status", "episode_status"),
    [
        ("timeout", "oracle_timeout"),
        ("tool_error", "oracle_call_failed"),
        ("infrastructure_error", "oracle_call_failed"),
    ],
)
def test_oracle_execution_failures_determine_the_episode_terminal_status(
    tool_status: str, episode_status: str
) -> None:
    execution = ExecutedToolCall(
        **{
            **_execution().model_dump(mode="python"),
            "status": tool_status,
            "state_commit": "not_committed",
            "result": None,
            "result_hash": None,
            "released_to_model": False,
            "reason_code": f"tool_execution.{tool_status}",
        }
    )
    base = _episode().model_dump(mode="python")
    episode = ExecutableEpisode(
        **{
            **base,
            "status": episode_status,
            "reason_code": f"episode.{episode_status}",
            "observed": (_turn().model_copy(update={"advanced": False}),),
            "executions": (execution,),
            "final_state_hash": None,
            "assertions": (),
        }
    )
    assert episode.status == episode_status

    with pytest.raises(ValidationError, match="highest-priority tool outcome"):
        ExecutableEpisode(
            **{
                **base,
                "status": "candidate_mismatch",
                "reason_code": "episode.candidate_mismatch",
                "observed": (_turn().model_copy(update={"advanced": False}),),
                "executions": (execution,),
                "final_state_hash": None,
                "assertions": (),
            }
        )


def test_a_non_object_result_cannot_pose_as_a_conforming_one() -> None:
    for payload in ("plain text", [1, 2, 3], 42):
        with pytest.raises(ValidationError, match="must be a JSON object"):
            ExecutedToolCall(
                **{
                    **_execution().model_dump(mode="python"),
                    "result": payload,
                }
            )


@pytest.mark.parametrize(
    ("field", "value"),
    [("turn_index", 7), ("execution_index", 7)],
)
def test_an_event_cannot_cite_evidence_the_episode_never_recorded(field: str, value: int) -> None:
    kind = "tool_execution" if field == "execution_index" else "terminal"
    companion = {"turn_index": 0} if field == "execution_index" else {}
    with pytest.raises(ValidationError, match="an executable event cites"):
        ExecutableEpisode(
            **{
                **_episode().model_dump(mode="python"),
                "events": (
                    ExecutableEvent(
                        index=0,
                        kind=kind,
                        reason_code="event.cited",
                        **companion,
                        **{field: value},
                    ),
                ),
            }
        )


def test_a_user_message_release_is_bound_to_the_turn_and_derived() -> None:
    released = _turn().model_copy(update={"released_user_message_hash": HASH})
    episode = _episode(turn=released)

    assert episode.released_user_turns == 1
    assert episode.semantic_payload()["released_user_turns"] == 1

    with pytest.raises(ValidationError, match="only a turn that advanced"):
        ExecutableTurn(
            **{
                **_turn().model_dump(mode="python"),
                "advanced": False,
                "released_user_message_hash": HASH,
            }
        )


def test_a_tool_execution_event_identifies_the_outcomes_turn() -> None:
    with pytest.raises(ValidationError, match="identifies its executable turn"):
        ExecutableEvent(
            index=0,
            kind="tool_execution",
            execution_index=0,
            reason_code="tool_execution.started",
        )


def test_a_tool_execution_event_cannot_cite_another_recorded_turn() -> None:
    second_turn = _turn().model_copy(
        update={"turn_index": 1, "tool_call_outcome_indexes": (1,)}
    )
    second_execution = _execution().model_copy(
        update={"execution_index": 1, "turn_index": 1}
    )
    with pytest.raises(ValidationError, match="tool outcome's turn"):
        ExecutableEpisode(
            **{
                **_episode().model_dump(mode="python"),
                "assistant_turns": 2,
                "observed": (_turn(), second_turn),
                "executions": (_execution(), second_execution),
                "events": (
                    ExecutableEvent(
                        index=0,
                        kind="tool_execution",
                        turn_index=0,
                        execution_index=1,
                        reason_code="tool_execution.started",
                    ),
                ),
            }
        )


@pytest.mark.parametrize(
    "envelope",
    [
        {"tool_calls": (_candidate_call(),), "tool_call_outcome_indexes": (0,)},
        {"finish_reason": "stop"},
        {"assistant_content": "an answer"},
    ],
)
def test_a_candidate_call_that_never_completed_returned_no_envelope(
    envelope: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError, match="no envelope to read"):
        ExecutableTurn(
            **{
                **_turn().model_dump(mode="python"),
                "call_status": "timeout",
                "response_hash": None,
                "tool_calls": (),
                "tool_call_outcome_indexes": (),
                "finish_reason": None,
                "advanced": False,
                **envelope,
            }
        )


def test_only_a_call_that_received_a_body_can_identify_one() -> None:
    with pytest.raises(ValidationError, match="received a response body"):
        ExecutableTurn(
            **{
                **_turn().model_dump(mode="python"),
                "call_status": "transport_error",
                "tool_calls": (),
                "tool_call_outcome_indexes": (),
                "finish_reason": None,
                "advanced": False,
            }
        )


def test_a_completed_episode_requires_a_final_state_and_advanced_turns() -> None:
    with pytest.raises(ValidationError, match="final oracle state"):
        ExecutableEpisode(
            **{
                **_episode().model_dump(mode="python"),
                "final_state_hash": None,
            }
        )


def test_assertion_infrastructure_failure_is_terminal_evidence() -> None:
    broken = _assertion().model_copy(
        update={
            "status": "infrastructure_error",
            "reason_code": "assertion.import_failed",
        }
    )
    with pytest.raises(ValidationError, match="assertion infrastructure error"):
        _episode(assertion=broken)


def test_unknown_commit_status_identifies_the_ambiguous_call() -> None:
    unknown = ExecutedToolCall(
        **{
            **_execution().model_dump(mode="python"),
            "status": "unknown_commit_state",
            "state_commit": "unknown",
            "result": None,
            "result_hash": None,
            "released_to_model": False,
            "reason_code": "tool_execution.unknown_commit_state",
        }
    )
    turn = _turn().model_copy(update={"advanced": False})
    base = _episode().model_dump(mode="python")
    episode = ExecutableEpisode(
        **{
            **base,
            "status": "unknown_commit_state",
            "reason_code": "episode.unknown_commit_state",
            "observed": (turn,),
            "executions": (unknown,),
            "final_state_hash": None,
            "assertions": (),
        }
    )

    assert episode.status == "unknown_commit_state"

    returned_without_a_verdict = ExecutedToolCall(
        **{
            **_execution().model_dump(mode="python"),
            "state_commit": "unknown",
            "released_to_model": False,
        }
    )
    unresolved = ExecutableEpisode(
        **{
            **base,
            "status": "unknown_commit_state",
            "reason_code": "episode.unknown_commit_state",
            "observed": (turn,),
            "executions": (returned_without_a_verdict,),
            "final_state_hash": None,
            "assertions": (),
        }
    )
    assert unresolved.executions[0].has_result


def test_a_malformed_result_can_retain_an_independent_unknown_commit_verdict() -> None:
    unknown = ExecutedToolCall(
        **{
            **_execution().model_dump(mode="python"),
            "status": "malformed_result",
            "state_commit": "unknown",
            "result": None,
            "result_hash": None,
            "released_to_model": False,
            "malformed_result": "not an object",
            "malformed_result_type": "str",
            "malformed_result_hash": _hash("not an object"),
            "reason_code": "tool_execution.malformed_result",
        }
    )
    episode = ExecutableEpisode(
        **{
            **_episode().model_dump(mode="python"),
            "status": "unknown_commit_state",
            "reason_code": "episode.unknown_commit_state",
            "observed": (_turn().model_copy(update={"advanced": False}),),
            "executions": (unknown,),
            "final_state_hash": None,
            "assertions": (),
        }
    )

    assert episode.executions[0].status == "malformed_result"
    assert episode.executions[0].state_commit == "unknown"


def test_diagnostic_wording_does_not_change_executable_evidence_identity() -> None:
    episode = _episode()
    execution = episode.executions[0].model_copy(update={"detail": "rewritten execution detail"})
    turn = episode.observed[0].model_copy(update={"detail": "rewritten turn detail"})
    assertion = episode.assertions[0].model_copy(update={"detail": "rewritten assertion detail"})
    event = episode.events[0].model_copy(update={"detail": "rewritten event detail"})
    rewritten = episode.model_copy(
        update={
            "detail": "rewritten episode detail",
            "executions": (execution,),
            "observed": (turn,),
            "assertions": (assertion,),
            "events": (event,),
        }
    )

    assert rewritten.episode_hash == episode.episode_hash


def test_reason_codes_and_structured_results_change_executable_evidence_identity() -> None:
    episode = _episode()
    changed_reason = episode.model_copy(update={"reason_code": "episode.other"})
    changed_execution = _execution(result={"account_id": "A-1", "balance": 501})
    changed_result = episode.model_copy(update={"executions": (changed_execution,)})
    result_detail = _execution(
        result={"account_id": "A-1", "balance": 500, "detail": "oracle-owned result field"}
    )
    changed_result_detail = episode.model_copy(update={"executions": (result_detail,)})

    assert changed_reason.episode_hash != episode.episode_hash
    assert changed_result.episode_hash != episode.episode_hash
    assert changed_result_detail.episode_hash != episode.episode_hash


def test_cache_replay_does_not_change_executable_evidence_identity() -> None:
    assert _episode(replayed=True).episode_hash == _episode(replayed=False).episode_hash


def test_verified_oracle_identity_is_path_free_and_content_bound() -> None:
    def oracle(root: Path, *, resource_hash: str = OTHER_HASH) -> VerifiedOracleSource:
        return VerifiedOracleSource(
            kind="python",
            pack_id="banking",
            pack_version="1.0",
            expected_pack_content_hash=HASH,
            actual_pack_content_hash=HASH,
            pack_root=root,
            pack_manifest_path=root / "manifest.yaml",
            pack_file_count=3,
            resource_role="backend",
            resource_path=root / "backend.py",
            resource_content_hash=resource_hash,
            interface_probed=True,
            backend_interface=("call_tool", "get_state", "list_tools", "reset"),
        )

    first = oracle(Path("/machine-a/pack"))
    moved = oracle(Path("/machine-b/pack"))
    changed = oracle(Path("/machine-a/pack"), resource_hash="sha256:" + "7" * 64)

    assert first.verification_identity == moved.verification_identity
    assert first.verification_identity != changed.verification_identity
