from __future__ import annotations

import asyncio
import hashlib
import json
import queue
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval import (
    eval_runner as batch_runner,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval import (
    oracle_session as oracle_session_module,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.call_matching import (
    CanonicalCallMatchGate,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_cache import (
    CandidateIOCache,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_contract import (
    CandidateAttempt,
    CandidateCallOutcome,
    CandidateResponse,
    CandidateToolCall,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.contamination_contract import (
    CandidateEligibility,
    CommonEvaluationTaskSet,
    EligibleEvalPlan,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.conversation_contract import (
    ConversationScript,
    ScriptedCall,
    ScriptedTurn,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.error_taxonomy import (
    ERROR_TAXONOMY_HASH,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.eval_artifacts import (
    EVAL_MANIFEST_FILE,
    EVAL_REPORT_FILE,
    EVAL_TASK_RESULTS_FILE,
    EvalArtifactError,
    executable_task_result,
    write_executable_eval_artifacts,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_aggregation import (
    aggregate_executable_scores,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver import (
    run_executable_episode,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_errors import (
    ExecutableAuthorizationError,
    OracleSessionError,
    OracleStateError,
    ToolTraceCacheConflictError,
    ToolTraceCacheError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_projection import (
    ExecutableAssertionSpec,
    ExecutableDependency,
    ExecutableProjectionError,
    ExecutableTaskSpec,
    ExecutableToolPolicy,
    _assertion_bindings,
    _dependency_specs,
    build_executable_task_spec,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_scorer import (
    score_executable_episode,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_scoring_contract import (
    ExecutableMetricResult,
    ExecutableTaskScore,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_scoring_errors import (
    ExecutableAggregationError,
    ExecutableEvidenceError,
    ExecutableScoringPolicyError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_trace_parser import (
    parse_executable_trace,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.identity import (
    candidate_identity_claim,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.oracle_session import (
    _ProcessEpisodeBridge,
    open_oracle_session,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.schemas import (
    CandidateApi,
    CandidateInference,
    CandidateModelIdentity,
    EvalCandidate,
    EvalFileRef,
    EvalLimits,
    EvalScoringConfig,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.source_contract import (
    VerifiedOracleSource,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.tool_trace_cache import (
    ToolTraceCache,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.tool_trace_contract import (
    build_tool_trace_request,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.trace_scorer import (
    score_normalized_trace,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    CanonicalExportRow,
    thaw_json,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_projection import (
    CanonicalExportProjection,
    ProjectionSource,
    conversation_plan,
    derive_provenance,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.held_out_contract import (
    fixture_ref,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.isolation import (
    _assertion_verdict,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pack_loader import (
    project_model_facing_tools,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json
from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.state_machine import (
    build_plan,
)

HASH = "sha256:" + "1" * 64
OTHER_HASH = "sha256:" + "2" * 64
SOURCE_IDENTITY = "sha256:" + "3" * 64
TASK_ID = "task-live-1"
PACK_ROOT = Path(__file__).parents[3] / "src" / "nemotron" / "steps" / "byob" / "data" / "tiny_oracle_pack"


def _hash(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


def _hash_file(path: Path | None) -> str:
    assert path is not None
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _candidate() -> EvalCandidate:
    return EvalCandidate(
        alias="candidate_a",
        model="candidate-route",
        provider="nvidia",
        provider_api_version="v1",
        api=CandidateApi(
            base_url="https://candidate.example.com/v1",
            api_key_env="CANDIDATE_API_KEY",
        ),
        model_identity=CandidateModelIdentity(
            source="huggingface",
            model="org/candidate",
            revision="a" * 40,
        ),
        inference=CandidateInference(
            temperature=0.0,
            top_p=1.0,
            max_tokens=512,
            seed=7,
            tool_choice="auto",
            provider_extensions={},
        ),
    )


def _limits() -> EvalLimits:
    return EvalLimits(
        max_turns=4,
        tool_timeout_s=3.0,
        candidate_timeout_s=3.0,
        episode_timeout_s=20.0,
        max_parallel_tasks=1,
        max_retries=0,
    )


def _scoring() -> EvalScoringConfig:
    return EvalScoringConfig(
        contract=EvalFileRef(path="/contract.md", content_hash=HASH),
        argument_matching="schema_then_canonical",
        insert_declared_defaults=True,
        respect_call_order=True,
        respect_call_group=True,
        allow_llm_repair=False,
        task_success="all_applicable_gates",
    )


def _plan(*, source_task_ids_hash: str = OTHER_HASH) -> EligibleEvalPlan:
    candidate = _candidate()
    scoring = _scoring()
    eligibility = CandidateEligibility(
        alias=candidate.alias,
        identity=candidate_identity_claim(candidate),
        canonical_model_identity=candidate.canonical_model_identity,
        eligible_task_ids=(TASK_ID,),
    )
    return EligibleEvalPlan(
        eval_config_hash=HASH,
        scoring_policy_hash=scoring.scoring_policy_hash,
        source_verification_identity=SOURCE_IDENTITY,
        source_run_id="run-1",
        source_task_ids_hash=source_task_ids_hash,
        enforce=True,
        on_violation="fail_run",
        comparison_set="common_intersection",
        candidates=(eligibility,),
        common=CommonEvaluationTaskSet(
            comparison_set="common_intersection",
            task_ids=(TASK_ID,),
            candidate_aliases=(candidate.alias,),
        ),
        publication_allowed=True,
    )


def _oracle() -> VerifiedOracleSource:
    return VerifiedOracleSource(
        kind="python",
        pack_id="tiny_library",
        pack_version="0.1.0",
        expected_pack_content_hash=HASH,
        actual_pack_content_hash=HASH,
        pack_root=PACK_ROOT,
        pack_manifest_path=PACK_ROOT / "manifest.yaml",
        pack_file_count=6,
        resource_role="backend",
        resource_path=PACK_ROOT / "backend.py",
        resource_content_hash=OTHER_HASH,
        interface_probed=True,
        backend_interface=("call_tool", "get_state", "list_tools", "reset"),
    )


def _tools() -> list[dict[str, Any]]:
    full = json.loads((PACK_ROOT / "tools.json").read_text(encoding="utf-8"))
    return project_model_facing_tools(full)


def _wire_call(call_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "get_book_status",
                    "arguments": canonical_json(arguments),
                },
            }
        ],
    }


def _row() -> CanonicalExportRow:
    arguments = {"book_id": "BK-100"}
    return CanonicalExportRow(
        task_id=TASK_ID,
        template_id="lib_status_single",
        variant_index=0,
        messages=(
            {"role": "system", "content": "You are a library assistant."},
            {"role": "user", "content": "Is BK-100 available?"},
            _wire_call("gold-call", arguments),
            {
                "role": "tool",
                "tool_call_id": "gold-call",
                "content": canonical_json({"book_id": "BK-100", "status": "gold-only"}),
            },
            {"role": "assistant", "content": "BK-100 is available."},
        ),
        tools=_tools(),
        expected_tool_calls=(
            {
                "turn_index": 0,
                "call_group": 0,
                "position_in_group": 0,
                "function_name": "get_book_status",
                "arguments": arguments,
            },
        ),
        success_assertions=("assert_book_status_reported",),
        fixture_refs=(canonical_json(["books", "BK-100"]),),
        intent="check_book_status",
        category="circulation",
        difficulty="easy",
        required_tools=("get_book_status",),
        required_tools_fingerprint=canonical_json(["get_book_status"]),
        tools_present=("get_book_status", "checkout_book"),
        turn_policy="single_turn",
        is_multi_turn=False,
        num_tool_calls=1,
        call_order="strict",
        system_prompt_id="library",
        tier="gold",
        gold_eligible=True,
        validated_by=("schema", "replay", "assertions"),
        pack_id="tiny_library",
        pack_version="0.1.0",
        seed=7,
        src="tiny_library:lib_status_single",
        metadata={
            "language": "en",
            "expt_name": "runner",
            "base_task_id": TASK_ID,
            "surface_source": "oracle",
            "profile_hash": "profile",
        },
    )


def _projection(row: CanonicalExportRow) -> CanonicalExportProjection:
    return CanonicalExportProjection(
        source=ProjectionSource(
            file="benchmark.parquet",
            content_hash=OTHER_HASH,
            rows=1,
        ),
        provenance=derive_provenance((row,)),
        rows=(row,),
        plans=(conversation_plan(row),),
    )


def _source(tmp_path: Path, oracle: VerifiedOracleSource) -> Any:
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(
        json.dumps({"oracle_clock": "2026-03-02T09:00:00+07:00"}),
        encoding="utf-8",
    )
    return SimpleNamespace(
        executable=True,
        gold_eligible=True,
        oracle=oracle,
        verification_identity=SOURCE_IDENTITY,
        eval_config_hash=HASH,
        source_run_id="run-1",
        task_ids=(TASK_ID,),
        task_index=SimpleNamespace(task_ids_hash=OTHER_HASH),
        evaluation_benchmark=SimpleNamespace(content_hash=OTHER_HASH, rows=1),
        source_manifest_path=manifest,
    )


def _script(oracle: VerifiedOracleSource) -> ConversationScript:
    tools = _tools()
    call = ScriptedCall(
        call_index=0,
        position_in_group=0,
        function_name="get_book_status",
        arguments={"book_id": "BK-100"},
        recorded_result="",
    )
    return ConversationScript(
        task_id=TASK_ID,
        source_verification_identity=SOURCE_IDENTITY,
        seed_messages=(
            {"role": "system", "content": "You are a library assistant."},
            {"role": "user", "content": "Is BK-100 available?"},
        ),
        tools=tools,
        turns=(
            ScriptedTurn(
                turn_index=0,
                user_turn_index=0,
                call_group=0,
                calls=(call,),
            ),
            ScriptedTurn(
                turn_index=1,
                user_turn_index=0,
                expected_assistant_content="BK-100 is available.",
                is_terminal=True,
            ),
        ),
        user_turns=1,
        required_tools=("get_book_status",),
        call_order="strict",
    )


def _task(oracle: VerifiedOracleSource) -> ExecutableTaskSpec:
    plan = _plan()
    return ExecutableTaskSpec(
        task_id=TASK_ID,
        candidate_alias="candidate_a",
        canonical_model_identity=_candidate().canonical_model_identity,
        plan_identity=plan.plan_identity,
        eval_config_hash=plan.eval_config_hash,
        scoring_policy_hash=plan.scoring_policy_hash,
        source_verification_identity=SOURCE_IDENTITY,
        oracle_verification_identity=oracle.verification_identity,
        source_content_hash=OTHER_HASH,
        oracle_clock="2026-03-02T09:00:00+07:00",
        seed=7,
        fixture_refs=("books:BK-100",),
        script=_script(oracle),
        success_assertions=(),
        turn_policy="single_turn",
        assistant_milestones=(
            {"type": "tool_call", "tool": "get_book_status"},
            {"type": "final_answer"},
        ),
        tool_policies=(
            ExecutableToolPolicy(
                function_name="get_book_status",
                mutates=False,
            ),
            ExecutableToolPolicy(
                function_name="checkout_book",
                mutates=True,
                requires_confirmation=True,
                confirmation_parameter="confirm",
            ),
        ),
        assertion_task={"task_id": TASK_ID},
    )


def _with_assertions(
    task: ExecutableTaskSpec,
    *names: str,
    category: str = "result",
) -> ExecutableTaskSpec:
    """Declare assertions the way the projection does, specs included."""
    return task.model_copy(
        update={
            "success_assertions": names,
            "assertion_specs": tuple(ExecutableAssertionSpec(name=name, category=category) for name in names),
        }
    )


def _confirmation_task(oracle: VerifiedOracleSource) -> ExecutableTaskSpec:
    task = _task(oracle)
    call = ScriptedCall(
        call_index=0,
        position_in_group=0,
        function_name="checkout_book",
        arguments={
            "book_id": "BK-100",
            "patron_id": "P-1",
            "confirm": True,
        },
        recorded_result="",
    )
    script = ConversationScript(
        task_id=TASK_ID,
        source_verification_identity=SOURCE_IDENTITY,
        seed_messages=task.script.seed_messages,
        tools=task.script.tools,
        turns=(
            ScriptedTurn(
                turn_index=0,
                user_turn_index=0,
                expected_assistant_content="Please confirm the checkout.",
                releases_user_message={
                    "role": "user",
                    "content": "Yes, check out the book.",
                },
            ),
            ScriptedTurn(
                turn_index=1,
                user_turn_index=1,
                call_group=0,
                calls=(call,),
            ),
            ScriptedTurn(
                turn_index=2,
                user_turn_index=1,
                expected_assistant_content="The checkout is complete.",
                is_terminal=True,
            ),
        ),
        user_turns=2,
        required_tools=("checkout_book",),
        call_order="strict",
    )
    return task.model_copy(
        update={
            "script": script,
            "turn_policy": "confirmation",
            "confirmed_call_turns": (1,),
        }
    )


def _confirmation_probe_task(oracle: VerifiedOracleSource) -> ExecutableTaskSpec:
    """A gold trace that probes with ``confirm: false`` before the user answers."""

    task = _task(oracle)
    probe = ScriptedCall(
        call_index=0,
        position_in_group=0,
        function_name="checkout_book",
        arguments={"book_id": "BK-100", "patron_id": "P-1", "confirm": False},
        recorded_result="",
    )
    commit = ScriptedCall(
        call_index=1,
        position_in_group=0,
        function_name="checkout_book",
        arguments={"book_id": "BK-100", "patron_id": "P-1", "confirm": True},
        recorded_result="",
    )
    script = ConversationScript(
        task_id=TASK_ID,
        source_verification_identity=SOURCE_IDENTITY,
        seed_messages=task.script.seed_messages,
        tools=task.script.tools,
        turns=(
            ScriptedTurn(turn_index=0, user_turn_index=0, call_group=0, calls=(probe,)),
            ScriptedTurn(
                turn_index=1,
                user_turn_index=0,
                expected_assistant_content="Please confirm the checkout.",
                releases_user_message={
                    "role": "user",
                    "content": "Yes, check out the book.",
                },
            ),
            ScriptedTurn(turn_index=2, user_turn_index=1, call_group=1, calls=(commit,)),
            ScriptedTurn(
                turn_index=3,
                user_turn_index=1,
                expected_assistant_content="The checkout is complete.",
                is_terminal=True,
            ),
        ),
        user_turns=2,
        required_tools=("checkout_book",),
        call_order="strict",
    )
    return task.model_copy(
        update={
            "script": script,
            "turn_policy": "confirmation",
            "confirmed_call_turns": (2,),
        }
    )


def _dependent_task(oracle: VerifiedOracleSource) -> ExecutableTaskSpec:
    task = _task(oracle)
    producer = ScriptedCall(
        call_index=0,
        position_in_group=0,
        function_name="get_book_status",
        arguments={"book_id": "BK-100"},
        recorded_result="",
    )
    consumer = ScriptedCall(
        call_index=1,
        position_in_group=0,
        function_name="get_book_status",
        arguments={"book_id": "BK-GOLD"},
        recorded_result="",
    )
    script = ConversationScript(
        task_id=TASK_ID,
        source_verification_identity=SOURCE_IDENTITY,
        seed_messages=task.script.seed_messages,
        tools=task.script.tools,
        turns=(
            ScriptedTurn(
                turn_index=0,
                user_turn_index=0,
                call_group=0,
                calls=(producer,),
            ),
            ScriptedTurn(
                turn_index=1,
                user_turn_index=0,
                call_group=1,
                calls=(consumer,),
            ),
            ScriptedTurn(
                turn_index=2,
                user_turn_index=0,
                expected_assistant_content="The latest book is available.",
                is_terminal=True,
            ),
        ),
        user_turns=1,
        required_tools=("get_book_status",),
        call_order="strict",
    )
    return task.model_copy(
        update={
            "script": script,
            "turn_policy": "dependent_call",
            "dependencies": (
                ExecutableDependency(
                    dependency_index=0,
                    consumer_call_index=1,
                    consumer_turn_index=1,
                    consumer_position_in_turn=0,
                    argument_path=("book_id",),
                    producer_call_index=0,
                    producer_turn_index=0,
                    result_path="book_id",
                    expected_value_type="string",
                ),
            ),
        }
    )


def _scripted_user_task(
    oracle: VerifiedOracleSource,
    *,
    turn_policy: str,
    user_content: str,
) -> ExecutableTaskSpec:
    task = _task(oracle)
    call = ScriptedCall(
        call_index=0,
        position_in_group=0,
        function_name="get_book_status",
        arguments={"book_id": "BK-100"},
        recorded_result="",
    )
    script = ConversationScript(
        task_id=TASK_ID,
        source_verification_identity=SOURCE_IDENTITY,
        seed_messages=task.script.seed_messages,
        tools=task.script.tools,
        turns=(
            ScriptedTurn(
                turn_index=0,
                user_turn_index=0,
                expected_assistant_content="Which book should I check?",
                releases_user_message={"role": "user", "content": user_content},
            ),
            ScriptedTurn(
                turn_index=1,
                user_turn_index=1,
                call_group=0,
                calls=(call,),
            ),
            ScriptedTurn(
                turn_index=2,
                user_turn_index=1,
                expected_assistant_content="BK-100 is available.",
                is_terminal=True,
            ),
        ),
        user_turns=2,
        required_tools=("get_book_status",),
        call_order="strict",
    )
    return task.model_copy(update={"script": script, "turn_policy": turn_policy})


class _FakeOracle:
    def __init__(self, identity: str) -> None:
        self.oracle_verification_identity = identity
        self.calls: list[tuple[str, dict[str, Any], int]] = []
        self.closed = False

    async def reset(self) -> None:
        return None

    async def call_tool(self, function_name: str, arguments: dict[str, Any], *, turn_index: int) -> Any:
        self.calls.append((function_name, arguments, turn_index))
        return {"book_id": "BK-100", "status": "available"}

    async def get_state(self) -> dict[str, Any]:
        return {"books": [{"book_id": "BK-100", "status": "available"}]}

    async def run_assertion(self, name: str, *, task: dict[str, Any]) -> dict[str, Any]:
        return {"name": name, "status": "passed", "passed": True, "detail": None}

    async def close(self) -> None:
        self.closed = True


class _TimeoutOracle(_FakeOracle):
    async def call_tool(self, function_name: str, arguments: dict[str, Any], *, turn_index: int) -> Any:
        self.calls.append((function_name, arguments, turn_index))
        raise TimeoutError("response lost")


class _FatalThenAssertionOracle(_TimeoutOracle):
    def __init__(self, identity: str) -> None:
        super().__init__(identity)
        self.assertion_calls = 0

    async def run_assertion(self, name: str, *, task: dict[str, Any]) -> dict[str, Any]:
        self.assertion_calls += 1
        return {
            "name": name,
            "status": "infrastructure_error",
            "passed": False,
            "detail": "must not run after fatal tool evidence",
        }


class _SlowResetStateFailingOracle(_FakeOracle):
    async def reset(self) -> None:
        await asyncio.sleep(0.05)

    async def get_state(self) -> dict[str, Any]:
        raise OracleStateError(
            "eval.oracle.state",
            "could not read the final state",
            expected="a state object",
            recovery="restart the session",
        )


class _InvalidJsonObjectOracle(_FakeOracle):
    async def call_tool(self, function_name: str, arguments: dict[str, Any], *, turn_index: int) -> Any:
        self.calls.append((function_name, arguments, turn_index))
        return {1: "object keys must be strings"}


class _UnreadableVerdictOracle(_FakeOracle):
    async def run_assertion(self, name: str, *, task: dict[str, Any]) -> dict[str, Any]:
        return {"name": name, "outcome": "probably fine"}

    async def close(self) -> None:
        self.closed = True
        raise OracleSessionError(
            "eval.oracle.close",
            "failed during cleanup",
            expected="a closed session",
            recovery="discard the session",
        )


class _PendingMutationOracle(_FakeOracle):
    async def call_tool(self, function_name: str, arguments: dict[str, Any], *, turn_index: int) -> Any:
        self.calls.append((function_name, arguments, turn_index))
        return {"status": "awaiting_confirmation"}


class _FakeClient:
    def __init__(self, responses: list[CandidateResponse]) -> None:
        self.responses = responses
        self.requests: list[Any] = []
        self.outcomes: list[CandidateCallOutcome] = []

    async def complete(self, request: Any, *, deadline: float) -> CandidateCallOutcome:
        del deadline
        self.requests.append(request)
        response = self.responses.pop(0)
        outcome = CandidateCallOutcome(
            request_hash=request.request_hash,
            status="completed",
            attempts=(
                CandidateAttempt(
                    attempt_index=0,
                    observed_at="2026-01-01T00:00:00+00:00",
                    status="completed",
                    retryable=False,
                    http_status=200,
                    latency_s=0.01,
                    raw_response="{}",
                    raw_response_hash=("sha256:" + hashlib.sha256(b"{}").hexdigest()),
                ),
            ),
            response=response,
        )
        self.outcomes.append(outcome)
        return outcome


def _response_with_call(
    arguments: dict[str, Any],
    *,
    function_name: str = "get_book_status",
) -> CandidateResponse:
    raw = canonical_json(arguments)
    return CandidateResponse(
        assistant_content=None,
        tool_calls=(
            CandidateToolCall(
                index=0,
                id="candidate-call",
                type="function",
                function_name=function_name,
                raw_arguments=raw,
                parsed_arguments=arguments,
                arguments_status="valid_object",
            ),
        ),
        finish_reason="tool_calls",
        selected_attempt=0,
        raw_response_hash=HASH,
    )


def _text_response(text: str) -> CandidateResponse:
    return CandidateResponse(
        assistant_content=text,
        finish_reason="stop",
        selected_attempt=0,
        raw_response_hash=HASH,
    )


def test_executable_projection_binds_plan_source_oracle_and_pack_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle = _oracle()
    source = _source(tmp_path, oracle)
    row = _row()
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_projection.assert_source_unchanged",
        lambda _source: None,
    )

    task = build_executable_task_spec(
        _projection(row),
        TASK_ID,
        candidate_alias="candidate_a",
        source=source,
        plan=_plan(),
    )

    assert task.oracle_verification_identity == oracle.verification_identity
    assert task.script.source_verification_identity == SOURCE_IDENTITY
    assert task.tool_policy("checkout_book").mutates
    assert task.assistant_milestones[-1]["type"] == "final_answer"
    assert task.assertion_task["slots"] == {"book_id": "BK-100"}
    assert task.assertion_task["assertion_task_schema_version"] == "1.0"
    assert task.assertion_task["intent"] == "check_book_status"
    assert task.assertion_spec("assert_book_status_reported").category == "result"
    assert task.assertion_spec("assert_book_status_reported").trace_compatible
    assert "user_turn_templates" in task.assertion_task
    assert "recorded_result" not in task.assertion_task
    assert "gold-only" not in canonical_json(task.model_dump(mode="json"))


def test_direct_publication_tool_description_must_match_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = _oracle()
    source = _source(tmp_path, oracle)
    row = _row()
    tools = json.loads(canonical_json(row.tools))
    tools[0]["function"]["description"] = "Unverified description."
    changed = row.model_copy(update={"tools": tools})
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_projection.assert_source_unchanged",
        lambda _source: None,
    )

    with pytest.raises(ExecutableProjectionError, match="do not match"):
        build_executable_task_spec(
            _projection(changed),
            TASK_ID,
            candidate_alias="candidate_a",
            source=source,
            plan=_plan(),
        )


def test_executable_projection_preserves_from_result_dependency_coordinates() -> None:
    template = {
        "template_id": "dependent",
        "turn_policy": "dependent_call",
        "assistant_milestones": [
            {
                "id": "producer",
                "type": "tool_call",
                "tool": "get_book_status",
                "call_group": 0,
            },
            {
                "type": "tool_call",
                "tool": "get_book_status",
                "call_group": 1,
                "args": {
                    "book_id": {
                        "from_result": {
                            "call": "producer",
                            "path": "book_id",
                        }
                    }
                },
            },
            {"type": "final_answer"},
        ],
    }
    plan = build_plan(
        template,
        {"task_id": TASK_ID, "template_id": "dependent"},
    )

    dependencies = _dependency_specs(
        row=_row(),
        plan=plan,
        script=_dependent_task(_oracle()).script,
    )

    assert len(dependencies) == 1
    assert dependencies[0].producer_call_index == 0
    assert dependencies[0].consumer_call_index == 1
    assert dependencies[0].consumer_turn_index == 1
    assert dependencies[0].argument_path == ("book_id",)
    assert dependencies[0].result_path == "book_id"


def _binding_row(
    *,
    opening: str,
    surface_source: str = "template",
    expected: tuple[tuple[str, dict[str, Any]], ...] = (),
    fixture_refs: tuple[str, ...] = (),
) -> Any:
    return SimpleNamespace(
        task_id=TASK_ID,
        messages=(SimpleNamespace(role="user", content=opening),),
        metadata={"surface_source": surface_source, "language": "en"},
        expected_tool_calls=tuple(
            SimpleNamespace(
                function_name=tool,
                arguments=arguments,
                call_group=0,
                position_in_group=position,
            )
            for position, (tool, arguments) in enumerate(expected)
        ),
        fixture_refs=fixture_refs,
    )


def test_a_slot_the_expected_trace_never_names_is_read_back_from_the_surface() -> None:
    # Two calls to one tool make milestone-to-call pairing ambiguous, and the
    # tool argument is named book_id rather than either slot, so the published
    # opening turn is the only evidence that says which book is which.
    template = {
        "slots": {
            "book_a": {"source": "fixture:books.book_id"},
            "book_b": {"source": "fixture:books.book_id"},
        },
        "user_turn_templates": {"en": "Please check status for {book_a} and {book_b}."},
    }
    bindings = _assertion_bindings(
        _binding_row(
            opening="Please check status for BK-100 and BK-200.",
            expected=(
                ("get_book_status", {"book_id": "BK-100"}),
                ("get_book_status", {"book_id": "BK-200"}),
            ),
            fixture_refs=(
                canonical_json(["books", "BK-100"]),
                canonical_json(["books", "BK-200"]),
            ),
        ),
        template,
        [
            {"type": "tool_call", "tool": "get_book_status", "args": {"book_id": "{book_a}"}},
            {"type": "tool_call", "tool": "get_book_status", "args": {"book_id": "{book_b}"}},
        ],
        fixtures={
            "books": [
                {"book_id": "BK-100", "status": "available"},
                {"book_id": "BK-200", "status": "on_loan"},
            ]
        },
    )

    assert bindings["slots"] == {"book_a": "BK-100", "book_b": "BK-200"}
    assert bindings["unresolved_slots"] == []


def test_a_slot_no_fixture_holds_is_still_recovered_from_the_surface() -> None:
    # A negative-path template binds a value that deliberately exists in no
    # fixture, so only the surface can state it.
    bindings = _assertion_bindings(
        _binding_row(opening="How is transaction TXN-MISSING going?"),
        {
            "slots": {"transaction_id": {"source": "absent:transactions"}},
            "user_turn_templates": {"en": "How is transaction {transaction_id} going?"},
        },
        [],
        pack_manifest={"absent_ids": {"transactions": ["TXN-MISSING"]}},
    )

    assert bindings["slots"] == {"transaction_id": "TXN-MISSING"}
    assert bindings["unresolved_slots"] == []


def test_a_paraphrased_surface_is_never_read_back_as_a_slot_value() -> None:
    bindings = _assertion_bindings(
        _binding_row(
            opening="Hey, could you look up how BK-999 is doing?",
            surface_source="model",
        ),
        {
            "slots": {"book_id": {"source": "absent:books"}},
            "user_turn_templates": {"en": "Can you check whether book {book_id} is available?"},
        },
        [],
    )

    assert bindings["slots"] == {}
    assert bindings["unresolved_slots"] == ["book_id"]


def test_an_unrecoverable_slot_is_named_rather_than_guessed_or_refused() -> None:
    # The pack offers two literals and the row publishes nothing that picks one.
    # Projection must not invent a value, and must not deny the task either: an
    # assertion that reads the slot fails as infrastructure, one that does not
    # still runs.
    bindings = _assertion_bindings(
        _binding_row(opening="Is BK-100 available?"),
        {"slots": {"unbound": {"source": "literal:[1, 2]"}}},
        [],
    )

    assert bindings["slots"] == {}
    assert bindings["unresolved_slots"] == ["unbound"]


def test_a_surface_slot_keeps_the_declared_type_its_rendering_erased() -> None:
    bindings = _assertion_bindings(
        _binding_row(opening="Send 200000 to 970436."),
        {
            "slots": {
                "amount": {"source": "literal:[200000]"},
                "bank": {"source": "literal:['970436']"},
            },
            "user_turn_templates": {"en": "Send {amount} to {bank}."},
        },
        [],
    )

    assert bindings["slots"] == {"amount": 200000, "bank": "970436"}


def test_surface_recovery_uses_every_verified_pack_source_kind() -> None:
    bindings = _assertion_bindings(
        _binding_row(
            opening="Book The Trial, mode compact, page 2, missing 404.",
            fixture_refs=(canonical_json(["books", "BK-100"]),),
        ),
        {
            "slots": {
                "title": {"source": "fixture:books.title"},
                "mode": {"source": "enum:search_books.mode"},
                "page": {"source": "range:{'min': 1, 'max': 3}"},
                "missing": {"source": "absent:books"},
            },
            "user_turn_templates": {"en": "Book {title}, mode {mode}, page {page}, missing {missing}."},
        },
        [],
        pack_manifest={"absent_ids": {"books": [404]}},
        fixtures={"books": [{"book_id": "BK-100", "title": "The Trial"}]},
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search_books",
                    "parameters": {
                        "type": "object",
                        "properties": {"mode": {"type": "string", "enum": ["full", "compact"]}},
                    },
                },
            }
        ],
    )

    assert bindings["slots"] == {
        "title": "The Trial",
        "mode": "compact",
        "page": 2,
        "missing": 404,
    }
    assert bindings["unresolved_slots"] == []


def test_a_declared_null_slot_is_not_confused_with_an_unresolved_value() -> None:
    bindings = _assertion_bindings(
        _binding_row(opening="Optional value: None."),
        {
            "slots": {"optional": {"source": "literal:[None]"}},
            "user_turn_templates": {"en": "Optional value: {optional}."},
        },
        [],
    )

    assert "optional" in bindings["slots"]
    assert bindings["slots"]["optional"] is None
    assert bindings["unresolved_slots"] == []


def test_multiple_corrections_leave_the_last_delivered_value_in_force() -> None:
    bindings = _assertion_bindings(
        _binding_row(opening="Use original.", expected=()),
        {
            "slots": {"value": {"source": "literal:['original']"}},
            "user_turn_templates": {"en": "Use {value}."},
            "user_simulator_turns": [
                {
                    "after": "first",
                    "slot_updates": {"value": {"source": "literal:['middle']"}},
                },
                {
                    "after": "second",
                    "slot_updates": {"value": {"source": "literal:['final']"}},
                },
            ],
        },
        [{"id": "first"}, {"id": "second"}],
    )

    assert bindings["slots_initial"] == {"value": "original"}
    assert bindings["slot_updates"] == [
        {"entry_index": 0, "values": {"value": "middle"}, "aliases": {}},
        {"entry_index": 1, "values": {"value": "final"}, "aliases": {}},
    ]
    assert bindings["slots"] == {"value": "final"}
    assert bindings["unresolved_slots"] == []


def test_an_unknown_initial_value_is_distinct_from_a_known_corrected_value() -> None:
    bindings = _assertion_bindings(
        _binding_row(
            opening="Use the value I mentioned.",
            surface_source="model",
            expected=(("set_value", {"value": 20}),),
        ),
        {
            "slots": {"value": {"source": "literal:[10, 15]"}},
            "user_simulator_turns": [
                {
                    "after": "correction",
                    "slot_updates": {"value": {"source": "literal:[20]"}},
                }
            ],
        },
        [
            {"id": "correction"},
            {
                "type": "tool_call",
                "tool": "set_value",
                "args": {"value": "{value}"},
                "call_group": 0,
            },
        ],
    )

    assert bindings["slots"] == {"value": 20}
    assert bindings["slots_initial"] == {}
    assert bindings["unresolved_slots"] == []
    assert bindings["unresolved_slots_initial"] == ["value"]


def test_an_unknown_correction_keeps_its_timeline_position() -> None:
    bindings = _assertion_bindings(
        _binding_row(opening="Use original."),
        {
            "slots": {"value": {"source": "literal:['original']"}},
            "user_turn_templates": {"en": "Use {value}."},
            "user_simulator_turns": [
                {
                    "after": "correction",
                    "slot_updates": {"value": {"source": "literal:['middle', 'final']"}},
                }
            ],
        },
        [{"id": "correction"}],
    )

    assert bindings["slot_updates"] == [{"entry_index": 0, "values": {}, "aliases": {}}]
    assert bindings["unresolved_slot_updates"] == [{"update_index": 0, "entry_index": 0, "slots": ["value"]}]


def test_repeated_tool_milestones_pair_by_call_group_position() -> None:
    bindings = _assertion_bindings(
        _binding_row(
            opening="Check the hidden books.",
            surface_source="model",
            expected=(
                ("get_book_status", {"book_id": "BK-100"}),
                ("get_book_status", {"book_id": "BK-200"}),
            ),
        ),
        {
            "slots": {
                "book_a": {"source": "absent:books"},
                "book_b": {"source": "absent:books"},
            }
        },
        [
            {
                "type": "tool_call",
                "tool": "get_book_status",
                "args": {"book_id": "{book_a}"},
                "call_group": 0,
            },
            {
                "type": "tool_call",
                "tool": "get_book_status",
                "args": {"book_id": "{book_b}"},
                "call_group": 0,
            },
        ],
    )

    assert bindings["slots"] == {"book_a": "BK-100", "book_b": "BK-200"}


def test_assertion_reading_an_unresolved_slot_is_infrastructure() -> None:
    def needs_slot(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
        if task.get("slots", {}).get("book_id") is None:
            raise AssertionError("candidate omitted the expected book")

    verdict = _assertion_verdict(
        needs_slot,
        name="needs_slot",
        state={},
        trace=[],
        task={"slots": {}, "unresolved_slots": ["book_id"]},
        ctx=None,
    )

    assert verdict["status"] == "infrastructure_error"
    assert "book_id" in verdict["detail"]


def test_assertion_can_explicitly_report_not_applicable() -> None:
    def optional_assertion(*, state: dict, trace: list, task: dict, ctx: Any) -> dict[str, str]:
        return {
            "status": "not_applicable",
            "detail": "the task declares no final-answer requirement",
        }

    verdict = _assertion_verdict(
        optional_assertion,
        name="optional_assertion",
        state={},
        trace=[],
        task={},
        ctx=None,
    )

    assert verdict == {
        "name": "optional_assertion",
        "status": "not_applicable",
        "passed": False,
        "detail": "the task declares no final-answer requirement",
    }


def test_malformed_assertion_return_is_infrastructure() -> None:
    def malformed(*, state: dict, trace: list, task: dict, ctx: Any) -> bool:
        return True

    verdict = _assertion_verdict(
        malformed,
        name="malformed",
        state={},
        trace=[],
        task={},
        ctx=None,
    )

    assert verdict["status"] == "infrastructure_error"
    assert "AssertionContractError" in verdict["detail"]


def test_unresolved_slot_unrelated_to_an_assertion_does_not_mask_failure() -> None:
    def no_tools(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
        if trace:
            raise AssertionError("a tool was called")

    verdict = _assertion_verdict(
        no_tools,
        name="no_tools",
        state={},
        trace=[{"tool": "get_book_status"}],
        task={"slots": {}, "unresolved_slots": ["book_id"]},
        ctx=None,
    )

    assert verdict["status"] == "failed"


def test_an_empty_slots_object_still_reports_the_slot_an_assertion_needed() -> None:
    # Packs read slots through `task.get("slots") or {}`. An empty object is
    # falsy, so the fallback literal escapes read tracking unless the parent
    # records the whole child. Without that, an unbindable slot is charged to
    # the candidate as a failure instead of to the evaluator.
    def needs_slot(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
        slots = task.get("slots") or {}
        if slots.get("transaction_id") is None:
            raise AssertionError("candidate never named the transaction")

    verdict = _assertion_verdict(
        needs_slot,
        name="needs_slot",
        state={},
        trace=[],
        task={"slots": {}, "unresolved_slots": ["transaction_id"]},
        ctx=None,
    )

    assert verdict["status"] == "infrastructure_error"
    assert "transaction_id" in verdict["detail"]


def test_reading_an_unknown_initial_slot_is_infrastructure() -> None:
    def needs_initial(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
        initial = task.get("slots_initial") or {}
        if initial.get("amount") is None:
            raise AssertionError("the original amount is missing")

    verdict = _assertion_verdict(
        needs_initial,
        name="needs_initial",
        state={},
        trace=[],
        task={
            "slots": {"amount": 20},
            "slots_initial": {},
            "unresolved_slots": [],
            "unresolved_slots_initial": ["amount"],
        },
        ctx=None,
    )

    assert verdict["status"] == "infrastructure_error"
    assert "slots_initial.amount" in verdict["detail"]


def test_all_dictionary_reads_preserve_unresolved_slot_evidence() -> None:
    def by_length(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
        if len(task["slots"]) != 2:
            raise AssertionError("a slot is missing")

    def by_inequality(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
        if task["slots"] != {"known": 1, "missing": 2}:
            raise AssertionError("a slot is missing")

    def by_pop(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
        if task["slots"].pop("missing", None) is None:
            raise AssertionError("a slot is missing")

    def by_popitem(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
        task["slots"].popitem()
        raise AssertionError("the complete mapping was required")

    for assertion in (by_length, by_inequality, by_pop, by_popitem):
        verdict = _assertion_verdict(
            assertion,
            name=assertion.__name__,
            state={},
            trace=[],
            task={"slots": {"known": 1}, "unresolved_slots": ["missing"]},
            ctx=None,
        )
        assert verdict["status"] == "infrastructure_error"
        assert "slots.missing" in verdict["detail"]


def test_testing_a_populated_slots_object_is_not_reading_every_slot() -> None:
    # `task.get("slots") or {}` is the pack idiom, and on a mapping that still
    # holds something its answer does not depend on the slot that went unbound.
    # Charging the idiom itself as a read of every slot turned correct pack
    # behavior into evaluator evidence and lost the verdict the assertion reached.
    def reads_one_bound_slot(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
        slots = task.get("slots") or {}
        if slots.get("account_id") != "ACC-002":
            raise AssertionError("candidate reported another account")

    task = {"slots": {"account_id": "ACC-002"}, "unresolved_slots": ["service"]}

    assert _assertion_verdict(
        reads_one_bound_slot,
        name="reads_one_bound_slot",
        state={},
        trace=[],
        task=task,
        ctx=None,
    )["status"] == "passed"

    def reads_the_unbound_slot(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
        slots = task.get("slots") or {}
        if slots.get("service") is None:
            raise AssertionError("candidate never named the service")

    verdict = _assertion_verdict(
        reads_the_unbound_slot,
        name="reads_the_unbound_slot",
        state={},
        trace=[],
        task=task,
        ctx=None,
    )

    assert verdict["status"] == "infrastructure_error"
    assert "slots.service" in verdict["detail"]


def test_reading_an_unresolved_slot_update_is_infrastructure() -> None:
    def needs_update(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
        values = task["slot_updates"][0].get("values") or {}
        if values.get("amount") is None:
            raise AssertionError("the corrected amount is missing")

    verdict = _assertion_verdict(
        needs_update,
        name="needs_update",
        state={},
        trace=[],
        task={
            "slot_updates": [{"entry_index": 0, "values": {}, "aliases": {}}],
            "unresolved_slot_updates": [{"update_index": 0, "entry_index": 0, "slots": ["amount"]}],
        },
        ctx=None,
    )

    assert verdict["status"] == "infrastructure_error"
    assert "slot_updates[0].values.amount" in verdict["detail"]


def test_a_fixture_slot_binds_when_the_collection_keys_rows_by_number() -> None:
    # fixture_ref renders the cited primary id with str, so a numeric key is
    # published as text while the fixture row keeps the number.
    bindings = _assertion_bindings(
        _binding_row(
            opening="Check book The Trial.",
            fixture_refs=(fixture_ref("books", 100),),
        ),
        {
            "slots": {"title": {"source": "fixture:books.title"}},
            "user_turn_templates": {"en": "Check book {title}."},
        },
        [],
        fixtures={"books": [{"book_id": 100, "title": "The Trial"}]},
    )

    assert bindings["slots"] == {"title": "The Trial"}
    assert bindings["unresolved_slots"] == []


def test_fixture_shorthand_and_filters_match_generation_semantics() -> None:
    bindings = _assertion_bindings(
        _binding_row(
            opening="Check the selected books.",
            surface_source="model",
            fixture_refs=(
                fixture_ref("books", "BK-100"),
                fixture_ref("books", "BK-200"),
            ),
        ),
        {
            "slots": {
                "available_book": {
                    "source": "books.book_id",
                    "filter": "status == 'available'",
                },
                "loaned_book": {
                    "source": "books.book_id",
                    "filter": "status == 'on_loan'",
                },
            }
        },
        [],
        fixtures={
            "books": [
                {"book_id": "BK-100", "status": "available"},
                {"book_id": "BK-200", "status": "on_loan"},
            ]
        },
    )

    assert bindings["slots"] == {
        "available_book": "BK-100",
        "loaned_book": "BK-200",
    }
    assert bindings["unresolved_slots"] == []


async def _a_state_failure_keeps_the_reason_the_episode_already_ended_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver.assert_source_unchanged",
        lambda _source: None,
    )

    return await run_executable_episode(
        candidate=_candidate(),
        limits=EvalLimits(
            max_turns=4,
            tool_timeout_s=0.01,
            candidate_timeout_s=0.01,
            episode_timeout_s=0.01,
            max_parallel_tasks=1,
            max_retries=0,
        ),
        client=_FakeClient([]),  # type: ignore[arg-type]
        task=_task(oracle_source),
        source=source,
        plan=_plan(),
        oracle=_SlowResetStateFailingOracle(oracle_source.verification_identity),
        gate=CanonicalCallMatchGate(_scoring()),
    )


def test_a_state_failure_keeps_the_reason_the_episode_already_ended_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The deadline elapses before the candidate is asked anything, so the
    # episode ends with no observed turn. Adopting the later state-snapshot
    # failure as the terminal status would describe an episode shape the
    # contract forbids, and the whole record would be lost to a validation
    # error instead of reporting the timeout that actually ended it.
    episode = asyncio.run(_a_state_failure_keeps_the_reason_the_episode_already_ended_for(tmp_path, monkeypatch))

    assert episode.status == "episode_timeout"
    assert episode.observed == ()
    assert episode.final_state_hash is None


def test_scorer_classifies_missing_assertions_when_final_state_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    task = _with_assertions(_task(oracle_source), "assert_book_status_reported")
    oracle = _SlowResetStateFailingOracle(oracle_source.verification_identity)
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=oracle,
            responses=[_text_response("This does not match the expected call.")],
        )
    )

    assert episode.status == "candidate_mismatch"
    assert episode.final_state_hash is None
    assert episode.assertions == ()
    score = score_executable_episode(
        episode=episode,
        task=task,
        scoring=_scoring(),
        plan=_plan(),
    )
    assert score.gate("assertions").failure_class == "infrastructure"
    assert score.gate("assertions").reason_code == "executable.assertion_state_unavailable"
    assert score.non_candidate_stop


def test_closing_a_failed_bridge_still_signals_its_episode_iterator() -> None:
    class FinishedThread:
        def join(self, timeout: float) -> None:
            self.timeout = timeout

        def is_alive(self) -> bool:
            return False

    bridge = object.__new__(_ProcessEpisodeBridge)
    bridge._exchange_lock = threading.Lock()
    bridge._closed = True
    bridge._close_requested = False
    bridge._started = True
    bridge._commands = queue.Queue()
    bridge._thread = FinishedThread()
    bridge._limits = SimpleNamespace(tool_timeout_s=1)

    bridge._close_sync()

    assert bridge._commands.get_nowait() is None


async def _fatal_tool_status_is_not_overwritten_by_later_assertions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    task = _with_assertions(_task(oracle_source), "assert_book_status_reported")
    oracle = _FatalThenAssertionOracle(oracle_source.verification_identity)
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver.assert_source_unchanged",
        lambda _source: None,
    )

    episode = await run_executable_episode(
        candidate=_candidate(),
        limits=_limits(),
        client=_FakeClient(  # type: ignore[arg-type]
            [_response_with_call({"book_id": "BK-100"})]
        ),
        task=task,
        source=source,
        plan=_plan(),
        oracle=oracle,
        gate=CanonicalCallMatchGate(_scoring()),
    )

    assert episode.status == "oracle_timeout"
    assert episode.assertions == ()
    assert oracle.assertion_calls == 0


def test_a_fatal_tool_status_is_not_overwritten_by_later_assertions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(_fatal_tool_status_is_not_overwritten_by_later_assertions(tmp_path, monkeypatch))


async def _python_oracle_session_keeps_state_in_one_isolated_worker(
    tmp_path: Path,
) -> None:
    oracle = _oracle()
    source = _source(tmp_path, oracle)
    task = _task(oracle)
    session = open_oracle_session(
        source=source,
        task=task,
        limits=_limits(),
    )
    assert not session._bridge._started
    assert not session._bridge._thread.is_alive()
    try:
        await session.reset()
        result = await session.call_tool(
            "get_book_status",
            {"book_id": "BK-100"},
            turn_index=0,
        )
        state = await session.get_state()
    finally:
        await session.close()

    assert result["book_id"] == "BK-100"
    assert state["books"][0]["book_id"] == "BK-100"


def test_python_oracle_session_keeps_state_in_one_isolated_worker(
    tmp_path: Path,
) -> None:
    asyncio.run(_python_oracle_session_keeps_state_in_one_isolated_worker(tmp_path))


async def _python_oracle_session_excludes_held_out_fixture_state(
    tmp_path: Path,
) -> None:
    pack_root = tmp_path / "pack"
    shutil.copytree(PACK_ROOT, pack_root)
    (pack_root / "held_out.yaml").write_text(
        yaml.safe_dump(
            {
                "version": "1",
                "fixtures": {"books": ["BK-200"]},
                "templates": [],
                "policy": {"fixtures_in_backend_state": False, "seed": 0},
            }
        ),
        encoding="utf-8",
    )
    manifest_path = pack_root / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["held_out"] = "held_out.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    # A backend must not recover the full inventory through its source location.
    # Redefine reset after the pack's implementation to model that bypass.
    backend_path = pack_root / "backend.py"
    backend_path.write_text(
        backend_path.read_text(encoding="utf-8")
        + """

def reset(*, ctx, fixtures=None):
    import json
    from pathlib import Path
    global _FIXTURE_SNAPSHOT, _STATE, _CTX, _ID_COUNTER
    _FIXTURE_SNAPSHOT = json.loads(Path(__file__).with_name("fixtures.json").read_text())
    _STATE = copy.deepcopy(_FIXTURE_SNAPSHOT)
    _CTX = ctx
    _ID_COUNTER = 0
""",
        encoding="utf-8",
    )
    oracle = _oracle().model_copy(
        update={
            "pack_root": pack_root,
            "pack_manifest_path": manifest_path,
            "pack_file_count": 7,
            "resource_path": backend_path,
        }
    )
    session = open_oracle_session(
        source=_source(tmp_path, oracle),
        task=_task(oracle),
        limits=_limits(),
    )
    try:
        await session.reset()
        state = await session.get_state()
    finally:
        await session.close()

    book_ids = {book["book_id"] for book in state["books"]}
    assert "BK-100" in book_ids
    assert "BK-200" not in book_ids


def test_python_oracle_session_excludes_held_out_fixture_state(
    tmp_path: Path,
) -> None:
    asyncio.run(_python_oracle_session_excludes_held_out_fixture_state(tmp_path))


def test_pack_fixtures_are_parsed_once_per_revision_across_task_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch run opens one session per task against the same pack revision."""
    oracle = _oracle()
    source = _source(tmp_path, oracle)
    task = _task(oracle)
    validated = 0
    validate = oracle_session_module.validate_json_value

    def counting(value: Any, *, label: str) -> None:
        nonlocal validated
        if label == "oracle fixtures":
            validated += 1
        validate(value, label=label)

    monkeypatch.setattr(oracle_session_module, "_FIXTURES_CACHE", {})
    monkeypatch.setattr(oracle_session_module, "validate_json_value", counting)
    first = open_oracle_session(source=source, task=task, limits=_limits())
    second = open_oracle_session(source=source, task=task, limits=_limits())
    try:
        assert validated == 1
        assert first._bridge._fixtures  # type: ignore[attr-defined]
        assert first._bridge._fixtures is second._bridge._fixtures  # type: ignore[attr-defined]
    finally:
        asyncio.run(first.close())
        asyncio.run(second.close())


def test_a_changed_fixture_file_is_never_served_from_the_parse_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(oracle_session_module, "_FIXTURES_CACHE", {})
    path = tmp_path / "fixtures.json"
    path.write_text('{"books": []}', encoding="utf-8")
    first = oracle_session_module._load_pack_fixtures(path)
    assert oracle_session_module._load_pack_fixtures(path) is first

    path.write_text('{"books": [{"book_id": "BK-1"}]}', encoding="utf-8")
    assert oracle_session_module._load_pack_fixtures(path) != first

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="not an object"):
        oracle_session_module._load_pack_fixtures(path)


async def _projected_task_runs_its_pack_assertions_in_the_isolated_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    plan = _plan()
    for module in ("executable_projection", "executable_driver"):
        monkeypatch.setattr(
            f"nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.{module}.assert_source_unchanged",
            lambda _source: None,
        )
    task = build_executable_task_spec(
        _projection(_row()),
        TASK_ID,
        candidate_alias="candidate_a",
        source=source,
        plan=plan,
    )
    session = open_oracle_session(source=source, task=task, limits=_limits())
    client = _FakeClient(
        [
            _response_with_call({"book_id": "BK-100"}),
            _text_response("BK-100 is available."),
        ]
    )

    episode = await run_executable_episode(
        candidate=_candidate(),
        limits=_limits(),
        client=client,  # type: ignore[arg-type]
        task=task,
        source=source,
        plan=plan,
        oracle=session,
        gate=CanonicalCallMatchGate(_scoring()),
    )

    assert episode.status == "completed"
    assert [outcome.status for outcome in episode.assertions] == ["passed"]


def test_projected_task_runs_its_pack_assertions_in_the_isolated_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(_projected_task_runs_its_pack_assertions_in_the_isolated_session(tmp_path, monkeypatch))


async def _live_driver_returns_live_result_to_candidate_not_recorded_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    task = _task(oracle_source)
    plan = _plan()
    oracle = _FakeOracle(oracle_source.verification_identity)
    client = _FakeClient(
        [
            _response_with_call({"book_id": "BK-100"}),
            _text_response("BK-100 is available."),
        ]
    )
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver.assert_source_unchanged",
        lambda _source: None,
    )

    episode = await run_executable_episode(
        candidate=_candidate(),
        limits=_limits(),
        client=client,  # type: ignore[arg-type]
        task=task,
        source=source,
        plan=plan,
        oracle=oracle,
        gate=CanonicalCallMatchGate(_scoring()),
    )

    second_messages = client.requests[1].body["messages"]
    tool_message = next(message for message in second_messages if message["role"] == "tool")
    assert json.loads(tool_message["content"]) == {
        "book_id": "BK-100",
        "status": "available",
    }
    assert "gold-only" not in canonical_json(second_messages)
    assert episode.status == "completed"
    assert episode.executions[0].released_to_model
    assert episode.final_state_hash is not None
    assert oracle.closed


def test_live_driver_returns_live_result_to_candidate_not_recorded_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(_live_driver_returns_live_result_to_candidate_not_recorded_gold(tmp_path, monkeypatch))


async def _authorization_failure_closes_the_open_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    source.verification_identity = "sha256:" + "b" * 64
    oracle = _FakeOracle(oracle_source.verification_identity)
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver.assert_source_unchanged",
        lambda _source: None,
    )

    with pytest.raises(ExecutableAuthorizationError, match="different verified source"):
        await run_executable_episode(
            candidate=_candidate(),
            limits=_limits(),
            client=_FakeClient([]),  # type: ignore[arg-type]
            task=_task(oracle_source),
            source=source,
            plan=_plan(),
            oracle=oracle,
            gate=CanonicalCallMatchGate(_scoring()),
        )

    assert oracle.closed


def test_authorization_failure_closes_the_open_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_authorization_failure_closes_the_open_session(tmp_path, monkeypatch))


async def _schema_invalid_candidate_call_is_recorded_but_not_executed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    task = _task(oracle_source)
    oracle = _FakeOracle(oracle_source.verification_identity)
    client = _FakeClient([_response_with_call({})])
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver.assert_source_unchanged",
        lambda _source: None,
    )

    episode = await run_executable_episode(
        candidate=_candidate(),
        limits=_limits(),
        client=client,  # type: ignore[arg-type]
        task=task,
        source=source,
        plan=_plan(),
        oracle=oracle,
        gate=CanonicalCallMatchGate(_scoring()),
    )

    assert episode.status == "candidate_mismatch"
    assert episode.executions[0].status == "not_executed"
    assert episode.executions[0].schema_valid is False
    assert not oracle.calls


def test_schema_invalid_candidate_call_is_recorded_but_not_executed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(_schema_invalid_candidate_call_is_recorded_but_not_executed(tmp_path, monkeypatch))


async def _invalid_json_object_result_is_recorded_as_an_oracle_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver.assert_source_unchanged",
        lambda _source: None,
    )

    episode = await run_executable_episode(
        candidate=_candidate(),
        limits=_limits(),
        client=_FakeClient(  # type: ignore[arg-type]
            [_response_with_call({"book_id": "BK-100"})]
        ),
        task=_task(oracle_source),
        source=source,
        plan=_plan(),
        oracle=_InvalidJsonObjectOracle(oracle_source.verification_identity),
        gate=CanonicalCallMatchGate(_scoring()),
    )

    assert episode.status == "oracle_call_failed"
    assert episode.executions[0].status == "tool_error"
    assert episode.executions[0].result is None


def test_invalid_json_object_result_is_recorded_as_an_oracle_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(_invalid_json_object_result_is_recorded_as_an_oracle_failure(tmp_path, monkeypatch))


async def _cleanup_failure_does_not_replace_a_candidate_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    oracle = _UnreadableVerdictOracle(oracle_source.verification_identity)
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver.assert_source_unchanged",
        lambda _source: None,
    )

    episode = await run_executable_episode(
        candidate=_candidate(),
        limits=_limits(),
        client=_FakeClient([_response_with_call({})]),  # type: ignore[arg-type]
        task=_task(oracle_source),
        source=source,
        plan=_plan(),
        oracle=oracle,
        gate=CanonicalCallMatchGate(_scoring()),
    )

    assert episode.status == "candidate_mismatch"
    assert episode.reason_code == "episode.candidate_mismatch"
    assert oracle.closed


def test_cleanup_failure_does_not_replace_a_candidate_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_cleanup_failure_does_not_replace_a_candidate_outcome(tmp_path, monkeypatch))


async def _drive_score_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    task: ExecutableTaskSpec,
    oracle: _FakeOracle,
    responses: list[CandidateResponse],
    tool_trace_cache: ToolTraceCache | None = None,
    client: _FakeClient | None = None,
) -> Any:
    source = _source(tmp_path, _oracle())
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver.assert_source_unchanged",
        lambda _source: None,
    )
    active_client = client or _FakeClient(responses)
    return await run_executable_episode(
        candidate=_candidate(),
        limits=_limits(),
        client=active_client,  # type: ignore[arg-type]
        task=task,
        source=source,
        plan=_plan(),
        oracle=oracle,
        gate=CanonicalCallMatchGate(_scoring()),
        tool_trace_cache=tool_trace_cache,
    )


def test_confirmation_protected_call_is_not_executed_before_user_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    task = _confirmation_task(oracle_source)
    oracle = _FakeOracle(oracle_source.verification_identity)
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=oracle,
            responses=[
                _response_with_call(
                    {
                        "book_id": "BK-100",
                        "patron_id": "P-1",
                        "confirm": True,
                    },
                    function_name="checkout_book",
                )
            ],
        )
    )

    assert episode.status == "confirmation_not_earned"
    assert episode.reason_code == "episode.confirmation_not_earned"
    assert episode.executions[0].status == "not_executed"
    assert episode.executions[0].reason_code == "tool_execution.confirmation_not_earned"
    assert oracle.calls == []


def test_confirmation_protected_call_executes_after_matching_confirmed_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    task = _confirmation_task(oracle_source)
    oracle = _FakeOracle(oracle_source.verification_identity)
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=oracle,
            responses=[
                _text_response("Please confirm the checkout."),
                _response_with_call(
                    {
                        "book_id": "BK-100",
                        "patron_id": "P-1",
                        "confirm": True,
                    },
                    function_name="checkout_book",
                ),
                _text_response("The checkout is complete."),
            ],
        )
    )

    assert episode.status == "completed"
    assert episode.executions[0].status == "completed"
    assert [call[0] for call in oracle.calls] == ["checkout_book"]


def test_a_blocked_confirmation_ends_the_episode_instead_of_running_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A withheld call must stop the episode, not leave an unanswered tool call."""

    oracle_source = _oracle()
    task = _confirmation_task(oracle_source).model_copy(update={"confirmed_call_turns": ()})
    oracle = _FakeOracle(oracle_source.verification_identity)
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=oracle,
            responses=[
                _text_response("Please confirm the checkout."),
                _response_with_call(
                    {"book_id": "BK-100", "patron_id": "P-1", "confirm": True},
                    function_name="checkout_book",
                ),
                _text_response("The checkout is complete."),
            ],
        )
    )

    assert episode.status == "confirmation_not_earned"
    assert episode.executions[0].reason_code == "tool_execution.confirmation_not_earned"
    assert not episode.observed[-1].advanced
    assert oracle.calls == []


def test_an_unconfirmed_probe_the_gold_trace_makes_still_reaches_the_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    task = _confirmation_probe_task(oracle_source)
    oracle = _FakeOracle(oracle_source.verification_identity)
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=oracle,
            responses=[
                _response_with_call(
                    {"book_id": "BK-100", "patron_id": "P-1", "confirm": False},
                    function_name="checkout_book",
                ),
                _text_response("Please confirm the checkout."),
                _response_with_call(
                    {"book_id": "BK-100", "patron_id": "P-1", "confirm": True},
                    function_name="checkout_book",
                ),
                _text_response("The checkout is complete."),
            ],
        )
    )

    assert episode.status == "completed"
    assert [execution.status for execution in episode.executions] == [
        "completed",
        "completed",
    ]
    assert [call[1]["confirm"] for call in oracle.calls] == [False, True]


def test_dependent_call_uses_the_prior_live_result_for_driving_and_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class LiveResultOracle(_FakeOracle):
        async def call_tool(
            self,
            function_name: str,
            arguments: dict[str, Any],
            *,
            turn_index: int,
        ) -> Any:
            self.calls.append((function_name, arguments, turn_index))
            return {
                "book_id": ("BK-LIVE" if len(self.calls) == 1 else arguments["book_id"]),
                "status": "available",
            }

    oracle_source = _oracle()
    task = _dependent_task(oracle_source)
    oracle = LiveResultOracle(oracle_source.verification_identity)
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=oracle,
            responses=[
                _response_with_call({"book_id": "BK-100"}),
                _response_with_call({"book_id": "BK-LIVE"}),
                _text_response("The latest book is available."),
            ],
        )
    )
    score = score_executable_episode(
        episode=episode,
        task=task,
        scoring=_scoring(),
        plan=_plan(),
    )

    assert episode.status == "completed"
    assert [call[1]["book_id"] for call in oracle.calls] == ["BK-100", "BK-LIVE"]
    assert episode.dependencies[0].resolved_value == "BK-LIVE"
    assert score.gate("arguments").outcome == "passed"
    assert score.gate("dependency_resolution").outcome == "passed"
    assert score.metric("tool_name_accuracy").value == 1.0
    assert score.metric("argument_accuracy").value == 1.0
    assert score.metric("tool_execution_success_rate").value == 1.0
    assert score.metric("path_success_rate").value == 1.0
    assert score.metric("state_match_rate").value is None
    assert score.metric("final_answer_success_rate").value is None
    assert score.metric("task_success_rate").value == 1.0
    assert score.task_success


def test_a_producer_result_naming_a_null_error_still_resolves_its_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NullErrorFieldOracle(_FakeOracle):
        async def call_tool(
            self,
            function_name: str,
            arguments: dict[str, Any],
            *,
            turn_index: int,
        ) -> Any:
            self.calls.append((function_name, arguments, turn_index))
            return {
                "book_id": ("BK-LIVE" if len(self.calls) == 1 else arguments["book_id"]),
                "status": "available",
                "error": None,
            }

    oracle_source = _oracle()
    task = _dependent_task(oracle_source)
    oracle = NullErrorFieldOracle(oracle_source.verification_identity)
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=oracle,
            responses=[
                _response_with_call({"book_id": "BK-100"}),
                _response_with_call({"book_id": "BK-LIVE"}),
                _text_response("The latest book is available."),
            ],
        )
    )

    assert episode.status == "completed"
    assert episode.dependencies[0].status == "resolved"
    assert episode.dependencies[0].resolved_value == "BK-LIVE"


@pytest.mark.parametrize(
    ("producer_result", "dependency_status"),
    [
        ({"status": "available"}, "result_path_missing"),
        ({"book_id": 100, "status": "available"}, "result_type_mismatch"),
        (
            {"book_id": "BK-LIVE", "error": {"code": "not_found"}},
            "result_unavailable",
        ),
    ],
)
def test_invalid_live_dependency_stops_before_the_consumer_request(
    producer_result: dict[str, Any],
    dependency_status: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidDependencyOracle(_FakeOracle):
        async def call_tool(
            self,
            function_name: str,
            arguments: dict[str, Any],
            *,
            turn_index: int,
        ) -> Any:
            self.calls.append((function_name, arguments, turn_index))
            return producer_result

    oracle_source = _oracle()
    task = _dependent_task(oracle_source)
    oracle = InvalidDependencyOracle(oracle_source.verification_identity)
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=oracle,
            responses=[_response_with_call({"book_id": "BK-100"})],
        )
    )
    score = score_executable_episode(
        episode=episode,
        task=task,
        scoring=_scoring(),
        plan=_plan(),
    )

    assert episode.status == "dependency_resolution_failed"
    assert len(episode.observed) == 1
    assert episode.released_tool_results == 0
    assert episode.dependencies[0].status == dependency_status
    assert score.gate("dependency_resolution").failure_class == "infrastructure"
    assert score.non_candidate_stop
    assert not score.task_success


def test_executable_scorer_rejects_dependency_value_not_derived_from_live_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class LiveResultOracle(_FakeOracle):
        async def call_tool(
            self,
            function_name: str,
            arguments: dict[str, Any],
            *,
            turn_index: int,
        ) -> Any:
            self.calls.append((function_name, arguments, turn_index))
            return {
                "book_id": ("BK-LIVE" if len(self.calls) == 1 else arguments["book_id"]),
                "status": "available",
            }

    oracle_source = _oracle()
    task = _dependent_task(oracle_source)
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=LiveResultOracle(oracle_source.verification_identity),
            responses=[
                _response_with_call({"book_id": "BK-100"}),
                _response_with_call({"book_id": "BK-LIVE"}),
                _text_response("The latest book is available."),
            ],
        )
    )
    forged = episode.dependencies[0].model_copy(
        update={
            "resolved_value": "BK-GOLD",
            "resolved_value_hash": _hash("BK-GOLD"),
        }
    )

    with pytest.raises(ExecutableEvidenceError, match="dependencies"):
        score_executable_episode(
            episode=episode.model_copy(update={"dependencies": (forged,)}),
            task=task,
            scoring=_scoring(),
            plan=_plan(),
        )


@pytest.mark.parametrize(
    ("turn_policy", "user_content"),
    [
        ("missing_slot", "Check book BK-100."),
        ("correction", "Use BK-100 instead."),
    ],
)
def test_multiturn_policies_release_only_the_published_scripted_user_turn(
    turn_policy: str,
    user_content: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle_source = _oracle()
    task = _scripted_user_task(
        oracle_source,
        turn_policy=turn_policy,
        user_content=user_content,
    )
    oracle = _FakeOracle(oracle_source.verification_identity)
    client = _FakeClient(
        [
            _text_response("Which book should I check?"),
            _response_with_call({"book_id": "BK-100"}),
            _text_response("BK-100 is available."),
        ]
    )
    source = _source(tmp_path, oracle_source)
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver.assert_source_unchanged",
        lambda _source: None,
    )

    episode = asyncio.run(
        run_executable_episode(
            candidate=_candidate(),
            limits=_limits(),
            client=client,  # type: ignore[arg-type]
            task=task,
            source=source,
            plan=_plan(),
            oracle=oracle,
            gate=CanonicalCallMatchGate(_scoring()),
        )
    )

    second_request_users = [
        message["content"] for message in thaw_json(client.requests[1].body)["messages"] if message["role"] == "user"
    ]
    assert episode.status == "completed"
    assert second_request_users[-1] == user_content
    assert episode.released_user_turns == 1


def test_executable_scorer_happy_path_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    oracle_source = _oracle()
    task = _task(oracle_source)
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=_FakeOracle(oracle_source.verification_identity),
            responses=[
                _response_with_call({"book_id": "BK-100"}),
                _text_response("BK-100 is available."),
            ],
        )
    )

    first = score_executable_episode(
        episode=episode,
        task=task,
        scoring=_scoring(),
        plan=_plan(),
    )
    second = score_executable_episode(
        episode=episode,
        task=task,
        scoring=_scoring(),
        plan=_plan(),
    )

    assert first.task_success
    assert first.score_hash == second.score_hash
    assert first.gate("oracle_execution").outcome == "passed"
    assert first.gate("commit_state_known").outcome == "not_applicable"
    assert first.gate("assertions").outcome == "not_applicable"
    assert first.attempted_calls == 1
    assert first.successful_executions == 1

    aggregate = aggregate_executable_scores(
        scores=(first,),
        plan=_plan(),
        candidate_alias="candidate_a",
    )
    assert aggregate.task_count == 1
    assert aggregate.successful_tasks == 1
    assert aggregate.metric("tool_execution_success_rate").value == 1.0
    assert aggregate.metric("state_match_rate").not_applicable_reason == "metric.no_state_assertion"
    assert (
        aggregate.aggregate_hash
        == aggregate_executable_scores(
            scores=(second,),
            plan=_plan(),
            candidate_alias="candidate_a",
        ).aggregate_hash
    )


def test_executable_aggregation_refuses_duplicate_or_foreign_task_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    task = _task(oracle_source)
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=_FakeOracle(oracle_source.verification_identity),
            responses=[
                _response_with_call({"book_id": "BK-100"}),
                _text_response("BK-100 is available."),
            ],
        )
    )
    score = score_executable_episode(
        episode=episode,
        task=task,
        scoring=_scoring(),
        plan=_plan(),
    )

    with pytest.raises(ExecutableAggregationError, match="publication order"):
        aggregate_executable_scores(
            scores=(score, score),
            plan=_plan(),
            candidate_alias="candidate_a",
        )

    foreign = score.model_copy(update={"plan_identity": "sha256:" + "9" * 64})
    with pytest.raises(ExecutableAggregationError, match="authorization boundary"):
        aggregate_executable_scores(
            scores=(foreign,),
            plan=_plan(),
            candidate_alias="candidate_a",
        )


def test_executable_task_contract_binds_terminal_completion_and_infrastructure_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    task = _task(oracle_source)
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=_FakeOracle(oracle_source.verification_identity),
            responses=[
                _response_with_call({"book_id": "BK-100"}),
                _text_response("BK-100 is available."),
            ],
        )
    )
    score = score_executable_episode(
        episode=episode,
        task=task,
        scoring=_scoring(),
        plan=_plan(),
    )

    completed_with_failed_completion = score.semantic_payload()
    completion = next(
        gate for gate in completed_with_failed_completion["gates"] if gate["gate"] == "executable_completion"
    )
    completion.update(
        {
            "outcome": "failed",
            "failure_class": "candidate",
            "reason_code": "executable.episode_incomplete",
            "turn_index": 0,
        }
    )
    with pytest.raises(ValueError, match="completed episode"):
        ExecutableTaskScore.model_validate(completed_with_failed_completion)

    infrastructure_without_stop = score.semantic_payload()
    infrastructure_without_stop["episode_status"] = "oracle_timeout"
    infrastructure_without_stop["task_success"] = False
    task_success_metric = next(
        metric for metric in infrastructure_without_stop["metrics"] if metric["metric"] == "task_success_rate"
    )
    task_success_metric.update({"numerator": 0, "value": 0.0})
    completion = next(gate for gate in infrastructure_without_stop["gates"] if gate["gate"] == "executable_completion")
    completion.update(
        {
            "outcome": "failed",
            "failure_class": "infrastructure",
            "reason_code": "executable.episode_incomplete",
            "turn_index": 0,
        }
    )
    with pytest.raises(ValueError, match="non_candidate_stop"):
        ExecutableTaskScore.model_validate(infrastructure_without_stop)


def test_metric_na_reasons_must_belong_to_the_registered_taxonomy() -> None:
    with pytest.raises(ValueError, match="registered taxonomy"):
        ExecutableMetricResult(
            metric="schema_valid_rate",
            numerator=0,
            denominator=0,
            value=None,
            not_applicable_reason="schema_valid.not_applicable",
        )
    metric = ExecutableMetricResult(
        metric="schema_valid_rate",
        numerator=0,
        denominator=0,
        value=None,
        not_applicable_reason="metric.gate_not_applicable",
    )
    assert metric.not_applicable_reason == "metric.gate_not_applicable"


def test_tool_trace_cache_replays_the_complete_episode_without_external_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle_source = _oracle()
    task = _task(oracle_source)
    cache = ToolTraceCache(tmp_path / "tool_trace_cache.jsonl")
    first_oracle = _FakeOracle(oracle_source.verification_identity)
    first = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=first_oracle,
            responses=[
                _response_with_call({"book_id": "BK-100"}),
                _text_response("BK-100 is available."),
            ],
            tool_trace_cache=cache,
        )
    )
    replay_oracle = _FakeOracle(oracle_source.verification_identity)
    replay = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=replay_oracle,
            responses=[],
            tool_trace_cache=ToolTraceCache(cache.path),
        )
    )

    assert not first.replayed
    assert replay.replayed
    assert replay.episode_hash == first.episode_hash
    assert replay.executions[0].result == first.executions[0].result
    assert replay_oracle.calls == []
    assert replay_oracle.closed
    assert cache.content_hash is not None
    assert (
        score_executable_episode(
            episode=replay,
            task=task,
            scoring=_scoring(),
            plan=_plan(),
        ).score_hash
        == score_executable_episode(
            episode=first,
            task=task,
            scoring=_scoring(),
            plan=_plan(),
        ).score_hash
    )

    request = build_tool_trace_request(
        candidate=_candidate(),
        task=task,
        source=_source(tmp_path, oracle_source),
        plan=_plan(),
    )
    with pytest.raises(ToolTraceCacheConflictError):
        cache.put_completion(
            request,
            first.model_copy(update={"detail": "different diagnostic wording"}),
        )


def test_an_unfinished_tool_trace_is_crash_evidence_not_a_cache_hit(
    tmp_path: Path,
) -> None:
    oracle_source = _oracle()
    request = build_tool_trace_request(
        candidate=_candidate(),
        task=_task(oracle_source),
        source=_source(tmp_path, oracle_source),
        plan=_plan(),
    )
    cache = ToolTraceCache(tmp_path / "tool_trace_cache.jsonl")
    assert cache.put_request(request)

    with pytest.raises(ToolTraceCacheError, match="unfinished executable request"):
        cache.get(request)


def test_publication_evidence_streams_identities_and_refuses_a_claim_without_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle_source = _oracle()
    task = _task(oracle_source)
    cache = ToolTraceCache(tmp_path / "tool_trace_cache.jsonl")
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=_FakeOracle(oracle_source.verification_identity),
            responses=[
                _response_with_call({"book_id": "BK-100"}),
                _text_response("BK-100 is available."),
            ],
            tool_trace_cache=cache,
        )
    )
    published = list(cache.publication_evidence())
    assert [(item.candidate_alias, item.task_id) for item in published] == [(episode.candidate_alias, episode.task_id)]
    assert published[0].episode_hash == episode.episode_hash
    assert [turn.request_hash for turn in published[0].turns] == [turn.request_hash for turn in episode.observed]

    unfinished = build_tool_trace_request(
        candidate=_candidate(),
        task=_confirmation_task(oracle_source),
        source=_source(tmp_path, oracle_source),
        plan=_plan(),
    )
    assert cache.put_request(unfinished)
    with pytest.raises(ToolTraceCacheError, match="unfinished or orphan"):
        list(cache.publication_evidence())


def test_a_truncated_tool_trace_cache_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "tool_trace_cache.jsonl"
    path.write_text('{"partial": true}', encoding="utf-8")

    with pytest.raises(ToolTraceCacheError, match="record terminator"):
        ToolTraceCache(path)


def test_executable_artifacts_bind_aggregates_task_rows_and_both_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyarrow.parquet as pq

    oracle_source = _oracle()
    task = _task(oracle_source)
    output_dir = tmp_path / "eval-output"
    output_dir.mkdir()
    tool_cache = output_dir / "tool_trace_cache.jsonl"
    tool_trace_cache = ToolTraceCache(tool_cache)
    client = _FakeClient(
        [
            _response_with_call({"book_id": "BK-100"}),
            _text_response("BK-100 is available."),
        ]
    )
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=_FakeOracle(oracle_source.verification_identity),
            responses=[],
            tool_trace_cache=tool_trace_cache,
            client=client,
        )
    )
    task_score = score_executable_episode(
        episode=episode,
        task=task,
        scoring=_scoring(),
        plan=_plan(),
    )
    aggregate = aggregate_executable_scores(
        scores=(task_score,),
        plan=_plan(),
        candidate_alias="candidate_a",
    )
    candidate_cache = output_dir / "candidate_io_cache.jsonl"
    candidate_io_cache = CandidateIOCache(candidate_cache)
    for request, outcome in zip(client.requests, client.outcomes, strict=True):
        candidate_io_cache.put_request(request)
        for attempt in outcome.attempts:
            candidate_io_cache.put_attempt(request.request_hash, attempt)
        candidate_io_cache.put_completion(outcome)
    candidate = _candidate()
    config = SimpleNamespace(
        eval_config_hash=HASH,
        publication_allowed=True,
        non_publication_reasons=(),
        source=SimpleNamespace(
            semantic_payload=lambda: {
                "run_manifest": {"content_hash": HASH},
                "benchmark": {"content_hash": OTHER_HASH},
            }
        ),
        outputs=SimpleNamespace(
            output_dir=output_dir,
            write_task_results=True,
            write_eval_manifest=True,
            cache_candidate_responses=True,
            cache_tool_results=True,
        ),
        candidate=lambda alias: candidate if alias == candidate.alias else None,
    )

    artifacts = write_executable_eval_artifacts(
        eval_run_id="eval-run-1",
        config=config,  # type: ignore[arg-type]
        plan=_plan(),
        candidate_scores=(aggregate,),
        task_scores=(task_score,),
        candidate_io_cache_path=candidate_cache,
        tool_trace_cache_path=tool_cache,
    )

    report = json.loads((output_dir / EVAL_REPORT_FILE).read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / EVAL_MANIFEST_FILE).read_text(encoding="utf-8"))
    rows = pq.read_table(output_dir / EVAL_TASK_RESULTS_FILE).to_pylist()
    assert report["candidates"][0]["metrics"]["task_success_rate"]["value"] == 1.0
    assert report["error_taxonomy_hash"] == ERROR_TAXONOMY_HASH
    assert rows[0]["task_success"] is True
    assert rows[0]["episode_status"] == "completed"
    assert rows[0]["failure_records"] == []
    assert rows[0]["final_answer_passed"] is None
    assert manifest["artifacts"]["candidate_io_cache"]["content_hash"] == _hash_file(candidate_cache)
    assert manifest["artifacts"]["tool_trace_cache"]["content_hash"] == _hash_file(tool_cache)
    assert manifest["error_taxonomy_hash"] == ERROR_TAXONOMY_HASH
    assert set(manifest["runtime"]) >= {
        "python",
        "platform",
        "pipeline_git_sha",
        "pipeline_source_hash",
        "dependency_lock_hash",
        "worker_image_digest",
    }
    assert artifacts.manifest_hash == _hash_file(artifacts.manifest_path)

    with candidate_cache.open("ab") as handle:
        handle.write(b'{"corrupt":true}\n')
    with pytest.raises(EvalArtifactError, match="cache failed publication validation"):
        write_executable_eval_artifacts(
            eval_run_id="eval-run-1",
            config=config,  # type: ignore[arg-type]
            plan=_plan(),
            candidate_scores=(aggregate,),
            task_scores=(task_score,),
            candidate_io_cache_path=candidate_cache,
            tool_trace_cache_path=tool_cache,
        )


def test_executable_artifacts_preserve_uncached_provider_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RejectedClient:
        async def complete(self, request: Any, *, deadline: float) -> CandidateCallOutcome:
            del deadline
            return CandidateCallOutcome(
                request_hash=request.request_hash,
                status="provider_rejected",
                attempts=(
                    CandidateAttempt(
                        attempt_index=0,
                        observed_at="2026-01-01T00:00:00+00:00",
                        status="provider_rejected",
                        retryable=False,
                        http_status=400,
                        latency_s=0.01,
                    ),
                ),
            )

    output_dir = tmp_path / "provider-rejected"
    output_dir.mkdir()
    candidate_cache = output_dir / "candidate_io_cache.jsonl"
    candidate_cache.touch()
    tool_cache = output_dir / "tool_trace_cache.jsonl"
    oracle_source = _oracle()
    task = _task(oracle_source)
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=_FakeOracle(oracle_source.verification_identity),
            responses=[],
            tool_trace_cache=ToolTraceCache(tool_cache),
            client=RejectedClient(),  # type: ignore[arg-type]
        )
    )
    task_score = score_executable_episode(
        episode=episode,
        task=task,
        scoring=_scoring(),
        plan=_plan(),
    )
    aggregate = aggregate_executable_scores(
        scores=(task_score,),
        plan=_plan(),
        candidate_alias="candidate_a",
    )

    artifacts = write_executable_eval_artifacts(
        eval_run_id="eval-run-provider-rejected",
        config=_artifact_config(output_dir),  # type: ignore[arg-type]
        plan=_plan(),
        candidate_scores=(aggregate,),
        task_scores=(task_score,),
        candidate_io_cache_path=candidate_cache,
        tool_trace_cache_path=tool_cache,
    )

    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    published_episode = next(iter(ToolTraceCache(tool_cache).publication_evidence()))
    assert published_episode.turns[0].call_status == "provider_rejected"
    assert report["candidates"][0]["non_candidate_stops"] == 1
    assert artifacts.manifest_path is not None


def _published_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_dir: Path,
) -> tuple[Any, Any, _FakeClient]:
    """Drive one scored episode and return what an artifact set needs."""
    oracle_source = _oracle()
    task = _task(oracle_source)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = _FakeClient(
        [
            _response_with_call({"book_id": "BK-100"}),
            _text_response("BK-100 is available."),
        ]
    )
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=_FakeOracle(oracle_source.verification_identity),
            responses=[],
            client=client,
        )
    )
    task_score = score_executable_episode(
        episode=episode,
        task=task,
        scoring=_scoring(),
        plan=_plan(),
    )
    aggregate = aggregate_executable_scores(
        scores=(task_score,),
        plan=_plan(),
        candidate_alias="candidate_a",
    )
    return task_score, aggregate, client


def _artifact_config(
    output_dir: Path,
    *,
    write_task_results: bool = True,
    write_eval_manifest: bool = True,
    cache_candidate_responses: bool = True,
    cache_tool_results: bool = True,
) -> Any:
    candidate = _candidate()
    return SimpleNamespace(
        eval_config_hash=HASH,
        publication_allowed=True,
        non_publication_reasons=(),
        source=SimpleNamespace(
            semantic_payload=lambda: {
                "run_manifest": {"content_hash": HASH},
                "benchmark": {"content_hash": OTHER_HASH},
            }
        ),
        outputs=SimpleNamespace(
            output_dir=output_dir,
            write_task_results=write_task_results,
            write_eval_manifest=write_eval_manifest,
            cache_candidate_responses=cache_candidate_responses,
            cache_tool_results=cache_tool_results,
        ),
        candidate=lambda alias: candidate if alias == candidate.alias else None,
    )


def test_a_published_candidate_cache_is_checked_without_episode_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run may keep only the candidate cache, and it is still evidence."""
    output_dir = tmp_path / "complete"
    task_score, aggregate, client = _published_fixture(tmp_path, monkeypatch, output_dir)
    complete_path = output_dir / "candidate_io_cache.jsonl"
    complete = CandidateIOCache(complete_path)
    for request, outcome in zip(client.requests, client.outcomes, strict=True):
        complete.put_request(request)
        for attempt in outcome.attempts:
            complete.put_attempt(request.request_hash, attempt)
        complete.put_completion(outcome)

    artifacts = write_executable_eval_artifacts(
        eval_run_id="eval-run-1",
        config=_artifact_config(output_dir, cache_tool_results=False),  # type: ignore[arg-type]
        plan=_plan(),
        candidate_scores=(aggregate,),
        task_scores=(task_score,),
        candidate_io_cache_path=complete_path,
        tool_trace_cache_path=None,
    )
    assert artifacts.manifest_path is not None

    unfinished_dir = tmp_path / "unfinished"
    unfinished_dir.mkdir()
    unfinished_path = unfinished_dir / "candidate_io_cache.jsonl"
    unfinished = CandidateIOCache(unfinished_path)
    for index, (request, outcome) in enumerate(zip(client.requests, client.outcomes, strict=True)):
        unfinished.put_request(request)
        if index == 0:
            for attempt in outcome.attempts:
                unfinished.put_attempt(request.request_hash, attempt)
            unfinished.put_completion(outcome)

    with pytest.raises(EvalArtifactError, match="cache failed publication validation"):
        write_executable_eval_artifacts(
            eval_run_id="eval-run-1",
            config=_artifact_config(unfinished_dir, cache_tool_results=False),  # type: ignore[arg-type]
            plan=_plan(),
            candidate_scores=(aggregate,),
            task_scores=(task_score,),
            candidate_io_cache_path=unfinished_path,
            tool_trace_cache_path=None,
        )
    assert not (unfinished_dir / EVAL_REPORT_FILE).exists()


def test_task_results_immutability_is_a_property_of_rows_not_parquet_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_dir = tmp_path / "eval-output"
    task_score, aggregate, _client = _published_fixture(tmp_path, monkeypatch, output_dir)
    config = _artifact_config(
        output_dir,
        write_eval_manifest=False,
        cache_candidate_responses=False,
        cache_tool_results=False,
    )
    arguments: dict[str, Any] = {
        "eval_run_id": "eval-run-1",
        "config": config,
        "plan": _plan(),
        "candidate_scores": (aggregate,),
        "task_scores": (task_score,),
        "candidate_io_cache_path": None,
        "tool_trace_cache_path": None,
    }
    first = write_executable_eval_artifacts(**arguments)
    assert first.task_results_path is not None

    # Another writer build re-encodes the same rows into different bytes.
    table = pq.read_table(first.task_results_path)
    pq.write_table(table, first.task_results_path, compression="gzip")
    reencoded = write_executable_eval_artifacts(**arguments)
    assert reencoded.task_results_hash == _hash_file(first.task_results_path)
    assert reencoded.task_results_hash != first.task_results_hash

    rows = table.to_pylist()
    rows[0]["task_success"] = not rows[0]["task_success"]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=table.schema),
        first.task_results_path,
        compression="zstd",
    )
    with pytest.raises(EvalArtifactError, match="different immutable evidence"):
        write_executable_eval_artifacts(**arguments)


def test_batch_pipeline_replays_end_to_end_from_both_valid_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "batch-output"
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    source.evaluation_benchmark.path = tmp_path / "benchmark.parquet"
    plan = _plan()
    projection = _projection(_row())
    candidate = _candidate()
    responses = [
        _response_with_call({"book_id": "BK-100"}),
        _text_response("BK-100 is available."),
    ]
    candidate_calls = 0
    oracle_resets = 0
    oracle_closes = 0

    class CachingClient(_FakeClient):
        def __init__(
            self,
            values: list[CandidateResponse],
            cache: CandidateIOCache,
        ) -> None:
            super().__init__(values)
            self.cache = cache

        async def complete(self, request: Any, *, deadline: float) -> CandidateCallOutcome:
            nonlocal candidate_calls
            candidate_calls += 1
            outcome = await super().complete(request, deadline=deadline)
            self.cache.put_request(request)
            for attempt in outcome.attempts:
                self.cache.put_attempt(request.request_hash, attempt)
            self.cache.put_completion(outcome)
            return outcome

        async def aclose(self) -> None:
            return None

    class CountingOracle(_FakeOracle):
        async def reset(self) -> None:
            nonlocal oracle_resets
            oracle_resets += 1
            await super().reset()

        async def close(self) -> None:
            nonlocal oracle_closes
            oracle_closes += 1
            await super().close()

    config = SimpleNamespace(
        settings=SimpleNamespace(executable=True, held_out_eval=False),
        eval_config_hash=HASH,
        publication_allowed=True,
        non_publication_reasons=(),
        limits=_limits(),
        scoring=_scoring(),
        source=SimpleNamespace(
            semantic_payload=lambda: {
                "run_manifest": {"content_hash": HASH},
                "benchmark": {"content_hash": OTHER_HASH},
            }
        ),
        outputs=SimpleNamespace(
            output_dir=output_dir,
            write_task_results=True,
            write_eval_manifest=True,
            cache_candidate_responses=True,
            cache_tool_results=True,
        ),
        candidate=lambda alias: candidate if alias == candidate.alias else None,
    )
    monkeypatch.setattr(batch_runner, "write_resolved_eval_config", lambda *_args: HASH)
    monkeypatch.setattr(
        batch_runner,
        "verify_eval_source",
        lambda *_args, **_kwargs: source,
    )
    monkeypatch.setattr(
        batch_runner,
        "write_source_verification_report",
        lambda *_args: (output_dir / "source.json", HASH),
    )
    monkeypatch.setattr(
        batch_runner,
        "evaluate_contamination",
        lambda *_args: plan,
    )
    monkeypatch.setattr(
        batch_runner,
        "write_contamination_report",
        lambda *_args: (output_dir / "contamination.json", HASH),
    )
    monkeypatch.setattr(
        batch_runner,
        "project_published_benchmark",
        lambda *_args, **_kwargs: projection,
    )
    monkeypatch.setattr(batch_runner, "assert_plan_unchanged", lambda *_args: None)
    for module in ("executable_projection", "executable_driver"):
        monkeypatch.setattr(
            f"nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.{module}.assert_source_unchanged",
            lambda _source: None,
        )

    def client_factory(
        _candidate: Any,
        _limits: Any,
        cache: CandidateIOCache,
    ) -> CachingClient:
        return CachingClient(list(responses), cache)

    def oracle_factory(_source: Any, task: Any, _limits: Any) -> CountingOracle:
        return CountingOracle(task.oracle_verification_identity)

    first = asyncio.run(
        batch_runner.run_bfcl_eval(
            config,  # type: ignore[arg-type]
            client_factory=client_factory,
            oracle_factory=oracle_factory,
        )
    )
    first_hashes = tuple(score.score_hash for score in first.task_scores)
    first_parquet_hash = first.artifacts.task_results_hash
    assert candidate_calls == 2
    assert oracle_resets == 1
    assert oracle_closes == 1

    second = asyncio.run(
        batch_runner.run_bfcl_eval(
            config,  # type: ignore[arg-type]
            client_factory=client_factory,
            oracle_factory=oracle_factory,
        )
    )
    assert tuple(score.score_hash for score in second.task_scores) == first_hashes
    assert second.artifacts.task_results_hash == first_parquet_hash
    assert second.eval_run_id == first.eval_run_id
    assert candidate_calls == 2
    assert oracle_resets == 1
    assert oracle_closes == 2


def test_business_rejection_is_execution_success_and_assertions_decide_correctness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BusinessRejectionOracle(_FakeOracle):
        async def call_tool(
            self,
            function_name: str,
            arguments: dict[str, Any],
            *,
            turn_index: int,
        ) -> Any:
            self.calls.append((function_name, arguments, turn_index))
            return {"error": {"code": "not_found", "id": arguments["book_id"]}}

    oracle_source = _oracle()
    task = _with_assertions(_task(oracle_source), "assert_book_status_reported")
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=BusinessRejectionOracle(oracle_source.verification_identity),
            responses=[
                _response_with_call({"book_id": "BK-100"}),
                _text_response("The book was not found."),
            ],
        )
    )
    score = score_executable_episode(episode=episode, task=task, scoring=_scoring(), plan=_plan())

    assert episode.executions[0].status == "business_rejection"
    assert score.gate("oracle_execution").outcome == "passed"
    assert score.gate("assertions").outcome == "passed"
    assert score.task_success


@pytest.mark.parametrize(
    ("assertion_status", "episode_status", "failure_class"),
    [
        ("failed", "completed", "candidate"),
        (
            "infrastructure_error",
            "assertion_infrastructure_failed",
            "infrastructure",
        ),
    ],
)
def test_assertion_verdicts_keep_candidate_and_infrastructure_failures_separate(
    assertion_status: str,
    episode_status: str,
    failure_class: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class VerdictOracle(_FakeOracle):
        async def run_assertion(self, name: str, *, task: dict[str, Any]) -> dict[str, Any]:
            return {
                "name": name,
                "status": assertion_status,
                "passed": assertion_status == "passed",
                "detail": f"assertion ended as {assertion_status}",
            }

    oracle_source = _oracle()
    task = _with_assertions(_task(oracle_source), "assert_book_status_reported")
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=VerdictOracle(oracle_source.verification_identity),
            responses=[
                _response_with_call({"book_id": "BK-100"}),
                _text_response("BK-100 is available."),
            ],
        )
    )
    score = score_executable_episode(episode=episode, task=task, scoring=_scoring(), plan=_plan())

    assert episode.status == episode_status
    assert score.gate("assertions").failure_class == failure_class
    assert score.non_candidate_stop is (failure_class == "infrastructure")
    assert not score.task_success
    metric = score.metric("assertion_success_rate")
    if failure_class == "infrastructure":
        # A broken assertion produced no evidence about the answer, so it cannot
        # be a denominator the candidate is measured against. The gate already
        # records the stop.
        assert metric.value is None
        assert metric.not_applicable_reason == "metric.assertion_evidence_incomplete"
    else:
        assert metric.value == 0.0
        assert metric.denominator == 1


def test_not_applicable_assertion_does_not_fail_the_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class VerdictOracle(_FakeOracle):
        async def run_assertion(self, name: str, *, task: dict[str, Any]) -> dict[str, Any]:
            return {
                "name": name,
                "status": "not_applicable",
                "passed": False,
                "detail": "the assertion's conditional state was not reached",
            }

    oracle_source = _oracle()
    task = _with_assertions(_task(oracle_source), "assert_conditional_state")
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=VerdictOracle(oracle_source.verification_identity),
            responses=[
                _response_with_call({"book_id": "BK-100"}),
                _text_response("BK-100 is available."),
            ],
        )
    )
    score = score_executable_episode(episode=episode, task=task, scoring=_scoring(), plan=_plan())

    assert episode.status == "completed"
    assert episode.assertions[0].status == "not_applicable"
    assert score.gate("assertions").outcome == "not_applicable"
    assert score.metric("assertion_success_rate").not_applicable_reason == "metric.all_assertions_not_applicable"
    assert score.task_success


def test_declared_assertion_category_reaches_the_episode_and_its_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    task = _with_assertions(
        _task(oracle_source),
        "assert_book_now_on_loan",
        category="state",
    )
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=_FakeOracle(oracle_source.verification_identity),
            responses=[
                _response_with_call({"book_id": "BK-100"}),
                _text_response("BK-100 is available."),
            ],
        )
    )
    score = score_executable_episode(episode=episode, task=task, scoring=_scoring(), plan=_plan())

    assert episode.assertions[0].category == "state"
    assert score.assertions[0].category == "state"
    assert score.metric("state_match_rate").value == 1.0
    assert score.metric("assertion_success_rate").value == 1.0
    assert score.metric("final_answer_success_rate").not_applicable_reason == "metric.no_final_answer_assertion"


def test_a_metric_quotient_is_not_part_of_the_score_identity() -> None:
    metric = ExecutableMetricResult(
        metric="assertion_success_rate",
        numerator=1,
        denominator=2,
        value=0.5,
    )

    assert "value" not in metric.identity_payload()
    assert metric.identity_payload()["numerator"] == 1
    assert metric.identity_payload()["denominator"] == 2


def test_oracle_timeout_is_a_non_candidate_scoring_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    oracle_source = _oracle()
    task = _with_assertions(_task(oracle_source), "assert_book_status_reported")
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=_TimeoutOracle(oracle_source.verification_identity),
            responses=[_response_with_call({"book_id": "BK-100"})],
        )
    )
    score = score_executable_episode(episode=episode, task=task, scoring=_scoring(), plan=_plan())

    assert score.gate("oracle_execution").failure_class == "infrastructure"
    assert score.gate("assertions").outcome == "not_applicable"
    assert score.required_assertions == 1
    assert score.assertions == ()
    assert score.gate("executable_completion").failure_class == "infrastructure"
    assert score.non_candidate_stop
    assert not score.task_success
    # A declared assertion the timeout prevented from running is missing
    # evidence, not a candidate failure, so no assertion-derived metric may
    # report a number the episode never produced.
    assert score.metric("assertion_success_rate").value is None
    assert score.metric("assertion_success_rate").not_applicable_reason == "metric.assertion_evidence_incomplete"
    assert score.metric("path_success_rate").value is None
    assert score.metric("path_success_rate").not_applicable_reason == "metric.path_evidence_incomplete"
    result = executable_task_result(score)
    assert result["episode_status"] == "oracle_timeout"
    assert result["non_candidate_stop"] is True
    assert result["failure_records"]
    assert {record["attribution"] for record in result["failure_records"]} == {"infrastructure"}
    # Every incomplete episode fails the same completion gate, so only the
    # episode-layer record says which terminal ended this one.
    assert result["failure_records"][0] == {
        "layer": "episode",
        "code": "episode.oracle_timeout",
        "attribution": "infrastructure",
        "subject": "episode",
    }


def test_the_executable_score_reports_the_trace_layers_attribution_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One layer decides whose failure a shared gate is, and the other carries it.

    Deriving the classification a second time from the terminal status would let
    the executable score and the trace score disagree about the same evidence.
    """
    oracle_source = _oracle()
    task = _with_assertions(_task(oracle_source), "assert_book_status_reported")
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=_TimeoutOracle(oracle_source.verification_identity),
            responses=[_response_with_call({"book_id": "BK-100"})],
        )
    )
    score = score_executable_episode(episode=episode, task=task, scoring=_scoring(), plan=_plan())
    trace = parse_executable_trace(episode, task)
    normalized = score_normalized_trace(
        trace=trace,
        script=task.script,
        scoring=_scoring(),
        completion_detail=episode.detail,
    )

    assert trace.non_candidate_stop
    assert normalized.gates
    for gate in normalized.gates:
        lifted = score.gate(gate.gate)
        assert (lifted.outcome, lifted.failure_class) == (
            gate.outcome,
            gate.failure_class,
        ), gate.gate


def test_executable_trace_parser_refuses_a_task_from_another_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    task = _task(oracle_source)
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=_FakeOracle(oracle_source.verification_identity),
            responses=[
                _response_with_call({"book_id": "BK-100"}),
                _text_response("BK-100 is available."),
            ],
        )
    )

    with pytest.raises(ExecutableEvidenceError, match="exact ExecutableTaskSpec"):
        parse_executable_trace(
            episode.model_copy(update={"script_hash": OTHER_HASH}),
            task,
        )


def test_completed_executable_evidence_cannot_hide_unobserved_script_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    task = _task(oracle_source)
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=_FakeOracle(oracle_source.verification_identity),
            responses=[
                _response_with_call({"book_id": "BK-100"}),
                _text_response("BK-100 is available."),
            ],
        )
    )

    with pytest.raises(ExecutableEvidenceError, match="claims completion"):
        parse_executable_trace(
            episode.model_copy(update={"observed": episode.observed[:-1]}),
            task,
        )


@pytest.mark.parametrize("changes_state", [False, True])
def test_determined_mutation_commit_verdicts_pass_the_evidence_gate(
    changes_state: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MutationOracle(_FakeOracle):
        def __init__(self, identity: str) -> None:
            super().__init__(identity)
            self.version = 0

        async def call_tool(
            self,
            function_name: str,
            arguments: dict[str, Any],
            *,
            turn_index: int,
        ) -> Any:
            self.calls.append((function_name, arguments, turn_index))
            if changes_state:
                self.version = 1
            return {"book_id": arguments["book_id"], "status": "available"}

        async def get_state(self) -> dict[str, Any]:
            return {"version": self.version}

    oracle_source = _oracle()
    task = _task(oracle_source).model_copy(
        update={
            "tool_policies": (
                ExecutableToolPolicy(function_name="get_book_status", mutates=True),
                ExecutableToolPolicy(
                    function_name="checkout_book",
                    mutates=True,
                    requires_confirmation=True,
                    confirmation_parameter="confirm",
                ),
            )
        }
    )
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=MutationOracle(oracle_source.verification_identity),
            responses=[
                _response_with_call({"book_id": "BK-100"}),
                _text_response("BK-100 is available."),
            ],
        )
    )
    score = score_executable_episode(episode=episode, task=task, scoring=_scoring(), plan=_plan())

    assert episode.executions[0].state_commit == ("committed" if changes_state else "not_committed")
    assert score.gate("commit_state_known").outcome == "passed"
    assert score.task_success


def test_unknown_mutation_commit_state_is_an_infrastructure_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class UnknownCommitOracle(_FakeOracle):
        def __init__(self, identity: str) -> None:
            super().__init__(identity)
            self.state_reads = 0

        async def get_state(self) -> dict[str, Any]:
            self.state_reads += 1
            if self.state_reads == 1:
                return {"version": 0}
            raise OracleStateError(
                "eval.oracle.state",
                "state disappeared after mutation",
                expected="a state object",
                recovery="discard the task",
            )

    oracle_source = _oracle()
    task = _task(oracle_source).model_copy(
        update={
            "tool_policies": (
                ExecutableToolPolicy(function_name="get_book_status", mutates=True),
                ExecutableToolPolicy(
                    function_name="checkout_book",
                    mutates=True,
                    requires_confirmation=True,
                    confirmation_parameter="confirm",
                ),
            )
        }
    )
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=UnknownCommitOracle(oracle_source.verification_identity),
            responses=[_response_with_call({"book_id": "BK-100"})],
        )
    )
    score = score_executable_episode(episode=episode, task=task, scoring=_scoring(), plan=_plan())

    assert episode.status == "unknown_commit_state"
    assert score.gate("commit_state_known").failure_class == "infrastructure"
    assert score.non_candidate_stop
    assert not score.task_success


def test_result_evidence_changes_executable_score_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class AlternateResultOracle(_FakeOracle):
        async def call_tool(
            self,
            function_name: str,
            arguments: dict[str, Any],
            *,
            turn_index: int,
        ) -> Any:
            self.calls.append((function_name, arguments, turn_index))
            return {"book_id": "BK-100", "status": "reserved"}

    oracle_source = _oracle()
    task = _task(oracle_source)
    responses = [
        _response_with_call({"book_id": "BK-100"}),
        _text_response("BK-100 is available."),
    ]
    first_episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=_FakeOracle(oracle_source.verification_identity),
            responses=list(responses),
        )
    )
    second_episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=AlternateResultOracle(oracle_source.verification_identity),
            responses=list(responses),
        )
    )
    first = score_executable_episode(episode=first_episode, task=task, scoring=_scoring(), plan=_plan())
    second = score_executable_episode(episode=second_episode, task=task, scoring=_scoring(), plan=_plan())

    assert first_episode.executions[0].result_hash != second_episode.executions[0].result_hash
    assert first.score_hash != second.score_hash


def test_executable_score_hash_ignores_diagnostic_wording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    oracle_source = _oracle()
    task = _task(oracle_source)
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=_FakeOracle(oracle_source.verification_identity),
            responses=[
                _response_with_call({"book_id": "BK-100"}),
                _text_response("BK-100 is available."),
            ],
        )
    )
    reworded = episode.model_copy(
        update={
            "detail": "different diagnostic",
            "observed": tuple(
                turn.model_copy(update={"detail": "different turn diagnostic"}) for turn in episode.observed
            ),
            "executions": tuple(
                item.model_copy(update={"detail": "different execution diagnostic"}) for item in episode.executions
            ),
        }
    )

    original = score_executable_episode(episode=episode, task=task, scoring=_scoring(), plan=_plan())
    changed = score_executable_episode(episode=reworded, task=task, scoring=_scoring(), plan=_plan())

    assert episode.episode_hash == reworded.episode_hash
    assert original.score_hash == changed.score_hash


def test_executable_scorer_refuses_identity_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    oracle_source = _oracle()
    task = _task(oracle_source)
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=_FakeOracle(oracle_source.verification_identity),
            responses=[
                _response_with_call({"book_id": "BK-100"}),
                _text_response("BK-100 is available."),
            ],
        )
    )

    with pytest.raises(ExecutableEvidenceError, match="plan_identity"):
        score_executable_episode(
            episode=episode.model_copy(update={"plan_identity": OTHER_HASH}),
            task=task,
            scoring=_scoring(),
            plan=_plan(),
        )


def test_executable_scorer_rejects_a_task_spec_not_used_to_drive_the_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    base_task = _task(oracle_source)
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=base_task,
            oracle=_FakeOracle(oracle_source.verification_identity),
            responses=[
                _response_with_call({"book_id": "BK-100"}),
                _text_response("BK-100 is available."),
            ],
        )
    )
    requiring_assertion = _with_assertions(base_task, "assert_book_status_reported")

    with pytest.raises(ExecutableEvidenceError, match="task_spec_hash"):
        score_executable_episode(
            episode=episode,
            task=requiring_assertion,
            scoring=_scoring(),
            plan=_plan(),
        )


def test_assertions_only_accepts_candidate_trace_mismatch_when_assertions_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    task = _with_assertions(_task(oracle_source), "assert_book_status_reported")
    episode = asyncio.run(
        _drive_score_fixture(
            tmp_path,
            monkeypatch,
            task=task,
            oracle=_FakeOracle(oracle_source.verification_identity),
            responses=[_response_with_call({"book_id": "BK-200"})],
        )
    )
    assertions_only = _scoring().model_copy(update={"task_success": "assertions_only"})
    assertions_only_plan = _plan().model_copy(update={"scoring_policy_hash": assertions_only.scoring_policy_hash})
    assertions_only_task = task.model_copy(
        update={
            "scoring_policy_hash": assertions_only.scoring_policy_hash,
            "plan_identity": assertions_only_plan.plan_identity,
        }
    )

    score = score_executable_episode(
        episode=episode.model_copy(
            update={
                "plan_identity": assertions_only_plan.plan_identity,
                "task_spec_hash": assertions_only_task.task_spec_hash,
            }
        ),
        task=assertions_only_task,
        scoring=assertions_only,
        plan=assertions_only_plan,
    )

    assert episode.status == "candidate_mismatch"
    assert score.gate("tool_selection").outcome == "passed"
    assert score.gate("arguments").outcome == "failed"
    assert score.gate("assertions").outcome == "passed"
    assert score.gate("executable_completion").failure_class == "candidate"
    assert score.task_success


def test_assertions_only_requires_at_least_one_assertion() -> None:
    oracle_source = _oracle()
    task = _task(oracle_source)
    assertions_only = _scoring().model_copy(update={"task_success": "assertions_only"})

    with pytest.raises(ExecutableScoringPolicyError, match="no required assertion"):
        score_executable_episode(
            episode=None,  # type: ignore[arg-type]
            task=task,
            scoring=assertions_only,
            plan=_plan(),
        )


async def _unreadable_assertion_verdict_keeps_the_episode_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    task = _with_assertions(_task(oracle_source), "assert_book_status_reported")
    oracle = _UnreadableVerdictOracle(oracle_source.verification_identity)
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver.assert_source_unchanged",
        lambda _source: None,
    )

    episode = await run_executable_episode(
        candidate=_candidate(),
        limits=_limits(),
        client=_FakeClient(  # type: ignore[arg-type]
            [
                _response_with_call({"book_id": "BK-100"}),
                _text_response("BK-100 is available."),
            ]
        ),
        task=task,
        source=source,
        plan=_plan(),
        oracle=oracle,
        gate=CanonicalCallMatchGate(_scoring()),
    )

    assert episode.status == "assertion_infrastructure_failed"
    assert [outcome.status for outcome in episode.assertions] == ["infrastructure_error"]
    assert oracle.closed


def test_an_unreadable_assertion_verdict_keeps_the_episode_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(_unreadable_assertion_verdict_keeps_the_episode_evidence(tmp_path, monkeypatch))


async def _pending_mutation_is_not_recorded_as_committed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    task = _task(oracle_source).model_copy(
        update={
            "tool_policies": (
                ExecutableToolPolicy(
                    function_name="get_book_status",
                    mutates=True,
                ),
                ExecutableToolPolicy(
                    function_name="checkout_book",
                    mutates=True,
                    requires_confirmation=True,
                    confirmation_parameter="confirm",
                ),
            )
        }
    )
    oracle = _PendingMutationOracle(oracle_source.verification_identity)
    client = _FakeClient(
        [
            _response_with_call({"book_id": "BK-100"}),
            _text_response("BK-100 is available."),
        ]
    )
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver.assert_source_unchanged",
        lambda _source: None,
    )

    episode = await run_executable_episode(
        candidate=_candidate(),
        limits=_limits(),
        client=client,  # type: ignore[arg-type]
        task=task,
        source=source,
        plan=_plan(),
        oracle=oracle,
        gate=CanonicalCallMatchGate(_scoring()),
    )

    assert episode.status == "completed"
    assert episode.executions[0].state_commit == "not_committed"
    assert episode.executions[0].state_before_hash is not None
    assert episode.executions[0].state_before_hash == episode.executions[0].state_after_hash


def test_a_pending_mutation_is_not_recorded_as_committed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_pending_mutation_is_not_recorded_as_committed(tmp_path, monkeypatch))


async def _terminal_tool_call_can_complete_without_releasing_its_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    task = _task(oracle_source)
    terminal_call = task.script.turns[0].model_copy(update={"is_terminal": True})
    task = task.model_copy(update={"script": task.script.model_copy(update={"turns": (terminal_call,)})})
    oracle = _FakeOracle(oracle_source.verification_identity)
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver.assert_source_unchanged",
        lambda _source: None,
    )

    episode = await run_executable_episode(
        candidate=_candidate(),
        limits=_limits(),
        client=_FakeClient(  # type: ignore[arg-type]
            [_response_with_call({"book_id": "BK-100"})]
        ),
        task=task,
        source=source,
        plan=_plan(),
        oracle=oracle,
        gate=CanonicalCallMatchGate(_scoring()),
    )

    assert episode.status == "completed"
    assert episode.observed[0].advanced
    assert not episode.executions[0].released_to_model


def test_a_terminal_tool_call_can_complete_without_releasing_its_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asyncio.run(_terminal_tool_call_can_complete_without_releasing_its_result(tmp_path, monkeypatch))


async def _mutating_timeout_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    oracle_source = _oracle()
    source = _source(tmp_path, oracle_source)
    task = _task(oracle_source).model_copy(
        update={
            "tool_policies": (
                ExecutableToolPolicy(
                    function_name="get_book_status",
                    mutates=True,
                ),
                ExecutableToolPolicy(
                    function_name="checkout_book",
                    mutates=True,
                    requires_confirmation=True,
                    confirmation_parameter="confirm",
                ),
            )
        }
    )
    oracle = _TimeoutOracle(oracle_source.verification_identity)
    client = _FakeClient([_response_with_call({"book_id": "BK-100"})])
    monkeypatch.setattr(
        "nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.executable_driver.assert_source_unchanged",
        lambda _source: None,
    )

    episode = await run_executable_episode(
        candidate=_candidate(),
        limits=_limits(),
        client=client,  # type: ignore[arg-type]
        task=task,
        source=source,
        plan=_plan(),
        oracle=oracle,
        gate=CanonicalCallMatchGate(_scoring()),
    )

    assert len(oracle.calls) == 1
    assert episode.status == "unknown_commit_state"
    assert episode.executions[0].status == "unknown_commit_state"
    assert episode.executions[0].state_commit == "unknown"


def test_a_mutating_timeout_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_mutating_timeout_is_not_retried(tmp_path, monkeypatch))
