"""Publishing a trace-only run: aggregation, artifacts, and orchestration.

A trace-only run measures fewer dimensions than an executable one, so the danger
these tests guard against is not a wrong number but a number that looks like
another measurement. Every claim here is taken over evidence a driver produced:
episodes are driven against a mock provider, scored, aggregated, and published,
so a claim about an artifact is a claim about the pipeline.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_args

import httpx
import pytest

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval import (
    METRIC_NOT_APPLICABLE_CODES,
    SCORING_GATES,
    TRACE_METRIC_TAXONOMY,
    CandidateApi,
    CandidateEligibility,
    CandidateInference,
    CandidateModelIdentity,
    CanonicalCallMatchGate,
    CommonEvaluationTaskSet,
    ConversationScript,
    EligibleEvalPlan,
    EvalArtifactError,
    EvalCandidate,
    EvalFileRef,
    EvalLimits,
    EvalScoringConfig,
    TraceAggregationError,
    TraceCandidateScore,
    TraceMetricName,
    TraceMetricResult,
    TraceTaskScore,
    aggregate_trace_scores,
    build_candidate_request,
    build_conversation_script,
    candidate_identity_claim,
    eval_report_document,
    eval_runner,
    run_candidate_episode,
    score_trace_episode,
    write_trace_eval_artifacts,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_cache import (
    CandidateIOCache,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_client import (
    NativeFunctionCallingClient,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
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
EVAL_CONFIG_HASH = "sha256:" + "3" * 64
CALL_TASK = "t__call"
TEXT_TASK = "t__text"
TASK_IDS = (CALL_TASK, TEXT_TASK)

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_balance",
            "description": "Read the balance of one account.",
            "parameters": {
                "type": "object",
                "properties": {"account_id": {"type": "string"}},
                "required": ["account_id"],
                "additionalProperties": False,
            },
        },
    }
]


def _row(
    *,
    task_id: str,
    messages: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    turn_policy: str = "single_turn",
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
        tools_present=("get_balance",),
        turn_policy=turn_policy,
        is_multi_turn=False,
        num_tool_calls=len(expected),
        call_order="strict",
        system_prompt_id="sp1",
        tier="gold",
        gold_eligible=True,
        validated_by=("schema", "replay"),
        pack_id="banking_vn",
        pack_version="1.0.0",
        seed=7,
        src="banking_vn:t1",
        metadata={
            "base_task_id": "b1",
            "expt_name": "publication",
            "language": "vi",
            "profile_hash": "ph",
            "surface_source": "oracle",
        },
    )


def _call_row() -> CanonicalExportRow:
    """One request answered with one call and then one sentence."""
    return _row(
        task_id=CALL_TASK,
        messages=[
            {"role": "system", "content": "You are a bank assistant."},
            {"role": "user", "content": "Balance of account 1?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "get_balance",
                            "arguments": canonical_json({"account_id": "1"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0",
                "content": canonical_json({"balance": 500}),
            },
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


def _text_row() -> CanonicalExportRow:
    """A request no declared tool can serve, so the trace answers in words."""
    return _row(
        task_id=TEXT_TASK,
        turn_policy="irrelevant",
        messages=[
            {"role": "system", "content": "You are a bank assistant."},
            {"role": "user", "content": "What is the weather in Hanoi?"},
            {"role": "assistant", "content": "I can only help with banking."},
        ],
        expected=[],
    )


ROWS = (_call_row(), _text_row())


def _projection() -> CanonicalExportProjection:
    return CanonicalExportProjection(
        source=ProjectionSource(
            file="benchmark.parquet",
            content_hash=BENCHMARK_HASH,
            rows=len(ROWS),
        ),
        provenance=derive_provenance(ROWS),
        rows=ROWS,
        plans=tuple(conversation_plan(row) for row in ROWS),
    )


def _source(tmp_path: Path) -> Any:
    return SimpleNamespace(
        evaluation_benchmark=SimpleNamespace(
            path=tmp_path / "benchmark.parquet",
            content_hash=BENCHMARK_HASH,
            rows=len(ROWS),
        ),
        task_ids=TASK_IDS,
        verification_identity=SOURCE_IDENTITY,
        source_run_id="run-1",
    )


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
            seed=42,
            tool_choice="auto",
            provider_extensions={},
        ),
    )


def _limits(*, max_parallel_tasks: int = 2) -> EvalLimits:
    return EvalLimits(
        max_turns=6,
        tool_timeout_s=1.0,
        candidate_timeout_s=5.0,
        episode_timeout_s=30.0,
        max_parallel_tasks=max_parallel_tasks,
        max_retries=0,
    )


def _scoring(**overrides: Any) -> EvalScoringConfig:
    fields: dict[str, Any] = {
        "contract": EvalFileRef(
            path="/refs/bfcl-eval-scoring-contract.md",
            content_hash=CONTRACT_HASH,
        ),
        "argument_matching": "schema_then_canonical",
        "insert_declared_defaults": True,
        "respect_call_order": True,
        "respect_call_group": True,
        "allow_llm_repair": False,
        "task_success": "all_applicable_gates",
    }
    fields.update(overrides)
    return EvalScoringConfig(**fields)


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


def _config(
    output_dir: Path,
    *,
    executable: bool = False,
    write_task_results: bool = True,
    write_eval_manifest: bool = True,
    cache_candidate_responses: bool = True,
    max_parallel_tasks: int = 2,
    scoring: EvalScoringConfig | None = None,
) -> Any:
    candidate = _candidate()
    return SimpleNamespace(
        settings=SimpleNamespace(executable=executable, held_out_eval=False),
        eval_config_hash=EVAL_CONFIG_HASH,
        publication_allowed=True,
        non_publication_reasons=(),
        limits=_limits(max_parallel_tasks=max_parallel_tasks),
        scoring=scoring or _scoring(),
        source=SimpleNamespace(
            semantic_payload=lambda: {"benchmark": {"content_hash": BENCHMARK_HASH}}
        ),
        outputs=SimpleNamespace(
            output_dir=output_dir,
            write_task_results=write_task_results,
            write_eval_manifest=write_eval_manifest,
            cache_candidate_responses=cache_candidate_responses,
            # Trace evaluation executes no tool, so this flag has no tool result
            # to persist; publication policy still requires it to be set.
            cache_tool_results=True,
        ),
        candidate=lambda alias: candidate if alias == candidate.alias else None,
    )


def _script(task_id: str, tmp_path: Path) -> ConversationScript:
    return build_conversation_script(_projection(), task_id, source=_source(tmp_path))


_CALL_REPLY: dict[str, Any] = {
    "id": "chatcmpl-calls",
    "object": "chat.completion",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "get_balance",
                            "arguments": '{"account_id":"1"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
}


def _text_reply(text: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-text",
        "object": "chat.completion",
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


def _replies_by_task() -> dict[str, list[dict[str, Any]]]:
    """What the mock provider answers, per task, in turn order."""
    return {
        CALL_TASK: [_CALL_REPLY, _text_reply("Account 1 holds 500.")],
        TEXT_TASK: [_text_reply("I can only help with banking.")],
    }


def _transport(pending: dict[str, list[dict[str, Any]]]) -> httpx.MockTransport:
    """Answer each task's turns in order, whichever order the batch asks them in.

    The wire request carries no task id — nothing but the conversation itself is
    sent — so the task is recognized from the request the prompt opens with.
    """

    def handle(request: httpx.Request) -> httpx.Response:
        body = request.content.decode("utf-8")
        task_id = TEXT_TASK if "weather" in body else CALL_TASK
        return httpx.Response(200, json=pending[task_id].pop(0))

    return httpx.MockTransport(handle)


def _client_factory(
    pending: dict[str, list[dict[str, Any]]],
) -> Any:
    transport = _transport(pending)

    def factory(
        candidate: EvalCandidate,
        limits: EvalLimits,
        cache: CandidateIOCache,
    ) -> NativeFunctionCallingClient:
        return NativeFunctionCallingClient(
            candidate,
            limits,
            cache,
            transport=transport,
        )

    return factory


def _scored(
    task_id: str,
    tmp_path: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
    scoring: EvalScoringConfig | None = None,
) -> TraceTaskScore:
    """Drive one task against the mock provider and score what it produced."""
    monkeypatch.setenv("CANDIDATE_API_KEY", "secret-key")
    policy = scoring or _scoring()
    script = _script(task_id, tmp_path)
    candidate = _candidate()
    limits = _limits()
    pending = _replies_by_task()

    async def execute() -> Any:
        client = NativeFunctionCallingClient(
            candidate,
            limits,
            CandidateIOCache(tmp_path / f"cache-{task_id}.jsonl"),
            transport=_transport(pending),
        )
        try:
            return await run_candidate_episode(
                candidate=candidate,
                limits=limits,
                client=client,
                script=script,
                plan=_plan(scoring=policy),
                gate=CanonicalCallMatchGate(policy),
            )
        finally:
            await client.aclose()

    episode = asyncio.run(execute())
    return score_trace_episode(
        episode=episode,
        script=script,
        scoring=policy,
        plan=_plan(scoring=policy),
    )


def _authorized_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    plan: EligibleEvalPlan | None = None,
) -> None:
    """Replace the gates a run passes before its first candidate request."""
    source = _source(tmp_path)
    monkeypatch.setattr(eval_runner, "write_resolved_eval_config", lambda *_args: None)
    monkeypatch.setattr(
        eval_runner,
        "verify_eval_source",
        lambda *_args, **_kwargs: source,
    )
    monkeypatch.setattr(
        eval_runner,
        "write_source_verification_report",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        eval_runner,
        "evaluate_contamination",
        lambda *_args: plan or _plan(),
    )
    monkeypatch.setattr(eval_runner, "write_contamination_report", lambda *_args: None)
    monkeypatch.setattr(
        eval_runner,
        "project_published_benchmark",
        lambda *_args, **_kwargs: _projection(),
    )
    monkeypatch.setattr(eval_runner, "assert_plan_unchanged", lambda *_args: None)


# --------------------------------------------------------------------------------------
# The metric taxonomy: one published rate per gate the scorer computes.
# --------------------------------------------------------------------------------------


def test_every_trace_gate_has_exactly_one_published_metric() -> None:
    """A gate without a metric would be scored per task and dropped per run."""
    assert TRACE_METRIC_TAXONOMY == tuple(
        f"{gate}_pass_rate" for gate in SCORING_GATES
    ) + ("task_success_rate",)
    assert set(get_args(TraceMetricName)) == set(TRACE_METRIC_TAXONOMY)


def test_trace_metric_names_are_not_the_executable_ones() -> None:
    """The two taxonomies count different things, so they never share a name."""
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval import (
        EXECUTABLE_METRIC_TAXONOMY,
    )

    shared = set(TRACE_METRIC_TAXONOMY) & set(EXECUTABLE_METRIC_TAXONOMY)
    assert shared == {"task_success_rate"}


# --------------------------------------------------------------------------------------
# Aggregation over one candidate's authorized task set.
# --------------------------------------------------------------------------------------


def test_an_aggregate_counts_a_gate_only_over_the_tasks_it_applied_to(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = (
        _scored(CALL_TASK, tmp_path, monkeypatch=monkeypatch),
        _scored(TEXT_TASK, tmp_path, monkeypatch=monkeypatch),
    )
    assert all(score.task_success for score in scores)

    aggregate = aggregate_trace_scores(
        scores=scores,
        plan=_plan(),
        candidate_alias="candidate_a",
    )

    assert aggregate.scope == "trace"
    assert aggregate.task_ids == TASK_IDS
    assert aggregate.task_score_hashes == tuple(score.score_hash for score in scores)
    assert aggregate.successful_tasks == 2
    assert aggregate.non_candidate_stops == 0
    # The text-only task asks for no call, so its argument gate never applied and
    # is not counted as a pass it did not earn.
    arguments = aggregate.metric("arguments_pass_rate")
    assert (arguments.numerator, arguments.denominator, arguments.value) == (1, 1, 1.0)
    completion = aggregate.metric("trace_completion_pass_rate")
    assert (completion.numerator, completion.denominator) == (2, 2)
    success = aggregate.metric("task_success_rate")
    assert (success.numerator, success.denominator, success.value) == (2, 2, 1.0)
    assert aggregate.aggregate_hash.startswith("sha256:")
    assert aggregate.aggregate_hash == aggregate_trace_scores(
        scores=scores,
        plan=_plan(),
        candidate_alias="candidate_a",
    ).aggregate_hash


def test_a_gate_no_task_applied_is_reported_na_rather_than_a_vacuous_rate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under canonical_only there is no schema step, so the gate applies nowhere."""
    policy = _scoring(argument_matching="canonical_only")
    scores = (
        _scored(CALL_TASK, tmp_path, monkeypatch=monkeypatch, scoring=policy),
        _scored(TEXT_TASK, tmp_path, monkeypatch=monkeypatch, scoring=policy),
    )

    aggregate = aggregate_trace_scores(
        scores=scores,
        plan=_plan(scoring=policy),
        candidate_alias="candidate_a",
    )

    schema = aggregate.metric("schema_valid_pass_rate")
    assert (schema.numerator, schema.denominator, schema.value) == (0, 0, None)
    assert schema.not_applicable_reason == "metric.no_applicable_task"
    assert schema.not_applicable_reason in METRIC_NOT_APPLICABLE_CODES


def test_trace_metric_na_reasons_must_belong_to_the_registered_taxonomy() -> None:
    with pytest.raises(ValueError, match="registered taxonomy"):
        TraceMetricResult(
            metric="schema_valid_pass_rate",
            numerator=0,
            denominator=0,
            value=None,
            not_applicable_reason="schema_valid.not_applicable",
        )


def test_an_aggregate_refuses_a_partial_or_reordered_task_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _scored(CALL_TASK, tmp_path, monkeypatch=monkeypatch)
    second = _scored(TEXT_TASK, tmp_path, monkeypatch=monkeypatch)

    with pytest.raises(TraceAggregationError) as partial:
        aggregate_trace_scores(
            scores=(first,),
            plan=_plan(),
            candidate_alias="candidate_a",
        )
    assert partial.value.code == "eval_trace_aggregation_invalid"

    with pytest.raises(TraceAggregationError, match="publication order"):
        aggregate_trace_scores(
            scores=(second, first),
            plan=_plan(),
            candidate_alias="candidate_a",
        )


def test_an_aggregate_refuses_scores_taken_under_another_policy_or_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = (
        _scored(CALL_TASK, tmp_path, monkeypatch=monkeypatch),
        _scored(TEXT_TASK, tmp_path, monkeypatch=monkeypatch),
    )
    relaxed = _scoring(respect_call_order=False)
    foreign_policy = (
        scores[0].model_copy(update={"scoring_policy": relaxed.semantic_payload()}),
        scores[1],
    )
    with pytest.raises(TraceAggregationError, match="authorization boundary"):
        aggregate_trace_scores(
            scores=foreign_policy,
            plan=_plan(),
            candidate_alias="candidate_a",
        )

    foreign_contract = (
        scores[0],
        scores[1].model_copy(
            update={"scoring_contract_hash": "sha256:" + "9" * 64}
        ),
    )
    with pytest.raises(TraceAggregationError, match="scoring-contract"):
        aggregate_trace_scores(
            scores=foreign_contract,
            plan=_plan(),
            candidate_alias="candidate_a",
        )


def test_a_report_refuses_aggregates_that_measured_different_things(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One artifact set is one measurement, whatever its rows happen to cover."""
    aggregate = aggregate_trace_scores(
        scores=(
            _scored(CALL_TASK, tmp_path, monkeypatch=monkeypatch),
            _scored(TEXT_TASK, tmp_path, monkeypatch=monkeypatch),
        ),
        plan=_plan(),
        candidate_alias="candidate_a",
    )
    restamped: TraceCandidateScore = aggregate.model_copy(
        update={"scope": "trace_and_executable"}
    )

    with pytest.raises(EvalArtifactError, match="mix evaluation scopes"):
        eval_report_document(
            eval_run_id="eval-run-1",
            config=_config(tmp_path / "report"),  # type: ignore[arg-type]
            plan=_plan(),
            candidate_scores=(aggregate, restamped),
        )


def test_a_report_refuses_incomplete_duplicate_or_restamped_aggregates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = (
        _scored(CALL_TASK, tmp_path, monkeypatch=monkeypatch),
        _scored(TEXT_TASK, tmp_path, monkeypatch=monkeypatch),
    )
    aggregate = aggregate_trace_scores(
        scores=scores,
        plan=_plan(),
        candidate_alias="candidate_a",
    )
    arguments = {
        "eval_run_id": "eval-run-1",
        "config": _config(tmp_path / "report"),
        "plan": _plan(),
    }

    with pytest.raises(EvalArtifactError, match="do not cover the plan aliases"):
        eval_report_document(candidate_scores=(), **arguments)  # type: ignore[arg-type]
    with pytest.raises(EvalArtifactError, match="duplicate alias"):
        eval_report_document(
            candidate_scores=(aggregate, aggregate),
            **arguments,  # type: ignore[arg-type]
        )
    with pytest.raises(EvalArtifactError, match="authorization boundary"):
        eval_report_document(
            candidate_scores=(
                aggregate.model_copy(update={"plan_identity": "sha256:" + "9" * 64}),
            ),
            **arguments,  # type: ignore[arg-type]
        )


def test_publication_refuses_task_scores_that_do_not_match_their_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores = (
        _scored(CALL_TASK, tmp_path, monkeypatch=monkeypatch),
        _scored(TEXT_TASK, tmp_path, monkeypatch=monkeypatch),
    )
    aggregate = aggregate_trace_scores(
        scores=scores,
        plan=_plan(),
        candidate_alias="candidate_a",
    )
    restamped_hashes = aggregate.model_copy(
        update={
            "task_score_hashes": (
                "sha256:" + "9" * 64,
                aggregate.task_score_hashes[1],
            )
        }
    )

    with pytest.raises(EvalArtifactError, match="score identities"):
        write_trace_eval_artifacts(
            eval_run_id="eval-run-1",
            config=_config(tmp_path / "output"),  # type: ignore[arg-type]
            plan=_plan(),
            candidate_scores=(restamped_hashes,),
            task_scores=scores,
            candidate_io_cache_path=None,
        )

    restamped_score = scores[0].model_copy(
        update={"source_verification_identity": "sha256:" + "8" * 64}
    )
    aggregate_for_restamped_score = aggregate.model_copy(
        update={
            "task_score_hashes": (
                restamped_score.score_hash,
                aggregate.task_score_hashes[1],
            )
        }
    )
    with pytest.raises(EvalArtifactError, match="aggregate boundary"):
        write_trace_eval_artifacts(
            eval_run_id="eval-run-1",
            config=_config(tmp_path / "restamped-output"),  # type: ignore[arg-type]
            plan=_plan(),
            candidate_scores=(aggregate_for_restamped_score,),
            task_scores=(restamped_score, scores[1]),
            candidate_io_cache_path=None,
        )


# --------------------------------------------------------------------------------------
# Publication: the same three immutable files, stamped as a trace measurement.
# --------------------------------------------------------------------------------------


def test_a_trace_run_publishes_one_immutable_artifact_set_it_can_account_for(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyarrow.parquet as pq

    monkeypatch.setenv("CANDIDATE_API_KEY", "secret-key")
    output_dir = tmp_path / "trace-output"
    _authorized_run(monkeypatch, tmp_path)
    config = _config(output_dir)

    result = asyncio.run(
        eval_runner.run_bfcl_trace_eval(
            config,  # type: ignore[arg-type]
            client_factory=_client_factory(_replies_by_task()),
        )
    )

    assert tuple(score.task_id for score in result.task_scores) == TASK_IDS
    assert all(score.task_success for score in result.task_scores)
    assert [score.scope for score in result.candidate_scores] == ["trace"]

    report = json.loads(result.artifacts.report_path.read_text(encoding="utf-8"))
    assert report["eval_scope"] == "trace"
    assert report["eval_run_id"] == result.eval_run_id
    assert report["candidates"][0]["scope"] == "trace"
    assert set(report["candidates"][0]["metrics"]) == set(TRACE_METRIC_TAXONOMY)

    assert result.artifacts.manifest_path is not None
    manifest = json.loads(result.artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["eval_scope"] == "trace"
    # There is no live tool result to persist, so no tool-trace cache is claimed.
    assert set(manifest["artifacts"]) == {
        "eval_report",
        "eval_task_results",
        "candidate_io_cache",
    }
    assert not (output_dir / "tool_trace_cache.jsonl").exists()

    assert result.artifacts.task_results_path is not None
    rows = pq.read_table(result.artifacts.task_results_path).to_pylist()
    assert [row["task_id"] for row in rows] == list(TASK_IDS)
    assert {row["mode"] for row in rows} == {"trace"}
    # A trace row never implies evidence its scorer had no way to observe.
    for row in rows:
        assert row["execution_success"] is None
        assert row["assertions_passed"] is None
        assert row["milestones_correct"] is None
        assert row["final_answer_passed"] is None
        assert row["failure_codes"] == []
        assert row["failure_records"] == []

    # Replaying the same run into the same tree reproduces every identity.
    replay = asyncio.run(
        eval_runner.run_bfcl_trace_eval(
            config,  # type: ignore[arg-type]
            client_factory=_client_factory(_replies_by_task()),
        )
    )
    assert replay.eval_run_id == result.eval_run_id
    assert replay.artifacts.report_hash == result.artifacts.report_hash
    assert replay.artifacts.task_results_hash == result.artifacts.task_results_hash
    assert tuple(score.score_hash for score in replay.task_scores) == tuple(
        score.score_hash for score in result.task_scores
    )


def test_a_failed_trace_task_publishes_its_terminal_and_its_gate_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrong tool is the model's failure, and the row says so in one vocabulary."""
    import pyarrow.parquet as pq

    monkeypatch.setenv("CANDIDATE_API_KEY", "secret-key")
    output_dir = tmp_path / "mismatch-output"
    _authorized_run(monkeypatch, tmp_path)
    replies = _replies_by_task()
    replies[CALL_TASK] = [_text_reply("I will not check anything.")]

    result = asyncio.run(
        eval_runner.run_bfcl_trace_eval(
            _config(output_dir),  # type: ignore[arg-type]
            client_factory=_client_factory(replies),
        )
    )

    failed = next(score for score in result.task_scores if score.task_id == CALL_TASK)
    assert not failed.task_success
    assert not failed.non_candidate_stop
    aggregate = result.candidate_scores[0]
    assert aggregate.successful_tasks == 1
    assert aggregate.metric("task_success_rate").denominator == 2

    assert result.artifacts.task_results_path is not None
    row = next(
        row
        for row in pq.read_table(result.artifacts.task_results_path).to_pylist()
        if row["task_id"] == CALL_TASK
    )
    assert row["task_success"] is False
    assert row["episode_status"] == "candidate_mismatch"
    assert row["execution_success"] is None
    assert row["assertions_passed"] is None
    assert row["milestones_correct"] is None
    assert row["final_answer_passed"] is None
    layers = {record["layer"] for record in row["failure_records"]}
    assert layers == {"episode", "gate"}
    assert {record["attribution"] for record in row["failure_records"]} == {"candidate"}
    assert "episode.candidate_mismatch" in {
        record["code"] for record in row["failure_records"]
    }


def test_a_trace_run_refuses_to_publish_a_candidate_cache_it_cannot_account_for(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A claimed request without a completion is crash evidence, not a cache."""
    scores = (
        _scored(CALL_TASK, tmp_path, monkeypatch=monkeypatch),
        _scored(TEXT_TASK, tmp_path, monkeypatch=monkeypatch),
    )
    aggregate = aggregate_trace_scores(
        scores=scores,
        plan=_plan(),
        candidate_alias="candidate_a",
    )
    output_dir = tmp_path / "crashed-output"
    output_dir.mkdir()
    cache_path = output_dir / "candidate_io_cache.jsonl"
    CandidateIOCache(cache_path).put_request(
        build_candidate_request(
            _candidate(),
            request_id="candidate_a:t__call:0",
            task_id=CALL_TASK,
            turn_index=0,
            messages=_script(CALL_TASK, tmp_path).seed_messages,
            tools=TOOLS,
        )
    )

    with pytest.raises(EvalArtifactError, match="cache failed publication validation"):
        write_trace_eval_artifacts(
            eval_run_id="eval-run-1",
            config=_config(output_dir),  # type: ignore[arg-type]
            plan=_plan(),
            candidate_scores=(aggregate,),
            task_scores=scores,
            candidate_io_cache_path=cache_path,
        )
    assert not (output_dir / "eval_report.json").exists()


def test_an_unpublished_candidate_cache_leaves_nothing_beside_the_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CANDIDATE_API_KEY", "secret-key")
    output_dir = tmp_path / "no-cache-output"
    _authorized_run(monkeypatch, tmp_path)

    result = asyncio.run(
        eval_runner.run_bfcl_trace_eval(
            _config(output_dir, cache_candidate_responses=False),  # type: ignore[arg-type]
            client_factory=_client_factory(_replies_by_task()),
        )
    )

    assert not (output_dir / "candidate_io_cache.jsonl").exists()
    assert result.artifacts.manifest_path is not None
    manifest = json.loads(result.artifacts.manifest_path.read_text(encoding="utf-8"))
    assert "candidate_io_cache" not in manifest["artifacts"]


# --------------------------------------------------------------------------------------
# Orchestration: one measurement per run, bounded and fail-closed.
# --------------------------------------------------------------------------------------


def test_the_trace_runner_refuses_an_executable_config_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        eval_runner,
        "write_resolved_eval_config",
        lambda *_args: pytest.fail("an executable config must fail before writing"),
    )

    with pytest.raises(eval_runner.UnsupportedRunnerModeError) as caught:
        asyncio.run(
            eval_runner.run_bfcl_trace_eval(
                _config(tmp_path / "executable", executable=True),  # type: ignore[arg-type]
            )
        )

    assert caught.value.code == "eval_runner_mode_unsupported"
    assert "run_bfcl_eval" in caught.value.recovery


def test_the_declared_mode_chooses_the_runner_rather_than_the_caller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []

    async def executable(config: Any, **_kwargs: Any) -> str:
        called.append("executable")
        return "executable"

    async def trace(config: Any, **_kwargs: Any) -> str:
        called.append("trace")
        return "trace"

    monkeypatch.setattr(eval_runner, "run_bfcl_eval", executable)
    monkeypatch.setattr(eval_runner, "run_bfcl_trace_eval", trace)
    modes = iter([True, False])
    monkeypatch.setattr(
        eval_runner,
        "load_eval_config",
        lambda _path: SimpleNamespace(
            settings=SimpleNamespace(executable=next(modes), held_out_eval=False)
        ),
    )

    assert eval_runner.run_declared_eval_sync(tmp_path / "eval.yaml") == "executable"
    assert eval_runner.run_declared_eval_sync(tmp_path / "eval.yaml") == "trace"
    assert called == ["executable", "trace"]


def test_the_trace_batch_bounds_parallelism_and_keeps_publication_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    peak = 0

    monkeypatch.setattr(
        eval_runner,
        "build_conversation_script",
        lambda _projection, task_id, **_kwargs: SimpleNamespace(task_id=task_id),
    )

    async def drive(**kwargs: Any) -> str:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.02)
            return kwargs["script"].task_id
        finally:
            active -= 1

    monkeypatch.setattr(eval_runner, "run_candidate_episode", drive)
    monkeypatch.setattr(
        eval_runner,
        "score_trace_episode",
        lambda **kwargs: f"score:{kwargs['episode']}",
    )

    class Plan:
        candidate_aliases = ("candidate_a",)
        scoring_policy_hash = _scoring().scoring_policy_hash

        def evaluation_task_ids(self, alias: str) -> tuple[str, ...]:
            assert alias == "candidate_a"
            return tuple(f"task-{index}" for index in range(6))

    scores = asyncio.run(
        eval_runner._run_candidate_trace_tasks(
            config=_config(tmp_path, max_parallel_tasks=2),  # type: ignore[arg-type]
            source=object(),  # type: ignore[arg-type]
            plan=Plan(),  # type: ignore[arg-type]
            projection=object(),  # type: ignore[arg-type]
            candidate=SimpleNamespace(alias="candidate_a"),  # type: ignore[arg-type]
            client=SimpleNamespace(),  # type: ignore[arg-type]
        )
    )

    assert scores == tuple(f"score:task-{index}" for index in range(6))
    assert peak == 2


def test_the_first_trace_failure_cancels_its_siblings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[str] = []

    monkeypatch.setattr(
        eval_runner,
        "build_conversation_script",
        lambda _projection, task_id, **_kwargs: SimpleNamespace(task_id=task_id),
    )

    async def drive(**kwargs: Any) -> str:
        task_id = kwargs["script"].task_id
        started.append(task_id)
        if task_id == "task-1":
            raise RuntimeError("the benchmark projection changed under the run")
        await asyncio.sleep(10)
        return task_id

    monkeypatch.setattr(eval_runner, "run_candidate_episode", drive)

    class Plan:
        candidate_aliases = ("candidate_a",)

        def evaluation_task_ids(self, alias: str) -> tuple[str, ...]:
            return ("task-0", "task-1", "task-2")

    with pytest.raises(RuntimeError, match="changed under the run"):
        asyncio.run(
            eval_runner._run_candidate_trace_tasks(
                config=_config(tmp_path, max_parallel_tasks=2),  # type: ignore[arg-type]
                source=object(),  # type: ignore[arg-type]
                plan=Plan(),  # type: ignore[arg-type]
                projection=object(),  # type: ignore[arg-type]
                candidate=SimpleNamespace(alias="candidate_a"),  # type: ignore[arg-type]
                client=SimpleNamespace(),  # type: ignore[arg-type]
            )
        )

    assert "task-2" not in started
