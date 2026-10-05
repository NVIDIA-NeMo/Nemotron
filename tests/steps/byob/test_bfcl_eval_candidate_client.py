"""Native function-calling transport and its immutable observations."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval import (
    CandidateApi,
    CandidateInference,
    CandidateModelIdentity,
    EvalCandidate,
    EvalLimits,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_cache import (
    CandidateIOCache,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_client import (
    NativeFunctionCallingClient,
    build_candidate_request,
    parse_candidate_response,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_contract import (
    CandidateAttempt,
    CandidateCallOutcome,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_errors import (
    CandidateAuthenticationError,
    CandidateCacheError,
    CandidateCredentialMissingError,
    CandidateProviderExtensionError,
    CandidateRequestError,
    CandidateResponseError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import thaw_json

MESSAGES = [{"role": "user", "content": "Balance of account 1?"}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_balance",
            "description": "Read the balance.",
            "parameters": {
                "type": "object",
                "properties": {"account_id": {"type": "string"}},
                "required": ["account_id"],
                "additionalProperties": False,
            },
        },
    }
]


def _candidate(
    *,
    extensions: dict[str, Any] | None = None,
    top_p: float | None = 1.0,
) -> EvalCandidate:
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
            top_p=top_p,
            max_tokens=512,
            seed=42,
            tool_choice="auto",
            provider_extensions=extensions or {},
        ),
    )


def _limits(*, retries: int = 2) -> EvalLimits:
    return EvalLimits(
        max_turns=6,
        tool_timeout_s=1.0,
        candidate_timeout_s=5.0,
        episode_timeout_s=10.0,
        max_parallel_tasks=2,
        max_retries=retries,
    )


def _request(candidate: EvalCandidate | None = None):
    return build_candidate_request(
        candidate or _candidate(),
        request_id="candidate_a:t__1:0",
        task_id="t__1",
        turn_index=0,
        messages=MESSAGES,
        tools=TOOLS,
    )


def _completion(*, arguments: Any = '{"account_id":"1"}') -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_balance", "arguments": arguments},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _run(client: NativeFunctionCallingClient, request=None):
    async def execute():
        try:
            return await client.complete(request or _request())
        finally:
            await client.aclose()

    return asyncio.run(execute())


def test_request_contains_native_tools_and_every_pinned_inference_parameter() -> None:
    request = _request()

    assert thaw_json(request.body) == {
        "model": "candidate-route",
        "messages": MESSAGES,
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "seed": 42,
    }
    assert request.request_hash.startswith("sha256:")
    assert request.request_body_hash.startswith("sha256:")


def test_an_unset_nucleus_cutoff_leaves_the_field_out_instead_of_sending_a_null() -> None:
    """Providers that refuse the pair refuse it on presence, null included.

    Every Anthropic model rejects a request carrying both temperature and top_p, so a
    null that reaches the wire fails the run exactly as a number would. The field has
    to be absent, and the two bodies must stay distinguishable by hash so provenance
    cannot confuse an omitted cutoff with a pinned one.
    """
    request = _request(_candidate(top_p=None))

    body = thaw_json(request.body)
    assert "top_p" not in body
    assert body["temperature"] == 0.0
    assert body["seed"] == 42
    assert request.request_body_hash != _request(_candidate(top_p=1.0)).request_body_hash


def test_nucleus_sampling_may_only_go_unpinned_where_the_decode_is_greedy() -> None:
    """The escape hatch is bounded by the reason it is safe.

    At temperature 0 the distribution is a point mass, so no cutoff changes the chosen
    token and omitting the field pins nothing away. Above 0 an absent top_p would hand
    real sampling behaviour to a provider default this config never recorded, which is
    the drift the pinned-inference contract exists to prevent.
    """
    greedy = CandidateInference(temperature=0.0, top_p=None, max_tokens=512, tool_choice="auto")
    assert greedy.top_p is None
    # null is recorded rather than dropped, so the hash states which case this is.
    assert greedy.semantic_payload()["top_p"] is None

    with pytest.raises(ValidationError, match="only be null when temperature is 0"):
        CandidateInference(temperature=0.7, top_p=None, max_tokens=512, tool_choice="auto")


def test_only_the_matching_provider_extension_is_sent() -> None:
    candidate = _candidate(extensions={"nvidia.v1": {"repetition_penalty": 1.1}})

    request = _request(candidate)

    assert request.body["repetition_penalty"] == 1.1


def test_an_extension_cannot_replace_the_standard_contract() -> None:
    candidate = _candidate(extensions={"nvidia.v1": {"temperature": 0.7}})

    with pytest.raises(CandidateProviderExtensionError, match="standard request field"):
        _request(candidate)


def test_answer_key_fields_cannot_enter_model_facing_messages() -> None:
    with pytest.raises(CandidateRequestError, match="answer-key"):
        build_candidate_request(
            _candidate(),
            request_id="candidate_a:t__1:0",
            task_id="t__1",
            turn_index=0,
            messages=[
                {
                    "role": "user",
                    "content": "Balance?",
                    "expected_tool_calls": [{"function_name": "get_balance"}],
                }
            ],
            tools=TOOLS,
        )


@pytest.mark.parametrize(
    ("arguments", "status"),
    [
        ('{"account_id":"1"}', "valid_object"),
        ('{"account_id":', "invalid_json"),
        ('{"account_id":NaN}', "invalid_json"),
        ('["1"]', "json_not_object"),
        (None, "missing"),
        ({"account_id": "1"}, "wrong_type"),
    ],
)
def test_function_arguments_are_parsed_once_without_repair(arguments: Any, status: str) -> None:
    raw = json.dumps(_completion(arguments=arguments))

    response = parse_candidate_response(raw, selected_attempt=0)

    call = response.tool_calls[0]
    assert call.raw_arguments == arguments
    assert call.arguments_status == status
    if status == "valid_object":
        assert call.parsed_arguments == {"account_id": "1"}
    else:
        assert call.parsed_arguments is None


def test_nested_candidate_output_is_frozen() -> None:
    response = parse_candidate_response(
        json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": {"parts": ["done"]},
                            "tool_calls": [],
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        ),
        selected_attempt=0,
    )

    with pytest.raises(TypeError):
        response.assistant_content["new"] = True


def test_a_non_openai_envelope_is_a_model_failure_not_a_retry() -> None:
    with pytest.raises(CandidateResponseError, match="exactly one choice"):
        parse_candidate_response('{"choices":[]}', selected_attempt=0)


def test_a_live_completion_is_cached_and_replayed_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url == "https://candidate.example.com/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer top-secret"
        sent = json.loads(request.content)
        assert sent["tools"] == TOOLS
        return httpx.Response(
            200,
            json=_completion(),
            headers={"x-request-id": "provider-request-1"},
        )

    monkeypatch.setenv("CANDIDATE_API_KEY", "top-secret")
    cache = CandidateIOCache(tmp_path / "candidate_io_cache.jsonl")
    first = _run(
        NativeFunctionCallingClient(
            _candidate(), _limits(), cache, transport=httpx.MockTransport(handler)
        )
    )
    monkeypatch.delenv("CANDIDATE_API_KEY")
    second = _run(
        NativeFunctionCallingClient(
            _candidate(),
            _limits(),
            CandidateIOCache(cache.path),
            transport=httpx.MockTransport(handler),
        )
    )

    assert calls == 1
    assert first.status == second.status == "completed"
    assert first.response is not None and second.response is not None
    assert first.response.response_hash == second.response.response_hash
    assert second.replayed is True
    assert "top-secret" not in cache.path.read_text(encoding="utf-8")


def test_a_missing_credential_fails_before_network_or_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CANDIDATE_API_KEY", raising=False)
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json=_completion())

    cache = CandidateIOCache(tmp_path / "candidate_io_cache.jsonl")
    client = NativeFunctionCallingClient(
        _candidate(), _limits(), cache, transport=httpx.MockTransport(handler)
    )

    with pytest.raises(CandidateCredentialMissingError):
        _run(client)

    assert called is False
    assert not cache.path.exists()


def test_transient_provider_errors_are_retried_and_all_attempts_are_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                503,
                text="temporarily unavailable",
                headers={"retry-after": "0"},
            )
        return httpx.Response(200, json=_completion())

    monkeypatch.setenv("CANDIDATE_API_KEY", "secret")
    cache = CandidateIOCache(tmp_path / "candidate_io_cache.jsonl")
    outcome = _run(
        NativeFunctionCallingClient(
            _candidate(),
            _limits(),
            cache,
            transport=httpx.MockTransport(handler),
            backoff_base_s=0,
        )
    )

    assert outcome.status == "completed"
    assert [attempt.status for attempt in outcome.attempts] == [
        "provider_unavailable",
        "completed",
    ]
    assert outcome.attempts[0].retry_after_s == 0.0
    documents = [json.loads(line) for line in cache.path.read_text(encoding="utf-8").splitlines()]
    assert [document["record_type"] for document in documents] == [
        "request",
        "attempt",
        "attempt",
        "completion",
    ]
    assert "temporarily unavailable" not in cache.path.read_text(encoding="utf-8")


def test_a_rejected_credential_stops_the_run_rather_than_scoring_the_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every task presents the same key, so one refusal settles all of them.

    Recording the refusal as this task's outcome would let the run continue and
    publish a report whose zeroes read like a measurement of the model. It would
    also cache a rejected credential as the task's immutable answer, so a rerun
    with a working key would replay the refusal instead of asking the endpoint.
    """
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="invalid token secret")

    monkeypatch.setenv("CANDIDATE_API_KEY", "secret")
    cache = CandidateIOCache(tmp_path / "candidate_io_cache.jsonl")
    with pytest.raises(CandidateAuthenticationError) as raised:
        _run(
            NativeFunctionCallingClient(
                _candidate(),
                _limits(),
                cache,
                transport=httpx.MockTransport(handler),
            )
        )

    assert calls == 1
    assert raised.value.code == "eval_candidate_authentication_failed"
    assert "CANDIDATE_API_KEY" in raised.value.recovery
    recorded = (
        cache.path.read_text(encoding="utf-8") if cache.path.exists() else ""
    )
    documents = [json.loads(line) for line in recorded.splitlines()]
    assert documents == []
    assert "invalid token secret" not in recorded


def test_malformed_http_200_is_preserved_and_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, text='{"choices":[]}')

    monkeypatch.setenv("CANDIDATE_API_KEY", "secret")
    outcome = _run(
        NativeFunctionCallingClient(
            _candidate(),
            _limits(),
            CandidateIOCache(tmp_path / "candidate_io_cache.jsonl"),
            transport=httpx.MockTransport(handler),
        )
    )

    assert calls == 1
    assert outcome.status == "malformed_response"
    assert outcome.attempts[0].raw_response == '{"choices":[]}'
    assert not (tmp_path / "candidate_io_cache.jsonl").exists()


def test_a_response_over_the_bound_is_stopped_and_not_cached_as_model_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CANDIDATE_API_KEY", "secret")
    outcome = _run(
        NativeFunctionCallingClient(
            _candidate(),
            _limits(),
            CandidateIOCache(tmp_path / "candidate_io_cache.jsonl"),
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, content=b"x" * 128)
            ),
            max_response_bytes=32,
        )
    )

    assert outcome.status == "malformed_response"
    assert outcome.attempts[0].raw_response is None
    assert "exceeded 32 bytes" in str(outcome.attempts[0].diagnostic)


def test_invalid_utf8_is_a_malformed_observation_not_a_parser_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CANDIDATE_API_KEY", "secret")
    outcome = _run(
        NativeFunctionCallingClient(
            _candidate(),
            _limits(),
            CandidateIOCache(tmp_path / "candidate_io_cache.jsonl"),
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, content=b"\xff\xfe")
            ),
        )
    )

    assert outcome.status == "malformed_response"
    assert outcome.attempts[0].raw_response is None
    assert "UTF-8" in str(outcome.attempts[0].diagnostic)


def test_timeouts_exhaust_the_pinned_retry_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out", request=request)

    monkeypatch.setenv("CANDIDATE_API_KEY", "secret")
    cache_path = tmp_path / "candidate_io_cache.jsonl"
    outcome = _run(
        NativeFunctionCallingClient(
            _candidate(),
            _limits(retries=2),
            CandidateIOCache(cache_path),
            transport=httpx.MockTransport(handler),
            backoff_base_s=0,
        )
    )

    assert calls == 3
    assert outcome.status == "retry_exhausted"
    assert [attempt.status for attempt in outcome.attempts] == ["timeout"] * 3
    assert not cache_path.exists()

    retried = _run(
        NativeFunctionCallingClient(
            _candidate(),
            _limits(retries=0),
            CandidateIOCache(cache_path),
            transport=httpx.MockTransport(handler),
            backoff_base_s=0,
        )
    )
    assert retried.status == "retry_exhausted"
    assert calls == 4
    assert not cache_path.exists()


def test_an_interrupted_cache_is_evidence_not_a_cache_hit(tmp_path: Path) -> None:
    cache = CandidateIOCache(tmp_path / "candidate_io_cache.jsonl")
    request = _request()
    cache.put_request(request)

    with pytest.raises(CandidateCacheError, match="unfinished request"):
        CandidateIOCache(cache.path).get(request.request_hash)


def test_cancellation_is_recorded_then_propagated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CANDIDATE_API_KEY", "secret")
    cache = CandidateIOCache(tmp_path / "candidate_io_cache.jsonl")

    async def handler(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(60)
        return httpx.Response(200, json=_completion())

    async def cancel_call() -> None:
        client = NativeFunctionCallingClient(
            _candidate(),
            _limits(),
            cache,
            transport=httpx.MockTransport(handler),
        )
        try:
            task = asyncio.create_task(client.complete(_request()))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            await client.aclose()

    asyncio.run(cancel_call())
    assert not cache.path.exists()
    assert CandidateIOCache(cache.path).get(_request().request_hash) is None


def test_duplicate_concurrent_calls_pay_for_one_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CANDIDATE_API_KEY", "secret")
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return httpx.Response(200, json=_completion())

    async def run_both():
        client = NativeFunctionCallingClient(
            _candidate(),
            _limits(),
            CandidateIOCache(tmp_path / "candidate_io_cache.jsonl"),
            transport=httpx.MockTransport(handler),
        )
        try:
            request = _request()
            return await asyncio.gather(client.complete(request), client.complete(request))
        finally:
            await client.aclose()

    first, second = asyncio.run(run_both())
    assert calls == 1
    assert first.status == second.status == "completed"
    assert {first.replayed, second.replayed} == {False, True}


def test_one_response_body_is_stored_once_however_many_records_cite_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CANDIDATE_API_KEY", "secret")
    cache = CandidateIOCache(tmp_path / "candidate_io_cache.jsonl")
    _run(
        NativeFunctionCallingClient(
            _candidate(),
            _limits(),
            cache,
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, json=_completion())
            ),
        )
    )

    # ``object`` survives only in the raw envelope, so it counts stored copies.
    assert cache.path.read_text(encoding="utf-8").count("chat.completion") == 1


def test_a_transient_outcome_cannot_be_committed_as_a_completion(tmp_path: Path) -> None:
    cache = CandidateIOCache(tmp_path / "candidate_io_cache.jsonl")
    request = _request()
    cache.put_request(request)
    recorded = CandidateAttempt(
        attempt_index=0,
        observed_at="2026-01-01T00:00:00+00:00",
        status="timeout",
        retryable=True,
        latency_s=0.5,
    )
    cache.put_attempt(request.request_hash, recorded)

    with pytest.raises(CandidateCacheError, match="semantically completed"):
        cache.put_completion(
            CandidateCallOutcome(
                request_hash=request.request_hash,
                status="retry_exhausted",
                attempts=(recorded, recorded.model_copy(update={"attempt_index": 1})),
            )
        )


def test_a_cancelled_call_never_becomes_an_outcome() -> None:
    with pytest.raises(ValidationError, match="attempt evidence"):
        CandidateCallOutcome(
            request_hash=_request().request_hash,
            status="cancelled",
            attempts=(
                CandidateAttempt(
                    attempt_index=0,
                    observed_at="2026-01-01T00:00:00+00:00",
                    status="cancelled",
                    retryable=False,
                    latency_s=0.0,
                ),
            ),
        )


def test_a_writer_sees_records_another_writer_appended_after_it_opened(tmp_path: Path) -> None:
    path = tmp_path / "candidate_io_cache.jsonl"
    first = CandidateIOCache(path)
    second = CandidateIOCache(path)
    request = _request()

    assert first.put_request(request) is True
    assert second.put_request(request) is False


def test_tampering_with_a_cache_record_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CANDIDATE_API_KEY", "secret")
    cache = CandidateIOCache(tmp_path / "candidate_io_cache.jsonl")
    _run(
        NativeFunctionCallingClient(
            _candidate(),
            _limits(),
            cache,
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, json=_completion())
            ),
        )
    )
    contents = cache.path.read_text(encoding="utf-8")
    cache.path.write_text(contents.replace("candidate-route", "other-route", 1), encoding="utf-8")

    with pytest.raises(CandidateCacheError, match="record_hash"):
        CandidateIOCache(cache.path)
