# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native OpenAI-compatible function-calling transport."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_cache import (
    CandidateIOCache,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_contract import (
    CallStatus,
    CandidateAttempt,
    CandidateCallOutcome,
    CandidateRequest,
    CandidateResponse,
    CandidateToolCall,
    parse_function_arguments,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.candidate_errors import (
    CandidateAuthenticationError,
    CandidateCredentialMissingError,
    CandidateProviderExtensionError,
    CandidateRequestError,
    CandidateResponseError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.eval.schemas import (
    EvalCandidate,
    EvalLimits,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.export_contract import (
    thaw_json,
    validate_json_value,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

_STANDARD_REQUEST_FIELDS = frozenset(
    {"model", "messages", "tools", "tool_choice", "temperature", "top_p", "max_tokens", "seed"}
)
_RETRYABLE_HTTP = frozenset({408, 429, 500, 502, 503, 504})
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_ANSWER_KEY_FIELDS = frozenset(
    {"expected_tool_calls", "success_assertions", "fixture_refs", "oracle_results", "reference_trace"}
)


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(canonical_json(value).encode("utf-8"))


def build_candidate_request(
    candidate: EvalCandidate,
    *,
    request_id: str,
    task_id: str,
    turn_index: int,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
) -> CandidateRequest:
    """Build and hash the exact provider body without contacting the endpoint."""
    if not request_id.strip() or not task_id.strip() or type(turn_index) is not int or turn_index < 0:
        raise CandidateRequestError(
            f"candidates[{candidate.alias}]",
            "has an invalid request, task, or turn identity",
            actual={"request_id": request_id, "task_id": task_id, "turn_index": turn_index},
            expected="non-empty request_id/task_id and a non-negative integer turn_index",
            recovery="construct candidate calls from the authorized task and conversation turn",
        )
    message_list = [dict(message) for message in messages]
    tool_list = [dict(tool) for tool in tools]
    validate_json_value(message_list, label="candidate messages")
    validate_json_value(tool_list, label="candidate tools")
    if not message_list:
        raise CandidateRequestError(
            f"candidates[{candidate.alias}]",
            "would send no messages",
            expected="at least one model-facing message",
            recovery="start the conversation from the answer-free seed messages",
        )
    _validate_model_facing_messages(candidate, message_list)
    _validate_model_facing_tools(candidate, tool_list)
    body: dict[str, Any] = {
        "model": candidate.model,
        "messages": message_list,
        "tools": tool_list,
        "tool_choice": candidate.inference.tool_choice,
        "temperature": candidate.inference.temperature,
        "max_tokens": candidate.inference.max_tokens,
    }
    # A null top_p is omitted rather than sent as null: providers that reject the pair
    # reject it on presence, and a null would also read as a value the endpoint should
    # interpret. The schema only permits null at temperature 0, where the decode is
    # greedy and the field cannot change the chosen token.
    if candidate.inference.top_p is not None:
        body["top_p"] = candidate.inference.top_p
    if candidate.inference.seed is not None:
        body["seed"] = candidate.inference.seed
    body.update(_provider_extensions(candidate))
    validate_json_value(body, label="candidate request body")
    identity = {
        "schema_version": "1.0",
        "request_id": request_id.strip(),
        "candidate_alias": candidate.alias,
        "canonical_model_identity": candidate.canonical_model_identity,
        "task_id": task_id.strip(),
        "turn_index": turn_index,
        "provider": candidate.provider,
        "provider_api_version": candidate.provider_api_version,
        "base_url": candidate.api.base_url,
        "model": candidate.model,
        "body": body,
    }
    return CandidateRequest(
        **{key: value for key, value in identity.items() if key not in {"schema_version", "body"}},
        body=body,
        request_body_hash=_sha256_json(body),
        request_hash=_sha256_json(identity),
    )


def _validate_model_facing_messages(
    candidate: EvalCandidate,
    messages: Sequence[Mapping[str, Any]],
) -> None:
    for index, message in enumerate(messages):
        role = message.get("role")
        if not isinstance(role, str) or not role.strip():
            raise CandidateRequestError(
                f"candidates[{candidate.alias}].messages[{index}]",
                "has no message role",
                expected="a non-empty OpenAI-compatible role",
                recovery="project the canonical conversation into model-facing messages",
            )
        if leaked := sorted(set(message) & _ANSWER_KEY_FIELDS):
            raise CandidateRequestError(
                f"candidates[{candidate.alias}].messages[{index}]",
                f"contains answer-key field(s) {leaked}",
                expected="only model-facing conversation fields",
                recovery="remove gold calls, assertions, fixtures, oracle results, and reference traces",
            )


def _validate_model_facing_tools(
    candidate: EvalCandidate,
    tools: Sequence[Mapping[str, Any]],
) -> None:
    if not tools:
        raise CandidateRequestError(
            f"candidates[{candidate.alias}].tools",
            "is empty",
            expected="the function definitions published for this BFCL task",
            recovery="decode the row's model-facing tools before building the request",
        )
    for index, tool in enumerate(tools):
        function = tool.get("function")
        if tool.get("type") != "function" or not isinstance(function, Mapping):
            raise CandidateRequestError(
                f"candidates[{candidate.alias}].tools[{index}]",
                "is not an OpenAI-compatible function definition",
                expected="type=function and a function mapping",
                recovery="send the decoded model-facing tools from the verified source",
            )
        if not isinstance(function.get("name"), str) or not isinstance(function.get("parameters"), Mapping):
            raise CandidateRequestError(
                f"candidates[{candidate.alias}].tools[{index}].function",
                "does not name a function with a JSON Schema parameter mapping",
                expected="non-empty name and parameters object",
                recovery="send the decoded model-facing tools from the verified source",
            )


def _provider_extensions(candidate: EvalCandidate) -> dict[str, Any]:
    extensions = thaw_json(candidate.inference.provider_extensions)
    if not extensions:
        return {}
    expected = f"{candidate.provider}.{candidate.provider_api_version}".casefold()
    normalized = {str(name).casefold(): settings for name, settings in extensions.items()}
    unknown = sorted(set(normalized) - {expected})
    if unknown:
        raise CandidateProviderExtensionError(
            f"candidates[{candidate.alias}].inference.provider_extensions",
            f"declares namespace(s) {unknown} that do not describe this endpoint",
            expected=expected,
            recovery="remove the namespace or align provider/provider_api_version with the endpoint contract",
        )
    settings = normalized.get(expected, {})
    if not isinstance(settings, dict):
        raise CandidateProviderExtensionError(
            f"candidates[{candidate.alias}].inference.provider_extensions.{expected}",
            "is not a mapping",
            expected="a JSON mapping of provider request fields",
            recovery="move provider-specific fields under the matching versioned namespace",
        )
    if conflicts := sorted(set(settings) & _STANDARD_REQUEST_FIELDS):
        raise CandidateProviderExtensionError(
            f"candidates[{candidate.alias}].inference.provider_extensions.{expected}",
            f"tries to replace standard request field(s) {conflicts}",
            expected="provider-only fields that do not override the pinned inference contract",
            recovery="set standard fields in candidates[].inference and remove them from the extension",
        )
    return dict(settings)


def parse_candidate_response(
    raw_response: str,
    *,
    selected_attempt: int,
    provider_request_id: str | None = None,
) -> CandidateResponse:
    """Parse the provider envelope once; malformed function JSON is preserved."""
    try:
        document = json.loads(
            raw_response,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-standard JSON constant {token}")),
        )
        validate_json_value(document, label="candidate response")
    except (json.JSONDecodeError, ValueError) as exc:
        raise CandidateResponseError(
            "candidate.response",
            "is not JSON",
            actual=raw_response,
            expected="an OpenAI-compatible chat completion object",
            recovery="fix the endpoint compatibility; model output is never repaired by another LLM",
        ) from exc
    if not isinstance(document, dict):
        raise _response_error("is not a JSON object", document)
    choices = document.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise _response_error("must contain exactly one choice", choices)
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise _response_error("choice[0].message is not an object", choice)
    message = choice["message"]
    calls_value = message.get("tool_calls", [])
    if calls_value is None:
        calls_value = []
    if not isinstance(calls_value, list):
        raise _response_error("message.tool_calls is not a list", calls_value)
    calls: list[CandidateToolCall] = []
    for index, item in enumerate(calls_value):
        if not isinstance(item, dict):
            raise _response_error(f"message.tool_calls[{index}] is not an object", item)
        function = item.get("function")
        if not isinstance(function, dict):
            raise _response_error(f"message.tool_calls[{index}].function is not an object", function)
        if not isinstance(function.get("name"), str) or not function["name"].strip():
            raise _response_error(
                f"message.tool_calls[{index}].function.name is not a non-empty string",
                function.get("name"),
            )
        raw, parsed, status = parse_function_arguments(function.get("arguments"))
        calls.append(
            CandidateToolCall(
                index=index,
                id=item.get("id") if isinstance(item.get("id"), str) else None,
                type=item.get("type") if isinstance(item.get("type"), str) else None,
                function_name=function["name"],
                raw_arguments=raw,
                parsed_arguments=parsed,
                arguments_status=status,
            )
        )
    content = message.get("content")
    validate_json_value(content, label="candidate assistant content")
    usage = document.get("usage")
    if usage is not None and not isinstance(usage, dict):
        raise _response_error("usage is present but is not an object", usage)
    validate_json_value(usage, label="candidate response usage")
    response_id = document.get("id")
    finish_reason = choice.get("finish_reason")
    return CandidateResponse(
        provider_response_id=response_id if isinstance(response_id, str) else None,
        provider_request_id=provider_request_id,
        assistant_content=content,
        tool_calls=tuple(calls),
        finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        usage=usage,
        selected_attempt=selected_attempt,
        raw_response_hash=_sha256_bytes(raw_response.encode("utf-8")),
    )


def _response_error(problem: str, actual: Any) -> CandidateResponseError:
    return CandidateResponseError(
        "candidate.response",
        problem,
        actual=actual,
        expected="one OpenAI-compatible chat completion choice",
        recovery="fix the endpoint compatibility; malformed model output is recorded and never repaired",
    )


class NativeFunctionCallingClient:
    """Send one native function-calling assistant turn, with replayable evidence."""

    def __init__(
        self,
        candidate: EvalCandidate,
        limits: EvalLimits,
        cache: CandidateIOCache,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        backoff_base_s: float = 0.25,
        max_response_bytes: int = _MAX_RESPONSE_BYTES,
    ) -> None:
        self.candidate = candidate
        self.limits = limits
        self.cache = cache
        self.backoff_base_s = backoff_base_s
        self.max_response_bytes = max_response_bytes
        self._semaphore = asyncio.Semaphore(limits.max_parallel_tasks)
        self._request_locks: dict[str, asyncio.Lock] = {}
        self._request_callers: dict[str, int] = {}
        self._client = httpx.AsyncClient(
            transport=transport,
            follow_redirects=False,
            timeout=httpx.Timeout(limits.candidate_timeout_s),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> NativeFunctionCallingClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        await self.aclose()

    async def complete(
        self,
        request: CandidateRequest,
        *,
        deadline: float | None = None,
    ) -> CandidateCallOutcome:
        request_hash = request.request_hash
        lock = self._request_locks.setdefault(request_hash, asyncio.Lock())
        self._request_callers[request_hash] = self._request_callers.get(request_hash, 0) + 1
        try:
            async with lock, self._semaphore:
                return await self._complete(request, deadline=deadline)
        finally:
            # One lock per logical call would otherwise outlive every call in a
            # run that makes millions of them.
            remaining = self._request_callers[request_hash] - 1
            if remaining:
                self._request_callers[request_hash] = remaining
            else:
                del self._request_callers[request_hash]
                del self._request_locks[request_hash]

    async def _complete(
        self,
        request: CandidateRequest,
        *,
        deadline: float | None,
    ) -> CandidateCallOutcome:
        cached = self.cache.get(request.request_hash)
        if cached is not None:
            return cached
        self._check_request_candidate(request)
        api_key = os.environ.get(self.candidate.api.api_key_env)
        if not api_key:
            raise CandidateCredentialMissingError(
                f"candidates[{self.candidate.alias}].api.api_key_env",
                "does not resolve to a non-empty credential",
                actual=api_key,
                secret=True,
                expected=f"environment variable {self.candidate.api.api_key_env} to be set",
                recovery="export the endpoint credential in the runner environment; never put it in YAML",
            )
        started = time.monotonic()
        logical_deadline = started + self.limits.candidate_timeout_s
        if deadline is not None:
            logical_deadline = min(logical_deadline, deadline)
        attempts: list[CandidateAttempt] = []
        for attempt_index in range(self.limits.max_retries + 1):
            remaining = logical_deadline - time.monotonic()
            if remaining <= 0:
                attempt = CandidateAttempt(
                    attempt_index=attempt_index,
                    observed_at=_observed_at(),
                    status="timeout",
                    retryable=False,
                    latency_s=0.0,
                    diagnostic="logical candidate deadline expired before the attempt",
                )
                attempts.append(attempt)
                break
            attempt, response = await self._attempt(
                request,
                attempt_index=attempt_index,
                api_key=api_key,
                timeout_s=remaining,
            )
            attempts.append(attempt)
            if attempt.status == "cancelled":
                # No cache record is written: an interruption is not a model
                # answer and the logical request must remain retryable.
                raise asyncio.CancelledError
            if attempt.status == "authentication_failed":
                # Every task in the run presents the same credential, so a
                # rejection is a property of the configuration rather than
                # evidence about this task, and no later task can do better.
                # As with a cancellation no cache record is written: persisting
                # this would harden a rejected credential into the task's answer.
                raise CandidateAuthenticationError(
                    f"candidates[{self.candidate.alias}].api",
                    "rejected the credential the runner supplied",
                    actual=f"HTTP {attempt.http_status}",
                    expected="a credential the endpoint accepts",
                    recovery=(
                        f"check that {self.candidate.api.api_key_env} holds a key for "
                        f"{self.candidate.api.base_url}, then re-run into a new output directory"
                    ),
                )
            if response is not None:
                outcome = CandidateCallOutcome(
                    request_hash=request.request_hash,
                    status="completed",
                    attempts=tuple(attempts),
                    response=response,
                )
                self._cache_completed(request, outcome)
                return outcome
            if not attempt.retryable or attempt_index == self.limits.max_retries:
                break
            requested_delay = max(
                self._backoff(request.request_hash, attempt_index),
                attempt.retry_after_s or 0.0,
            )
            delay = min(requested_delay, logical_deadline - time.monotonic())
            if delay > 0:
                await asyncio.sleep(delay)
        status: CallStatus = "retry_exhausted" if attempts[-1].retryable else attempts[-1].status
        outcome = CandidateCallOutcome(
            request_hash=request.request_hash,
            status=status,
            attempts=tuple(attempts),
        )
        return outcome

    def _cache_completed(
        self,
        request: CandidateRequest,
        outcome: CandidateCallOutcome,
    ) -> None:
        """Commit only a semantic model response as replayable evidence.

        Attempts are buffered until a valid provider completion exists. Transport,
        provider, authentication, and malformed-envelope failures therefore remain
        retryable and cannot become a task's immutable answer.
        """
        self.cache.put_request(request)
        for attempt in outcome.attempts:
            self.cache.put_attempt(request.request_hash, attempt)
        self.cache.put_completion(outcome)

    async def _attempt(
        self,
        request: CandidateRequest,
        *,
        attempt_index: int,
        api_key: str,
        timeout_s: float,
    ) -> tuple[CandidateAttempt, CandidateResponse | None]:
        started = time.monotonic()
        try:
            async with self._client.stream(
                "POST",
                f"{self.candidate.api.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                content=canonical_json(thaw_json(request.body)).encode("utf-8"),
                timeout=httpx.Timeout(timeout_s),
            ) as response:
                status_code = response.status_code
                headers = response.headers
                raw = bytearray()
                oversized = False
                async for chunk in response.aiter_bytes():
                    if len(raw) + len(chunk) > self.max_response_bytes:
                        oversized = True
                        break
                    raw.extend(chunk)
        except asyncio.CancelledError:
            attempt = CandidateAttempt(
                attempt_index=attempt_index,
                observed_at=_observed_at(),
                status="cancelled",
                retryable=False,
                latency_s=float(time.monotonic() - started),
                diagnostic="candidate call cancelled",
            )
            return attempt, None
        except httpx.TimeoutException:
            return (
                CandidateAttempt(
                    attempt_index=attempt_index,
                    observed_at=_observed_at(),
                    status="timeout",
                    retryable=True,
                    latency_s=float(time.monotonic() - started),
                    diagnostic="candidate endpoint timed out",
                ),
                None,
            )
        except httpx.TransportError:
            return (
                CandidateAttempt(
                    attempt_index=attempt_index,
                    observed_at=_observed_at(),
                    status="transport_error",
                    retryable=True,
                    latency_s=float(time.monotonic() - started),
                    diagnostic="candidate endpoint transport failed",
                ),
                None,
            )
        latency = float(time.monotonic() - started)
        provider_request_id = _safe_request_id(headers)
        raw_bytes = bytes(raw)
        raw_hash = _sha256_bytes(raw_bytes)
        if oversized:
            return (
                CandidateAttempt(
                    attempt_index=attempt_index,
                    observed_at=_observed_at(),
                    status="malformed_response",
                    retryable=False,
                    http_status=status_code,
                    latency_s=latency,
                    provider_request_id=provider_request_id,
                    diagnostic=f"response body exceeded {self.max_response_bytes} bytes",
                ),
                None,
            )
        if status_code != 200:
            status, retryable = _classify_http(status_code)
            return (
                CandidateAttempt(
                    attempt_index=attempt_index,
                    observed_at=_observed_at(),
                    status=status,
                    retryable=retryable,
                    http_status=status_code,
                    latency_s=latency,
                    raw_response_hash=raw_hash,
                    provider_request_id=provider_request_id,
                    diagnostic=f"provider returned HTTP {status_code} with {len(raw_bytes)} body bytes",
                    retry_after_s=_retry_after(headers) if retryable else None,
                ),
                None,
            )
        try:
            raw_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return (
                CandidateAttempt(
                    attempt_index=attempt_index,
                    observed_at=_observed_at(),
                    status="malformed_response",
                    retryable=False,
                    http_status=200,
                    latency_s=latency,
                    raw_response_hash=raw_hash,
                    provider_request_id=provider_request_id,
                    diagnostic="HTTP 200 response body is not valid UTF-8",
                ),
                None,
            )
        try:
            parsed = parse_candidate_response(
                raw_text,
                selected_attempt=attempt_index,
                provider_request_id=provider_request_id,
            )
        except CandidateResponseError as exc:
            return (
                CandidateAttempt(
                    attempt_index=attempt_index,
                    observed_at=_observed_at(),
                    status="malformed_response",
                    retryable=False,
                    http_status=200,
                    latency_s=latency,
                    raw_response=raw_text,
                    raw_response_hash=raw_hash,
                    provider_request_id=provider_request_id,
                    diagnostic=exc.problem,
                ),
                None,
            )
        return (
            CandidateAttempt(
                attempt_index=attempt_index,
                observed_at=_observed_at(),
                status="completed",
                retryable=False,
                http_status=200,
                latency_s=latency,
                raw_response=raw_text,
                raw_response_hash=raw_hash,
                provider_request_id=provider_request_id,
            ),
            parsed,
        )

    def _check_request_candidate(self, request: CandidateRequest) -> None:
        if (
            request.candidate_alias != self.candidate.alias
            or request.canonical_model_identity != self.candidate.canonical_model_identity
            or request.base_url != self.candidate.api.base_url
            or request.model != self.candidate.model
        ):
            raise CandidateRequestError(
                f"candidate.request[{request.request_id}]",
                "was built for another candidate or endpoint",
                actual=request.request_hash,
                expected=f"a request built for candidate {self.candidate.alias}",
                recovery="build and execute the request with the same EvalCandidate contract",
            )

    def _backoff(self, request_hash: str, attempt_index: int) -> float:
        if self.backoff_base_s <= 0:
            return 0.0
        digest = hashlib.sha256(f"{request_hash}:{attempt_index}".encode()).digest()
        jitter = int.from_bytes(digest[:2], "big") / 65535
        return self.backoff_base_s * (2**attempt_index) * (0.75 + 0.5 * jitter)


def _classify_http(status_code: int) -> tuple[CallStatus, bool]:
    if status_code in {401, 403}:
        return "authentication_failed", False
    if status_code == 429:
        return "rate_limited", True
    if status_code in _RETRYABLE_HTTP:
        return "provider_unavailable", True
    return "provider_rejected", False


def _safe_request_id(headers: httpx.Headers) -> str | None:
    for name in ("x-request-id", "request-id", "x-nv-request-id"):
        value = headers.get(name)
        if value and len(value) <= 256:
            return value
    return None


def _retry_after(headers: httpx.Headers) -> float | None:
    value = headers.get("retry-after")
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            moment = parsedate_to_datetime(value)
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            seconds = (moment - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds)


def _observed_at() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "NativeFunctionCallingClient",
    "build_candidate_request",
    "parse_candidate_response",
]
