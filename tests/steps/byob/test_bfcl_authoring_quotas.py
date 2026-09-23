from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from nemotron.steps.byob.runtime.authoring_workflow.quota import (
    RunQuota,
    RunQuotaError,
    RunQuotaLimits,
    estimate_structured_token_units,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.model_io_cache import (
    ImmutableModelIOCache,
)
from nemotron.steps.byob.runtime.pack_authoring.model_client import (
    AuthoringModel,
    ModelCallRecord,
    call_structured,
)


class _Output(BaseModel):
    value: str


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def _limits(
    *,
    calls: int = 1,
    tokens: int = 100_000,
    batch: int = 1,
    wall_ms: int = 1_000,
) -> RunQuotaLimits:
    return RunQuotaLimits(
        max_provider_calls=calls,
        max_token_units=tokens,
        max_batch_size=batch,
        max_wall_time_ms=wall_ms,
    )


def _call(
    tmp_path: Path,
    *,
    cache: ImmutableModelIOCache,
    quota: RunQuota,
    caller: Any,
) -> tuple[dict[str, Any], ModelCallRecord]:
    return call_structured(
        AuthoringModel(
            alias="author",
            provider="test",
            model="recorded",
            canonical_id="test/recorded",
            inference_parameters={"max_output_tokens": 10},
        ),
        stage_name="quota_test",
        prompt_version="v1",
        system_prompt="Return structured output.",
        prompt="Draft one bounded value.",
        columns={"evidence": "{}"},
        output_format=_Output,
        cache=cache,
        run_dir=tmp_path / "runs",
        caller=caller,
        quota=quota,
    )


def test_quota_accounting_is_deterministic() -> None:
    first = RunQuota(_limits(calls=2, batch=2))
    first.reserve_provider_call(token_units=25, batch_size=2)
    first.record_cache_hit(batch_size=1)

    second = RunQuota(_limits(calls=2, batch=2))
    second.reserve_provider_call(token_units=25, batch_size=2)
    second.record_cache_hit(batch_size=1)

    assert first.snapshot() == second.snapshot()
    assert first.snapshot().provider_calls == 1
    assert first.snapshot().token_units == 25
    assert first.snapshot().cache_hits == 1
    assert first.snapshot().largest_batch == 2


def test_cache_hit_consumes_no_provider_call_or_token_quota(tmp_path: Path) -> None:
    cache = ImmutableModelIOCache(tmp_path / "cache.jsonl")
    provider_calls = 0

    def caller(*_args: Any, **_kwargs: Any) -> dict[str, dict[str, str]]:
        nonlocal provider_calls
        provider_calls += 1
        return {"quota_test": {"value": "cached"}}

    response, first_record = _call(
        tmp_path,
        cache=cache,
        quota=RunQuota(_limits()),
        caller=caller,
    )
    replay_quota = RunQuota(_limits(calls=0, tokens=0))
    replay, replay_record = _call(
        tmp_path,
        cache=cache,
        quota=replay_quota,
        caller=lambda *_args, **_kwargs: pytest.fail("cache replay called provider"),
    )

    assert response == replay == {"value": "cached"}
    assert first_record.served_from_cache is False
    assert replay_record.served_from_cache is True
    assert provider_calls == 1
    assert replay_quota.snapshot().provider_calls == 0
    assert replay_quota.snapshot().token_units == 0
    assert replay_quota.snapshot().cache_hits == 1


def test_cache_miss_refuses_before_provider_when_call_quota_is_zero(
    tmp_path: Path,
) -> None:
    called = False

    def caller(*_args: Any, **_kwargs: Any) -> dict[str, dict[str, str]]:
        nonlocal called
        called = True
        return {"quota_test": {"value": "forbidden"}}

    with pytest.raises(RunQuotaError) as refused:
        _call(
            tmp_path,
            cache=ImmutableModelIOCache(tmp_path / "cache.jsonl"),
            quota=RunQuota(_limits(calls=0)),
            caller=caller,
        )
    assert refused.value.code == "provider_call_quota_exhausted"
    assert called is False


def test_token_batch_and_wall_clock_limits_fail_closed() -> None:
    token_quota = RunQuota(_limits(tokens=9))
    with pytest.raises(RunQuotaError) as tokens:
        token_quota.reserve_provider_call(token_units=10, batch_size=1)
    assert tokens.value.code == "token_quota_exhausted"

    batch_quota = RunQuota(_limits(batch=1))
    with pytest.raises(RunQuotaError) as batch:
        batch_quota.reserve_provider_call(token_units=1, batch_size=2)
    assert batch.value.code == "batch_quota_exceeded"

    clock = _Clock()
    wall_quota = RunQuota(_limits(wall_ms=10), clock=clock)
    clock.value = 0.011
    with pytest.raises(RunQuotaError) as wall:
        wall_quota.reserve_provider_call(token_units=1, batch_size=1)
    assert wall.value.code == "wall_clock_quota_exhausted"


def test_failed_provider_call_still_consumes_reserved_quota(tmp_path: Path) -> None:
    quota = RunQuota(_limits(calls=1))

    def unavailable(*_args: Any, **_kwargs: Any) -> dict[str, dict[str, str]]:
        raise RuntimeError("provider unavailable")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        _call(
            tmp_path,
            cache=ImmutableModelIOCache(tmp_path / "cache.jsonl"),
            quota=quota,
            caller=unavailable,
        )
    assert quota.snapshot().provider_calls == 1
    with pytest.raises(RunQuotaError) as retried:
        _call(
            tmp_path,
            cache=ImmutableModelIOCache(tmp_path / "cache.jsonl"),
            quota=quota,
            caller=unavailable,
        )
    assert retried.value.code == "provider_call_quota_exhausted"


def test_token_estimate_is_canonical_and_reserves_output_bound() -> None:
    first = estimate_structured_token_units(
        model_canonical="TEST/MODEL",
        system_prompt="system",
        prompt="prompt",
        model_input={"b": "2", "a": "1"},
        inference_parameters={"max_output_tokens": 7},
        output_schema={"type": "object"},
    )
    second = estimate_structured_token_units(
        model_canonical="test/model",
        system_prompt="system",
        prompt="prompt",
        model_input={"a": "1", "b": "2"},
        inference_parameters={"max_output_tokens": 7},
        output_schema={"type": "object"},
    )
    assert first == second
    assert first > 7
