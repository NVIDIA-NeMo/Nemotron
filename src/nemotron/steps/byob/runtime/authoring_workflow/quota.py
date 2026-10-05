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

"""Deterministic per-run budgets for assisted-authoring model calls."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictInt, model_validator

from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import (
    canonical_json,
)
from nemotron.steps.byob.runtime.pack_authoring.artifacts import sha256_json

QUOTA_POLICY_VERSION: Literal["bfcl-authoring-quota-v1"] = (
    "bfcl-authoring-quota-v1"
)
QUOTA_SNAPSHOT_VERSION: Literal["bfcl-authoring-quota-snapshot-v1"] = (
    "bfcl-authoring-quota-snapshot-v1"
)
DEFAULT_OUTPUT_TOKEN_RESERVE = 16_384

MonotonicClock = Callable[[], float]


class RunQuotaError(RuntimeError):
    """A stable quota refusal with a safe recovery instruction."""

    def __init__(self, code: str, detail: str, *, recovery: str) -> None:
        self.code = code
        self.detail = detail
        self.recovery = recovery
        super().__init__(f"{code}: {detail}; recovery: {recovery}")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunQuotaLimits(_StrictModel):
    schema_version: Literal["bfcl-authoring-quota-v1"] = QUOTA_POLICY_VERSION
    max_provider_calls: StrictInt
    max_token_units: StrictInt
    max_batch_size: StrictInt
    max_wall_time_ms: StrictInt

    @model_validator(mode="after")
    def _validate(self) -> RunQuotaLimits:
        if self.max_provider_calls < 0 or self.max_token_units < 0:
            raise ValueError("provider-call and token quotas cannot be negative")
        if self.max_batch_size <= 0 or self.max_wall_time_ms <= 0:
            raise ValueError("batch and wall-clock quotas must be positive")
        return self

    @property
    def policy_digest(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


class RunQuotaSnapshot(_StrictModel):
    schema_version: Literal["bfcl-authoring-quota-snapshot-v1"]
    policy_digest: str
    provider_calls: StrictInt
    token_units: StrictInt
    cache_hits: StrictInt
    largest_batch: StrictInt
    snapshot_digest: str

    @model_validator(mode="after")
    def _validate(self) -> RunQuotaSnapshot:
        if min(
            self.provider_calls,
            self.token_units,
            self.cache_hits,
            self.largest_batch,
        ) < 0:
            raise ValueError("quota usage cannot be negative")
        unsigned = self.model_dump(mode="json", exclude={"snapshot_digest"})
        if self.snapshot_digest != sha256_json(unsigned):
            raise ValueError("quota snapshot digest mismatch")
        return self


DEFAULT_AUTHORING_QUOTA = RunQuotaLimits(
    max_provider_calls=4,
    max_token_units=8_000_000,
    max_batch_size=1,
    max_wall_time_ms=15 * 60 * 1000,
)


def estimate_structured_token_units(
    *,
    model_canonical: str,
    system_prompt: str,
    prompt: str,
    model_input: Mapping[str, str],
    inference_parameters: Mapping[str, Any],
    output_schema: Mapping[str, Any],
) -> int:
    """Return a conservative, tokenizer-independent UTF-8-byte token bound."""
    configured_output = inference_parameters.get(
        "max_output_tokens",
        inference_parameters.get("max_tokens", DEFAULT_OUTPUT_TOKEN_RESERVE),
    )
    if (
        not isinstance(configured_output, int)
        or isinstance(configured_output, bool)
        or configured_output < 0
    ):
        raise RunQuotaError(
            "token_quota_configuration_invalid",
            "max_output_tokens/max_tokens must be a non-negative integer",
            recovery="pin a valid output-token bound in resolved authoring config",
        )
    request = {
        "model_canonical": model_canonical.strip().lower(),
        "system_prompt": system_prompt,
        "prompt": prompt,
        "input": dict(sorted(model_input.items())),
        "inference_parameters": dict(inference_parameters),
        "output_schema": output_schema,
    }
    input_units = len(canonical_json(request).encode("utf-8"))
    return input_units + configured_output


class RunQuota:
    """Reserve deterministic usage before every provider interaction."""

    def __init__(
        self,
        limits: RunQuotaLimits,
        *,
        clock: MonotonicClock = time.monotonic,
    ) -> None:
        self.limits = limits
        self._clock = clock
        self._started_at = clock()
        self._provider_calls = 0
        self._token_units = 0
        self._cache_hits = 0
        self._largest_batch = 0
        self._lock = threading.Lock()

    def _check_wall_clock(self) -> None:
        elapsed_ms = int((self._clock() - self._started_at) * 1000)
        if elapsed_ms > self.limits.max_wall_time_ms:
            raise RunQuotaError(
                "wall_clock_quota_exhausted",
                f"run elapsed {elapsed_ms}ms, limit is {self.limits.max_wall_time_ms}ms",
                recovery="start a new authorized run or increase the reviewed quota",
            )

    def record_cache_hit(self, *, batch_size: int) -> None:
        with self._lock:
            self._check_batch(batch_size)
            self._check_wall_clock()
            self._cache_hits += 1
            self._largest_batch = max(self._largest_batch, batch_size)

    def reserve_provider_call(self, *, token_units: int, batch_size: int) -> None:
        if token_units < 0:
            raise RunQuotaError(
                "token_quota_configuration_invalid",
                "reserved token units cannot be negative",
                recovery="fix the deterministic token estimator",
            )
        with self._lock:
            self._check_batch(batch_size)
            self._check_wall_clock()
            next_calls = self._provider_calls + 1
            if next_calls > self.limits.max_provider_calls:
                raise RunQuotaError(
                    "provider_call_quota_exhausted",
                    f"provider call {next_calls} exceeds limit "
                    f"{self.limits.max_provider_calls}",
                    recovery="replay from cache or start a new authorized run",
                )
            next_tokens = self._token_units + token_units
            if next_tokens > self.limits.max_token_units:
                raise RunQuotaError(
                    "token_quota_exhausted",
                    f"token units {next_tokens} exceed limit "
                    f"{self.limits.max_token_units}",
                    recovery="reduce the bounded input or increase the reviewed quota",
                )
            self._provider_calls = next_calls
            self._token_units = next_tokens
            self._largest_batch = max(self._largest_batch, batch_size)

    def check_after_provider_call(self) -> None:
        with self._lock:
            self._check_wall_clock()

    def _check_batch(self, batch_size: int) -> None:
        if batch_size <= 0 or batch_size > self.limits.max_batch_size:
            raise RunQuotaError(
                "batch_quota_exceeded",
                f"batch size {batch_size} exceeds limit {self.limits.max_batch_size}",
                recovery="split the request into policy-compliant batches",
            )

    def snapshot(self) -> RunQuotaSnapshot:
        with self._lock:
            unsigned = {
                "schema_version": QUOTA_SNAPSHOT_VERSION,
                "policy_digest": self.limits.policy_digest,
                "provider_calls": self._provider_calls,
                "token_units": self._token_units,
                "cache_hits": self._cache_hits,
                "largest_batch": self._largest_batch,
            }
            return RunQuotaSnapshot.model_validate(
                {**unsigned, "snapshot_digest": sha256_json(unsigned)}
            )
