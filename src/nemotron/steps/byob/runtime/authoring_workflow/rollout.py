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

"""Fail-closed per-adapter rollout policy for live authoring operations."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

AdapterKind = Literal["local_python", "http_package", "mcp_mode_a"]
RolloutOrigin = Literal["environment", "policy", "default"]

ADAPTER_ROLLOUT_ENV: Mapping[AdapterKind, str] = {
    "local_python": "BFCL_ENABLE_LOCAL_PYTHON",
    "http_package": "BFCL_ENABLE_HTTP_PACKAGE",
    "mcp_mode_a": "BFCL_ENABLE_MCP_MODE_A",
}
LEGACY_MCP_ROLLOUT_ENV = "BFCL_ENABLE_EXPERIMENTAL_MCP"
_KINDS = frozenset(ADAPTER_ROLLOUT_ENV)
_TRUE = frozenset({"1", "true", "yes"})
_FALSE = frozenset({"0", "false", "no"})


class RolloutPolicyError(ValueError):
    def __init__(self, code: str, detail: str, *, recovery: str) -> None:
        self.code = code
        self.detail = detail
        self.recovery = recovery
        super().__init__(f"{code}: {detail}; recovery: {recovery}")


@dataclass(frozen=True)
class AdapterRolloutDecision:
    adapter_kind: AdapterKind
    enabled: bool
    origin: RolloutOrigin
    environment_variable: str | None
    legacy_alias_used: bool


def _kind(value: str) -> AdapterKind:
    if value not in _KINDS:
        raise RolloutPolicyError(
            "rollout_adapter_unknown",
            f"unknown authoring adapter kind {value!r}",
            recovery=f"use one of {sorted(_KINDS)}",
        )
    return cast(AdapterKind, value)


def _boolean(raw: str, variable: str) -> bool:
    normalized = raw.strip().casefold()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    raise RolloutPolicyError(
        "rollout_value_malformed",
        f"{variable} must be one of {sorted(_TRUE | _FALSE)}, got {raw!r}",
        recovery=f"set {variable} to 1 or 0 explicitly",
    )


def _policy_values(
    policy: Mapping[str, bool] | None,
) -> dict[AdapterKind, bool]:
    if policy is None:
        return {}
    values: dict[AdapterKind, bool] = {}
    for raw_kind, enabled in policy.items():
        kind = _kind(raw_kind)
        if type(enabled) is not bool:
            raise RolloutPolicyError(
                "rollout_policy_malformed",
                f"policy value for {kind!r} must be a boolean",
                recovery="use true or false in the reviewed authoring policy",
            )
        values[kind] = enabled
    return values


def resolve_adapter_rollout(
    adapter_kind: str,
    *,
    environ: Mapping[str, str] | None = None,
    policy: Mapping[str, bool] | None = None,
) -> AdapterRolloutDecision:
    kind = _kind(adapter_kind)
    source = os.environ if environ is None else environ
    policy_values = _policy_values(policy)
    variable = ADAPTER_ROLLOUT_ENV[kind]
    current_raw = source.get(variable)
    legacy_raw = (
        source.get(LEGACY_MCP_ROLLOUT_ENV) if kind == "mcp_mode_a" else None
    )
    current = _boolean(current_raw, variable) if current_raw is not None else None
    legacy = (
        _boolean(legacy_raw, LEGACY_MCP_ROLLOUT_ENV)
        if legacy_raw is not None
        else None
    )
    if current is not None and legacy is not None and current != legacy:
        raise RolloutPolicyError(
            "rollout_settings_conflict",
            f"{variable} and {LEGACY_MCP_ROLLOUT_ENV} disagree",
            recovery="remove the legacy alias or set both flags to the same value",
        )
    if current is not None:
        return AdapterRolloutDecision(
            adapter_kind=kind,
            enabled=current,
            origin="environment",
            environment_variable=variable,
            legacy_alias_used=legacy is not None,
        )
    if legacy is not None:
        return AdapterRolloutDecision(
            adapter_kind=kind,
            enabled=legacy,
            origin="environment",
            environment_variable=LEGACY_MCP_ROLLOUT_ENV,
            legacy_alias_used=True,
        )
    if kind in policy_values:
        return AdapterRolloutDecision(
            adapter_kind=kind,
            enabled=policy_values[kind],
            origin="policy",
            environment_variable=None,
            legacy_alias_used=False,
        )
    return AdapterRolloutDecision(
        adapter_kind=kind,
        enabled=False,
        origin="default",
        environment_variable=None,
        legacy_alias_used=False,
    )


def adapter_rollout_enabled(
    adapter_kind: str,
    *,
    environ: Mapping[str, str] | None = None,
    policy: Mapping[str, bool] | None = None,
) -> bool:
    return resolve_adapter_rollout(
        adapter_kind,
        environ=environ,
        policy=policy,
    ).enabled


def require_adapter_rollout(
    adapter_kind: str,
    *,
    environ: Mapping[str, str] | None = None,
    policy: Mapping[str, bool] | None = None,
) -> AdapterRolloutDecision:
    decision = resolve_adapter_rollout(
        adapter_kind,
        environ=environ,
        policy=policy,
    )
    if not decision.enabled:
        variable = ADAPTER_ROLLOUT_ENV[decision.adapter_kind]
        raise RolloutPolicyError(
            "adapter_rollout_disabled",
            f"live {decision.adapter_kind} authoring is not enabled",
            recovery=f"set {variable}=1 or enable it in reviewed authoring policy",
        )
    return decision


def require_no_rollout_revocation(
    adapter_kind: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Honor an explicit current environment disable after config resolution.

    An omitted environment value does not override a reviewed policy decision already
    sealed in resolved configuration. Any present value is parsed strictly, including
    legacy/new MCP conflict detection.
    """
    kind = _kind(adapter_kind)
    source = os.environ if environ is None else environ
    relevant = [ADAPTER_ROLLOUT_ENV[kind]]
    if kind == "mcp_mode_a":
        relevant.append(LEGACY_MCP_ROLLOUT_ENV)
    if not any(variable in source for variable in relevant):
        return
    require_adapter_rollout(kind, environ=source)
