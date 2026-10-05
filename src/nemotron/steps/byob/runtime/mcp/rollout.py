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

"""Explicit rollout gate for operations that contact an MCP server."""

from __future__ import annotations

from collections.abc import Mapping

from nemotron.steps.byob.runtime.authoring_workflow.rollout import (
    LEGACY_MCP_ROLLOUT_ENV,
    RolloutPolicyError,
    adapter_rollout_enabled,
    require_adapter_rollout,
)
from nemotron.steps.byob.runtime.mcp.errors import McpConfigError

MCP_FEATURE_ENV = LEGACY_MCP_ROLLOUT_ENV


def mcp_feature_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Compatibility alias for the generic MCP Mode A rollout decision."""
    try:
        return adapter_rollout_enabled("mcp_mode_a", environ=environ)
    except RolloutPolicyError as exc:
        raise McpConfigError(str(exc)) from exc


def require_mcp_feature(environ: Mapping[str, str] | None = None) -> None:
    """Compatibility gate retained for one deprecation window."""
    try:
        require_adapter_rollout("mcp_mode_a", environ=environ)
    except RolloutPolicyError as exc:
        raise McpConfigError(
            f"{exc}; compatibility alias: set {MCP_FEATURE_ENV}=1"
        ) from exc
