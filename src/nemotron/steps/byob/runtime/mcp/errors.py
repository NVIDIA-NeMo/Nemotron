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

"""Typed failures for MCP discovery and normalization."""

from __future__ import annotations


class McpIntegrationError(RuntimeError):
    """Base class for BFCL MCP integration failures."""


class McpConfigError(McpIntegrationError, ValueError):
    """The strict MCP oracle configuration is invalid."""


class McpCredentialError(McpIntegrationError, ValueError):
    """A named credential is absent or unsafe to place on the wire."""


class McpExecutablePolicyError(McpIntegrationError, ValueError):
    """A stdio executable does not satisfy its trusted policy."""


class McpTransportError(McpIntegrationError):
    """The MCP transport could not be opened or completed safely."""


class McpProtocolError(McpIntegrationError):
    """The server did not satisfy the negotiated MCP discovery contract."""


class McpCatalogError(McpIntegrationError):
    """The complete selected tool catalog could not be established."""


class McpNormalizationError(McpIntegrationError):
    """A selected MCP tool cannot be represented by BFCL."""


class McpIdentityMismatchError(McpIntegrationError):
    """Observed MCP identity differs from the operator-pinned identity."""
