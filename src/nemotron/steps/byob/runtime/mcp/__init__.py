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

"""BFCL MCP client, discovery, and normalization contracts."""

from nemotron.steps.byob.runtime.mcp.config import (
    LoadedMcpOracleConfig,
    McpOracleConfig,
    load_mcp_oracle_config,
    load_trusted_executable_policies,
)
from nemotron.steps.byob.runtime.mcp.discovery import (
    DiscoveryReport,
    discover_mcp_oracle,
    write_discovery_report,
)

__all__ = [
    "DiscoveryReport",
    "LoadedMcpOracleConfig",
    "McpOracleConfig",
    "discover_mcp_oracle",
    "load_mcp_oracle_config",
    "load_trusted_executable_policies",
    "write_discovery_report",
]
