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

"""Deriving BFCL pack files from an approved evidence bundle.

This package deliberately imports nothing from `runtime/mcp/`. The drafting phase runs in
the `byob` environment, where Data Designer pins MCP SDK v1, while discovery and the
gateway run in `bfcl-mcp` with SDK v2; the two extras are declared mutually exclusive. The
only thing that crosses between them is a file. Keeping that separation structural rather
than incidental is what stops a later module-level import on the MCP side from breaking
authoring in an environment where the SDK it wants cannot be installed.
"""
