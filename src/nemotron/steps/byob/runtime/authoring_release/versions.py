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

"""Release-envelope versions and compatibility constants."""

from typing import Literal

REVIEW_PACKET_VERSION_V2: Literal["bfcl-authoring-review-packet-v2"] = "bfcl-authoring-review-packet-v2"
REVIEW_APPROVAL_VERSION_V2: Literal["bfcl-authoring-review-approval-v2"] = "bfcl-authoring-review-approval-v2"
FREEZE_MANIFEST_VERSION_V2: Literal["bfcl-authoring-frozen-release-v2"] = "bfcl-authoring-frozen-release-v2"
FREEZE_MANIFEST_VERSION_V3: Literal["bfcl-authoring-frozen-release-v3"] = "bfcl-authoring-frozen-release-v3"

MCP_REVIEW_PACKET_VERSION_V1 = "bfcl-mcp-review-packet-v1"
MCP_REVIEW_APPROVAL_VERSION_V1 = "bfcl-mcp-review-approval-v1"
MCP_FREEZE_MANIFEST_VERSION_V1 = "bfcl-mcp-frozen-release-v1"
