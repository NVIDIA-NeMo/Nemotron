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

"""HTTP-safe failures emitted by the MCP-to-BFCL gateway."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GatewayError(RuntimeError):
    """One infrastructure failure with a stable BFCL gateway error code."""

    code: str
    message: str
    http_status: int = 500
    poison_session: bool = False

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.message)

    def as_dict(self) -> dict[str, dict[str, str]]:
        return {"error": {"code": self.code, "message": self.message}}


def bad_request(code: str, message: str) -> GatewayError:
    return GatewayError(code=code, message=message, http_status=400)


def unavailable(code: str, message: str) -> GatewayError:
    return GatewayError(code=code, message=message, http_status=503)


def upstream_failure(
    code: str,
    message: str,
    *,
    timeout: bool = False,
) -> GatewayError:
    return GatewayError(
        code=code,
        message=message,
        http_status=504 if timeout else 502,
        poison_session=True,
    )
