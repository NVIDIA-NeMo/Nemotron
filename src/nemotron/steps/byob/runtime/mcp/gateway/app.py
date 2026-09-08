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

"""Starlette adapter exposing the BFCL Oracle HTTP v1 routes."""

from __future__ import annotations

import hmac
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from nemotron.steps.byob.runtime.mcp.gateway.conformance import attestation_bytes
from nemotron.steps.byob.runtime.mcp.gateway.errors import GatewayError
from nemotron.steps.byob.runtime.mcp.gateway.service import GatewayService

_MAX_ERROR_MESSAGE_CHARS = 1024


class _BearerAuthMiddleware:
    def __init__(self, app: ASGIApp, *, token: str | None) -> None:
        self.app = app
        # Compared as bytes: a header carrying non-ASCII would make the str form of
        # compare_digest raise, turning a rejected request into a 500.
        self.expected = None if token is None else f"Bearer {token}".encode()

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] == "http" and self.expected is not None:
            observed = Headers(scope=scope).get("authorization", "").encode("latin-1")
            if not hmac.compare_digest(observed, self.expected):
                response = JSONResponse(
                    {
                        "error": {
                            "code": "mcp_gateway_unauthorized",
                            "message": "gateway authentication failed",
                        }
                    },
                    status_code=401,
                    headers={"Cache-Control": "no-store"},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


async def _json_body(
    request: Request,
    *,
    max_request_bytes: int,
) -> dict[str, Any]:
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            declared_size = int(declared)
        except ValueError as exc:
            raise GatewayError(
                "mcp_request_invalid",
                "Content-Length must be an integer",
                http_status=400,
            ) from exc
        if declared_size > max_request_bytes:
            raise GatewayError(
                "mcp_request_too_large",
                "request exceeds gateway max_request_bytes",
                http_status=413,
            )
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > max_request_bytes:
            raise GatewayError(
                "mcp_request_too_large",
                "request exceeds gateway max_request_bytes",
                http_status=413,
            )
        body.extend(chunk)
    try:
        decoded = body.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise GatewayError(
            "mcp_request_invalid",
            "request body must be strict JSON without duplicate keys",
            http_status=400,
        ) from exc
    if not isinstance(value, dict):
        raise GatewayError(
            "mcp_request_invalid",
            "request body must be a JSON object",
            http_status=400,
        )
    return value


def _exact_fields(
    value: dict[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise GatewayError(
            "mcp_request_invalid",
            f"{label} fields mismatch: missing={missing}, unknown={unknown}",
            http_status=400,
        )


def create_gateway_app(
    service: GatewayService,
    *,
    max_request_bytes: int = 10 * 1024 * 1024,
    client_bearer_token: str | None = None,
) -> Starlette:
    """Build one HTTP adapter around an injected, transport-neutral service."""

    async def metadata(request: Request) -> JSONResponse:
        return JSONResponse(service.metadata())

    async def tools(request: Request) -> JSONResponse:
        return JSONResponse({"tools": service.list_tools()})

    async def conformance(request: Request) -> Response:
        # Served as canonical bytes rather than re-encoded JSON: the pack pins a digest over
        # this exact document, and a JSON encoder that reorders keys would break it.
        return Response(
            content=attestation_bytes(service.conformance()),
            media_type="application/json",
        )

    async def probe_report(request: Request) -> Response:
        return Response(
            content=attestation_bytes(service.probe_report()),
            media_type="application/json",
        )

    async def gateway_conformance_report(request: Request) -> Response:
        return Response(
            content=attestation_bytes(service.gateway_conformance_report()),
            media_type="application/json",
        )

    async def create_session(request: Request) -> JSONResponse:
        payload = await _json_body(request, max_request_bytes=max_request_bytes)
        _exact_fields(
            payload,
            frozenset({"context", "fixtures"}),
            label="session request",
        )
        result = await service.create_session(
            context=payload["context"],
            fixtures=payload["fixtures"],
        )
        return JSONResponse(result, status_code=201)

    async def call_tool(request: Request) -> JSONResponse:
        payload = await _json_body(request, max_request_bytes=max_request_bytes)
        _exact_fields(
            payload,
            frozenset({"name", "arguments", "turn_index"}),
            label="tool call request",
        )
        result = await service.call_tool(
            request.path_params["session_id"],
            name=payload["name"],
            arguments=payload["arguments"],
            turn_index=payload["turn_index"],
        )
        return JSONResponse(result)

    async def get_state(request: Request) -> JSONResponse:
        result = await service.get_state(request.path_params["session_id"])
        return JSONResponse(result)

    async def delete_session(request: Request) -> Response:
        await service.delete_session(request.path_params["session_id"])
        return Response(status_code=204)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        await service.start()
        try:
            yield
        finally:
            await service.shutdown()

    async def gateway_error_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        assert isinstance(exc, GatewayError)
        message = exc.message[:_MAX_ERROR_MESSAGE_CHARS]
        return JSONResponse(
            {"error": {"code": exc.code, "message": message}},
            status_code=exc.http_status,
            headers={"Cache-Control": "no-store"},
        )

    async def internal_error_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        return JSONResponse(
            {
                "error": {
                    "code": "mcp_gateway_internal",
                    "message": "gateway encountered an internal failure",
                }
            },
            status_code=500,
            headers={"Cache-Control": "no-store"},
        )

    app = Starlette(
        routes=[
            Route("/v1/metadata", metadata, methods=["GET"]),
            Route("/v1/conformance", conformance, methods=["GET"]),
            Route(
                "/v1/conformance/probe-report",
                probe_report,
                methods=["GET"],
            ),
            Route(
                "/v1/conformance/gateway-report",
                gateway_conformance_report,
                methods=["GET"],
            ),
            Route("/v1/tools", tools, methods=["GET"]),
            Route("/v1/sessions", create_session, methods=["POST"]),
            Route(
                "/v1/sessions/{session_id:str}/calls",
                call_tool,
                methods=["POST"],
            ),
            Route(
                "/v1/sessions/{session_id:str}/state",
                get_state,
                methods=["GET"],
            ),
            Route(
                "/v1/sessions/{session_id:str}",
                delete_session,
                methods=["DELETE"],
            ),
        ],
        exception_handlers={
            GatewayError: gateway_error_handler,
            Exception: internal_error_handler,
        },
        lifespan=lifespan,
        middleware=[Middleware(_BearerAuthMiddleware, token=client_bearer_token)],
    )
    return app
