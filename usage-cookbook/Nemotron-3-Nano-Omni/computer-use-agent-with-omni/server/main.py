"""FastAPI entrypoint for nano_omni_demo.

Endpoints
---------
GET  /                     -> redirects to web UI
GET  /health               -> health check + desktop status
GET  /env/screenshot       -> live PNG from desktop
POST /env/restart          -> restart the desktop container
POST /agent/start          -> {instruction} -> {job_id}
POST /agent/{job_id}/stop  -> stop the running job
GET  /agent/{job_id}/status -> job status
GET  /agent/{job_id}/events -> SSE stream of agent events
/vnc/*                     -> reverse proxy to KasmVNC (no cert/auth issues)
/vnc/websockify            -> WebSocket proxy for noVNC
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import socket
import ssl
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

import httpx
import websockets
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

from server.agent_runner import AgentJob, AgentRunner
from server.desktop_client import DesktopClient
from server.holotron_agent import HolotronAgent
from server.vllm_inference import VllmInferenceAgent

# ── Config (from .env) ──────────────────────────────────────────────────────

INFERENCE_PROVIDER = os.getenv("INFERENCE_PROVIDER", "vllm").strip().lower()
MODEL_FAMILY = os.getenv("MODEL_FAMILY", "nemotron").strip().lower()
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://host.docker.internal:8001/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
VLLM_MODEL = os.getenv("VLLM_MODEL", os.getenv("VLLM_SERVED_MODEL_NAME", "vllm_local"))
ENABLE_THINKING = os.getenv("ENABLE_THINKING", "true").lower() == "true"
TRUNCATE_HISTORY_THINKING = (
    os.getenv("TRUNCATE_HISTORY_THINKING", "false").lower() == "true"
)
MAX_STEPS = int(os.getenv("MAX_STEPS", "150"))
MAX_IMAGE_HISTORY = int(os.getenv("MAX_IMAGE_HISTORY", "3"))
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "20480"))
REASONING_BUDGET = int(os.getenv("REASONING_BUDGET", "16384"))
REASONING_GRACE_TOKENS = int(os.getenv("REASONING_GRACE_TOKENS", "1024"))
MODEL_ATTEMPT_TIMEOUT = float(os.getenv("MODEL_ATTEMPT_TIMEOUT", "120"))
MODEL_MAX_RETRIES = int(os.getenv("MODEL_MAX_RETRIES", "3"))
MODEL_RETRY_SLEEP = float(os.getenv("MODEL_RETRY_SLEEP", "5"))
COMPUTER_WAIT_SECONDS = max(0.0, float(os.getenv("COMPUTER_WAIT_SECONDS", "3")))
DESKTOP_HOST = os.getenv("DESKTOP_HOST", "localhost")
DESKTOP_API_PORT = int(os.getenv("DESKTOP_API_PORT", "5000"))
DESKTOP_VNC_PORT = int(os.getenv("DESKTOP_VNC_PORT", "6901"))
DESKTOP_PASSWORD = os.getenv("DESKTOP_PASSWORD", "password")
DOCKER_SOCKET = os.getenv("DOCKER_SOCKET", "/var/run/docker.sock")
DESKTOP_CONTAINER_NAME = os.getenv("DESKTOP_CONTAINER_NAME", "")
DESKTOP_CONTAINER_SERVICE = os.getenv("DESKTOP_CONTAINER_SERVICE", "desktop")
DOCKER_RESTART_TIMEOUT = int(os.getenv("DOCKER_RESTART_TIMEOUT", "10"))

# KasmVNC credentials (match the desktop container defaults)
KASM_VNC_USER = os.getenv("VNC_USER", "kasm_user")
KASM_VNC_PW = os.getenv("VNC_PW", "password")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _active_model() -> str:
    return VLLM_MODEL


def _active_api_base() -> str:
    return VLLM_API_BASE


def _active_api_key_configured() -> bool:
    return True

# ── State ───────────────────────────────────────────────────────────────────


class _State:
    desktop: DesktopClient
    jobs: dict[str, AgentJob] = {}


state = _State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if INFERENCE_PROVIDER != "vllm":
        logger.error(
            f"Unsupported INFERENCE_PROVIDER={INFERENCE_PROVIDER!r}; this demo currently supports only 'vllm'."
        )
    if MODEL_FAMILY not in {"nemotron", "holotron"}:
        logger.error(
            f"Unknown MODEL_FAMILY={MODEL_FAMILY!r}; expected 'nemotron' or 'holotron'."
        )
    state.desktop = DesktopClient(host=DESKTOP_HOST, api_port=DESKTOP_API_PORT)
    state.jobs = {}
    logger.info(
        "nano_omni_demo started (family={}, provider={}, model={}, base={}, thinking={})",
        MODEL_FAMILY,
        INFERENCE_PROVIDER,
        _active_model(),
        _active_api_base(),
        ENABLE_THINKING,
    )
    logger.info(f"Desktop: {DESKTOP_HOST}:{DESKTOP_API_PORT} (VNC: {DESKTOP_VNC_PORT})")
    yield


app = FastAPI(title="nano_omni_demo", lifespan=lifespan)


@app.middleware("http")
async def add_cross_origin_isolation_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    response.headers.setdefault("Cross-Origin-Embedder-Policy", "require-corp")
    response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
    return response


# Mount static web frontend
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


# ── DTOs ────────────────────────────────────────────────────────────────────


class StartAgentRequest(BaseModel):
    instruction: str = Field(..., min_length=1)
    max_steps: Optional[int] = None


# ── Helpers ─────────────────────────────────────────────────────────────────


async def _stop_job(job: AgentJob) -> None:
    job._stop.set()
    if job.status in {"pending", "running"}:
        job.status = "stopped"
        await job.events.put({"kind": "stopping", "ts": time.time()})
    if job._task is not None and not job._task.done():
        job._task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(job._task), timeout=2)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass


async def _stop_active_jobs() -> None:
    await asyncio.gather(
        *[
            _stop_job(job)
            for job in list(state.jobs.values())
            if job.status in {"pending", "running"}
        ],
        return_exceptions=True,
    )


def _docker_client() -> httpx.AsyncClient:
    if not os.path.exists(DOCKER_SOCKET):
        raise HTTPException(
            503,
            f"Docker socket not available at {DOCKER_SOCKET}; cannot restart desktop container.",
        )
    transport = httpx.AsyncHTTPTransport(uds=DOCKER_SOCKET)
    return httpx.AsyncClient(transport=transport, base_url="http://docker", timeout=30.0)


def _container_name(container: dict, container_id: str) -> str:
    names = container.get("Names")
    if isinstance(names, list) and names:
        return str(names[0]).lstrip("/")
    if isinstance(names, str) and names:
        return names.lstrip("/")
    name = container.get("Name")
    if isinstance(name, str) and name:
        return name.lstrip("/")
    return container_id[:12]


async def _compose_project_label(client: httpx.AsyncClient) -> str | None:
    try:
        r = await client.get(f"/containers/{socket.gethostname()}/json")
        r.raise_for_status()
        return r.json().get("Config", {}).get("Labels", {}).get("com.docker.compose.project")
    except Exception as exc:
        logger.warning(f"Could not inspect server container for compose project label: {exc}")
        return None


async def _find_desktop_container(client: httpx.AsyncClient) -> dict:
    if DESKTOP_CONTAINER_NAME:
        r = await client.get(f"/containers/{DESKTOP_CONTAINER_NAME}/json")
        if r.status_code == 404:
            raise HTTPException(404, f"Desktop container {DESKTOP_CONTAINER_NAME!r} not found")
        r.raise_for_status()
        data = r.json()
        return {
            "Id": data.get("Id"),
            "Names": data.get("Name", "").lstrip("/"),
        }

    labels = [f"com.docker.compose.service={DESKTOP_CONTAINER_SERVICE}"]
    project = await _compose_project_label(client)
    if project:
        labels.append(f"com.docker.compose.project={project}")

    filters = json.dumps({"label": labels})
    r = await client.get("/containers/json", params={"all": "true", "filters": filters})
    r.raise_for_status()
    containers = r.json()
    if not containers:
        raise HTTPException(
            404,
            f"No Docker container found for compose service {DESKTOP_CONTAINER_SERVICE!r}",
        )
    return containers[0]


# ── Routes ──────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/web/index.html")


@app.get("/health")
async def health() -> dict:
    desktop_ok = await state.desktop.is_ready()
    return {
        "status": "ok",
        "desktop": "ready" if desktop_ok else "not_ready",
        "inference_provider": INFERENCE_PROVIDER,
        "model_family": MODEL_FAMILY,
        "model": _active_model(),
        "api_base": _active_api_base(),
        "api_key_configured": _active_api_key_configured(),
        "enable_thinking": ENABLE_THINKING,
        "truncate_history_thinking": TRUNCATE_HISTORY_THINKING,
        "model_max_tokens": MODEL_MAX_TOKENS,
        "reasoning_budget": REASONING_BUDGET,
        "reasoning_grace_tokens": REASONING_GRACE_TOKENS,
        "model_attempt_timeout": MODEL_ATTEMPT_TIMEOUT,
        "model_max_retries": MODEL_MAX_RETRIES,
        "computer_wait_seconds": COMPUTER_WAIT_SECONDS,
        "vnc_password": KASM_VNC_PW,
    }


@app.get("/env/screenshot")
async def env_screenshot() -> Response:
    try:
        png = await state.desktop.screenshot()
        return Response(content=png, media_type="image/png")
    except Exception as e:
        raise HTTPException(503, f"Desktop not ready: {e}")


@app.post("/env/restart")
async def env_restart() -> dict:
    await _stop_active_jobs()

    async with _docker_client() as client:
        container = await _find_desktop_container(client)
        container_id = container.get("Id")
        if not container_id:
            raise HTTPException(500, "Docker did not return a desktop container id")

        logger.info(f"Restarting desktop container {container_id[:12]}")
        r = await client.post(
            f"/containers/{container_id}/restart",
            params={"t": str(DOCKER_RESTART_TIMEOUT)},
        )
        if r.status_code not in {204, 304}:
            raise HTTPException(r.status_code, f"Docker restart failed: {r.text}")

    try:
        await state.desktop.wait_ready(max_seconds=120, interval=2)
    except Exception as exc:
        raise HTTPException(503, f"Desktop restarted but did not become ready: {exc}")

    return {
        "status": "ready",
        "desktop": "ready",
        "container": _container_name(container, container_id),
        "model": _active_model(),
    }


@app.post("/agent/start")
async def agent_start(req: StartAgentRequest) -> dict:
    if INFERENCE_PROVIDER != "vllm":
        raise HTTPException(
            500,
            f"INFERENCE_PROVIDER={INFERENCE_PROVIDER!r} is not supported for this demo yet. Use 'vllm'.",
        )
    if MODEL_FAMILY not in {"nemotron", "holotron"}:
        raise HTTPException(
            500,
            f"Unknown MODEL_FAMILY={MODEL_FAMILY!r}. Use 'nemotron' or 'holotron'.",
        )
    if not await state.desktop.is_ready():
        raise HTTPException(503, "Desktop container not ready. Is it running? (docker compose up)")

    await _stop_active_jobs()

    if MODEL_FAMILY == "holotron":
        agent_cls = HolotronAgent
    else:
        agent_cls = VllmInferenceAgent
    agent = agent_cls(
        api_key=VLLM_API_KEY,
        api_base=VLLM_API_BASE,
        model=VLLM_MODEL,
        max_tokens=MODEL_MAX_TOKENS,
        max_image_history_length=MAX_IMAGE_HISTORY,
        thinking=ENABLE_THINKING,
        truncate_history_thinking=TRUNCATE_HISTORY_THINKING,
        reasoning_budget=REASONING_BUDGET,
        reasoning_grace_tokens=REASONING_GRACE_TOKENS,
        model_attempt_timeout=MODEL_ATTEMPT_TIMEOUT,
        max_retry=MODEL_MAX_RETRIES,
        retry_sleep=MODEL_RETRY_SLEEP,
        password=DESKTOP_PASSWORD,
    )
    runner = AgentRunner(
        agent,
        state.desktop,
        max_steps=req.max_steps or MAX_STEPS,
        computer_wait_seconds=COMPUTER_WAIT_SECONDS,
    )
    job = runner.start(req.instruction)
    state.jobs[job.job_id] = job
    return {"job_id": job.job_id, "status": job.status}


@app.post("/agent/{job_id}/stop")
async def agent_stop(job_id: str) -> dict:
    job = state.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    await _stop_job(job)
    return {"job_id": job_id, "status": job.status}


@app.get("/agent/{job_id}/status")
async def agent_status(job_id: str) -> dict:
    job = state.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "error": job.error,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }


@app.get("/agent/{job_id}/events")
async def agent_events(job_id: str, request: Request) -> StreamingResponse:
    job = state.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")

    async def gen():
        while True:
            if await request.is_disconnected():
                return
            try:
                ev = await asyncio.wait_for(job.events.get(), timeout=15)
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"
                if job.status not in {"running", "pending"}:
                    return
                continue
            yield f"data: {json.dumps(ev)}\n\n"
            if ev.get("kind") == "finished":
                return

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── KasmVNC Reverse Proxy ───────────────────────────────────────────────────
#
# Proxy KasmVNC through the FastAPI server so users only need to expose
# port 8000. No self-signed cert warnings, no Basic Auth prompts.
#
#   HTTP  /vnc/<path>       -> https://<desktop>:<vnc_port>/<path>
#   WS    /vnc/websockify   -> wss://<desktop>:<vnc_port>/websockify

_KASM_AUTH = "Basic " + base64.b64encode(
    f"{KASM_VNC_USER}:{KASM_VNC_PW}".encode()
).decode()

_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
    "content-encoding", "content-length",
}


@app.api_route(
    "/vnc/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
)
async def vnc_http_proxy(path: str, request: Request) -> Response:
    upstream = f"https://{DESKTOP_HOST}:{DESKTOP_VNC_PORT}/{path}"
    if request.url.query:
        upstream += f"?{request.url.query}"
    fwd_headers = {
        k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP
    }
    fwd_headers["host"] = f"{DESKTOP_HOST}:{DESKTOP_VNC_PORT}"
    fwd_headers["authorization"] = _KASM_AUTH
    body = await request.body()
    async with httpx.AsyncClient(verify=False, timeout=30.0, follow_redirects=False) as c:
        try:
            r = await c.request(request.method, upstream, content=body, headers=fwd_headers)
        except httpx.HTTPError as e:
            raise HTTPException(502, f"vnc proxy error: {e}")
    resp_headers = {
        k: v for k, v in r.headers.items() if k.lower() not in _HOP_BY_HOP
    }
    return Response(
        content=r.content,
        status_code=r.status_code,
        headers=resp_headers,
        media_type=r.headers.get("content-type"),
    )


@app.websocket("/vnc/websockify")
async def vnc_ws_proxy(client_ws: WebSocket) -> None:
    """Bidirectionally proxy the noVNC websocket to KasmVNC."""
    await client_ws.accept(subprotocol="binary")
    upstream_url = f"wss://{DESKTOP_HOST}:{DESKTOP_VNC_PORT}/websockify"
    sslctx = ssl.create_default_context()
    sslctx.check_hostname = False
    sslctx.verify_mode = ssl.CERT_NONE
    extra_headers = [
        ("Authorization", _KASM_AUTH),
        ("Origin", f"https://{DESKTOP_HOST}:{DESKTOP_VNC_PORT}"),
    ]
    try:
        async with websockets.connect(
            upstream_url,
            ssl=sslctx,
            subprotocols=["binary"],
            additional_headers=extra_headers,
            max_size=None,
            open_timeout=10,
            ping_interval=None,
        ) as upstream:

            async def c2u() -> None:
                try:
                    while True:
                        msg = await client_ws.receive()
                        t = msg.get("type")
                        if t == "websocket.disconnect":
                            return
                        if "bytes" in msg and msg["bytes"] is not None:
                            await upstream.send(msg["bytes"])
                        elif "text" in msg and msg["text"] is not None:
                            await upstream.send(msg["text"])
                except (WebSocketDisconnect, websockets.ConnectionClosed):
                    return

            async def u2c() -> None:
                try:
                    async for msg in upstream:
                        if isinstance(msg, (bytes, bytearray)):
                            await client_ws.send_bytes(bytes(msg))
                        else:
                            await client_ws.send_text(msg)
                except (RuntimeError, WebSocketDisconnect, websockets.ConnectionClosed):
                    return

            tasks = {asyncio.create_task(c2u()), asyncio.create_task(u2c())}
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*done, *pending, return_exceptions=True)
    except Exception as e:
        logger.warning(f"vnc ws proxy failed: {e}")
        try:
            await client_ws.close(code=1011)
        except Exception:
            pass
