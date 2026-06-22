"""Async HTTP client for the desktop container's minimal API.

The desktop container runs a Flask server that exposes:
    GET  /screenshot       -> image/png
    POST /screen_size      -> {"width": int, "height": int}
    POST /execute          -> {"output": str}  body: {"command": [...]}
    GET  /health           -> {"status": "ok"}
"""

from __future__ import annotations

import asyncio
import time

import httpx
from loguru import logger


class DesktopClient:
    """Client for communicating with the desktop container."""

    def __init__(self, host: str = "localhost", api_port: int = 5000, *, timeout: float = 60.0) -> None:
        self.base = f"http://{host}:{api_port}"
        self._timeout = timeout

    async def screenshot(self) -> bytes:
        """Capture current desktop as PNG bytes."""
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.get(f"{self.base}/screenshot")
            r.raise_for_status()
            return r.content

    async def screen_size(self) -> tuple[int, int]:
        """Get desktop resolution."""
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(f"{self.base}/screen_size")
            r.raise_for_status()
            data = r.json()
            return int(data["width"]), int(data["height"])

    async def execute(self, command: list[str]) -> str:
        """Execute a command inside the desktop container."""
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post(f"{self.base}/execute", json={"command": command})
            r.raise_for_status()
            return r.json().get("output", "")

    async def run_pyautogui(self, code: str) -> str:
        """Run a pyautogui snippet inside the desktop.

        Wraps the code with imports and executes via python3 -c.
        """
        wrapper = (
            "import pyautogui, time\n"
            "pyautogui.FAILSAFE = False\n"
            f"{code}\n"
        )
        return await self.execute(["python3", "-c", wrapper])

    async def wait_ready(self, max_seconds: float = 120.0, interval: float = 2.0) -> None:
        """Poll the desktop API until it's responsive."""
        deadline = time.time() + max_seconds
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                async with httpx.AsyncClient(timeout=10) as c:
                    r = await c.get(f"{self.base}/health")
                    if r.status_code == 200:
                        return
            except Exception as exc:
                last_err = exc
            await asyncio.sleep(interval)
        raise TimeoutError(f"Desktop API never became ready after {max_seconds}s: {last_err}")

    async def is_ready(self) -> bool:
        """Quick check if the desktop is responding."""
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{self.base}/health")
                return r.status_code == 200
        except Exception:
            return False
