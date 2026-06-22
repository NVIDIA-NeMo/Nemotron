"""Minimal desktop API used by nano_omni_demo.

The FastAPI server only needs four operations from the desktop container:
readiness, screenshot capture, screen size, and command execution. This file
keeps that surface intentionally small for public release.
"""

from __future__ import annotations

import os
import subprocess
from io import BytesIO
from typing import Any

import pyautogui
from flask import Flask, Response, jsonify, request


SERVER_VERSION = "2026.04.30.minimal"
MAX_COMMAND_TIMEOUT = 120
MAX_OUTPUT_CHARS = 20000

app = Flask(__name__)

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


def _json_error(message: str, status: int = 400):
    return jsonify({"status": "error", "message": message}), status


def _command_from_request(data: dict[str, Any]) -> list[str] | None:
    command = data.get("command")
    if not isinstance(command, list) or not command:
        return None
    if not all(isinstance(arg, str) for arg in command):
        return None
    return [
        os.path.expanduser(arg) if arg.startswith("~/") else arg
        for arg in command
    ]


@app.get("/health")
def health():
    return jsonify({"status": "ok", "version": SERVER_VERSION})


@app.get("/screenshot")
def screenshot():
    try:
        image = pyautogui.screenshot()
        if image.mode != "RGB":
            image = image.convert("RGB")
        buf = BytesIO()
        image.save(buf, format="PNG")
        return Response(buf.getvalue(), mimetype="image/png")
    except Exception as exc:
        return _json_error(f"screenshot failed: {exc}", 500)


@app.post("/screen_size")
def screen_size():
    try:
        size = pyautogui.size()
        return jsonify({"width": int(size.width), "height": int(size.height)})
    except Exception as exc:
        return _json_error(f"screen size failed: {exc}", 500)


@app.post("/execute")
def execute_command():
    data = request.get_json(silent=True) or {}
    command = _command_from_request(data)
    if command is None:
        return _json_error("command must be a non-empty list of strings")

    try:
        timeout = min(float(data.get("timeout", MAX_COMMAND_TIMEOUT)), MAX_COMMAND_TIMEOUT)
    except (TypeError, ValueError):
        timeout = MAX_COMMAND_TIMEOUT

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _json_error(f"command timed out after {timeout:g}s", 500)
    except Exception as exc:
        return _json_error(f"command failed: {exc}", 500)

    output = result.stdout or ""
    if result.stderr:
        output = f"{output}\n{result.stderr}" if output else result.stderr
    output = output[:MAX_OUTPUT_CHARS]

    return jsonify({
        "status": "success" if result.returncode == 0 else "error",
        "output": output,
        "error": (result.stderr or "")[:MAX_OUTPUT_CHARS],
        "returncode": result.returncode,
    })


if __name__ == "__main__":
    port = int(os.getenv("API_PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
