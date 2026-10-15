# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A minimal client for the selected OpenAI-compatible chat endpoint.

The public cookbook is configured for the hosted NVIDIA Build API. Advanced
users may point the same client at their own compatible endpoint.
"""

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request


def settings():
    return {"base_url": os.environ.get("MODEL_BASE_URL", "").rstrip("/"), "model": os.environ.get("MODEL_NAME", ""),
            "api_key": os.environ.get("MODEL_API_KEY") or os.environ.get("NVIDIA_API_KEY", "")}


def chat(messages, tools=None, max_tokens=4096, temperature=1.0, top_p=0.95, thinking=True, timeout=900, **extra):
    """Send one chat request. Returns (response payload, seconds)."""
    config = settings()
    if not config["base_url"] or not config["model"]:
        raise RuntimeError("Copy .env.example to .env and set NVIDIA_API_KEY; the official base URL and model ID are already filled in.")
    body = {"model": config["model"], "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "top_p": top_p, "chat_template_kwargs": {"enable_thinking": thinking}, **extra}
    if tools:
        body["tools"] = tools
    headers = {"Content-Type": "application/json"}
    if config["api_key"]:
        headers["Authorization"] = f"Bearer {config['api_key']}"
    request = urllib.request.Request(f"{config['base_url']}/chat/completions", data=json.dumps(body).encode(), headers=headers)
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            raise RuntimeError(
                f"HTTP {error.code}: NVIDIA Build rejected the API key. Generate a key at "
                "https://build.nvidia.com/settings and set NVIDIA_API_KEY in .env."
            ) from None
        if error.code == 404:
            raise RuntimeError(
                "HTTP 404: the endpoint or model was not found. Check MODEL_BASE_URL and MODEL_NAME against the "
                "published NVIDIA Build model page."
            ) from None
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode()[:2000]}") from None
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not reach MODEL_BASE_URL ({error.reason}). Check your network and the URL in .env."
        ) from None
    return payload, time.monotonic() - start


def _number_image(text="47"):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (240, 120), "white")
    ImageDraw.Draw(image).text((70, 20), text, fill="black", font_size=72)
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def check():
    """Four quick requests: text, thinking, an image, and a tool call. Returns a list of (check, passed, detail)."""
    results = []
    try:
        payload, seconds = chat([{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=512, thinking=False)
        text = (payload["choices"][0]["message"].get("content") or "").strip()
        results.append(("Text", "ready" in text.lower(), f"replied {text[:40]!r} in {seconds:.1f} s"))
        payload, seconds = chat([{"role": "user", "content": "What is 17 * 23? Answer with the number."}], max_tokens=4096, thinking=True)
        message = payload["choices"][0]["message"]
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        results.append(("Thinking", bool(reasoning) and "391" in (message.get("content") or ""),
                        f"returned {len(reasoning)} characters of reasoning in {seconds:.1f} s"))
        image = [{"type": "text", "text": "What number is shown? Reply with the number only."},
                 {"type": "image_url", "image_url": {"url": _number_image()}}]
        payload, seconds = chat([{"role": "user", "content": image}], max_tokens=512, thinking=False)
        text = (payload["choices"][0]["message"].get("content") or "").strip()
        results.append(("Image", "47" in text, f"read {text[:40]!r} in {seconds:.1f} s"))
        echo = {"type": "function", "function": {"name": "echo", "description": "Echo a value back.",
                                                 "parameters": {"type": "object", "properties": {"value": {"type": "integer"}},
                                                                "required": ["value"]}}}
        payload, seconds = chat([{"role": "user", "content": "Call the echo tool with value 7."}], tools=[echo], max_tokens=1024,
                                thinking=False)
        calls = payload["choices"][0]["message"].get("tool_calls") or []
        ok = bool(calls) and calls[0]["function"]["name"] == "echo"
        results.append(("Tool call", ok, f"called {calls[0]['function']['name'] if calls else 'nothing'} in {seconds:.1f} s"))
    except Exception as exc:
        results.append(("Endpoint", False, str(exc)[:300]))
    return results
