# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility helpers for Data Designer model-facade responses."""

from __future__ import annotations

from typing import Any

from data_designer.engine.models.utils import ChatMessage


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(_get(block, "text", "")) for block in content)
    return "" if content is None else str(content)


def completion_text(facade: Any, *, system_prompt: str, user_prompt: str) -> str:
    response = facade.completion(
        [ChatMessage.as_system(system_prompt), ChatMessage.as_user(user_prompt)],
        allow_multiple_choices=False,
    )
    choices = _get(response, "choices")
    if not choices:
        raw = _get(response, "raw", {})
        choices = _get(raw, "choices", [])
    if len(choices) != 1:
        raise ValueError(f"Expected one completion choice, received {len(choices)}")
    return _content_text(_get(_get(choices[0], "message", {}), "content", ""))
