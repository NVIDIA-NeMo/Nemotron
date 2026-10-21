# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crash-resumable async answer generation for Persona MCQ teacher models."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from nemotron.steps.sdg.persona_mcq.runtime.io import read_jsonl
from nemotron.steps.sdg.persona_mcq.runtime.languages import answer_instruction


def _answer_re(answer_label: str, answer_label_aliases: Iterable[str] = ()) -> re.Pattern[str]:
    labels = sorted({answer_label, *answer_label_aliases}, key=len, reverse=True)
    return re.compile(
        rf"(?:{'|'.join(re.escape(label) for label in labels)})"
        r"\s*[:：]?\s*[$*_{}\\\s]*(?:\\?text\{)?\s*\(?\s*([ABCD])\b",
        re.IGNORECASE,
    )


def answer_prompt(
    record: dict[str, Any],
    language_config: dict[str, Any],
    reasoning_config: dict[str, Any],
) -> str:
    options = "\n".join(f"{chr(65 + index)}) {choice}" for index, choice in enumerate(record["choices"]))
    instruction = answer_instruction(
        question_language=language_config["display_name"],
        answer_label=language_config["answer_label"],
        reasoning_language=reasoning_config["display_name"],
    )
    return f"{instruction}\n\n{record['question']}\n\n{options}"


def parse_answer_letter(
    content: str,
    answer_label: str = "Answer",
    answer_label_aliases: Iterable[str] = (),
) -> str | None:
    matches = list(_answer_re(answer_label, answer_label_aliases).finditer(content.strip()[-300:]))
    return matches[-1].group(1).upper() if matches else None


def split_visible_reasoning(
    content: str,
    answer_label: str,
    answer_label_aliases: Iterable[str] = (),
) -> tuple[str, str]:
    """Split a visible rationale from the configured final-answer marker."""
    stripped = content.strip()
    matches = list(_answer_re(answer_label, answer_label_aliases).finditer(stripped[-300:]))
    if not matches:
        return "", stripped
    match = matches[-1]
    absolute_start = max(0, len(stripped) - 300) + match.start()
    rationale = stripped[:absolute_start].strip()
    if not rationale:
        return "", stripped
    letter = match.group(1).upper()
    return rationale, f"{answer_label}: {letter}"


async def _request(
    client: Any,
    record: dict[str, Any],
    language_config: dict[str, Any],
    reasoning_config: dict[str, Any],
    model: dict[str, Any],
    max_retries: int,
) -> dict[str, Any]:
    import httpx

    body = {
        "model": model["model"],
        "messages": [{"role": "user", "content": answer_prompt(record, language_config, reasoning_config)}],
        "temperature": model.get("temperature", 1.0),
        "top_p": model.get("top_p", 1.0),
        "max_tokens": model.get("max_tokens", 16384),
        **(model.get("extra_body") or {}),
    }
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = await client.post("chat/completions", json=body)
            if response.status_code == 429 or response.status_code >= 500:
                raise httpx.HTTPStatusError("retryable model response", request=response.request, response=response)
            response.raise_for_status()
            payload = response.json()
            choice = payload["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            return {
                **record,
                "answer_model": model["model"],
                "answer": content,
                "reasoning": reasoning,
                "parsed_letter": parse_answer_letter(
                    content,
                    language_config["answer_label"],
                    language_config.get("answer_label_aliases") or (),
                ),
                "finish_reason": choice.get("finish_reason"),
                "completion_tokens": (payload.get("usage") or {}).get("completion_tokens"),
            }
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < max_retries:
                await asyncio.sleep(min(2**attempt, 30))
    raise RuntimeError(f"answer request failed after {max_retries} attempts: {last_error}")


async def generate_answers(
    records: list[dict[str, Any]],
    *,
    language_config: dict[str, Any],
    reasoning_config: dict[str, Any],
    model: dict[str, Any],
    output_path: Path,
    failure_path: Path,
    resume: bool,
    max_parallel: int,
    timeout: float,
    max_retries: int,
) -> dict[str, int]:
    import httpx

    output_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    existing_answers = read_jsonl(output_path) if resume and output_path.exists() else []
    existing_failures = read_jsonl(failure_path) if resume and failure_path.exists() else []
    done = {str(record["query_id"]) for record in existing_answers if record.get("query_id")}
    pending = [record for record in records if str(record["query_id"]) not in done]
    semaphore = asyncio.Semaphore(max_parallel)
    key_name = model.get("api_key_env", "NVIDIA_API_KEY")
    token = os.environ.get(key_name, model.get("api_key", ""))
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    limits = httpx.Limits(max_connections=max_parallel + 20, max_keepalive_connections=max_parallel)
    lock = asyncio.Lock()
    stats = {
        "pending": len(pending),
        "answered": len(existing_answers),
        "failed": len(existing_failures),
        "unparsed": sum(record.get("parsed_letter") is None for record in existing_answers),
    }

    async with httpx.AsyncClient(
        base_url=model["endpoint"].rstrip("/") + "/",
        timeout=timeout,
        headers=headers,
        limits=limits,
    ) as client:
        mode = "a" if resume else "w"
        with output_path.open(mode, encoding="utf-8") as output, failure_path.open(mode, encoding="utf-8") as failures:

            async def run_one(record: dict[str, Any]) -> None:
                async with semaphore:
                    try:
                        result = await _request(
                            client,
                            record,
                            language_config,
                            reasoning_config,
                            model,
                            max_retries,
                        )
                    except Exception as exc:  # noqa: BLE001 - persist per-row failure and continue.
                        async with lock:
                            failures.write(json.dumps({"query_id": record["query_id"], "error": str(exc)}) + "\n")
                            failures.flush()
                            stats["failed"] += 1
                        return
                    async with lock:
                        output.write(json.dumps(result, ensure_ascii=False) + "\n")
                        output.flush()
                        stats["answered"] += 1
                        stats["unparsed"] += int(result["parsed_letter"] is None)

            await asyncio.gather(*(run_one(record) for record in pending))
            os.fsync(output.fileno())
            os.fsync(failures.fileno())
    return stats
