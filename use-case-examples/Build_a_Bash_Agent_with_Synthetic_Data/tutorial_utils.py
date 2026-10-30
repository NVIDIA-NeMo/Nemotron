"""CPU-testable helpers shared by the tutorial notebooks."""

from __future__ import annotations

import json
import math
import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from numbers import Real
from typing import Any

from bash_agent.prompts import JSON_SYSTEM_PROMPT


COMMANDS = {"new", "dev", "up", "build", "dockerfile"}
TOOL_FIELDS = (
    "command",
    "template",
    "path",
    "port",
    "no_browser",
    "watch",
    "wait",
    "tag",
    "output_path",
)

SYSTEM_PROMPT = JSON_SYSTEM_PROMPT


def _truthy_sample(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def build_expected_output(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build a correct label from sampled values instead of asking an LLM to label data."""

    command = str(row["command"])
    if command not in COMMANDS:
        raise ValueError(f"Unsupported command: {command}")

    target = {field: None for field in TOOL_FIELDS}
    target["command"] = command
    if command == "new":
        target["template"] = str(row["template"])
        target["path"] = str(row["project_path"])
    elif command == "dev":
        target["port"] = int(row["port"])
        target["no_browser"] = True if _truthy_sample(row["no_browser"]) else None
    elif command == "up":
        target["port"] = int(row["port"])
        target["watch"] = True if _truthy_sample(row["watch"]) else None
        target["wait"] = True if _truthy_sample(row["wait"]) else None
    elif command == "build":
        target["tag"] = str(row["image_tag"])
    elif command == "dockerfile":
        target["output_path"] = str(row["dockerfile_path"])
    return target


def _plain_mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a mapping, got {type(value).__name__}")
    return dict(value)


def normalize_expected_output(value: Any) -> dict[str, Any]:
    """Restore schema types that tabular persistence can widen inside nested objects."""

    expected = _plain_mapping(value)
    port = expected.get("port")
    if port is not None:
        if (
            isinstance(port, bool)
            or not isinstance(port, Real)
            or not math.isfinite(float(port))
            or not float(port).is_integer()
        ):
            raise ValueError(f"Expected an integral port, got {port!r}")
        expected["port"] = int(port)
    return expected


def deterministic_request(expected_output: Any) -> str:
    """Create an unambiguous natural-language request for a validated target."""

    expected = normalize_expected_output(expected_output)
    command = expected.get("command")
    if command == "new":
        return (
            f"Create a new LangGraph project at `{expected['path']}` using the exact "
            f"`{expected['template']}` template."
        )
    if command == "dev":
        request = "Start the LangGraph development server"
        if expected.get("port") is not None:
            request += f" on port {expected['port']}"
        if expected.get("no_browser") is True:
            request += " without opening a browser"
        return request + "."
    if command == "up":
        request = "Bring the LangGraph services up"
        if expected.get("port") is not None:
            request += f" on port {expected['port']}"
        if expected.get("watch") is True:
            request += " with file watching enabled"
        if expected.get("wait") is True:
            request += " and wait until the services are ready"
        return request + "."
    if command == "build":
        return f"Build the LangGraph image with the exact tag `{expected['tag']}`."
    if command == "dockerfile":
        return f"Generate the LangGraph Dockerfile at `{expected['output_path']}`."
    raise ValueError(f"Unsupported command: {command!r}")


def generated_request_matches_expected_output(user_input: Any, expected_output: Any) -> bool:
    """Conservatively accept LLM phrasing only when it states exactly the target options."""

    if not isinstance(user_input, str) or not user_input.strip():
        return False
    expected = normalize_expected_output(expected_output)
    command = expected.get("command")
    text = user_input.casefold()

    relevant_fields = {
        "new": ("template", "path"),
        "dev": ("port",),
        "up": ("port",),
        "build": ("tag",),
        "dockerfile": ("output_path",),
    }
    if command not in relevant_fields:
        return False
    for field in relevant_fields[command]:
        value = expected.get(field)
        if value is not None and str(value).casefold() not in text:
            return False

    has_browser = bool(re.search(r"\bbrowser\b", text))
    has_no_browser = bool(
        re.search(
            r"(?:\bwithout\b|\bdo not\b|\bdon't\b|\bno\b|\bdisable\w*\b)"
            r"[^.]{0,40}\bbrowser\b|\bbrowser\b[^.]{0,20}\b(?:disabled|off)\b",
            text,
        )
    )
    has_watch = bool(
        re.search(
            r"\bwatch mode\b|\bwatch files?\b|\bfile watching\b|"
            r"\bwatching enabled\b|--watch\b|\bauto[- ]?reload",
            text,
        )
    )
    mentions_wait = bool(re.search(r"\bwait(?:s|ed|ing)?\b|\bready\b|\breadiness\b", text))
    has_wait_until_ready = bool(
        re.search(
            r"\bwait(?:s|ed|ing)?\b[^.]{0,50}\bready\b|"
            r"\buntil\b[^.]{0,50}\bready\b|\breadiness\b",
            text,
        )
    )
    if has_browser != (command == "dev" and expected.get("no_browser") is True):
        return False
    if has_browser and not has_no_browser:
        return False
    if has_watch != (command == "up" and expected.get("watch") is True):
        return False
    if mentions_wait != (command == "up" and expected.get("wait") is True):
        return False
    if mentions_wait and not has_wait_until_ready:
        return False

    if command not in {"dev", "up"} and (
        re.search(r"\bport\b", text) or re.search(r"\b[1-9]\d{3,4}\b", text)
    ):
        return False
    if command != "new" and re.search(r"\btemplate\b", text):
        return False
    if command != "build" and re.search(r"\btag\b", text):
        return False
    if command != "dockerfile" and "dockerfile" in text:
        return False

    forbidden_patterns = {
        "new": r"\b(?:build|browser|deploy|dockerfile|port|run|start|tag|wait|watch)\b",
        "dev": r"\b(?:build|dockerfile|error|fix|help|tag|template|wait|watch)\b",
        "up": r"\b(?:browser|build|dockerfile|error|finish|fix|help|output|shutdown|stop|tag|template)\b",
        "build": r"\b(?:browser|deploy|dockerfile|output|port|push|run|save|start|store|template|wait|watch)\b",
        "dockerfile": r"\b(?:browser|build|configuration|port|push|run|tag|template|wait|watch)\b|\bdocker image\b",
    }
    if re.search(forbidden_patterns[command], text):
        return False
    if command == "dockerfile" and expected.get("output_path") == "Dockerfile":
        if not re.search(
            r"`dockerfile`|\b(?:as|at|to) dockerfile\b|\bcurrent directory\b|\bproject root\b",
            text,
        ):
            return False

    intent_patterns = {
        "new": r"\b(?:create|generate|new|set up)\b",
        "dev": r"\b(?:dev|development)\b",
        "up": r"\b(?:bring|launch|online|run|serve|start|up)\b",
        "build": r"\bbuild\b",
        "dockerfile": r"\bdockerfile\b",
    }
    return bool(re.search(intent_patterns[command], text))


def repair_generated_request(user_input: Any, expected_output: Any) -> str:
    """Keep a complete LLM request or replace it with a deterministic safe fallback."""

    if generated_request_matches_expected_output(user_input, expected_output):
        return str(user_input).strip()
    return deterministic_request(expected_output)


def build_gym_record(user_input: str, expected_output: Any) -> dict[str, Any]:
    """Convert one generated row to NeMo Gym 0.3's dataset contract."""

    expected = normalize_expected_output(expected_output)
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(user_input)},
            ],
            "parallel_tool_calls": False,
            "tools": [],
        },
        "expected_output": expected,
    }


def completion_text(completion: Any) -> str:
    """Read either TRL's conversational completion or a plain string."""

    if isinstance(completion, str):
        return completion
    if isinstance(completion, Sequence) and completion:
        first = completion[0]
        if isinstance(first, Mapping):
            return str(first.get("content", ""))
    return ""


def build_verify_payload(
    completion: Any,
    prompt: Sequence[Mapping[str, Any]],
    expected_output: Any,
    request_id: str,
) -> dict[str, Any]:
    """Wrap a TRL completion in NeMo Gym's OpenAI Responses API envelope."""

    text = completion_text(completion)
    return {
        "responses_create_params": {
            "input": [dict(message) for message in prompt],
            "parallel_tool_calls": False,
            "tools": [],
        },
        "response": {
            "id": f"resp-{request_id}",
            "created_at": 0.0,
            "model": "grpo-policy",
            "object": "response",
            "output": [
                {
                    "id": f"msg-{request_id}",
                    "content": [{"annotations": [], "text": text, "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
        },
        "expected_output": _plain_mapping(expected_output),
    }


def create_nemo_gym_reward_function(
    verify_endpoint: str,
    *,
    timeout: float = 30.0,
    post: Callable[..., Any] | None = None,
):
    """Create a TRL 0.24 reward function backed by a NeMo Gym verifier."""

    import requests

    post_request = post or requests.post

    def reward_fn(completions, expected_output, prompts=None, **kwargs) -> list[float]:
        del kwargs
        if prompts is None:
            prompts = [[{"role": "user", "content": ""}] for _ in completions]
        rewards: list[float] = []
        for index, (completion, prompt, expected) in enumerate(
            zip(completions, prompts, expected_output, strict=True)
        ):
            payload = build_verify_payload(completion, prompt, expected, f"train-{index}")
            try:
                response = post_request(verify_endpoint, json=payload, timeout=timeout)
                response.raise_for_status()
                reward = float(response.json()["reward"])
                if not math.isfinite(reward):
                    raise ValueError("Verifier returned a non-finite reward")
            except (requests.RequestException, KeyError, TypeError, ValueError) as error:
                warnings.warn(f"NeMo Gym verification failed: {error}", stacklevel=2)
                reward = -1.0
            rewards.append(reward)
        return rewards

    return reward_fn
