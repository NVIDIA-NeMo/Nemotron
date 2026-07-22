"""NeMo Gym 0.3 resource server for LangGraph CLI tool-call verification."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


Command = Literal["new", "dev", "up", "build", "dockerfile"]
FLAG_KEYS = ("template", "path", "port", "no_browser", "watch", "wait", "tag", "output_path")
TEMPLATE_IDS = frozenset(
    {
        "agent-python",
        "deep-agent-js",
        "deep-agent-python",
        "new-langgraph-project-js",
        "new-langgraph-project-python",
    }
)
RELEVANT_FIELDS = {
    "new": frozenset({"template", "path"}),
    "dev": frozenset({"port", "no_browser"}),
    "up": frozenset({"port", "watch", "wait"}),
    "build": frozenset({"tag"}),
    "dockerfile": frozenset({"output_path"}),
}
REQUIRED_FIELDS = {
    "new": frozenset({"template", "path"}),
    "dev": frozenset(),
    "up": frozenset(),
    "build": frozenset({"tag"}),
    "dockerfile": frozenset({"output_path"}),
}
IMAGE_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@-]{0,254}\Z")


class CLIToolCall(BaseModel):
    """Structured output produced by the policy model."""

    # Match the runtime boundary: JSON strings, floats, and integers must not be
    # silently coerced into ports or booleans that the agent itself rejects.
    model_config = ConfigDict(extra="forbid", strict=True)

    command: Command
    template: str | None = None
    path: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    no_browser: bool | None = None
    watch: bool | None = None
    wait: bool | None = None
    tag: str | None = None
    output_path: str | None = None

    @field_validator("template")
    @classmethod
    def validate_template(cls, value: str | None) -> str | None:
        if value is not None and value not in TEMPLATE_IDS:
            allowed = ", ".join(sorted(TEMPLATE_IDS))
            raise ValueError(f"Unknown template. Allowed templates: {allowed}.")
        return value

    @field_validator("path", "output_path")
    @classmethod
    def validate_child_relative_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("Paths must be non-empty and contain no control characters.")
        if value.startswith("-"):
            raise ValueError("Paths cannot be interpreted as command options.")

        path = Path(value)
        if path.is_absolute():
            raise ValueError("Paths must be relative to the working directory.")

        # Mirror the runtime's resolved-path confinement without depending on a
        # particular working directory. A relative path may contain internal
        # `..`, but it may neither escape above the root nor resolve to the root.
        depth = 0
        for part in path.parts:
            if part in {"", "."}:
                continue
            if part == "..":
                depth -= 1
                if depth < 0:
                    raise ValueError("Paths must stay inside the working directory.")
            else:
                depth += 1
        if depth == 0:
            raise ValueError("Paths must name a child of the working directory.")
        return value

    @field_validator("tag")
    @classmethod
    def validate_image_tag(cls, value: str | None) -> str | None:
        if value is not None and not IMAGE_TAG.fullmatch(value):
            raise ValueError("Image tag contains unsupported characters.")
        return value

    @model_validator(mode="after")
    def validate_command_fields(self) -> "CLIToolCall":
        relevant = RELEVANT_FIELDS[self.command]
        required = REQUIRED_FIELDS[self.command]

        missing = sorted(field for field in required if getattr(self, field) is None)
        if missing:
            raise ValueError(
                f"Missing required field(s) for '{self.command}': {', '.join(missing)}."
            )

        irrelevant = sorted(
            field
            for field in FLAG_KEYS
            if field not in relevant and getattr(self, field) is not None
        )
        if irrelevant:
            raise ValueError(f"Field(s) not valid for '{self.command}': {', '.join(irrelevant)}.")
        return self


class LangGraphCLIResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the stateless verifier."""


class LangGraphCLIRunRequest(BaseRunRequest):
    expected_output: CLIToolCall


class LangGraphCLIVerifyRequest(LangGraphCLIRunRequest, BaseVerifyRequest):
    pass


class LangGraphCLIVerifyResponse(BaseVerifyResponse):
    expected_output: CLIToolCall
    exact_match: bool = False
    command_correct: bool = False
    flag_accuracy: float = 0.0
    feedback: str = ""
    parsed_output: dict[str, Any] | None = None


def extract_json_from_response(response: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a plain, fenced, or reasoning response."""

    if not isinstance(response, str):
        return None
    if "</think>" in response:
        response = response.rsplit("</think>", 1)[-1]

    candidates = [response.strip()]
    candidates.extend(re.findall(r"```(?:json)?\s*([\s\S]*?)```", response))
    candidates.extend(re.findall(r"<answer>\s*([\s\S]*?)\s*</answer>", response))

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed

        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def score_cli_output(
    predicted: dict[str, Any], reference: dict[str, Any]
) -> tuple[float, dict[str, Any]]:
    """Return a reward in [-1, 1] plus interpretable scoring metrics."""

    metrics: dict[str, Any] = {
        "exact_match": False,
        "command_correct": False,
        "flag_accuracy": 0.0,
        "correct_flags": 0,
        "wrong_flags": 0,
        "extra_flags": 0,
        "total_flags": 0,
    }

    if predicted.get("command") != reference.get("command"):
        return -1.0, metrics
    metrics["command_correct"] = True

    expected = {key: reference.get(key) for key in FLAG_KEYS if reference.get(key) is not None}
    actual = {key: predicted.get(key) for key in FLAG_KEYS if predicted.get(key) is not None}
    metrics["total_flags"] = len(expected)

    if not expected:
        if not actual:
            metrics["exact_match"] = True
            return 1.0, metrics
        metrics["extra_flags"] = len(actual)
        return max(-1.0, -0.5 * len(actual)), metrics

    correct = sum(actual.get(key) == value for key, value in expected.items() if key in actual)
    wrong = sum(actual.get(key) != value for key, value in expected.items() if key in actual)
    extra = sum(key not in expected for key in actual)
    total = len(expected)

    metrics.update(
        correct_flags=correct,
        wrong_flags=wrong,
        extra_flags=extra,
        flag_accuracy=correct / total,
    )
    reward = max(-1.0, min(1.0, (correct - wrong - extra) / total))
    if correct == total and wrong == 0 and extra == 0:
        metrics["exact_match"] = True
        reward = 1.0
    return reward, metrics


def make_feedback(metrics: dict[str, Any]) -> str:
    if metrics["exact_match"]:
        return "Exact match: command and arguments are correct."
    if not metrics["command_correct"]:
        return "Incorrect LangGraph CLI command."
    return (
        f"Command is correct; {metrics['correct_flags']}/{metrics['total_flags']} expected arguments "
        f"matched, with {metrics['wrong_flags']} wrong and {metrics['extra_flags']} extra."
    )


class LangGraphCLIResourcesServer(SimpleResourcesServer):
    config: LangGraphCLIResourcesServerConfig

    async def verify(self, body: LangGraphCLIVerifyRequest) -> LangGraphCLIVerifyResponse:
        parsed = extract_json_from_response(body.response.output_text or "")
        if parsed is None:
            return LangGraphCLIVerifyResponse(
                **body.model_dump(),
                reward=-1.0,
                feedback="Failed to parse a JSON object from the model response.",
            )

        try:
            predicted = CLIToolCall.model_validate(parsed).model_dump(mode="json")
        except ValidationError as error:
            return LangGraphCLIVerifyResponse(
                **body.model_dump(),
                reward=-1.0,
                feedback=f"Invalid CLI tool-call schema: {error.errors()}",
                parsed_output=parsed,
            )

        reward, metrics = score_cli_output(predicted, body.expected_output.model_dump(mode="json"))
        return LangGraphCLIVerifyResponse(
            **body.model_dump(),
            reward=reward,
            exact_match=metrics["exact_match"],
            command_correct=metrics["command_correct"],
            flag_accuracy=metrics["flag_accuracy"],
            feedback=make_feedback(metrics),
            parsed_output=predicted,
        )


if __name__ == "__main__":
    LangGraphCLIResourcesServer.run_webserver()
