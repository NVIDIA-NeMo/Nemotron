"""Build and validate LangGraph CLI invocations without shell parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


TEMPLATE_IDS = frozenset(
    {
        "agent-python",
        "deep-agent-js",
        "deep-agent-python",
        "new-langgraph-project-js",
        "new-langgraph-project-python",
    }
)

_COMMAND_FIELDS = frozenset(
    {
        "command",
        "template",
        "path",
        "port",
        "no_browser",
        "watch",
        "wait",
        "tag",
        "output_path",
    }
)
_RELEVANT_FIELDS = {
    "new": frozenset({"template", "path"}),
    "dev": frozenset({"port", "no_browser"}),
    "up": frozenset({"port", "watch", "wait"}),
    "build": frozenset({"tag"}),
    "dockerfile": frozenset({"output_path"}),
}
_IMAGE_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@-]{0,254}\Z")


class CommandValidationError(ValueError):
    """Raised when model output is not an allowed LangGraph invocation."""


@dataclass(frozen=True, slots=True)
class LangGraphInvocation:
    """A validated, immutable process argument vector."""

    argv: tuple[str, ...]


def _validate_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CommandValidationError(f"'{field}' must be a non-empty string.")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise CommandValidationError(f"'{field}' cannot contain control characters.")
    return value


def _validate_relative_path(
    value: Any,
    field: str,
    root_dir: str | Path,
    *,
    allow_root: bool = False,
) -> str:
    raw_path = _validate_text(value, field)
    if raw_path.startswith("-"):
        raise CommandValidationError(f"'{field}' cannot be interpreted as an option.")
    path = Path(raw_path)
    if path.is_absolute():
        raise CommandValidationError(f"'{field}' must be relative to the working directory.")

    root = Path(root_dir).resolve()
    target = (root / path).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise CommandValidationError(f"'{field}' must stay inside the working directory.") from exc
    if not allow_root and target == root:
        raise CommandValidationError(f"'{field}' must name a child of the working directory.")
    return raw_path


def _validate_port(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CommandValidationError("'port' must be an integer.")
    if not 1 <= value <= 65535:
        raise CommandValidationError("'port' must be between 1 and 65535.")
    return value


def _validate_optional_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise CommandValidationError(f"'{field}' must be a boolean.")
    return value


def _validate_payload_fields(payload: Mapping[str, Any], command: str) -> None:
    unknown = set(payload) - _COMMAND_FIELDS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise CommandValidationError(f"Unknown command field(s): {names}.")

    relevant = _RELEVANT_FIELDS[command]
    irrelevant = [
        field
        for field in _COMMAND_FIELDS - relevant - {"command"}
        if payload.get(field) is not None
    ]
    if irrelevant:
        names = ", ".join(sorted(irrelevant))
        raise CommandValidationError(f"Field(s) not valid for 'langgraph {command}': {names}.")


def invocation_from_payload(
    payload: Mapping[str, Any], root_dir: str | Path
) -> LangGraphInvocation:
    """Convert the model's structured command object to validated argv."""
    if not isinstance(payload, Mapping):
        raise CommandValidationError("The command payload must be a JSON object.")

    command = payload.get("command")
    if not isinstance(command, str) or command not in _RELEVANT_FIELDS:
        allowed = ", ".join(_RELEVANT_FIELDS)
        raise CommandValidationError(f"Unsupported command. Allowed commands: {allowed}.")
    _validate_payload_fields(payload, command)

    argv: list[str] = ["langgraph", command]
    if command == "new":
        template = _validate_text(payload.get("template"), "template")
        if template not in TEMPLATE_IDS:
            allowed = ", ".join(sorted(TEMPLATE_IDS))
            raise CommandValidationError(f"Unknown template. Allowed templates: {allowed}.")
        path = _validate_relative_path(payload.get("path"), "path", root_dir)
        argv.extend([path, "--template", template])
    elif command == "dev":
        port = payload.get("port")
        if port is not None:
            argv.extend(["--port", str(_validate_port(port))])
        no_browser = payload.get("no_browser")
        if no_browser is not None and _validate_optional_bool(no_browser, "no_browser"):
            argv.append("--no-browser")
    elif command == "up":
        port = payload.get("port")
        if port is not None:
            argv.extend(["--port", str(_validate_port(port))])
        watch = payload.get("watch")
        if watch is not None and _validate_optional_bool(watch, "watch"):
            argv.append("--watch")
        wait = payload.get("wait")
        if wait is not None and _validate_optional_bool(wait, "wait"):
            argv.append("--wait")
    elif command == "build":
        tag = _validate_text(payload.get("tag"), "tag")
        if not _IMAGE_TAG.fullmatch(tag):
            raise CommandValidationError("'tag' contains unsupported characters.")
        argv.extend(["-t", tag])
    else:
        output_path = _validate_relative_path(
            payload.get("output_path"), "output_path", root_dir, allow_root=False
        )
        # SAVE_PATH is positional in current langgraph-cli releases.
        argv.append(output_path)

    return LangGraphInvocation(tuple(argv))


def _parse_port_arg(value: str) -> int:
    if not value.isascii() or not value.isdecimal():
        raise CommandValidationError("'port' must be a base-10 integer.")
    return _validate_port(int(value))


def _parse_switches(args: tuple[str, ...], *, boolean_flags: Mapping[str, str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    index = 0
    while index < len(args):
        token = args[index]
        if token in {"--port", "-p"}:
            if "port" in payload or index + 1 >= len(args):
                raise CommandValidationError("'--port' must appear once with a value.")
            payload["port"] = _parse_port_arg(args[index + 1])
            index += 2
        elif token in boolean_flags:
            boolean_field = boolean_flags[token]
            if boolean_field in payload:
                raise CommandValidationError(f"'{token}' cannot be repeated.")
            payload[boolean_field] = True
            index += 1
        else:
            raise CommandValidationError(f"Unsupported argument: {token!r}.")
    return payload


def invocation_from_argv(argv: Sequence[str], root_dir: str | Path) -> LangGraphInvocation:
    """Validate model-generated argv and normalize it to canonical ordering."""
    if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence):
        raise CommandValidationError("'argv' must be an array of strings.")
    tokens = tuple(argv)
    if not 2 <= len(tokens) <= 8 or not all(isinstance(arg, str) for arg in tokens):
        raise CommandValidationError("'argv' must contain 2 to 8 string arguments.")
    for argument in tokens:
        _validate_text(argument, "argv item")
    if tokens[0] != "langgraph":
        raise CommandValidationError("Only the 'langgraph' executable is allowed.")

    command = tokens[1]
    args = tokens[2:]
    payload: dict[str, Any] = {"command": command}
    if command == "new":
        if len(args) != 3:
            raise CommandValidationError("'langgraph new' requires PATH and '--template TEMPLATE'.")
        if args[1] == "--template":
            payload.update(path=args[0], template=args[2])
        elif args[0] == "--template":
            payload.update(template=args[1], path=args[2])
        else:
            raise CommandValidationError("'langgraph new' requires PATH and '--template TEMPLATE'.")
    elif command == "dev":
        payload.update(_parse_switches(args, boolean_flags={"--no-browser": "no_browser"}))
    elif command == "up":
        payload.update(
            _parse_switches(
                args,
                boolean_flags={"--watch": "watch", "--wait": "wait"},
            )
        )
    elif command == "build":
        if len(args) != 2 or args[0] not in {"-t", "--tag"}:
            raise CommandValidationError("'langgraph build' requires '-t IMAGE_TAG'.")
        payload["tag"] = args[1]
    elif command == "dockerfile":
        if len(args) != 1:
            raise CommandValidationError(
                "'langgraph dockerfile' requires one positional SAVE_PATH."
            )
        payload["output_path"] = args[0]
    else:
        allowed = ", ".join(_RELEVANT_FIELDS)
        raise CommandValidationError(f"Unsupported command. Allowed commands: {allowed}.")

    return invocation_from_payload(payload, root_dir)
