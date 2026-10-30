#!/usr/bin/env python3
"""Interactive runtime for the trained LangGraph CLI agent."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .bash import Bash
from .commands import CommandValidationError, LangGraphInvocation
from .config import Config
from .helpers import Messages, get_llm


def format_argv(argv: tuple[str, ...]) -> str:
    """Format argv for humans only; this string is never executed."""
    return shlex.join(argv)


def confirm_execution(argv: tuple[str, ...]) -> bool:
    """Ask the user whether the exact validated argv should be executed."""
    response = input(f"    Execute {format_argv(argv)!r}? [y/N]: ").strip().lower()
    return response in {"y", "yes"}


def execute_with_confirmation(
    executor: Bash,
    invocation: LangGraphInvocation,
    confirm: Callable[[tuple[str, ...]], bool] = confirm_execution,
) -> dict[str, Any]:
    """Execute only after a human approves the same immutable argv."""
    if not confirm(invocation.argv):
        return {"error": "The user declined the execution of this command."}
    return executor.exec_langgraph(invocation)


def _tool_call_parts(tool_call: Any) -> tuple[str, str, str]:
    if isinstance(tool_call, Mapping):
        tool_id = tool_call.get("id")
        function = tool_call.get("function")
    else:
        tool_id = getattr(tool_call, "id", None)
        function = getattr(tool_call, "function", None)

    if isinstance(function, Mapping):
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", None)
    if not all(isinstance(value, str) and value for value in (tool_id, name, arguments)):
        raise ValueError("Malformed function tool call.")
    return tool_id, name, arguments


def _tool_call_id(tool_call: Any) -> str:
    if isinstance(tool_call, Mapping):
        tool_id = tool_call.get("id")
    else:
        tool_id = getattr(tool_call, "id", None)
    return tool_id if isinstance(tool_id, str) and tool_id else "call_invalid"


def _display_tool_result(result: Mapping[str, Any]) -> None:
    if result.get("stdout"):
        print(f"\nOutput:\n{result['stdout']}")
    if result.get("stderr"):
        print(f"\nError:\n{result['stderr']}")
    if result.get("error"):
        print(f"\nError:\n{result['error']}")


def _without_thinking(response: str) -> str:
    return response.split("</think>")[-1].strip()


def existing_directory(value: str) -> str:
    """Resolve an argparse value and require an existing directory."""
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"not an existing directory: {value!r}")
    return str(path)


def positive_seconds(value: str) -> float:
    """Parse a finite, positive duration for argparse."""
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number of seconds") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return seconds


def environment_name(value: str) -> str:
    """Validate an environment variable name passed through to child commands."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise argparse.ArgumentTypeError(f"invalid environment variable name: {value!r}")
    return value


def main(config: Config) -> None:
    executor = Bash(config)
    try:
        _run_agent(config, executor)
    finally:
        executor.close()


def _run_agent(config: Config, executor: Bash) -> None:
    llm = get_llm(config)
    messages = Messages(config.json_system_prompt)

    print("\n" + "=" * 60)
    print("LangGraph CLI Agent")
    print("=" * 60)
    if config.use_api:
        print(f"Model: {config.api_model_name} at {config.api_base_url}")
    else:
        print(f"Model: {config.model_path}")
    print(f"Working directory: {executor.cwd}")
    print("Type 'quit' or 'exit' to stop; type 'clear' to reset history.")
    print("=" * 60 + "\n")

    while True:
        try:
            user = input(f"['{executor.cwd}'] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nShutting down. Bye!")
            break

        if user.lower() in {"quit", "exit"}:
            print("\nShutting down. Bye!")
            break
        if user.lower() == "clear":
            messages.clear()
            print("Conversation cleared.\n")
            continue
        if not user:
            continue

        messages.add_user_message(user)

        while True:
            print("\nThinking...")
            try:
                raw_response, tool_calls = llm.query(messages, [executor.to_json_schema()])
            except Exception as exc:
                print(f"Error querying model: {exc}")
                break

            display_response = _without_thinking(raw_response)
            if tool_calls:
                # Chat Completions requires this assistant message immediately before
                # the corresponding role=tool messages.
                messages.add_assistant_tool_calls(raw_response or None, tool_calls)
            elif raw_response:
                messages.add_assistant_message(raw_response)

            if not tool_calls:
                if display_response:
                    print(f"\n{display_response}")
                    print("-" * 60)
                break

            for tool_call in tool_calls:
                tool_id = _tool_call_id(tool_call)
                try:
                    tool_id, function_name, raw_arguments = _tool_call_parts(tool_call)
                    arguments = json.loads(raw_arguments)
                    if (
                        function_name != "run_langgraph"
                        or not isinstance(arguments, dict)
                        or set(arguments) != {"argv"}
                    ):
                        raise CommandValidationError(
                            "Expected run_langgraph with exactly one 'argv' field."
                        )
                    invocation = executor.prepare_argv(arguments["argv"])
                except (CommandValidationError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    tool_result: dict[str, Any] = {"error": str(exc)}
                else:
                    print(f"\nProposed command: {format_argv(invocation.argv)}")
                    tool_result = execute_with_confirmation(executor, invocation)

                _display_tool_result(tool_result)
                messages.add_tool_message(json.dumps(tool_result), tool_id)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangGraph CLI agent runtime")
    parser.add_argument("--model-path", help="Local trained model checkpoint")
    parser.add_argument("--use-api", action="store_true", help="Use an OpenAI-compatible API")
    parser.add_argument("--api-url", default=None, help="OpenAI-compatible API base URL")
    parser.add_argument("--api-model", default=None, help="Served API model name")
    parser.add_argument(
        "--omit-api-thinking-override",
        action="store_true",
        help=(
            "Do not send vLLM's chat_template_kwargs reasoning override; "
            "configure reasoning-off at the server instead"
        ),
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--device", default=None, help="cuda, cpu, mps, or auto")
    parser.add_argument(
        "--root-dir",
        type=existing_directory,
        default=None,
        help="Existing trusted directory used as the command cwd and path boundary",
    )
    parser.add_argument(
        "--command-timeout",
        type=positive_seconds,
        default=None,
        metavar="SECONDS",
        help="Timeout for finite LangGraph commands (default: 600 seconds)",
    )
    parser.add_argument(
        "--pass-env",
        action="append",
        type=environment_name,
        default=None,
        metavar="NAME",
        help="Allow one environment variable in child commands; repeat as needed",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Config:
    """Apply validated CLI arguments before the runtime is initialized."""
    runtime_config = Config()
    if args.model_path is not None:
        runtime_config.model_path = args.model_path
    if args.use_api:
        runtime_config.use_api = True
    if args.api_url is not None:
        runtime_config.api_base_url = args.api_url
    if args.api_model is not None:
        runtime_config.api_model_name = args.api_model
    if args.omit_api_thinking_override:
        runtime_config.api_send_thinking_override = False
    if args.temperature is not None:
        runtime_config.temperature = args.temperature
    if args.device is not None:
        runtime_config.device = args.device
    if args.root_dir is not None:
        runtime_config.root_dir = args.root_dir
    if args.command_timeout is not None:
        runtime_config.command_timeout_seconds = args.command_timeout
    if args.pass_env is not None:
        runtime_config.pass_environment.update(args.pass_env)
    return runtime_config


if __name__ == "__main__":
    main(config_from_args(parse_args()))
