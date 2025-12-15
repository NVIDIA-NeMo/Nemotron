# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transform functions for JSONL output formats.

Provides factory functions and TypedDicts for common SFT/RL data formats.
"""

from collections.abc import Callable
from typing import TypedDict

# =============================================================================
# Type definitions for common output formats
# =============================================================================


class SftRecord(TypedDict):
    """Megatron-Bridge GPTSFTDataset format."""

    input: str
    output: str


class SftRecordWithSystem(TypedDict, total=False):
    """SFT format with optional system prompt."""

    input: str
    output: str
    system: str


class Message(TypedDict):
    """OpenAI chat message."""

    role: str  # "system" | "user" | "assistant"
    content: str


class OpenAIChatRecord(TypedDict):
    """OpenAI/RL format - used by OpenAIFormatDataset."""

    messages: list[Message]


class Conversation(TypedDict):
    """ShareGPT conversation turn."""

    from_: str  # "human" | "gpt" | "system" (serialized as "from")
    value: str


class ShareGPTRecord(TypedDict):
    """ShareGPT format - used by GPTSFTChatDataset."""

    conversations: list[Conversation]


# Transform is any callable: dict -> dict | None (None = skip record)
Transform = Callable[[dict], dict | None]


# =============================================================================
# Factory functions for common transforms
# =============================================================================


def sft(*, input: str = "input", output: str = "output", system: str | None = None) -> Transform:
    """Create SFT transform: extracts input/output fields.

    Args:
        input: Source field name for input text.
        output: Source field name for output text.
        system: Optional source field name for system prompt.

    Returns:
        Transform function producing SftRecord.

    Example:
        >>> transform = sft(input="instruction", output="response")
        >>> transform({"instruction": "Hello", "response": "Hi there!"})
        {'input': 'Hello', 'output': 'Hi there!'}
    """
    input_field = input  # Avoid shadowing builtin
    output_field = output

    def transform(record: dict) -> SftRecord | SftRecordWithSystem | None:
        try:
            result: dict = {
                "input": record[input_field],
                "output": record[output_field],
            }
            if system and system in record:
                result["system"] = record[system]
            return result  # type: ignore
        except KeyError:
            return None

    return transform


def openai_chat(*, messages: str = "messages") -> Transform:
    """Create OpenAI chat transform: extracts messages field.

    Args:
        messages: Source field name for messages list.

    Returns:
        Transform function producing OpenAIChatRecord.

    Example:
        >>> transform = openai_chat()
        >>> transform({"messages": [{"role": "user", "content": "Hi"}]})
        {'messages': [{'role': 'user', 'content': 'Hi'}]}
    """
    messages_field = messages

    def transform(record: dict) -> OpenAIChatRecord | None:
        try:
            return {"messages": record[messages_field]}
        except KeyError:
            return None

    return transform


def sharegpt(*, conversations: str = "conversations") -> Transform:
    """Create ShareGPT transform: extracts conversations field.

    Args:
        conversations: Source field name for conversations list.

    Returns:
        Transform function producing ShareGPTRecord.

    Example:
        >>> transform = sharegpt(conversations="turns")
        >>> transform({"turns": [{"from": "human", "value": "Hi"}]})
        {'conversations': [{'from': 'human', 'value': 'Hi'}]}
    """
    conversations_field = conversations

    def transform(record: dict) -> ShareGPTRecord | None:
        try:
            return {"conversations": record[conversations_field]}
        except KeyError:
            return None

    return transform


def nemotron_rl() -> Transform:
    """Extract messages and tools from Nemotron RL dataset format.

    Nemotron RL datasets store messages in `responses_create_params.input`
    and optionally tools in `responses_create_params.tools`.

    Returns:
        Transform function producing OpenAIChatRecord with optional tools.

    Example:
        >>> transform = nemotron_rl()
        >>> transform({
        ...     "responses_create_params": {
        ...         "input": [{"role": "user", "content": "Hi"}],
        ...         "tools": [{"name": "search", ...}]
        ...     }
        ... })
        {'messages': [{'role': 'user', 'content': 'Hi'}], 'tools': [...]}
    """

    def transform(record: dict) -> dict | None:
        try:
            params = record["responses_create_params"]
            result: dict = {"messages": params["input"]}
            if "tools" in params and params["tools"]:
                result["tools"] = params["tools"]
            return result
        except (KeyError, TypeError):
            return None

    return transform


def passthrough() -> Transform:
    """Pass records through unchanged.

    Returns:
        Transform function that returns records as-is.

    Example:
        >>> transform = passthrough()
        >>> transform({"any": "data"})
        {'any': 'data'}
    """
    return lambda record: record


def select(*fields: str) -> Transform:
    """Create transform that selects specific fields.

    Args:
        *fields: Field names to include in output.

    Returns:
        Transform function that extracts only the specified fields.

    Example:
        >>> transform = select("id", "text")
        >>> transform({"id": 1, "text": "hello", "extra": "ignored"})
        {'id': 1, 'text': 'hello'}
    """

    def transform(record: dict) -> dict | None:
        try:
            return {f: record[f] for f in fields}
        except KeyError:
            return None

    return transform


def rename(**field_mapping: str) -> Transform:
    """Create transform that renames fields.

    Args:
        **field_mapping: Mapping from new names to source field names.

    Returns:
        Transform function that extracts and renames fields.

    Example:
        >>> transform = rename(input="question", output="answer")
        >>> transform({"question": "What?", "answer": "This."})
        {'input': 'What?', 'output': 'This.'}
    """

    def transform(record: dict) -> dict | None:
        try:
            return {new_name: record[old_name] for new_name, old_name in field_mapping.items()}
        except KeyError:
            return None

    return transform


__all__ = [
    # Type definitions
    "Transform",
    "SftRecord",
    "SftRecordWithSystem",
    "Message",
    "OpenAIChatRecord",
    "Conversation",
    "ShareGPTRecord",
    # Factory functions
    "sft",
    "openai_chat",
    "nemotron_rl",
    "sharegpt",
    "passthrough",
    "select",
    "rename",
]
