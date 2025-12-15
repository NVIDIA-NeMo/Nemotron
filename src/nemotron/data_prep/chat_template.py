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

"""Chat template utilities for SFT data preparation.

Exact port of materialize.py logic for chat template application and
loss mask generation. Provides functions for:
- Converting JSON string arguments to dicts in tool calls
- Finding message boundaries in rendered templates
- Splitting templates into role-labeled chunks for loss masking
- Multi-turn conversation splitting for thinking/reasoning content
"""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def replace_json_args(messages: list[dict]) -> list[dict]:
    """Convert JSON string arguments to dict objects in tool calls.

    Modifies messages in-place to ensure tool call arguments are dicts,
    not JSON strings.

    Args:
        messages: List of OpenAI-format messages.

    Returns:
        The modified messages list.

    Note:
        Exact port of materialize.py::replace_json_args()
    """
    messages = copy.deepcopy(messages)
    for i in range(len(messages)):
        if messages[i]["role"] == "assistant":
            if messages[i].get("tool_calls"):
                for j in range(len(messages[i]["tool_calls"])):
                    if isinstance(messages[i]["tool_calls"][j]["function"]["arguments"], str):
                        messages[i]["tool_calls"][j]["function"]["arguments"] = json.loads(
                            messages[i]["tool_calls"][j]["function"]["arguments"]
                        )
    return messages


def find_last_user_message_end(
    messages: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    enable_thinking: bool = True,
    tools: list | None = None,
) -> int:
    """Find where the last user message ends in the rendered template.

    Uses incremental template rendering to find the exact character position
    where the last user message (plus generation prompt) ends.

    Args:
        messages: List of OpenAI-format messages.
        tokenizer: HuggingFace tokenizer with chat_template.
        enable_thinking: Whether thinking mode is enabled.
        tools: Optional list of tool definitions.

    Returns:
        Character position where last user message ends.

    Note:
        Exact port of materialize.py::find_last_user_message_end()
    """
    # Find the last user message index
    last_user_idx = max(i for i, msg in enumerate(messages) if msg["role"] == "user")

    # Render up to the last user message (inclusive)
    if enable_thinking and (
        "reasoning_content" not in messages[last_user_idx + 1]
        or messages[last_user_idx + 1]["reasoning_content"] == ""
    ):
        # Manual hack for empty reasoning content mismatch
        template_up_to_last_user = tokenizer.apply_chat_template(
            messages[: last_user_idx + 1],
            tokenize=False,
            add_generation_prompt=False,
            tools=tools,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
        template_up_to_last_user += "<|im_start|>assistant\n<think></think>"
    else:
        template_up_to_last_user = tokenizer.apply_chat_template(
            messages[: last_user_idx + 1],
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": enable_thinking},
            tools=tools,
        )

    return len(template_up_to_last_user)


def split_template_into_messages(
    messages: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    start_from_last_user: bool = True,
    enable_thinking: bool = True,
    tools: list | None = None,
) -> list[dict]:
    """Split rendered template back into individual message chunks.

    Each chunk has 'role' and 'content' fields where content is the
    raw rendered template text for that turn. This enables downstream
    loss masking based on role.

    Args:
        messages: List of OpenAI-format messages.
        tokenizer: HuggingFace tokenizer with chat_template.
        start_from_last_user: If True, first chunk includes all prior turns.
        enable_thinking: Whether thinking mode is enabled.
        tools: Optional list of tool definitions.

    Returns:
        List of chunks with 'role' and 'content' fields.

    Raises:
        ValueError: If incremental rendering doesn't match full template.

    Note:
        Exact port of materialize.py::split_template_into_messages()
    """
    # Render full template
    full_template = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )

    # Get first "message": if starting from last user, this includes all prior turns
    if start_from_last_user:
        system_end = full_template.find("<|im_end|>\n") + len("<|im_end|>\n")
        last_user_idx = max(i for i, msg in enumerate(messages) if msg["role"] == "user")
        last_user_pos = find_last_user_message_end(
            messages, tokenizer, enable_thinking=enable_thinking, tools=tools
        )
        previous_pos = last_user_pos
        # First chunk: everything up to last user message, split at system boundary
        result = [
            {"role": "system", "content": full_template[:system_end]},
            {"role": "user", "content": full_template[system_end:last_user_pos]},
        ]
        message_range = range(last_user_idx + 1, len(messages))
    else:
        previous_pos = 0
        result = []
        message_range = range(len(messages))

    for i in message_range:
        # Parallel tool calls
        if (
            i + 1 < len(messages)
            and messages[i]["role"] == "tool"
            and messages[i + 1]["role"] == "tool"
        ):
            continue

        # Render up to this message
        if (
            enable_thinking
            and messages[i]["role"] != "assistant"
            and i + 1 < len(messages)
            and (
                "reasoning_content" not in messages[i + 1]
                or messages[i + 1]["reasoning_content"] == ""
            )
        ):
            # Manual hack for empty reasoning content mismatch
            template_up_to_here = tokenizer.apply_chat_template(
                messages[: i + 1],
                tokenize=False,
                add_generation_prompt=False,
                tools=tools,
                chat_template_kwargs={"enable_thinking": enable_thinking},
            )
            template_up_to_here += "<|im_start|>assistant\n<think></think>"
        else:
            # Tool and user messages need generation prompt, others don't
            add_gen_prompt = messages[i]["role"] == "tool" or messages[i]["role"] == "user"
            template_up_to_here = tokenizer.apply_chat_template(
                messages[: i + 1],
                tokenize=False,
                add_generation_prompt=add_gen_prompt,
                tools=tools,
                chat_template_kwargs={"enable_thinking": enable_thinking},
            )

        current_pos = len(template_up_to_here)
        chunk_text = full_template[previous_pos:current_pos]

        # Verify incremental rendering matches full template
        if template_up_to_here != full_template[:current_pos]:
            raise ValueError(
                f"Template mismatch at message {i}: incremental rendering doesn't match full"
            )

        result.append({"role": messages[i]["role"], "content": chunk_text})
        previous_pos = current_pos

    return result


def create_masked_messages(
    messages: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    tools: list | None = None,
) -> list[tuple[list[dict], list[dict]]]:
    """Create message chunks for masking, optionally splitting at user turns.

    For conversations with thinking/reasoning content, splits into multiple
    sequences (one per user message). Each sequence is independently
    processable for training.

    Args:
        messages: List of OpenAI-format messages.
        tokenizer: HuggingFace tokenizer with chat_template.
        tools: Optional list of tool definitions.

    Returns:
        List of (chunks, original_messages) tuples. Each chunks list
        contains dicts with 'role' and 'content' for loss masking.

    Note:
        Exact port of materialize.py::create_masked_messages()
    """
    # Check if conversation has thinking (determines splitting strategy)
    has_thinking = any("reasoning_content" in msg and msg["reasoning_content"] for msg in messages)

    if has_thinking:
        # Split based on user messages - create chunks up to each user message
        user_idxs = [i for i, msg in enumerate(messages) if msg["role"] == "user"]
        result = []
        for i in range(len(user_idxs)):
            if i == len(user_idxs) - 1:
                # Last user message - include all remaining messages
                messages_i = messages
            else:
                # Include messages up to but not including the next user message
                messages_i = messages[: user_idxs[i + 1]]

            chunks = split_template_into_messages(
                messages_i,
                tokenizer,
                start_from_last_user=True,
                enable_thinking=has_thinking,
                tools=tools,
            )

            result.append((chunks, messages_i))  # Return both chunks and original messages
        return result
    else:
        # Generate one sequence
        chunks = split_template_into_messages(
            messages,
            tokenizer,
            start_from_last_user=False,
            enable_thinking=has_thinking,
            tools=tools,
        )
        return [(chunks, messages)]  # Return both chunks and original messages


def validate_conversation(
    messages: list[dict],
    tools: list | None = None,
) -> tuple[bool, str | None]:
    """Validate conversation for common issues.

    Checks from materialize_fast.py:
    - Tool calls present in message content but 'tools' key missing
    - <tool_call> in messages but no '# Tools' header

    Args:
        messages: List of OpenAI-format messages.
        tools: Optional list of tool definitions.

    Returns:
        Tuple of (is_valid, error_message). If is_valid is True,
        error_message is None.

    Note:
        Validation logic from materialize_fast.py
    """
    # Check if any message has <tool_call> but no message has # Tools
    any_tool_call = any(
        isinstance(m, dict)
        and (
            (isinstance(m.get("content"), str) and "<tool_call>" in m.get("content", ""))
            or (
                isinstance(m.get("reasoning_content"), str)
                and "<tool_call>" in m.get("reasoning_content", "")
            )
        )
        for m in messages
    )
    any_tools_header = any(
        isinstance(m, dict)
        and (
            (isinstance(m.get("content"), str) and "# Tools" in m.get("content", ""))
            or (
                isinstance(m.get("reasoning_content"), str)
                and "# Tools" in m.get("reasoning_content", "")
            )
        )
        for m in messages
    )
    if any_tool_call and not any_tools_header:
        return (
            False,
            "Message-level: <tool_call> present but # Tools missing",
        )

    return (True, None)


def split_system_user_chunks(chunks: list[dict]) -> list[dict]:
    """Split first chunk if it contains both system and user content.

    When start_from_last_user=True in split_template_into_messages,
    the first 'user' chunk may contain both system and user content.
    This function splits them into separate chunks.

    Args:
        chunks: List of chunks from split_template_into_messages.

    Returns:
        Processed chunks with system and user properly separated.

    Note:
        Post-processing logic from materialize_fast.py
    """
    processed = []
    for i, chunk in enumerate(chunks):
        if (
            i == 0
            and chunk["role"] == "user"
            and "<|im_start|>system" in chunk["content"]
            and "<|im_start|>user" in chunk["content"]
        ):
            # Split into separate system and user chunks
            content = chunk["content"]
            system_end = content.find("<|im_end|>\n") + len("<|im_end|>\n")
            user_start = content.find("<|im_start|>user")

            processed.append({"role": "system", "content": content[:system_end]})
            processed.append({"role": "user", "content": content[user_start:]})
        else:
            processed.append(chunk)
    return processed


__all__ = [
    "replace_json_args",
    "find_last_user_message_end",
    "split_template_into_messages",
    "create_masked_messages",
    "validate_conversation",
    "split_system_user_chunks",
]
