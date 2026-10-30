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

"""Regression tests for consecutive same-role turns in chat-template splitting.

NVIDIA-NeMo/Megatron-Bridge#5080 (§3): conversations containing consecutive
`user` turns crashed incremental rendering ("Template mismatch at message N")
and the rows were silently dropped. The splitter must handle such rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from nemotron.data_prep.core.chat_template import (
    create_masked_messages,
    split_template_into_messages,
)

TEMPLATES_DIR = Path(__file__).parents[2] / "src" / "nemotron" / "data_prep" / "templates"


class FakeChatTokenizer:
    """Minimal stand-in for a HF tokenizer's chat-template rendering."""

    def __init__(self, chat_template: str):
        self.chat_template = chat_template
        self._env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        self._env.filters["tojson"] = lambda value, **kw: json.dumps(value, **kw)
        self._env.globals["raise_exception"] = _raise_template_exception

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        chat_template_kwargs=None,
        **kwargs,
    ):
        assert tokenize is False
        context = {
            "messages": messages,
            "tools": tools,
            "add_generation_prompt": add_generation_prompt,
            "chat_template_kwargs": chat_template_kwargs or {},
            **(chat_template_kwargs or {}),
            **kwargs,
        }
        return self._env.from_string(self.chat_template).render(**context)


def _raise_template_exception(message):
    raise ValueError(message)


@pytest.fixture(params=["nano3", "lightning35"])
def tokenizer(request):
    template = (TEMPLATES_DIR / f"{request.param}.jinja").read_text()
    return FakeChatTokenizer(template)


# The representative record from the issue: consecutive `user` turns.
CONSECUTIVE_USER_MESSAGES = [
    {"role": "system", "content": "You are a shopping-intent assistant."},
    {"role": "user", "content": '{"query": "wireless earbuds", "selected_option": ""}'},
    {"role": "user", "content": "Please recommend wireless earbuds."},
    {"role": "assistant", "content": "The user wants to buy wireless earbuds."},
]


class TestConsecutiveUserTurns:
    def test_split_no_thinking_succeeds(self, tokenizer):
        """The exact issue repro must split cleanly instead of raising."""
        chunks = split_template_into_messages(
            CONSECUTIVE_USER_MESSAGES,
            tokenizer,
            start_from_last_user=False,
            enable_thinking=False,
        )
        assert [c["role"] for c in chunks] == ["system", "user", "user", "assistant"]

    def test_split_reconstructs_full_template(self, tokenizer):
        """Concatenated chunks must equal the full rendered conversation."""
        full = tokenizer.apply_chat_template(
            CONSECUTIVE_USER_MESSAGES,
            tokenize=False,
            add_generation_prompt=False,
            chat_template_kwargs={"enable_thinking": False},
            enable_thinking=False,
        )
        chunks = split_template_into_messages(
            CONSECUTIVE_USER_MESSAGES,
            tokenizer,
            start_from_last_user=False,
            enable_thinking=False,
        )
        assert "".join(c["content"] for c in chunks) == full

    def test_create_masked_messages_no_thinking(self, tokenizer):
        results = create_masked_messages(CONSECUTIVE_USER_MESSAGES, tokenizer)
        assert len(results) == 1
        chunks, original = results[0]
        assert original == CONSECUTIVE_USER_MESSAGES
        assert any(c["role"] == "assistant" for c in chunks)

    def test_create_masked_messages_with_thinking(self, tokenizer):
        """Thinking path: untrainable sub-sequences are skipped, not fatal."""
        messages = [
            {"role": "system", "content": "You are a shopping-intent assistant."},
            {"role": "user", "content": "first user turn"},
            {"role": "user", "content": "second user turn"},
            {
                "role": "assistant",
                "content": "Answer.",
                "reasoning_content": "Consider the request.",
            },
        ]
        results = create_masked_messages(messages, tokenizer)
        # One sub-sequence: the first user turn has no assistant continuation
        # (the next message is another user turn) and is skipped.
        assert len(results) == 1
        chunks, _ = results[0]
        assert any(c["role"] == "assistant" for c in chunks)

    def test_trailing_user_turn_does_not_crash(self, tokenizer):
        """A conversation ending on a user turn must not raise."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "dangling follow-up"},
        ]
        chunks = split_template_into_messages(
            messages,
            tokenizer,
            start_from_last_user=False,
            enable_thinking=False,
        )
        assert [c["role"] for c in chunks] == ["system", "user", "assistant", "user"]


class TestRegularConversationsUnchanged:
    def test_simple_conversation_roles(self, tokenizer):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        chunks = split_template_into_messages(
            messages,
            tokenizer,
            start_from_last_user=False,
            enable_thinking=False,
        )
        assert [c["role"] for c in chunks] == [
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        full = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            chat_template_kwargs={"enable_thinking": False},
            enable_thinking=False,
        )
        assert "".join(c["content"] for c in chunks) == full
