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

"""Tests for transform factories."""

import pytest

from nemotron.data_prep.formats.transforms import (
    nemotron_rl,
    openai_chat,
    passthrough,
    rename,
    select,
    sft,
    sharegpt,
)


# =============================================================================
# sft()
# =============================================================================


class TestSftTransform:
    def test_default_fields(self) -> None:
        t = sft()
        result = t({"input": "Hello", "output": "Hi"})
        assert result == {"input": "Hello", "output": "Hi"}

    def test_custom_fields(self) -> None:
        t = sft(input="question", output="answer")
        result = t({"question": "What?", "answer": "This."})
        assert result == {"input": "What?", "output": "This."}

    def test_with_system(self) -> None:
        t = sft(system="system")
        result = t({"input": "Hi", "output": "Hello", "system": "Be polite"})
        assert result == {"input": "Hi", "output": "Hello", "system": "Be polite"}

    def test_system_field_missing(self) -> None:
        t = sft(system="system")
        result = t({"input": "Hi", "output": "Hello"})
        assert result == {"input": "Hi", "output": "Hello"}

    def test_missing_required_field(self) -> None:
        t = sft()
        result = t({"input": "Hello"})  # missing "output"
        assert result is None

    def test_empty_record(self) -> None:
        t = sft()
        assert t({}) is None


# =============================================================================
# openai_chat()
# =============================================================================


class TestOpenaiChatTransform:
    def test_default_field(self) -> None:
        t = openai_chat()
        messages = [{"role": "user", "content": "Hi"}]
        result = t({"messages": messages})
        assert result == {"messages": messages}

    def test_custom_field(self) -> None:
        t = openai_chat(messages="conversation")
        msgs = [{"role": "user", "content": "Hi"}]
        result = t({"conversation": msgs})
        assert result == {"messages": msgs}

    def test_missing_field(self) -> None:
        t = openai_chat()
        assert t({"other": "data"}) is None


# =============================================================================
# sharegpt()
# =============================================================================


class TestSharegptTransform:
    def test_default_field(self) -> None:
        t = sharegpt()
        convos = [{"from": "human", "value": "Hi"}]
        result = t({"conversations": convos})
        assert result == {"conversations": convos}

    def test_custom_field(self) -> None:
        t = sharegpt(conversations="turns")
        convos = [{"from": "human", "value": "Hi"}]
        result = t({"turns": convos})
        assert result == {"conversations": convos}

    def test_missing_field(self) -> None:
        t = sharegpt()
        assert t({"other": "data"}) is None


# =============================================================================
# nemotron_rl()
# =============================================================================


class TestNemotronRlTransform:
    def test_basic(self) -> None:
        t = nemotron_rl()
        record = {
            "responses_create_params": {
                "input": [{"role": "user", "content": "Hi"}],
            }
        }
        result = t(record)
        assert result == {"messages": [{"role": "user", "content": "Hi"}]}

    def test_with_tools(self) -> None:
        t = nemotron_rl()
        record = {
            "responses_create_params": {
                "input": [{"role": "user", "content": "Search"}],
                "tools": [{"name": "search"}],
            }
        }
        result = t(record)
        assert result["messages"] == [{"role": "user", "content": "Search"}]
        assert result["tools"] == [{"name": "search"}]

    def test_empty_tools_excluded(self) -> None:
        t = nemotron_rl()
        record = {
            "responses_create_params": {
                "input": [{"role": "user", "content": "Hi"}],
                "tools": [],
            }
        }
        result = t(record)
        assert "tools" not in result

    def test_missing_params(self) -> None:
        t = nemotron_rl()
        assert t({"other": "data"}) is None

    def test_missing_input(self) -> None:
        t = nemotron_rl()
        assert t({"responses_create_params": {}}) is None


# =============================================================================
# passthrough()
# =============================================================================


class TestPassthroughTransform:
    def test_returns_same(self) -> None:
        t = passthrough()
        record = {"a": 1, "b": "two"}
        assert t(record) == record

    def test_identity(self) -> None:
        t = passthrough()
        record = {"x": [1, 2, 3]}
        assert t(record) is record


# =============================================================================
# select()
# =============================================================================


class TestSelectTransform:
    def test_select_fields(self) -> None:
        t = select("id", "text")
        result = t({"id": 1, "text": "hello", "extra": "ignored"})
        assert result == {"id": 1, "text": "hello"}

    def test_missing_field_returns_none(self) -> None:
        t = select("id", "missing")
        assert t({"id": 1}) is None

    def test_single_field(self) -> None:
        t = select("name")
        assert t({"name": "test", "other": "x"}) == {"name": "test"}


# =============================================================================
# rename()
# =============================================================================


class TestRenameTransform:
    def test_rename_fields(self) -> None:
        t = rename(input="question", output="answer")
        result = t({"question": "What?", "answer": "This."})
        assert result == {"input": "What?", "output": "This."}

    def test_missing_source_field(self) -> None:
        t = rename(input="question")
        assert t({"other": "data"}) is None

    def test_single_rename(self) -> None:
        t = rename(text="content")
        assert t({"content": "hello"}) == {"text": "hello"}
