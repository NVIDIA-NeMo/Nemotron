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

"""Tests for reasoning_content handling in chat template splitting.

Regression coverage for the packer dropping valid rows when an assistant turn
carried reasoning_content that the template renders as an empty <think></think>
but the consistency check treated as present (None and whitespace-only strings).
"""

import pytest

from nemotron.data_prep.core.chat_template import reasoning_renders_empty


def template_renders_empty(message: dict) -> bool:
    """Faithful model of the nano3.jinja reasoning rule, for cross-checking.

    The template keeps the reasoning only when reasoning_content is defined, is
    a string, and is non-empty after trimming. Everything else collapses to an
    empty <think></think>. reasoning_renders_empty must agree with this exactly.
    """
    val = message.get("reasoning_content")
    keeps_reasoning = (
        "reasoning_content" in message
        and isinstance(val, str)
        and len(val.strip()) > 0
    )
    return not keeps_reasoning


class TestReasoningRendersEmpty:
    @pytest.mark.parametrize(
        "message",
        [
            {},  # missing key
            {"reasoning_content": None},  # null (the reported bug)
            {"reasoning_content": ""},  # empty string (already handled before)
            {"reasoning_content": "\n\n"},  # whitespace-only (the reported bug)
            {"reasoning_content": "   "},  # spaces only
            {"reasoning_content": "\t \n"},  # mixed whitespace
            {"reasoning_content": 123},  # non-string, template treats as empty
            {"reasoning_content": ["x"]},  # non-string, template treats as empty
        ],
    )
    def test_empty_representations(self, message: dict) -> None:
        assert reasoning_renders_empty(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            {"reasoning_content": "thinking through the steps"},
            {"reasoning_content": "  padded but real  "},
            {"reasoning_content": "a"},
        ],
    )
    def test_present_representations(self, message: dict) -> None:
        assert reasoning_renders_empty(message) is False

    @pytest.mark.parametrize(
        "message",
        [
            {},
            {"reasoning_content": None},
            {"reasoning_content": ""},
            {"reasoning_content": "\n\n"},
            {"reasoning_content": "   "},
            {"reasoning_content": "\t \n"},
            {"reasoning_content": 123},
            {"reasoning_content": ["x"]},
            {"reasoning_content": "thinking through the steps"},
            {"reasoning_content": "  padded but real  "},
        ],
    )
    def test_agrees_with_template_rule(self, message: dict) -> None:
        # The whole point of the helper is to stay in lock-step with the
        # template, so it must match the template's own trim rule on every
        # representation a dataset can emit.
        assert reasoning_renders_empty(message) == template_renders_empty(message)
