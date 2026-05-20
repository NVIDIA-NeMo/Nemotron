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

"""Tests for chat_template incremental rendering.

Covers:

* Issue #184 regression: tokenizer chat templates that append an extra
  trailing newline after the generation prompt no longer cause rows to be
  filtered out (the rstrip fallback recovers them).
* Semantic preservation: rows the original ``materialize.py`` processed
  successfully produce byte-identical chunks. The rstrip fallback only fires
  when the strict prefix check fails, so well-behaved templates are
  unaffected.
* Input guards: empty / no-user / trailing-user conversations now raise
  typed ``ValueError`` instead of crashing with bare ``IndexError`` /
  ``ValueError`` from internal indexing.
"""

from __future__ import annotations

import pytest

from nemotron.data_prep.core.chat_template import (
    find_last_user_message_end,
    split_template_into_messages,
)


class _BuggyTokenizer:
    """Fake tokenizer that reproduces the issue #184 pathology.

    ``apply_chat_template`` renders messages naturally for the full template
    but, when ``add_generation_prompt=True``, appends an extra trailing
    newline after ``<|im_start|>assistant\\n``. This is the exact shape that
    causes the prefix check in ``split_template_into_messages`` to fail
    before the conditional-rstrip fallback was added.
    """

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        tools: list | None = None,
        chat_template_kwargs: dict | None = None,
    ) -> str:
        out = ""
        for m in messages:
            out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            # The bug: an extra \n that does NOT appear in the full template
            # at the corresponding position.
            out += "<|im_start|>assistant\n\n"
        return out


class _CleanTokenizer:
    """Fake tokenizer with no trailing-whitespace pathology.

    Used to confirm the conditional-rstrip fallback does NOT fire for
    well-behaved templates -- chunk boundaries must be byte-identical to
    the original implementation.
    """

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        tools: list | None = None,
        chat_template_kwargs: dict | None = None,
    ) -> str:
        out = ""
        for m in messages:
            out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
        return out


@pytest.fixture
def messages() -> list[dict]:
    return [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello"},
    ]


# =============================================================================
# Issue #184 -- trailing newline fallback
# =============================================================================


class TestIssue184TrailingNewline:
    """Regression tests for issue #184."""

    def test_buggy_template_does_not_filter_row(self, messages: list[dict]) -> None:
        """Trailing-newline pathology must NOT raise after the fallback.

        Before the fix this raised ``ValueError: Template mismatch at
        message 0`` and the whole conversation was dropped from the SFT
        dataset.
        """
        chunks = split_template_into_messages(
            messages,
            _BuggyTokenizer(),
            start_from_last_user=False,
            enable_thinking=False,
        )

        assert [c["role"] for c in chunks] == ["user", "assistant"]

    def test_buggy_template_chunks_reconstruct_full(self, messages: list[dict]) -> None:
        """Recovered chunks must concatenate back to the full template.

        Conditional rstrip means the fallback fires on exactly one
        iteration (the buggy boundary). The next iteration's strict check
        passes, so ``current_pos`` lands on the natural end of the full
        template -- joined chunks reconstruct ``full`` exactly, with no
        trailing-whitespace drift.
        """
        tokenizer = _BuggyTokenizer()
        full = tokenizer.apply_chat_template(messages, add_generation_prompt=False)

        chunks = split_template_into_messages(
            messages,
            tokenizer,
            start_from_last_user=False,
            enable_thinking=False,
        )

        assert "".join(c["content"] for c in chunks) == full


# =============================================================================
# Semantic preservation -- byte-identical output for clean templates
# =============================================================================


class TestSemanticPreservation:
    """The fallback must NOT fire for templates the original handled cleanly."""

    def test_clean_template_chunks_reconstruct_full_exactly(
        self,
        messages: list[dict],
    ) -> None:
        """Joined chunks must equal ``full`` byte-for-byte (no rstrip drift)."""
        tokenizer = _CleanTokenizer()
        full = tokenizer.apply_chat_template(messages, add_generation_prompt=False)

        chunks = split_template_into_messages(
            messages,
            tokenizer,
            start_from_last_user=False,
            enable_thinking=False,
        )

        assert [c["role"] for c in chunks] == ["user", "assistant"]
        assert "".join(c["content"] for c in chunks) == full

    def test_clean_template_preserves_trailing_newline(
        self,
        messages: list[dict],
    ) -> None:
        """The final chunk's content must keep the natural trailing newline.

        Regression guard for the previous unconditional-rstrip implementation,
        which dropped the trailing ``\\n`` of the last chunk on every row.
        """
        tokenizer = _CleanTokenizer()
        chunks = split_template_into_messages(
            messages,
            tokenizer,
            start_from_last_user=False,
            enable_thinking=False,
        )

        assert chunks[-1]["content"].endswith("<|im_end|>\n")


# =============================================================================
# Input guards -- typed errors instead of bare IndexError / ValueError
# =============================================================================


class TestInputGuards:
    """Edge cases that previously crashed must now raise informative errors."""

    def test_no_user_message_raises_value_error(self) -> None:
        """A conversation with no user turn must raise a clear ``ValueError``.

        The original implementation crashed with the opaque
        ``ValueError: max() iterable argument is empty``.
        """
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "Hi"},
        ]
        with pytest.raises(ValueError, match="no user message"):
            find_last_user_message_end(
                messages,
                _CleanTokenizer(),
                enable_thinking=False,
            )

    def test_trailing_user_message_raises_value_error(self) -> None:
        """A conversation ending with a user turn must raise a clear error.

        The original implementation crashed with ``IndexError`` while
        accessing ``messages[last_user_idx + 1]``.
        """
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        with pytest.raises(ValueError, match="ends with a user message"):
            find_last_user_message_end(
                messages,
                _CleanTokenizer(),
                enable_thinking=False,
            )

    def test_split_template_propagates_no_user_guard(self) -> None:
        """``split_template_into_messages`` should also surface the guard.

        The ``start_from_last_user=True`` path delegates to
        ``find_last_user_message_end``; the typed error must propagate.
        """
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "Hi"},
        ]
        with pytest.raises(ValueError, match="no user message"):
            split_template_into_messages(
                messages,
                _CleanTokenizer(),
                start_from_last_user=True,
                enable_thinking=False,
            )
