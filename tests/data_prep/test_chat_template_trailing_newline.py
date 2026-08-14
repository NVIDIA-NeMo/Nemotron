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

"""Regression test for issue #184.

Some chat templates append a trailing newline after the generation prompt that
is absent at the matching position in the full template. The strict prefix check
in ``split_template_into_messages`` then failed and the row was dropped from the
SFT dataset. The splitter must recover such rows without shifting chunk
boundaries on templates that already line up.
"""

from __future__ import annotations

from nemotron.data_prep.core.chat_template import split_template_into_messages

MESSAGES = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "Hello"},
]


class _Tokenizer:
    """Renders a full template cleanly but adds an extra newline after the
    generation prompt, reproducing the issue #184 pathology when ``extra`` is set."""

    def __init__(self, extra: str):
        self._extra = extra

    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            out += "<|im_start|>assistant\n" + self._extra
        return out


def test_trailing_newline_row_is_kept():
    tokenizer = _Tokenizer(extra="\n")
    full = tokenizer.apply_chat_template(MESSAGES)
    chunks = split_template_into_messages(MESSAGES, tokenizer, start_from_last_user=False, enable_thinking=False)
    assert [c["role"] for c in chunks] == ["user", "assistant"]
    assert "".join(c["content"] for c in chunks) == full


def test_clean_template_boundaries_unchanged():
    tokenizer = _Tokenizer(extra="")
    full = tokenizer.apply_chat_template(MESSAGES)
    chunks = split_template_into_messages(MESSAGES, tokenizer, start_from_last_user=False, enable_thinking=False)
    assert "".join(c["content"] for c in chunks) == full
    assert chunks[-1]["content"].endswith("<|im_end|>\n")
