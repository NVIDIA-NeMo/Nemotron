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

"""Inline judges — the embedded quality gate.

Judges return a ``<explanation>…</explanation><rating>success|failure</rating>``
verdict. ``run_inline_judge`` calls the judge alias, parses the verdict, and
retries once with a reformat nudge if the first response is unparseable.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

from ..core.llm import call_llm
from .prompts import JUDGE_REFORMAT_PROMPT

_EXPL = re.compile(r"<explanation>(.*?)</explanation>", re.DOTALL | re.IGNORECASE)
_RATE = re.compile(r"<rating>\s*(success|failure)\s*</rating>", re.IGNORECASE)


def parse_judge_response(text: str) -> Tuple[str, str, bool]:
    expl = _EXPL.search(text or "")
    rate = _RATE.search(text or "")
    explanation = expl.group(1).strip() if expl else ""
    rating = rate.group(1).lower() if rate else ""
    return explanation, rating, rating == "success"


def run_inline_judge(models: Dict[str, Any], alias: str, prompt_text: str) -> Dict[str, Any]:
    """Returns {'explanation', 'rating', 'success', 'parsed'}."""
    text = _content(call_llm(models, alias, [{"role": "user", "content": prompt_text}]))
    explanation, rating, success = parse_judge_response(text)
    if not rating:  # one reformat retry
        text = _content(call_llm(models, alias, [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": text},
            {"role": "user", "content": JUDGE_REFORMAT_PROMPT}]))
        explanation, rating, success = parse_judge_response(text)
    return {"explanation": explanation, "rating": rating or "failure",
            "success": success, "parsed": bool(rating)}


def _content(resp: Any) -> str:
    if isinstance(resp, dict):
        return resp.get("content", "") or ""
    return str(resp or "")
