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

"""Answerability check: does the generated query actually retrieve its source?

We can't match ids (the retrieval service re-ids chunks by content hash), so we
verify by TEXT: run the query through the live retriever and require that one of
the source passages is well-covered by the returned chunks. Char-n-gram overlap
keeps this language/domain-agnostic (no tokenizer, no regex, no stopwords).
"""

from __future__ import annotations

import random
from typing import List

from .corpus import Chunk


def _char_ngrams(text: str, n: int = 12) -> set:
    t = " ".join((text or "").split()).lower()
    return {t[i:i + n] for i in range(len(t) - n + 1)} if len(t) >= n else set()


def coverage(source_text: str, retrieved_text: str, n: int = 12) -> float:
    """Fraction of the SOURCE's char-n-grams present in the retrieved text."""
    src = _char_ngrams(source_text, n)
    if not src:
        return 0.0
    return len(src & _char_ngrams(retrieved_text, n)) / len(src)


def is_answerable(query: str, sources: List[Chunk], client, *, top_k: int,
                  min_coverage: float = 0.35, rng=None) -> bool:
    """True if the query retrieves text that covers at least one source passage.

    ``client`` is an HttpRetrievalClient; ``sources`` are the chunks the query was
    generated from. A source counts as retrieved if the concatenated retrieved text
    covers >= ``min_coverage`` of that source's n-grams.
    """
    if not query or not sources or client is None:
        return False
    chunks = client.retrieve(query, top_k, rng=rng or random.Random(0))
    hay = "\n".join(c.text for c in chunks)
    return any(coverage(s.text, hay) >= min_coverage for s in sources)
