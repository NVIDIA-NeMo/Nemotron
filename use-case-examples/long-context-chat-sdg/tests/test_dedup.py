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

import numpy as np

from long_context_chat_sdg.query_prep.dedup import dedup, normalize

_CATS = ["president", "prime", "right", "amend", "judge"]


def _fake_embed(texts):
    """One-hot by topic keyword, so same-topic paraphrases have cosine 1.0."""
    vecs = []
    for t in texts:
        tl = t.lower()
        idx = next((i for i, c in enumerate(_CATS) if c in tl), len(_CATS))
        v = np.zeros(len(_CATS) + 1, dtype="float32")
        v[idx] = 1.0
        vecs.append(v)
    return np.stack(vecs)


def test_normalize_collapses_case_and_punct():
    assert normalize("  What are the President's powers?? ") == "what are the president's powers"
    assert normalize("Hello, World!") == "hello, world"


def test_exact_normalized_dedup():
    rows = [{"query": "The President's powers"}, {"query": "the presidents powers"},
            {"query": "The President's powers"}]
    # threshold=1.0 => embedding pass skipped; only exact-normalized collapse runs
    out = dedup(rows, threshold=1.0)
    texts = [normalize(r["query"]) for r in out]
    assert len(texts) == len(set(texts))  # no exact-normalized duplicates remain


def test_embedding_near_dup_collapse():
    rows = [
        {"query": "What are the powers of the President?"},
        {"query": "Tell me about the President's authority"},   # same topic -> near-dup
        {"query": "How is the Prime Minister appointed?"},
        {"query": "Prime Minister appointment process"},        # same topic -> near-dup
        {"query": "How can the Constitution be amended?"},
    ]
    out = dedup(rows, threshold=0.9, embed_fn=_fake_embed)
    assert len(out) == 3  # one survivor per topic (president, prime, amend)
    # survivors keep the FIRST occurrence of each topic
    assert out[0]["query"].startswith("What are the powers")
