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

"""Dedup the incoming queries: normalize -> exact collapse -> embedding near-dup.

Two passes:
  1. normalize() + exact collapse — cheap, removes trivial casing/spacing dups.
  2. lightweight embedding (MiniLM) + greedy cosine collapse above ``threshold`` —
     removes paraphrase near-duplicates so the sampled set stays diverse.

``embed_fn`` is injectable (tests pass a fake) so this module is unit-testable
without downloading a model.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

_WS = re.compile(r"\s+")
_PUNCT_EDGE = re.compile(r"^[\W_]+|[\W_]+$")


def normalize(q: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation."""
    s = _WS.sub(" ", (q or "").lower()).strip()
    return _PUNCT_EDGE.sub("", s)


def _as_records(queries: List[Any]) -> List[Dict[str, Any]]:
    out = []
    for q in queries:
        if isinstance(q, dict):
            out.append(dict(q))
        else:
            out.append({"query": str(q)})
    return out


def dedup(queries: List[Any], *, threshold: float = 0.92, model_name: str = "",
          embed_fn: Optional[Callable[[List[str]], Any]] = None,
          query_field: str = "query") -> List[Dict[str, Any]]:
    """Return the deduped query records (originals preserved), in input order."""
    records = _as_records(queries)
    # pass 1: exact collapse on the normalized form
    seen_norm: set = set()
    stage1: List[Dict[str, Any]] = []
    for r in records:
        norm = normalize(str(r.get(query_field, "")))
        if not norm or norm in seen_norm:
            continue
        seen_norm.add(norm)
        stage1.append(r)
    if len(stage1) <= 1 or threshold >= 1.0:
        return stage1

    # pass 2: embedding near-dup collapse (greedy: keep if far from all kept)
    import numpy as np
    if embed_fn is None:
        from .embed import embed_texts
        embed_fn = lambda ts: embed_texts(ts, model_name)  # noqa: E731
    emb = np.asarray(embed_fn([str(r.get(query_field, "")) for r in stage1]), dtype="float32")
    kept_idx: List[int] = []
    for i in range(len(stage1)):
        if not kept_idx:
            kept_idx.append(i)
            continue
        sims = emb[kept_idx] @ emb[i]           # cosine (vectors are L2-normalized)
        if float(sims.max()) < threshold:
            kept_idx.append(i)
    return [stage1[i] for i in kept_idx]
