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

"""Stream the on-disk chunk corpus and reservoir-sample a bounded working pool.

The retrieval corpus is a (large, multi-GB) JSONL of chunks — one record per
page/chunk carrying at least ``text`` and a chunk id, usually a document id and a
source path too. We NEVER load it fully: a single streaming pass reservoir-samples
a uniform, bounded pool that later stages embed + cluster.

Schema is not hard-coded — a ``field_map`` names the fields, so a different corpus
(different language/domain/exporter) works by config alone. Defaults match the
NeMo-Retriever extraction output ({text, chunk_id, source_id, source_path}).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

# default field names — the NeMo-Retriever extraction chunk schema
DEFAULT_FIELD_MAP: Dict[str, str] = {
    "text_field": "text",
    "id_field": "chunk_id",
    "doc_id_field": "source_id",     # same across a document's chunks -> groups by doc/case
    "path_field": "source_path",     # human-readable provenance (e.g. encodes year/case)
}


@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str = ""
    source_path: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return {"id": self.id, "doc_id": self.doc_id, "source_path": self.source_path,
                "text": self.text}


def _get(rec: Dict[str, Any], key: str, default: str = "") -> str:
    v = rec.get(key, default)
    return v if isinstance(v, str) else ("" if v is None else str(v))


def stream_chunks(path: str, field_map: Optional[Dict[str, str]] = None, *,
                  min_text_chars: int = 1) -> Iterator[Chunk]:
    """Yield ``Chunk`` per JSONL line, skipping blanks and un-parseable rows.

    Streaming: memory stays O(1) regardless of corpus size. Rows whose text is
    shorter than ``min_text_chars`` (e.g. blank OCR pages) are skipped.
    """
    fm = {**DEFAULT_FIELD_MAP, **(field_map or {})}
    tf, idf, docf, pf = fm["text_field"], fm["id_field"], fm["doc_id_field"], fm["path_field"]
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            text = _get(rec, tf)
            if len(text.strip()) < min_text_chars:
                continue
            cid = _get(rec, idf) or _get(rec, docf)     # fall back to doc id if no chunk id
            if not cid:
                continue
            yield Chunk(id=cid, text=text, doc_id=_get(rec, docf), source_path=_get(rec, pf))


def reservoir_sample(path: str, n: int, *, seed: int = 7,
                     field_map: Optional[Dict[str, str]] = None,
                     min_text_chars: int = 1) -> List[Chunk]:
    """Uniform sample of ``n`` chunks in ONE streaming pass (Algorithm R).

    Deterministic given ``seed``; holds at most ``n`` chunks in memory. A uniform
    draw means the pool approximates the corpus topic distribution, so downstream
    embedding-clusters reflect the whole corpus without embedding all of it.
    """
    if n <= 0:
        return []
    rng = random.Random(seed)
    pool: List[Chunk] = []
    for i, ch in enumerate(stream_chunks(path, field_map, min_text_chars=min_text_chars)):
        if i < n:
            pool.append(ch)
        else:
            j = rng.randint(0, i)
            if j < n:
                pool[j] = ch
    return pool


def count_chunks(path: str, field_map: Optional[Dict[str, str]] = None) -> int:
    """Total usable chunks (one streaming pass). Handy for sizing a run."""
    return sum(1 for _ in stream_chunks(path, field_map))
