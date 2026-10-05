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

"""Read precomputed chunk embeddings from a user-provided LanceDB index.

The NeMo Retriever ingester already embedded every chunk (llama-nemotron-embed-1b-v2,
2048-dim, input_type=passage) and stored {text, path, metadata.chunk_id, vector} in
LanceDB. Reading those vectors clusters in the EXACT retrieval space at ZERO embedding
cost — no MiniLM, no endpoint calls. The document key is ``path`` (deterministic, shared
across a case's chunks); ``chunk_id`` lives inside the ``metadata`` struct.

The row-parsing core (``rows_to_pool``) is pure and testable with plain dicts; only
``read_lancedb`` touches the ``lancedb`` package (imported lazily).
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .corpus import Chunk

# Default row schema; field access remains tolerant of common LanceDB layouts.
DEFAULT_LANCEDB_FIELDS: Dict[str, str] = {
    "text_field": "text",
    "vector_field": "vector",
    "path_field": "path",          # deterministic document key
    "id_field": "chunk_id",        # nested inside the `metadata` struct
    "meta_field": "metadata",
}


def _row_to_chunk(row: Dict[str, Any], fm: Dict[str, str]) -> Tuple[Optional[Chunk], Any]:
    text = row.get(fm["text_field"]) or ""
    if not str(text).strip():
        return None, None
    vec = row.get(fm["vector_field"])
    if vec is None:
        return None, None
    path = str(row.get(fm["path_field"]) or "")
    meta = row.get(fm["meta_field"]) or {}
    if isinstance(meta, str):                          # LanceDB may store metadata as JSON text
        try:
            meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            meta = {}
    cid = (meta.get(fm["id_field"]) if isinstance(meta, dict) else None) or row.get(fm["id_field"]) or path
    return Chunk(id=str(cid), text=str(text), doc_id=path, source_path=path), vec


def rows_to_pool(rows: Iterable[Dict[str, Any]], field_map: Optional[Dict[str, str]] = None,
                 *, pool_size: int = 0, seed: int = 7) -> Tuple[List[Chunk], Any]:
    """(reservoir-)sample rows into (chunks, vectors). ``pool_size<=0`` keeps ALL rows.

    Deterministic given ``seed``. Rows without text or a vector are skipped.
    """
    import numpy as np
    fm = {**DEFAULT_LANCEDB_FIELDS, **(field_map or {})}
    rng = random.Random(seed)
    chunks: List[Chunk] = []
    vecs: List[Any] = []
    n_seen = 0
    for row in rows:
        ch, vec = _row_to_chunk(row, fm)
        if ch is None:
            continue
        if pool_size and pool_size > 0:
            if len(chunks) < pool_size:
                chunks.append(ch); vecs.append(vec)
            else:
                j = rng.randint(0, n_seen)
                if j < pool_size:
                    chunks[j] = ch; vecs[j] = vec
            n_seen += 1
        else:
            chunks.append(ch); vecs.append(vec)
    return chunks, np.asarray(vecs, dtype="float32")


def read_lancedb(cfg: Dict[str, Any], *, pool_size: int = 0, seed: int = 7,
                 batch_size: int = 8192) -> Tuple[List[Chunk], Any]:
    """Stream the LanceDB table and return (chunks, vectors). Bounded memory via batches.

    cfg keys: uri, table (required) + optional field overrides (text_field, vector_field,
    path_field, id_field, meta_field).
    """
    import lancedb
    tbl = lancedb.connect(cfg["uri"]).open_table(cfg["table"])

    def _rows() -> Iterable[Dict[str, Any]]:
        # pylance dataset scanner streams in batches (stable across lancedb versions)
        for batch in tbl.to_lance().to_batches(batch_size=batch_size):
            for r in batch.to_pylist():
                yield r

    return rows_to_pool(_rows(), cfg, pool_size=pool_size, seed=seed)
