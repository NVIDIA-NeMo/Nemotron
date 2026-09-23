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

"""Pluggable text embedder — used when clustering from raw JSONL (source: jsonl).

The other source (lancedb) reads PRECOMPUTED vectors and skips embedding entirely.
When we DO embed, prefer a HOSTED embedding model:

  - ``cohere`` : hosted Cohere-style /v2/embed endpoint (Nemotron-3-Embed on Lepton).
                 Reachable anywhere; the default hosted embedder.
  - ``nim``    : the retriever's own OpenAI-style /v1/embeddings (llama-nemotron-embed-1b-v2
                 at :8001, input_type=passage) — the SAME space the index uses (host-local).
  - ``minilm`` : local sentence-transformers fallback (256-tok, different space) — only
                 for a corpus with no hosted embedder available.

All return L2-normalized float32 vectors so the cluster backends work unchanged.
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional


def _l2_normalize(mat):
    import numpy as np
    a = np.asarray(mat, dtype="float32")
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (a / norms).astype("float32")


def minilm_embedder(model_name: str = "") -> Callable[[List[str]], Any]:
    from ..query_prep.embed import embed_texts  # MiniLM, already L2-normalized
    return lambda texts: embed_texts(texts, model_name)


def nim_embedder(model: str, endpoint: str, api_key_env: str = "", *, input_type: str = "passage",
                 batch_size: int = 64, timeout: int = 120) -> Callable[[List[str]], Any]:
    """OpenAI-style /v1/embeddings client (NVIDIA embedding NIM). ``endpoint`` is the base
    (…/v1). ``input_type`` is the NIM passage/query hint used by the retriever's ingester."""
    import numpy as np
    import requests

    url = endpoint.rstrip("/") + "/embeddings"
    key = os.environ.get(api_key_env, "") if api_key_env else ""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    def embed(texts: List[str]):
        vecs: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = [t or " " for t in texts[i:i + batch_size]]
            payload: dict = {"model": model, "input": batch}
            if input_type:
                payload["input_type"] = input_type
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
            vecs.extend(d["embedding"] for d in data)
        return _l2_normalize(np.asarray(vecs, dtype="float32"))

    return embed


def cohere_embedder(endpoint: str, *, input_type: str = "document", api_key_env: str = "",
                    batch_size: int = 64, timeout: int = 120,
                    retries: int = 3) -> Callable[[List[str]], Any]:
    """Hosted Cohere-style /v2/embed endpoint (e.g. Nemotron-3-Embed on Lepton/vLLM).

    Request  {"texts": [...], "input_type": "document"|"query"}
    Response {"embeddings": {"float": [[...]]}}   (already L2-normalized)
    Use ``document`` for chunks (== the model's "passage: " prefix).
    """
    import time
    import numpy as np
    import requests

    key = os.environ.get(api_key_env, "") if api_key_env else ""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    def embed(texts: List[str]):
        vecs: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = [t or " " for t in texts[i:i + batch_size]]
            for attempt in range(retries):
                try:
                    resp = requests.post(endpoint, json={"texts": batch, "input_type": input_type},
                                         headers=headers, timeout=timeout)
                    resp.raise_for_status()
                    vecs.extend(resp.json()["embeddings"]["float"])
                    break
                except requests.RequestException:
                    if attempt == retries - 1:
                        raise
                    time.sleep(2 * (attempt + 1))
        return _l2_normalize(np.asarray(vecs, dtype="float32"))

    return embed


def make_embedder(cfg: Optional[dict] = None) -> Callable[[List[str]], Any]:
    """Resolve an embedder from a config dict (only used for source: jsonl).

    cfg keys: backend ("cohere"|"nim"|"minilm"), endpoint, model, api_key_env,
    input_type, batch_size. Defaults to local MiniLM only if nothing is configured.
    """
    cfg = cfg or {}
    backend = (cfg.get("backend") or "minilm").lower()
    if backend in ("cohere", "v2embed"):
        return cohere_embedder(cfg["endpoint"], input_type=cfg.get("input_type", "document"),
                               api_key_env=cfg.get("api_key_env", ""),
                               batch_size=int(cfg.get("batch_size", 64)))
    if backend == "nim":
        return nim_embedder(cfg.get("model", ""), cfg["endpoint"], cfg.get("api_key_env", ""),
                            input_type=cfg.get("input_type", "passage"),
                            batch_size=int(cfg.get("batch_size", 64)))
    return minilm_embedder(cfg.get("model", ""))
