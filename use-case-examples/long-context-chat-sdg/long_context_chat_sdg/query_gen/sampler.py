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

"""Turn a sampled chunk pool into diverse GENERATION UNITS (kind × chunk GROUP).

The generation unit is a GROUP of tightly-related chunks, so the query must
SYNTHESIZE across them (hard, multi-hop) rather than restate one chunk. Diversity
is engineered, not hoped for:
  - Axis 1 (topic): embed the pool -> cluster -> round-robin ACROSS clusters, so
    units span topics instead of clumping in the biggest one.
  - Axis 2 (kind): each unit draws a KIND from a weighted distribution (aligned
    with the conversation planner's kinds).

Group formation is nearest-neighbour, NOT random-from-cluster: for a seed chunk we
take its closest neighbours IN EMBEDDING SPACE (tight => a natural question), and
prefer neighbours from DIFFERENT documents (cross-doc => the agent must chain
several searches to gather them => genuine multi-hop). Single-chunk kinds
(factual/ambiguous) keep the easy end for a difficulty spread.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .corpus import Chunk

# kinds whose question must span a GROUP of related chunks (the hard ones)
MULTI_CHUNK_KINDS = {"multi_hop", "comparative", "exploratory"}
DEFAULT_KIND_WEIGHTS: Dict[str, int] = {
    "multi_hop": 4, "comparative": 3, "exploratory": 2, "factual": 2, "ambiguous": 2,  # ~15% vague
}


@dataclass
class GenUnit:
    kind: str
    chunks: List[Chunk]

    @property
    def cluster_id(self) -> int:
        return int(self.chunks[0].meta.get("cluster", -1)) if self.chunks else -1

    @property
    def n_docs(self) -> int:
        return len({c.doc_id for c in self.chunks})


@dataclass
class ClusteredPool:
    chunks: List[Chunk]
    labels: List[int]
    emb: Any = None                    # np.ndarray [n, d] aligned with chunks (for NN grouping)

    def cluster_members(self) -> Dict[int, List[int]]:
        """cluster id -> list of GLOBAL chunk indices (also tags chunk.meta['cluster'])."""
        members: Dict[int, List[int]] = defaultdict(list)
        for i, lab in enumerate(self.labels):
            self.chunks[i].meta["cluster"] = int(lab)
            members[int(lab)].append(i)
        return members


def build_pool(chunks: List[Chunk], embed_fn: Optional[Callable[[List[str]], Any]] = None, *,
               emb: Any = None, algo: str = "kmeans", k: Optional[int] = None,
               min_cluster_size: int = 5) -> ClusteredPool:
    """Cluster the pool into topical groups (keeps embeddings for NN grouping).

    Pass ``emb`` to cluster PRECOMPUTED vectors (LanceDB source — no embedding), or
    ``embed_fn`` to embed the chunk text first (raw-JSONL source).
    """
    import numpy as np
    from ..query_prep.cluster import cluster_embeddings
    if not chunks:
        return ClusteredPool([], [], None)
    if emb is None:
        emb = embed_fn([c.text for c in chunks])
    emb = np.asarray(emb, dtype="float32")
    labels = cluster_embeddings(emb, algo=algo, k=k, min_cluster_size=min_cluster_size)
    return ClusteredPool(chunks, labels, emb)


def _weighted_kind(rng: random.Random, weights: Dict[str, int]) -> str:
    kinds, w = zip(*weights.items())
    return rng.choices(list(kinds), weights=list(w), k=1)[0]


def _nn_group(seed: int, members: List[int], emb, chunks: List[Chunk], size: int,
              cross_doc: bool) -> List[int]:
    """Seed chunk + its nearest neighbours within the cluster (cosine, emb is L2-normed).

    Prefer neighbours from other documents (cross_doc) so the group forces multi-hop;
    backfill with the next-nearest regardless of document if that leaves it short.
    """
    if len(members) <= 1 or size <= 1:
        return [seed]
    import numpy as np
    others = [m for m in members if m != seed]
    sims = emb[others] @ emb[seed]
    ranked = [others[j] for j in np.argsort(-sims)]        # nearest first
    group, docs = [seed], {chunks[seed].doc_id}
    for m in ranked:                                        # cross-doc pass
        if cross_doc and chunks[m].doc_id in docs:
            continue
        group.append(m); docs.add(chunks[m].doc_id)
        if len(group) >= size:
            return group
    for m in ranked:                                       # backfill (any doc)
        if m not in group:
            group.append(m)
            if len(group) >= size:
                break
    return group


def sample_units(pool: ClusteredPool, n_units: int, *,
                 kind_weights: Optional[Dict[str, int]] = None,
                 multi_hop_chunks: int = 3, cross_doc: bool = True,
                 seed: int = 7) -> List[GenUnit]:
    """Emit ``n_units`` GenUnits, round-robin across clusters for topic spread.

    Multi-chunk kinds build a nearest-neighbour GROUP (``multi_hop_chunks`` chunks,
    cross-document when possible); single-chunk kinds take just the seed. Each chunk
    seeds at most one unit.
    """
    weights = kind_weights or DEFAULT_KIND_WEIGHTS
    members = pool.cluster_members()
    if not members or pool.emb is None:
        return []
    rng = random.Random(seed)
    queues = {c: rng.sample(idxs, len(idxs)) for c, idxs in members.items()}  # seed order per cluster
    order = sorted(queues, key=lambda c: (-len(members[c]), c))               # big clusters first
    used: set = set()

    units: List[GenUnit] = []
    ci = guard = 0
    while len(units) < n_units and guard < n_units * 30:
        guard += 1
        c = order[ci % len(order)]
        ci += 1
        q = queues[c]
        seed_i = None
        while q:
            cand = q.pop()
            if cand not in used:
                seed_i = cand
                break
        if seed_i is None:
            continue
        kind = _weighted_kind(rng, weights)
        avail = [m for m in members[c] if m not in used or m == seed_i]
        if kind in MULTI_CHUNK_KINDS and len(avail) >= 2:
            grp = _nn_group(seed_i, avail, pool.emb, pool.chunks,
                            size=min(multi_hop_chunks, len(avail)), cross_doc=cross_doc)
        else:
            if kind in MULTI_CHUNK_KINDS:
                kind = "factual"                            # cluster too small to relate
            grp = [seed_i]
        used.update(grp)
        units.append(GenUnit(kind, [pool.chunks[i] for i in grp]))
    return units
