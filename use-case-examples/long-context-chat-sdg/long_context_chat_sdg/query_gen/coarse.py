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

"""First-level (coarse) clustering + stratified sampling — the cheap MiniLM pass.

Two-level design: MiniLM cheaply maps a large candidate set into coarse clusters, then
we draw an EQUAL number of chunks from each coarse cluster (flatten → maximise topical
diversity, beat corpus frequency bias). The resulting working set is then re-embedded
by the quality model (Nemotron-3) for the fine second-level clustering + grouping.

MiniLM is used only for the low-bar job (coarse partitioning + coverage), never for the
final grouping — so its weak topical precision never reaches the output.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable, List

from .corpus import Chunk


def coarse_stratified_sample(chunks: List[Chunk], embed_fn: Callable[[List[str]], Any], *,
                             coarse_k: int, sample_size: int, seed: int = 7) -> List[Chunk]:
    """MiniLM-embed + coarse-cluster ``chunks``, then draw EQUAL per cluster (round-robin)
    down to ``sample_size``. Returns all chunks if the candidate set is already small."""
    import numpy as np
    from ..query_prep.cluster import cluster_embeddings
    if not chunks or sample_size >= len(chunks):
        return list(chunks)
    emb = np.asarray(embed_fn([c.text for c in chunks]), dtype="float32")
    labels = cluster_embeddings(emb, algo="kmeans", k=coarse_k)

    buckets: dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        buckets[int(lab)].append(i)
    rng = random.Random(seed)
    for idxs in buckets.values():
        rng.shuffle(idxs)

    order = sorted(buckets)
    cursor = {c: 0 for c in order}
    out: List[Chunk] = []
    # round-robin one per cluster per pass -> equal representation across coarse topics
    while len(out) < sample_size and any(cursor[c] < len(buckets[c]) for c in order):
        for c in order:
            if len(out) >= sample_size:
                break
            if cursor[c] < len(buckets[c]):
                out.append(chunks[buckets[c][cursor[c]]])
                cursor[c] += 1
    return out
