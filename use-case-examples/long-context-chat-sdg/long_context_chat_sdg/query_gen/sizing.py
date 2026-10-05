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

"""Derive cluster count + pool size from the ONE knob that matters: n_queries.

You state how many queries you want and roughly how many to draw per topic cluster;
everything else follows. Any of the derived numbers can be overridden explicitly.

    n_clusters  = ceil(n_queries / queries_per_cluster)
    pool_size   = n_clusters * chunks_per_cluster     (reservoir caps at corpus size)

``chunks_per_cluster`` gives each cluster enough distinct chunks to (a) exist and
(b) supply multi-chunk (multi-hop/comparative) bundles, with headroom for the
overgenerate-then-validate drop. Default ~4x queries_per_cluster covers the ~1.9
avg chunks/query of the default kind mix plus validation overgen.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


def plan_sizes(n_queries: int, *, queries_per_cluster: int = 4,
               chunks_per_cluster: Optional[int] = None,
               n_clusters: Optional[int] = None,
               pool_size: Optional[int] = None) -> Dict[str, Any]:
    """Return {n_clusters, pool_size, queries_per_cluster, chunks_per_cluster}.

    Explicit ``n_clusters`` / ``pool_size`` win over the derivation.
    """
    n_queries = max(1, int(n_queries))
    qpc = max(1, int(queries_per_cluster))
    k = int(n_clusters) if n_clusters else max(1, math.ceil(n_queries / qpc))
    cpc = int(chunks_per_cluster) if chunks_per_cluster else max(qpc + 1, qpc * 4)
    pool = int(pool_size) if pool_size else k * cpc
    return {"n_clusters": k, "pool_size": pool, "queries_per_cluster": qpc,
            "chunks_per_cluster": cpc}
