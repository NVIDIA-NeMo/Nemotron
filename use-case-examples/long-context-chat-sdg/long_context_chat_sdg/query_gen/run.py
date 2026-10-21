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

"""Orchestrate query generation: corpus -> pool -> clusters -> units -> queries.

One entry point, ``run_query_gen``, used by pipeline.py's `query_gen` stage. The
target ``n_queries`` is a CONTRACT: we back-solve every size from it (candidates =
n_queries / expected-yield) and TOP UP in rounds — measuring the real validation
yield and provisioning the next round from it — until the target is met (or the
corpus is exhausted). LLM generation + validation run concurrently (IO-bound).
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from .corpus import reservoir_sample
from .embedder import make_embedder
from .generate import generate_query
from .sampler import build_pool, sample_units
from .sizing import plan_sizes
from .validate import is_answerable


def run_query_gen(models: Dict[str, Any], *,
                  source: str = "lancedb",
                  lancedb_cfg: Optional[dict] = None,
                  chunks_path: str = "",
                  coarse_cfg: Optional[dict] = None,     # first-level MiniLM cluster + stratified sample
                  model_alias: str = "user_model",
                  field_map: Optional[Dict[str, str]] = None,
                  n_queries: int = 400,
                  queries_per_cluster: int = 4,
                  chunks_per_cluster: Optional[int] = None,
                  pool_size: Optional[int] = None,       # None => derived per round
                  candidate_headroom: float = 1.5,       # generate headroom x n_queries (survive validation)
                  max_rounds: int = 3,                   # extra batches if still short of target
                  embed_cfg: Optional[dict] = None,
                  cluster_algo: str = "kmeans", n_clusters: Optional[int] = None,
                  min_cluster_size: int = 5,
                  kind_weights: Optional[Dict[str, int]] = None,
                  multi_hop_chunks: int = 3, cross_doc: bool = True,
                  client=None, top_k: int = 4, validate: bool = True,
                  min_coverage: float = 0.35,
                  seed: int = 7, max_workers: int = 8, dry_run: bool = False,
                  log: Callable[[str], None] = print) -> List[Dict[str, Any]]:

    # ── build a clustered pool of a given size (source-agnostic) ─────────────────
    def _make_cp(rseed: int, psize: int, nclusters: int):
        if source == "lancedb":
            from .lancedb_source import read_lancedb
            pool, emb = read_lancedb(lancedb_cfg or {}, pool_size=psize, seed=rseed)
            cp = build_pool(pool, emb=emb, algo=cluster_algo, k=nclusters, min_cluster_size=min_cluster_size)
            return cp, len(pool), "lancedb"
        if coarse_cfg and coarse_cfg.get("enabled"):
            from .coarse import coarse_stratified_sample
            cand = reservoir_sample(chunks_path, int(coarse_cfg.get("candidate_size", 10000)),
                                    seed=rseed, field_map=field_map)
            pool = coarse_stratified_sample(
                cand, make_embedder(coarse_cfg.get("embedding") or {"backend": "minilm"}),
                coarse_k=int(coarse_cfg.get("coarse_k", 64)), sample_size=psize, seed=rseed)
            return (build_pool(pool, make_embedder(embed_cfg), algo=cluster_algo, k=nclusters,
                               min_cluster_size=min_cluster_size), len(pool), "coarse")
        pool = reservoir_sample(chunks_path, psize, seed=rseed, field_map=field_map)
        return (build_pool(pool, make_embedder(embed_cfg), algo=cluster_algo, k=nclusters,
                           min_cluster_size=min_cluster_size), len(pool), "jsonl")

    # ── the per-unit worker: generate -> validate ───────────────────────────────
    from ..core.caller import make_openai_caller
    mc = models[model_alias] if models else {"model": "", "base_url": "", "api_key_env": "", "params": {}}
    caller = (make_openai_caller(mc["model"], mc["base_url"], mc.get("api_key_env", "NVIDIA_API_KEY"),
                                 params=dict(mc.get("params", {}))) if models else None)

    def _one(unit):
        # a transient LLM/retriever error drops THIS candidate, never the whole run
        try:
            row = generate_query(unit, caller)
            if not row:
                return None
            # vague ('ambiguous') queries are meant to be clarified, not to tightly retrieve a
            # source, so only require that the retriever responds (coverage gate relaxed).
            mc = 0.0 if unit.kind == "ambiguous" else min_coverage
            if validate and not is_answerable(row["query"], unit.chunks, client, top_k=top_k, min_coverage=mc):
                return None
            return row
        except Exception:
            return None

    log(f"[query_gen] target = {n_queries} validated queries "
        f"(generate ~{candidate_headroom:g}x to survive validation)")

    kept: List[Dict[str, Any]] = []
    seen: set = set()                                   # dedup queries across rounds

    for rnd in range(1, max_rounds + 1):
        if len(kept) >= n_queries:
            break
        need = n_queries - len(kept)
        # generate headroom x the shortfall; every size is derived from that count
        n_cand = need if not validate else math.ceil(need * candidate_headroom)
        plan = plan_sizes(n_cand, queries_per_cluster=queries_per_cluster,
                          chunks_per_cluster=chunks_per_cluster, n_clusters=n_clusters,
                          pool_size=pool_size)
        rseed = seed + (rnd - 1) * 1000
        cp, npool, kind = _make_cp(rseed, plan["pool_size"], plan["n_clusters"])
        nclf = len(set(cp.labels))
        units = sample_units(cp, n_cand, kind_weights=kind_weights,
                             multi_hop_chunks=multi_hop_chunks, cross_doc=cross_doc, seed=rseed)

        if dry_run:
            from collections import Counter
            sizes = sorted(Counter(cp.labels).values(), reverse=True)
            multi = [u for u in units if len(u.chunks) > 1]
            cross = sum(1 for u in multi if u.n_docs == len(u.chunks))
            log(f"[query_gen][dry-run] source={kind} pool={npool} {nclf} clusters "
                f"(sizes min/med/max={sizes[-1]}/{sizes[len(sizes) // 2]}/{sizes[0]})")
            log(f"[query_gen][dry-run] {len(units)} units; kind mix {dict(Counter(u.kind for u in units))}; "
                f"{cross}/{len(multi) or 1} multi-chunk groups fully cross-document")
            for u in units[:8]:
                snip = " ".join(u.chunks[0].text.split())[:100]
                log(f"[query_gen][dry-run]  · c{u.cluster_id:<4} {u.kind:<11} "
                    f"({len(u.chunks)} chunk / {u.n_docs} doc) {snip}…")
            return []

        log(f"[query_gen] round {rnd}: {kind} pool={npool}, {nclf} clusters, "
            f"generating {len(units)} candidates for {need} more")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = [s for s in ex.map(_one, units) if s]
        fresh = []
        for s in results:
            key = " ".join(s["query"].lower().split())
            if key and key not in seen:
                seen.add(key); fresh.append(s)
        kept.extend(fresh)
        log(f"[query_gen] round {rnd}: +{len(fresh)} new (kept {len(kept)}/{n_queries})")
        if not fresh:                                   # corpus exhausted / only duplicates
            log("[query_gen] no new queries this round — stopping short of target")
            break

    if len(kept) < n_queries:
        log(f"[query_gen] WARNING: produced {len(kept)}/{n_queries} (corpus/rounds exhausted)")
    return kept[:n_queries]
