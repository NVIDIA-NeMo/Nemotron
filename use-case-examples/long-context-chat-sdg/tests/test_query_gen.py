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

"""Unit tests for query_gen — no network, no real LLM, no sentence-transformers.

Uses a tiny JSONL fixture matching the NeMo-Retriever extraction schema and fake
embed/LLM/retriever callables so the pipeline logic is exercised deterministically.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from long_context_chat_sdg.query_gen.corpus import Chunk, count_chunks, reservoir_sample, stream_chunks
from long_context_chat_sdg.query_gen.generate import generate_query
from long_context_chat_sdg.query_gen.sampler import build_pool, sample_units
from long_context_chat_sdg.query_gen.sizing import plan_sizes
from long_context_chat_sdg.query_gen.lancedb_source import rows_to_pool
from long_context_chat_sdg.query_gen.validate import coverage, is_answerable


def test_lancedb_rows_to_pool_parses_and_samples():
    # rows mimic the retriever's LanceDB schema: text, path, vector, metadata.chunk_id
    rows = [{"text": f"legal passage {i}", "path": f"case_{i % 3}.PDF",
             "vector": [float(i), 1.0], "metadata": {"chunk_id": f"ck{i}"}} for i in range(20)]
    rows += [{"text": "", "path": "x", "vector": [0, 0], "metadata": {}},   # empty text -> skipped
             {"text": "no vector", "path": "y", "metadata": {}}]            # no vector -> skipped
    chunks, emb = rows_to_pool(rows)
    assert len(chunks) == 20 and emb.shape == (20, 2)
    assert chunks[0].id == "ck0" and chunks[0].doc_id == "case_0.PDF"       # chunk_id from metadata; path is doc key
    # metadata stored as a JSON STRING (LanceDB) must still yield the distinct chunk_id
    import json as _json
    sc, _ = rows_to_pool([{"text": "x y z", "path": "d.PDF", "vector": [1.0, 0.0],
                           "metadata": _json.dumps({"chunk_id": "ckX"})}])
    assert sc[0].id == "ckX"
    # reservoir sampling is bounded + deterministic
    c1, e1 = rows_to_pool(rows, pool_size=8, seed=7)
    c2, e2 = rows_to_pool(rows, pool_size=8, seed=7)
    assert len(c1) == 8 and [c.id for c in c1] == [c.id for c in c2]


def test_plan_sizes_derives_from_n_queries():
    # primary knob only: 4000 queries at ~4/cluster -> 1000 clusters, pool fills them
    p = plan_sizes(4000, queries_per_cluster=4)
    assert p["n_clusters"] == 1000
    assert p["pool_size"] == 1000 * p["chunks_per_cluster"] >= 1000
    # explicit overrides win
    assert plan_sizes(4000, n_clusters=250)["n_clusters"] == 250
    assert plan_sizes(400, pool_size=5000)["pool_size"] == 5000
    # small ask still valid
    assert plan_sizes(20, queries_per_cluster=4)["n_clusters"] == 5


def _write_corpus(tmp: Path, n: int = 40) -> Path:
    """n chunks across 2 fake 'documents', schema-faithful to the real corpus."""
    p = tmp / "chunks.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for i in range(n):
            doc = "docA" if i % 2 == 0 else "docB"
            topic = "constitutional amendment basic structure" if i % 2 == 0 else "criminal arrest bail procedure"
            rec = {"text": f"{topic} passage number {i} with enough words to be usable content here.",
                   "chunk_id": f"chunk{i:03d}", "source_id": doc, "page_number": i,
                   "source_path": f"judgments/{1950 + i}/case_{doc}_{i}.PDF",
                   "metadata": {"has_text": True}}
            f.write(json.dumps(rec) + "\n")
        f.write("\n")                                    # blank line -> skipped
        f.write('{"text": "", "chunk_id": "empty"}\n')   # empty text -> skipped
        f.write("not json\n")                            # bad line -> skipped
    return p


def test_stream_skips_bad_and_empty(tmp_path):
    p = _write_corpus(tmp_path, 10)
    chunks = list(stream_chunks(str(p)))
    assert len(chunks) == 10                              # blanks/empty/bad dropped
    assert count_chunks(str(p)) == 10
    assert all(c.id and c.text and c.doc_id for c in chunks)
    assert chunks[0].source_path.endswith(".PDF")


def test_reservoir_sample_bounded_and_deterministic(tmp_path):
    p = _write_corpus(tmp_path, 40)
    a = reservoir_sample(str(p), 12, seed=7)
    b = reservoir_sample(str(p), 12, seed=7)
    assert len(a) == 12
    assert [c.id for c in a] == [c.id for c in b]         # deterministic
    assert reservoir_sample(str(p), 999, seed=7).__len__() == 40  # n>corpus -> all


def _fake_embed(texts):
    """2 topical clusters with small within-cluster jitter so NN ordering is defined."""
    import numpy as np
    base = np.array([[1.0, 0.0] if "basic structure" in t else [0.0, 1.0] for t in texts],
                    dtype="float32")
    jitter = np.array([[0.0, (i % 5) * 0.01] for i in range(len(texts))], dtype="float32")
    v = base + jitter
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_build_pool_and_units_span_clusters_and_group(tmp_path):
    from long_context_chat_sdg.query_gen.sampler import MULTI_CHUNK_KINDS
    p = _write_corpus(tmp_path, 40)
    pool = reservoir_sample(str(p), 40, seed=7)
    cp = build_pool(pool, _fake_embed, algo="kmeans", k=2)
    assert len(set(cp.labels)) == 2 and cp.emb is not None
    units = sample_units(cp, 12, multi_hop_chunks=3, cross_doc=True, seed=7)
    assert len(units) == 12
    assert {u.cluster_id for u in units} == {0, 1}        # both topics represented
    for u in units:
        # every group stays within ONE topical cluster (NN grouping is intra-cluster)
        assert len({c.meta["cluster"] for c in u.chunks}) == 1
        if u.kind in MULTI_CHUNK_KINDS and len(u.chunks) > 1:
            assert 2 <= len(u.chunks) <= 3
    # each chunk seeds at most one unit (no reuse across groups)
    seen = [c.id for u in units for c in u.chunks]
    assert len(seen) == len(set(seen))


def test_generate_query_parses_json():
    unit = type("U", (), {"kind": "factual",
                          "chunks": [Chunk(id="c1", text="basic structure doctrine content", doc_id="d")],
                          "cluster_id": 0})()
    caller = lambda system, user: '{"query": "What is the basic structure doctrine?"}'  # noqa: E731
    row = generate_query(unit, caller)
    assert row["query"].endswith("?")
    assert row["kind"] == "factual" and row["source_ids"] == ["c1"]


def test_validate_coverage_and_answerable():
    src = Chunk(id="c1", text="the basic structure doctrine limits parliament amendment power", doc_id="d")
    assert coverage(src.text, "text about basic structure doctrine limits parliament amendment power") > 0.5
    assert coverage(src.text, "totally unrelated cooking recipe") < 0.1

    class FakeClient:
        def __init__(self, hit): self.hit = hit
        def retrieve(self, q, k, rng=None):
            return [Chunk(id="r", text=src.text if self.hit else "unrelated content")]

    assert is_answerable("q", [src], FakeClient(True), top_k=4, min_coverage=0.35)
    assert not is_answerable("q", [src], FakeClient(False), top_k=4, min_coverage=0.35)
