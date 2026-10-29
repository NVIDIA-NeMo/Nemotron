# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Reproducible ordering and sampling.

Python's built-in :func:`hash` is salted per process through ``PYTHONHASHSEED``,
so anything derived from it changes between runs and between workers. Every
ordering here goes through SHA-256 instead, which costs more and is worth it:
a profile that reports different quantiles on a rerun cannot be used to argue
about a threshold.

Sampling is hash-bottom-k. Taking the first *n* documents biases toward however
the shards happen to be ordered — usually crawl order, which correlates with
almost everything. Keeping the documents whose hash falls lowest is a
deterministic random subset that does not care about file order, worker count,
or how the corpus was sharded.
"""

from __future__ import annotations

import hashlib
import heapq
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

UINT64_MAX = (1 << 64) - 1


def stable_uint64(key: str, seed: int = 0) -> int:
    """A process-independent 64-bit value for a string key."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def stable_bucket(key: str, seed: int = 0, n_buckets: int = 10_000) -> int:
    """Assign a key to one of ``n_buckets``, reproducibly.

    Modulo bias is on the order of ``n_buckets / 2**64`` — about 5e-16 at ten
    thousand buckets — which is far below any effect a corpus measurement could
    detect.
    """
    if n_buckets <= 0:
        raise ValueError("n_buckets must be positive")
    return stable_uint64(key, seed) % n_buckets


def stable_sort_key(key: str, seed: int = 0) -> tuple[int, str]:
    """A total order: hash first, raw key as tiebreak.

    The tiebreak matters. Two documents sharing a hash would otherwise sort by
    whatever order they arrived in, which is exactly the run-dependence this
    module exists to remove.
    """
    return (stable_uint64(key, seed), key)


@dataclass(frozen=True)
class SourceAllocation:
    """How many documents one source contributes to a sample."""

    source: str
    population: int
    sampled: int

    @property
    def weight(self) -> float:
        """Documents each sampled document stands for, for the micro view."""
        return self.population / self.sampled if self.sampled else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "population": self.population,
            "sampled": self.sampled,
            "weight": self.weight,
        }


def allocate(populations: dict[str, int], max_total_docs: int) -> dict[str, SourceAllocation]:
    """Split a document budget across sources in proportion to their size.

    Proportional rather than equal, because equal-k oversamples small sources
    relative to the corpus. The macro view puts that back when it is wanted; a
    sample that cannot reconstruct either view is the one thing to avoid.

    ``max_total_docs <= 0`` means take everything. A source is never allocated
    zero while it has documents — losing a source entirely from the sample would
    silently drop it from every per-source figure in the report.
    """
    total = sum(populations.values())
    if total == 0:
        return {}
    if max_total_docs <= 0 or max_total_docs >= total:
        return {s: SourceAllocation(s, n, n) for s, n in sorted(populations.items())}

    allocations: dict[str, SourceAllocation] = {}
    for source, n in sorted(populations.items()):
        share = max(1, round(max_total_docs * n / total)) if n else 0
        allocations[source] = SourceAllocation(source, n, min(share, n))

    # The floor of one document per source can push the total past the budget
    # when there are many small sources. Trim the largest allocations back until
    # it fits, never below the floor, so the cap means what the caller asked.
    drawn = sum(a.sampled for a in allocations.values())
    while drawn > max_total_docs:
        trimmable = [a for a in allocations.values() if a.sampled > 1]
        if not trimmable:
            # More sources than the budget allows. Keeping one document each is
            # the lesser evil: dropping sources entirely would remove them from
            # every per-source figure without saying so.
            break
        biggest = max(trimmable, key=lambda a: (a.sampled, a.source))
        allocations[biggest.source] = SourceAllocation(biggest.source, biggest.population, biggest.sampled - 1)
        drawn -= 1

    return allocations


def bottom_k(
    items: Iterable[tuple[str, Any]],
    k: int,
    seed: int = 0,
) -> list[tuple[str, Any]]:
    """The ``k`` items whose key hashes lowest, in ascending hash order.

    Bounded memory: a max-heap of size ``k`` rather than sorting the corpus.
    """
    if k <= 0:
        return []

    # The arrival index is carried purely as a final tiebreaker. Without it, two
    # items sharing a hash *and* a key would make the heap compare payloads,
    # which raises for anything that is not order-comparable — a crash caused by
    # duplicate documents rather than by anything the caller did wrong.
    heap: list[tuple[int, str, int, Any]] = []
    for index, (key, payload) in enumerate(items):
        h = stable_uint64(key, seed)
        entry = (-h, key, -index, payload)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif -heap[0][0] > h:
            heapq.heapreplace(heap, entry)

    ordered = sorted(heap, key=lambda t: (-t[0], t[1], -t[2]))
    return [(key, payload) for _negh, key, _negindex, payload in ordered]


def sample_by_source(
    records: Iterable[tuple[str, str, Any]],
    allocations: dict[str, SourceAllocation],
    seed: int = 0,
) -> dict[str, list[tuple[str, Any]]]:
    """Draw the allocated number of documents from each source.

    ``records`` yields ``(source, key, payload)``. Selection within a source is
    independent of the order records arrive in.

    Records stream into one bounded heap per source rather than being collected
    first. Buffering the corpus would make ``max_total_docs`` a cap on the sample
    only, not on memory, and the payload here is document text — on a corpus of a
    hundred million documents that is the difference between a profile that runs
    and one that is killed before it samples anything.

    A source with no allocation is not capped, on the assumption the caller did
    not intend to sample it at all.
    """
    heaps: dict[str, list[tuple[int, str, int, Any]]] = {}
    uncapped: dict[str, list[tuple[str, Any]]] = {}

    for index, (source, key, payload) in enumerate(records):
        allocation = allocations.get(source)
        if allocation is None:
            uncapped.setdefault(source, []).append((key, payload))
            continue

        k = allocation.sampled
        if k <= 0:
            heaps.setdefault(source, [])
            continue

        heap = heaps.setdefault(source, [])
        h = stable_uint64(key, seed)
        entry = (-h, key, -index, payload)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif -heap[0][0] > h:
            heapq.heapreplace(heap, entry)

    per_source: dict[str, list[tuple[str, Any]]] = {
        source: [(key, payload) for _negh, key, _negindex, payload in sorted(heap, key=lambda t: (-t[0], t[1], -t[2]))]
        for source, heap in heaps.items()
    }
    per_source.update(uncapped)
    return per_source


def selection_threshold(k: int, population: int) -> int:
    """The hash value below which a document is in a bottom-k sample.

    Lets a second pass decide membership without holding the first pass's
    selection in memory.
    """
    if population <= 0 or k <= 0:
        return 0
    if k >= population:
        return UINT64_MAX
    return int(UINT64_MAX * (k / population))


def iter_stable(items: Sequence[str], seed: int = 0) -> Iterator[str]:
    """Iterate in the deterministic pseudo-random order used for selection."""
    return iter(sorted(items, key=lambda key: stable_sort_key(key, seed)))


def content_key(text: str) -> str:
    """A content-derived identity, for corpora that carry no id.

    Not unique when the corpus holds exact duplicates, so it does not by itself
    give a total order; pair it with a tiebreaker when one is needed.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_key(record: dict[str, Any], id_field: str | None, text_field: str, fallback: Callable[[], str]) -> str:
    """Identity for one record, preferring the corpus's own identifier."""
    if id_field and record.get(id_field) is not None:
        return str(record[id_field])
    text = record.get(text_field)
    return content_key(text) if isinstance(text, str) else fallback()
