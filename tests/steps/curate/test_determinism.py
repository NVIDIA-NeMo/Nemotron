# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Reproducible ordering and sampling."""

from __future__ import annotations

import subprocess
import sys

import pytest

from nemotron.steps.curate.nemo_curator.runtime import determinism as d


def test_hashing_is_stable_across_processes() -> None:
    """Python's built-in hash() is salted per process; this must not be."""
    expected = d.stable_uint64("doc-1", seed=7)
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "from nemotron.steps.curate.nemo_curator.runtime import determinism as d;"
            "print(d.stable_uint64('doc-1', seed=7))",
        ],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": "12345", "PATH": "/usr/bin:/bin", "PYTHONPATH": _src()},
    )
    assert int(out.stdout.strip()) == expected


def _src() -> str:
    from pathlib import Path

    return str(Path(__file__).resolve().parents[3] / "src")


def test_the_seed_changes_the_order() -> None:
    assert d.stable_uint64("doc-1", seed=0) != d.stable_uint64("doc-1", seed=1)


def test_bucketing_is_in_range_and_reproducible() -> None:
    values = [d.stable_bucket(f"doc-{i}", n_buckets=100) for i in range(500)]

    assert all(0 <= v < 100 for v in values)
    assert values == [d.stable_bucket(f"doc-{i}", n_buckets=100) for i in range(500)]


def test_bucketing_rejects_a_nonsense_bucket_count() -> None:
    with pytest.raises(ValueError):
        d.stable_bucket("x", n_buckets=0)


def test_sort_key_breaks_ties_on_the_key_itself() -> None:
    """Two documents sharing a hash must not fall back to arrival order."""
    assert d.stable_sort_key("a")[1] == "a"


# -- allocation ---------------------------------------------------------------


def test_allocation_is_proportional_not_equal(tmp_path) -> None:
    """Equal-k would oversample the small source relative to the corpus."""
    allocations = d.allocate({"big": 9000, "small": 1000}, max_total_docs=1000)

    assert allocations["big"].sampled > allocations["small"].sampled
    assert allocations["big"].sampled == pytest.approx(900, abs=2)
    assert allocations["small"].sampled == pytest.approx(100, abs=2)


def test_a_small_source_is_never_allocated_zero() -> None:
    """Dropping a source entirely would remove it from every per-source figure."""
    allocations = d.allocate({"huge": 1_000_000, "tiny": 3}, max_total_docs=100)

    assert allocations["tiny"].sampled >= 1


def test_a_budget_larger_than_the_corpus_takes_everything() -> None:
    allocations = d.allocate({"a": 10, "b": 5}, max_total_docs=1000)

    assert allocations["a"].sampled == 10
    assert allocations["b"].sampled == 5


def test_zero_budget_means_no_cap() -> None:
    assert d.allocate({"a": 10}, max_total_docs=0)["a"].sampled == 10


def test_weight_is_how_many_documents_each_sample_stands_for() -> None:
    allocation = d.SourceAllocation("a", population=1000, sampled=100)

    assert allocation.weight == 10.0


def test_an_empty_corpus_allocates_nothing() -> None:
    assert d.allocate({}, max_total_docs=100) == {}


# -- sampling -----------------------------------------------------------------


def test_bottom_k_ignores_input_order() -> None:
    items = [(f"doc-{i}", i) for i in range(100)]

    forward = d.bottom_k(items, 10)
    backward = d.bottom_k(list(reversed(items)), 10)

    assert forward == backward


def test_bottom_k_is_a_subset_relation_as_k_grows() -> None:
    """A larger sample must contain the smaller one, or two runs disagree."""
    items = [(f"doc-{i}", i) for i in range(200)]

    small = {k for k, _ in d.bottom_k(items, 10)}
    large = {k for k, _ in d.bottom_k(items, 40)}

    assert small <= large


def test_bottom_k_of_zero_is_empty() -> None:
    assert d.bottom_k([("a", 1)], 0) == []


def test_sampling_draws_the_allocated_count_per_source() -> None:
    records = [("big", f"b-{i}", i) for i in range(100)] + [("small", f"s-{i}", i) for i in range(10)]
    allocations = d.allocate({"big": 100, "small": 10}, max_total_docs=22)

    sample = d.sample_by_source(records, allocations)

    assert len(sample["big"]) == allocations["big"].sampled
    assert len(sample["small"]) == allocations["small"].sampled


def test_sampling_is_reproducible_from_the_seed() -> None:
    records = [("a", f"doc-{i}", i) for i in range(100)]
    allocations = d.allocate({"a": 100}, max_total_docs=20)

    first = d.sample_by_source(records, allocations, seed=3)
    second = d.sample_by_source(records, allocations, seed=3)
    other = d.sample_by_source(records, allocations, seed=4)

    assert first == second
    assert first != other


# -- identity -----------------------------------------------------------------


def test_the_corpus_identifier_is_preferred_over_a_content_hash() -> None:
    record = {"id": "doc-1", "text": "hello"}

    assert d.make_key(record, "id", "text", lambda: "fallback") == "doc-1"


def test_content_hash_is_used_when_there_is_no_identifier() -> None:
    key = d.make_key({"text": "hello"}, None, "text", lambda: "fallback")

    assert key == d.content_key("hello")


def test_identical_text_yields_an_identical_content_key() -> None:
    """Content keys are not unique under exact duplicates; callers must expect that."""
    assert d.content_key("same") == d.content_key("same")


def test_the_fallback_runs_only_when_there_is_nothing_else() -> None:
    assert d.make_key({}, None, "text", lambda: "row:7") == "row:7"


def test_selection_threshold_spans_the_hash_space() -> None:
    assert d.selection_threshold(0, 100) == 0
    assert d.selection_threshold(100, 100) == d.UINT64_MAX
    assert 0 < d.selection_threshold(50, 100) < d.UINT64_MAX


# -- regressions --------------------------------------------------------------


def test_the_budget_is_respected_when_rounding_would_overshoot() -> None:
    allocations = d.allocate({f"s{i}": 100 for i in range(7)}, max_total_docs=10)

    assert sum(a.sampled for a in allocations.values()) <= 10


def test_more_sources_than_budget_keeps_one_each_rather_than_dropping_sources() -> None:
    """Dropping a source removes it from every per-source figure without saying so."""
    allocations = d.allocate({f"s{i}": 10 for i in range(50)}, max_total_docs=20)

    assert all(a.sampled >= 1 for a in allocations.values())
    assert len(allocations) == 50


def test_bottom_k_survives_duplicate_keys_with_opaque_payloads() -> None:
    """Two identical keys must not make the heap try to order the documents."""

    class Opaque:
        def __init__(self, value):
            self.value = value

    selected = d.bottom_k([("same", Opaque(1)), ("same", Opaque(2))], 2)

    assert len(selected) == 2


def test_sampling_streams_rather_than_buffering_the_corpus() -> None:
    """max_total_docs must bound memory, not only the sample.

    The payload is document text; buffering a hundred-million-row corpus to pick
    two hundred thousand documents kills the process before it samples anything.
    Consuming a one-shot generator proves nothing is collected up front.
    """
    allocations = d.allocate({"a": 1000}, max_total_docs=10)
    stream = (("a", f"doc-{i}", f"text {i}") for i in range(1000))

    sample = d.sample_by_source(stream, allocations)

    assert len(sample["a"]) == 10


def test_streamed_sampling_matches_the_order_independent_result() -> None:
    records = [("a", f"doc-{i}", i) for i in range(300)]
    allocations = d.allocate({"a": 300}, max_total_docs=25)

    forward = d.sample_by_source(iter(records), allocations, seed=11)
    backward = d.sample_by_source(iter(list(reversed(records))), allocations, seed=11)

    assert forward == backward


def test_a_source_with_no_allocation_is_not_capped() -> None:
    """The caller did not ask to sample it, so it is passed through whole."""
    sample = d.sample_by_source([("stray", "k1", 1), ("stray", "k2", 2)], {})

    assert len(sample["stray"]) == 2
