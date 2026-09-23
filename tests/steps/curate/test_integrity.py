# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for corpus integrity measurement."""

from __future__ import annotations

from pathlib import Path

import pytest

from nemotron.steps.curate.nemo_curator.runtime import integrity


def write(path, *records: str):
    path.write_text("".join(r if r.endswith("\n") else r + "\n" for r in records), encoding="utf-8")
    return path


# -- reading a shard ----------------------------------------------------------


def test_a_clean_shard_reports_its_rows_and_a_digest(tmp_path) -> None:
    shard = write(tmp_path / "a.jsonl", '{"id":"1"}', '{"id":"2"}')

    report = integrity.scan_shard(shard)

    assert report.readable
    assert report.row_count == 2
    assert report.digest.startswith("sha256:")
    assert report.error is None


def test_blank_lines_are_not_rows(tmp_path) -> None:
    shard = tmp_path / "a.jsonl"
    shard.write_text('{"id":"1"}\n\n\n{"id":"2"}\n', encoding="utf-8")

    assert integrity.scan_shard(shard).row_count == 2


def test_a_truncated_final_record_is_named_with_its_byte_offset(tmp_path) -> None:
    """The offset is what makes the finding actionable rather than a scavenger hunt."""
    shard = tmp_path / "a.jsonl"
    intact = '{"id":"1"}\n'
    shard.write_text(intact + '{"id":"2", "text":"cut off here', encoding="utf-8")

    report = integrity.scan_shard(shard)

    assert not report.readable
    assert report.error == "truncated final record"
    assert report.error_byte_offset == len(intact)
    assert report.row_count == 1, "rows before the tear are still real"


def test_a_broken_record_mid_file_is_corrupt_not_truncated(tmp_path) -> None:
    """A complete line after the damage means the file was not merely cut short."""
    shard = tmp_path / "a.jsonl"
    shard.write_text('{"id":"1"}\nnot json at all\n{"id":"3"}\n', encoding="utf-8")

    report = integrity.scan_shard(shard)

    assert report.error == "unparsable record"
    assert report.row_count == 2


def test_a_zero_length_shard_is_readable_and_empty(tmp_path) -> None:
    shard = tmp_path / "a.jsonl"
    shard.write_text("", encoding="utf-8")

    report = integrity.scan_shard(shard)

    assert report.readable
    assert report.row_count == 0


def test_a_missing_file_is_reported_not_raised(tmp_path) -> None:
    report = integrity.scan_shard(tmp_path / "absent.jsonl")

    assert not report.readable
    assert report.error == "not a file"


def test_per_source_counts_only_when_asked(tmp_path) -> None:
    shard = write(tmp_path / "a.jsonl", '{"source":"wiki"}', '{"source":"news"}', '{"source":"wiki"}')

    assert integrity.scan_shard(shard).per_source == {}
    assert integrity.scan_shard(shard, source_field="source").per_source == {"news": 1, "wiki": 2}


def test_json_values_that_are_not_records_are_reported_as_damage(tmp_path) -> None:
    shard = write(tmp_path / "a.jsonl", "42", '["not", "an", "object"]', '{"source":"wiki"}')

    report = integrity.scan_shard(shard, source_field="source")

    assert report.row_count == 1
    assert report.readable is False
    assert report.bad_record_count == 2
    assert report.error == "non-object record"
    assert report.per_source == {"wiki": 1}


# -- corpus digest ------------------------------------------------------------


def test_digest_ignores_enumeration_order(tmp_path) -> None:
    """§8.2 guarantee (a): a filesystem listing files differently must not change it."""
    a = write(tmp_path / "a.jsonl", '{"id":"1"}')
    b = write(tmp_path / "b.jsonl", '{"id":"2"}')

    forward = integrity.corpus_digest(integrity.scan_corpus([a, b]), root=tmp_path)
    backward = integrity.corpus_digest(integrity.scan_corpus([b, a]), root=tmp_path)

    assert forward == backward


def test_digest_changes_when_content_changes(tmp_path) -> None:
    shard = write(tmp_path / "a.jsonl", '{"id":"1"}')
    before = integrity.corpus_digest(integrity.scan_corpus([shard]), root=tmp_path)

    write(shard, '{"id":"1"}', '{"id":"2"}')
    after = integrity.corpus_digest(integrity.scan_corpus([shard]), root=tmp_path)

    assert before != after


def test_digest_is_sensitive_to_shard_names(tmp_path) -> None:
    """Deliberately not name-independent: otherwise it could not say which shard moved."""
    a = write(tmp_path / "a.jsonl", '{"id":"1"}')
    before = integrity.corpus_digest(integrity.scan_corpus([a]), root=tmp_path)

    renamed = a.rename(tmp_path / "z.jsonl")
    after = integrity.corpus_digest(integrity.scan_corpus([renamed]), root=tmp_path)

    assert before != after


def test_digest_survives_moving_the_corpus_when_a_root_is_given(tmp_path) -> None:
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    write(one / "a.jsonl", '{"id":"1"}')
    write(two / "a.jsonl", '{"id":"1"}')

    assert integrity.corpus_digest(integrity.scan_corpus([one / "a.jsonl"]), root=one) == integrity.corpus_digest(
        integrity.scan_corpus([two / "a.jsonl"]), root=two
    )


# -- summary ------------------------------------------------------------------


def test_summary_rolls_up_and_names_the_damaged_shard(tmp_path) -> None:
    write(tmp_path / "a.jsonl", '{"source":"wiki"}')
    (tmp_path / "b.jsonl").write_text('{"source":"ne', encoding="utf-8")

    summary = integrity.summarize(integrity.scan_corpus(tmp_path.glob("*.jsonl"), source_field="source"))

    assert summary["file_count"] == 2
    assert summary["unreadable_count"] == 1
    assert summary["unreadable"][0]["path"].endswith("b.jsonl")
    assert summary["per_source"] == {"wiki": 1}


# -- containment --------------------------------------------------------------


def test_containment_requires_an_explicit_field_choice(tmp_path) -> None:
    """An implicit 'all common fields' default would flag language/domain as differences."""
    shard = write(tmp_path / "a.jsonl", '{"id":"1"}')

    with pytest.raises(integrity.ContainmentConfigError, match="comparison_fields must name"):
        integrity.row_keys([shard], [])


def test_a_subset_is_contained(tmp_path) -> None:
    parent = write(tmp_path / "parent.jsonl", '{"id":"1"}', '{"id":"2"}', '{"id":"3"}')
    child = write(tmp_path / "child.jsonl", '{"id":"1"}', '{"id":"3"}')

    result = integrity.containment([child], [parent], ["id"])

    assert result["contained"]
    assert result["missing_row_count"] == 0


def test_a_row_absent_from_the_parent_is_reported(tmp_path) -> None:
    parent = write(tmp_path / "parent.jsonl", '{"id":"1"}')
    child = write(tmp_path / "child.jsonl", '{"id":"1"}', '{"id":"99"}')

    result = integrity.containment([child], [parent], ["id"])

    assert not result["contained"]
    assert result["missing_row_count"] == 1


def test_containment_is_a_multiset_not_a_set(tmp_path) -> None:
    """Two copies in the subset need two copies in the superset."""
    parent = write(tmp_path / "parent.jsonl", '{"id":"1"}')
    child = write(tmp_path / "child.jsonl", '{"id":"1"}', '{"id":"1"}')

    result = integrity.containment([child], [parent], ["id"], duplicate_ids="multiset")

    assert not result["contained"]
    assert result["missing_row_count"] == 1


def test_repeated_keys_are_counted(tmp_path) -> None:
    shard = write(tmp_path / "a.jsonl", '{"id":"1"}', '{"id":"1"}')

    keyed = integrity.row_keys([shard], ["id"])

    assert keyed.duplicate_keys == 1
    assert keyed.rows_keyed == 2


def test_repeated_keys_become_a_finding_only_under_reject(tmp_path) -> None:
    """The policy lives in the manifest, so the same data is a fault or not depending on it."""
    parent = write(tmp_path / "parent.jsonl", '{"id":"1"}', '{"id":"1"}')
    child = write(tmp_path / "child.jsonl", '{"id":"1"}', '{"id":"1"}')

    rejected = integrity.containment([child], [parent], ["id"], duplicate_ids="reject")
    allowed = integrity.containment([child], [parent], ["id"], duplicate_ids="multiset")

    assert any("repeat" in problem for problem in rejected["problems"])
    assert allowed["problems"] == []


def test_a_record_missing_a_comparison_field_is_counted_not_dropped(tmp_path) -> None:
    """A row silently skipped is a row the containment result does not cover."""
    shard = write(tmp_path / "a.jsonl", '{"id":"1"}', '{"text":"no id"}')

    keyed = integrity.row_keys([shard], ["id"])

    assert keyed.rows_seen == 2
    assert keyed.rows_keyed == 1
    assert keyed.rows_missing_field == 1
    assert not keyed.complete
    assert any("missing ['id']" in example for example in keyed.examples)


def test_containment_is_not_claimed_over_rows_that_were_never_keyed(tmp_path) -> None:
    """The false all-clear: comparing nothing and reporting containment."""
    parent = write(tmp_path / "parent.jsonl", '{"id":"1"}')
    child = write(tmp_path / "child.jsonl", '{"text":"no id here"}')

    result = integrity.containment([child], [parent], ["id"])

    assert result["subset_rows"] == 0
    assert result["verifiable"] is False
    assert result["contained"] is False


def test_a_partially_keyable_target_is_not_verifiable(tmp_path) -> None:
    parent = write(tmp_path / "parent.jsonl", '{"id":"1"}', '{"id":"2"}')
    child = write(tmp_path / "child.jsonl", '{"id":"1"}', '{"text":"no id"}')

    result = integrity.containment([child], [parent], ["id"])

    assert result["verifiable"] is False
    assert result["target"]["rows_missing_field"] == 1


def test_reported_examples_are_capped(tmp_path) -> None:
    """A corpus missing the field on every row must not write a line per document."""
    shard = write(tmp_path / "a.jsonl", *['{"text":"x"}'] * 500)

    keyed = integrity.row_keys([shard], ["id"])

    assert keyed.rows_missing_field == 500
    assert len(keyed.examples) <= integrity.MAX_REPORTED_EXAMPLES


def test_comparing_several_fields_hashes_them_together(tmp_path) -> None:
    parent = write(tmp_path / "parent.jsonl", '{"text":"a","source":"wiki"}')
    same = write(tmp_path / "same.jsonl", '{"text":"a","source":"wiki","language":"en"}')
    other = write(tmp_path / "other.jsonl", '{"text":"a","source":"news"}')

    assert integrity.containment([same], [parent], ["text", "source"])["contained"]
    assert not integrity.containment([other], [parent], ["text", "source"])["contained"]


# -- resolving what the user pointed at ---------------------------------------
#
# Untested until a Curator comparison drew attention to it, and the gap had
# already cost a real defect: a bare directory matched only *.jsonl, so a
# directory of parquet — which `curate/ingest` reads — resolved to no files and
# was reported as a corpus that had been read.


def test_a_directory_of_parquet_resolves(tmp_path) -> None:
    """ingest reads parquet, so a directory naming a parquet corpus is not empty."""
    (tmp_path / "part_0.parquet").write_bytes(b"PAR1")
    (tmp_path / "part_1.parquet").write_bytes(b"PAR1")

    assert len(integrity.expand_inputs(str(tmp_path))) == 2


def test_a_directory_skips_files_that_are_not_corpus(tmp_path) -> None:
    """A README beside the shards would otherwise reach detect_format and stop the run."""
    write(tmp_path / "part_0.jsonl", '{"text":"a"}')
    (tmp_path / "README.md").write_text("notes", encoding="utf-8")
    (tmp_path / "_SUCCESS").write_text("", encoding="utf-8")

    assert [Path(p).name for p in integrity.expand_inputs(str(tmp_path))] == ["part_0.jsonl"]


def test_a_directory_skips_json_accounting_sidecars(tmp_path) -> None:
    write(tmp_path / "part_0.jsonl", '{"text":"a"}')
    for name in (
        "ingest_report.json",
        "run_manifest.json",
        "curation_ledger.json",
        "audit_report.json",
    ):
        (tmp_path / name).write_text('{"not":"a corpus row"}\n', encoding="utf-8")

    assert [Path(path).name for path in integrity.expand_inputs(str(tmp_path))] == ["part_0.jsonl"]


def test_a_directory_is_searched_recursively(tmp_path) -> None:
    nested = tmp_path / "lang" / "vi"
    nested.mkdir(parents=True)
    write(nested / "part_0.jsonl", '{"text":"a"}')

    assert len(integrity.expand_inputs(str(tmp_path))) == 1


def test_a_glob_is_taken_as_given(tmp_path) -> None:
    """A glob already says what it wants; the corpus-extension filter must not narrow it."""
    (tmp_path / "shard.bin").write_bytes(b"x")

    assert len(integrity.expand_inputs(str(tmp_path / "*.bin"))) == 1


def test_a_named_file_is_taken_as_given(tmp_path) -> None:
    named = tmp_path / "corpus.txt"
    named.write_text("x", encoding="utf-8")

    assert integrity.expand_inputs(str(named)) == [str(named)]


def test_a_list_of_references_is_merged_and_deduplicated(tmp_path) -> None:
    a = write(tmp_path / "a.jsonl", '{"text":"a"}')
    write(tmp_path / "b.jsonl", '{"text":"b"}')

    resolved = integrity.expand_inputs([str(tmp_path), str(a)])

    assert len(resolved) == 2, "a file named twice is one file"


def test_nothing_named_resolves_to_nothing() -> None:
    assert integrity.expand_inputs(None) == []
    assert integrity.expand_inputs([]) == []


# -- reading a corpus record by record -----------------------------------------
#
# Two runners carried the same 14-line reader. Shared here because "an unparsable
# line is counted, never silently skipped" is a property of the category, not of
# one step: curate/audit calls exactly that damage a finding, so a step that
# stayed quiet about it would describe a corpus its own sibling considers broken.


def test_records_are_yielded_with_their_shard(tmp_path) -> None:
    a = write(tmp_path / "a.jsonl", '{"id":"1"}', '{"id":"2"}')
    b = write(tmp_path / "b.jsonl", '{"id":"3"}')

    got = list(integrity.iter_records([str(a), str(b)]))

    assert [r["id"] for _, r in got] == ["1", "2", "3"]
    assert {Path(p).name for p, _ in got} == {"a.jsonl", "b.jsonl"}


def test_blank_lines_are_records_of_nothing(tmp_path) -> None:
    shard = tmp_path / "a.jsonl"
    shard.write_text('{"id":"1"}\n\n\n{"id":"2"}\n', encoding="utf-8")
    damage: dict[str, int] = {}

    got = list(integrity.iter_records([str(shard)], damage))

    assert len(got) == 2
    assert not damage, "a blank line is formatting, not corruption"


def test_an_unparsable_line_is_counted_not_skipped(tmp_path) -> None:
    shard = write(tmp_path / "a.jsonl", '{"id":"1"}', "{not json", '{"id":"2"}')
    damage: dict[str, int] = {}

    got = list(integrity.iter_records([str(shard)], damage))

    assert len(got) == 2, "the readable records still come through"
    assert damage[str(shard)] == 1


def test_a_non_mapping_json_value_is_counted_not_yielded(tmp_path) -> None:
    shard = write(tmp_path / "a.jsonl", "42", '["not", "a", "record"]', '{"id":"1"}')
    damage: dict[str, int] = {}

    got = list(integrity.iter_records([str(shard)], damage))

    assert [record["id"] for _, record in got] == ["1"]
    assert damage[str(shard)] == 2


def test_counting_damage_is_optional(tmp_path) -> None:
    """curate/profile reads the corpus twice and only one pass has a tally to fill."""
    shard = write(tmp_path / "a.jsonl", "{not json", '{"id":"1"}')

    assert len(list(integrity.iter_records([str(shard)]))) == 1


# -- a fingerprint over nothing is not a fingerprint ---------------------------
#
# corpus_fingerprint returned a digest for a corpus it could not read, and the
# same digest for every such corpus. The approve gate rests on this value: its
# whole promise is "thresholds calibrated on corpus A do not silently apply to
# corpus B". Two unreadable corpora sharing one value defeats that exactly where
# it matters, and the module already declares UnreadableCorpusError for the purpose.


def test_a_corpus_with_no_files_is_refused_not_digested(tmp_path) -> None:
    with pytest.raises(integrity.UnreadableCorpusError, match="no files"):
        integrity.corpus_fingerprint(str(tmp_path / "*.jsonl"), "text", "id")


def test_two_unreadable_corpora_cannot_share_a_fingerprint(tmp_path) -> None:
    """The failure this prevents: an approval granted before its corpus exists."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()

    for empty in (a, b):
        with pytest.raises(integrity.UnreadableCorpusError):
            integrity.corpus_fingerprint(f"{empty}/*.jsonl", "text", "id")


def test_a_corpus_whose_files_hold_no_documents_is_refused(tmp_path) -> None:
    """Files that exist but yield nothing readable are the same failure."""
    (tmp_path / "a.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(integrity.UnreadableCorpusError):
        integrity.corpus_fingerprint(f"{tmp_path}/*.jsonl", "text", "id")


def test_a_real_corpus_still_fingerprints(tmp_path) -> None:
    write(tmp_path / "a.jsonl", '{"id":"1","text":"hello"}')

    assert integrity.corpus_fingerprint(f"{tmp_path}/*.jsonl", "text", "id").startswith("sha256:")


# -- the sidecar denylist has exactly one exemption ---------------------------


def test_a_ledger_glob_still_finds_the_ledger(tmp_path) -> None:
    """curate/audit points AT curation_ledger.json on purpose.

    The denylist that stops a corpus reference reading a previous run's
    accounting would otherwise silently empty the audit's ledger list, and the
    audit would pass while reporting "no ledger_glob" -- losing the attribution
    it was asked for without ever failing.
    """
    # A realistic run directory: shards beside the accounting, not a directory
    # holding the ledger alone. With only the ledger present, allow_sidecars=True
    # would look right merely because there was nothing else to return.
    (tmp_path / "run1").mkdir()
    write(tmp_path / "run1" / "part_0.jsonl", '{"text":"a"}')
    (tmp_path / "run1" / "curation_ledger.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run1" / "run_manifest.json").write_text("{}", encoding="utf-8")

    for reference in (str(tmp_path / "**" / "curation_ledger.json"), str(tmp_path)):
        exempt = [Path(p).name for p in integrity.expand_inputs(reference, allow_sidecars=True)]
        default = [Path(p).name for p in integrity.expand_inputs(reference)]

        assert "curation_ledger.json" in exempt, reference
        assert "curation_ledger.json" not in default, f"still excluded by default: {reference}"


def test_the_audit_finds_the_ledger_through_its_own_entry_point(tmp_path) -> None:
    """Drives run_audit.audit(), not run_audit.expand().

    The previous version of this test called expand() and supplied
    allow_sidecars=True itself, so it never reached the call site it was meant to
    pin: reverting run_audit.py's `allow_sidecars=True` left it green. The
    exemption only matters if the audit asks for it, so the audit has to be what
    asks here.
    """
    import json

    from nemotron.steps.curate.nemo_curator.scripts import run_audit

    target = tmp_path / "out"
    target.mkdir()
    write(target / "part_0.jsonl", '{"id":"a","text":"x"}')
    (target / "curation_ledger.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "curate/nemo_curator",
                "source": None,
                "n_input": 1,
                "n_success": 1,
                "n_filtered": 0,
                "n_failed": 0,
                "n_quarantined": 0,
                "n_accounted": 1,
                "balanced": True,
                "filtered_by_reason": {},
                "failed_units": [],
                "quarantined_units": [],
                "notes": {},
            }
        ),
        encoding="utf-8",
    )

    report = run_audit.audit(
        {
            "target_glob": str(target),
            "ledger_glob": str(target / "*ledger.json"),
            "mode": "integrity",
        }
    )

    # A ledger that resolved is one the audit read: either it attributes, or it
    # reports the file as unreadable. What must NOT happen is the "no ledger_glob"
    # branch, where the audit passes having quietly lost the attribution it was
    # asked for -- which is what the sidecar denylist caused before the exemption.
    unresolved = [f for f in report["findings"] if "no ledger_glob" in str(f)]
    assert not unresolved, report["findings"]
    assert report.get("attribution", {}).get("available") is not False, report.get("attribution")


def test_a_glob_is_not_recursed_into_but_a_bare_directory_is(tmp_path) -> None:
    """A partitioned corpus is named by its directory, not by `<root>/*`.

    Recursing into glob-matched directories was tried and reverted: it changed
    what every pre-existing `<root>/*` config resolved to across all six
    resolvers, and turned a working `input: ./raw/*` over a parquet corpus with a
    converted/ subdirectory into an IngestError. The bare-directory spelling has
    always recursed and is the supported way to name a partitioned corpus.
    """
    for part in ("part=1", "part=2"):
        (tmp_path / part).mkdir()
        write(tmp_path / part / "shard.jsonl", '{"text":"a"}')

    assert integrity.expand_inputs(str(tmp_path / "*")) == [], "a glob match that is a directory is not a file"
    assert len(integrity.expand_inputs(str(tmp_path))) == 2, "the bare directory still recurses"
