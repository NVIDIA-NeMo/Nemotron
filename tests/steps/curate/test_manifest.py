# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit tests for the curate run manifest.

The manifest is the only channel through which a producing step tells an auditor
what it believed it did, so its schema is a contract between two steps rather
than an implementation detail of either.
"""

from __future__ import annotations

import json

import pytest

from nemotron.steps.curate.nemo_curator.runtime import manifest as m


def _valid(**overrides):
    document = m.build_manifest(
        step_id="curate/nemo_curator",
        config={"text_field": "text"},
        started_at="2026-08-24T00:00:00+00:00",
        input_glob="./in/*.jsonl",
        input_counts={"file_count": 2, "row_count": 10},
        output_counts={"file_count": 1, "row_count": 7},
        completed_at="2026-08-24T00:01:00+00:00",
    )
    document.update(overrides)
    return document


# -- canonical form -----------------------------------------------------------


def test_canonical_json_is_key_order_independent() -> None:
    assert m.canonical_json({"b": 1, "a": 2}) == m.canonical_json({"a": 2, "b": 1})


def test_canonical_json_does_not_escape_non_ascii() -> None:
    """A config holding a Vietnamese path must hash the same on any machine."""
    assert "Tiếng" in m.canonical_json({"note": "Tiếng Việt"})


def test_json_safe_turns_non_finite_floats_into_null() -> None:
    """A statistic over documents that all failed to score is absent, not a number."""
    inf = float("inf")
    cleaned = m.json_safe({"p50": float("nan"), "hi": inf, "lo": -inf, "kept": 0.5, "n": 3, "name": "x"})

    assert cleaned == {"p50": None, "hi": None, "lo": None, "kept": 0.5, "n": 3, "name": "x"}


def test_json_safe_reaches_into_nested_containers() -> None:
    cleaned = m.json_safe({"signals": [{"views": {"quantiles": [float("nan"), 1.0]}}]})

    assert cleaned == {"signals": [{"views": {"quantiles": [None, 1.0]}}]}


def test_json_safe_output_survives_a_strict_parser() -> None:
    """json.dumps writes bare NaN, which is not in RFC 8259 and no strict reader takes."""
    document = {"quantiles": {"p50": float("nan")}}

    with pytest.raises(ValueError, match="Out of range float"):
        json.dumps(document, allow_nan=False)

    assert json.loads(json.dumps(m.json_safe(document), allow_nan=False)) == {"quantiles": {"p50": None}}


def test_config_hash_carries_its_algorithm() -> None:
    digest = m.config_hash({"a": 1})
    assert digest.startswith("sha256:")
    assert digest == m.config_hash({"a": 1})
    assert digest != m.config_hash({"a": 2})


# -- counting -----------------------------------------------------------------


def test_count_jsonl_counts_rows_and_skips_blank_lines(tmp_path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"text":"x"}\n\n{"text":"y"}\n', encoding="utf-8")

    counted = m.count_jsonl([path])

    assert counted == {"file_count": 1, "row_count": 2}


def test_count_jsonl_tallies_per_source_only_when_asked(tmp_path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"source":"wiki"}\n{"source":"news"}\n{"source":"wiki"}\n', encoding="utf-8")

    assert "per_source" not in m.count_jsonl([path])
    assert m.count_jsonl([path], source_field="source")["per_source"] == {"news": 1, "wiki": 2}


def test_count_jsonl_records_a_missing_source_rather_than_dropping_the_row(tmp_path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"source":"wiki"}\n{"text":"no source here"}\n', encoding="utf-8")

    counted = m.count_jsonl([path], source_field="source")

    assert counted["row_count"] == 2
    assert counted["per_source"] == {"__missing__": 1, "wiki": 1}


def test_count_jsonl_survives_a_truncated_file(tmp_path) -> None:
    """A shard the writer left half-written still contributes its intact rows.

    The torn record is reported separately rather than counted as a row: the
    auditor's scan_shard makes the same call, and a manifest that disagreed with
    its own audit about what a row is would report the disagreement as data loss.
    """
    path = tmp_path / "a.jsonl"
    path.write_text('{"source":"wiki"}\n{"source":"ne', encoding="utf-8")

    counted = m.count_jsonl([path], source_field="source")

    assert counted["row_count"] == 1, "the torn record is not a row"
    assert counted["unparsable_rows"] == 1


def test_the_producer_and_the_auditor_count_a_damaged_shard_identically(tmp_path) -> None:
    """One definition of 'a row', or every audit of a damaged corpus is a false mismatch."""
    from nemotron.steps.curate.nemo_curator.runtime import integrity

    path = tmp_path / "a.jsonl"
    path.write_text('{"id":1}\n{"id":2,,,BROKEN\n{"id":3}\n', encoding="utf-8")

    assert m.count_jsonl([path])["row_count"] == integrity.scan_shard(path).row_count


def test_count_jsonl_ignores_paths_that_are_not_files(tmp_path) -> None:
    assert m.count_jsonl([tmp_path / "absent.jsonl", tmp_path]) == {"file_count": 0, "row_count": 0}


# -- assembly -----------------------------------------------------------------


def test_completed_at_is_absent_when_the_run_did_not_finish() -> None:
    document = m.build_manifest(
        step_id="curate/nemo_curator",
        config={},
        started_at="2026-08-24T00:00:00+00:00",
        input_glob="./in/*.jsonl",
        input_counts={"file_count": 1, "row_count": 5},
        output_counts={"file_count": 0, "row_count": 0},
        completed_at=None,
    )

    assert "completed_at" not in document["producer"]
    assert not m.is_complete(document)
    assert m.validate_manifest(document) == [], "an unfinished run still writes a valid manifest"


def test_absent_rows_are_reported_as_unattributed() -> None:
    """Curator reports no per-stage removals, so the manifest must not imply it does."""
    document = _valid()

    assert document["declared"]["attribution"] == m.ATTRIBUTION_UNAVAILABLE
    assert document["declared"]["rows_absent_from_output"] == 3
    assert document["declared"]["filtered"] is None


def test_a_caller_that_can_attribute_may_say_so() -> None:
    document = m.build_manifest(
        step_id="curate/nemo_curator",
        config={},
        started_at="2026-08-24T00:00:00+00:00",
        input_glob="./in/*.jsonl",
        input_counts={"file_count": 1, "row_count": 10},
        output_counts={"file_count": 1, "row_count": 7},
        declared={"attribution": m.ATTRIBUTION_DECLARED, "filtered": 3, "failed": 0, "quarantined": 0},
    )

    assert m.validate_manifest(document) == []


# -- validation ---------------------------------------------------------------


def test_a_well_formed_manifest_validates() -> None:
    assert m.validate_manifest(_valid()) == []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda d: d.__setitem__("schema_version", 99), "schema_version"),
        (lambda d: d["producer"].pop("config_hash"), "producer.config_hash"),
        (lambda d: d["producer"].__setitem__("config_hash", "deadbeef"), "algorithm prefix"),
        (lambda d: d["input"].__setitem__("row_count", -1), "input.row_count"),
        (lambda d: d["output"].__setitem__("file_count", "1"), "output.file_count"),
        (lambda d: d["declared"].__setitem__("attribution", "guessed"), "declared.attribution"),
        (lambda d: d["canonicalization"].__setitem__("duplicate_ids", "whatever"), "duplicate_ids"),
        (lambda d: d["canonicalization"].__setitem__("json", "pretty"), "canonicalization.json"),
        (lambda d: d.pop("declared"), "declared block"),
    ],
)
def test_validation_catches_each_contract_violation(mutate, expected) -> None:
    document = _valid()
    mutate(document)

    problems = m.validate_manifest(document)

    assert any(expected in p for p in problems), f"expected a problem mentioning {expected!r}, got {problems}"


def test_a_boolean_is_not_an_acceptable_count() -> None:
    """``True`` is an int in Python; a row count of ``True`` is a bug, not a count."""
    document = _valid()
    document["input"]["row_count"] = True

    assert any("input.row_count" in p for p in m.validate_manifest(document))


def test_validation_reports_every_fault_in_one_pass() -> None:
    document = _valid()
    document["producer"].pop("step_id")
    document["input"]["row_count"] = -5

    assert len(m.validate_manifest(document)) >= 2


def test_a_non_mapping_is_rejected_without_raising() -> None:
    assert m.validate_manifest(["not", "a", "manifest"])


def test_output_gain_is_a_manifest_contract_violation() -> None:
    document = _valid()
    document["input"]["row_count"] = 1
    document["output"]["row_count"] = 2
    document["declared"]["rows_absent_from_output"] = -1

    problems = m.validate_manifest(document)

    assert any("must not exceed" in problem for problem in problems)


def test_writing_refuses_a_manifest_with_output_gain(tmp_path) -> None:
    document = _valid()
    document["input"]["row_count"] = 1
    document["output"]["row_count"] = 2

    with pytest.raises(ValueError, match="must not exceed"):
        m.write_manifest(tmp_path / "run_manifest.json", document)


def test_an_injected_tool_revision_takes_precedence(monkeypatch) -> None:
    monkeypatch.setenv("NEMOTRON_TOOL_REVISION", "git:0123456789abcdef")

    assert m.tool_revision() == "git:0123456789abcdef"


def test_source_staging_without_distribution_metadata_uses_content_revision(monkeypatch, tmp_path) -> None:
    """Source hashing must precede the always-importable static version."""
    from nemotron import __version__

    assert __version__
    source = tmp_path / "nemo_curator" / "runtime" / "manifest.py"
    source.parent.mkdir(parents=True)
    source.write_text("revision = 1\n", encoding="utf-8")
    config = source.parents[1] / "config.yaml"
    config.write_text("mode: filter\n", encoding="utf-8")
    monkeypatch.delenv("NEMOTRON_TOOL_REVISION", raising=False)
    monkeypatch.setattr(m.Path, "resolve", lambda _self: source)
    monkeypatch.setattr("importlib.metadata.version", lambda _name: (_ for _ in ()).throw(LookupError()))

    first = m.tool_revision()
    config.write_text("mode: annotate\n", encoding="utf-8")
    second = m.tool_revision()

    assert first.startswith("source:sha256:")
    assert second.startswith("source:sha256:")
    assert first != second


def test_static_version_is_used_when_source_cannot_be_read(monkeypatch) -> None:
    from nemotron import __version__

    monkeypatch.delenv("NEMOTRON_TOOL_REVISION", raising=False)
    monkeypatch.setattr(m.Path, "resolve", lambda _self: m.Path("/staged/src/nemotron/runtime/manifest.py"))
    monkeypatch.setattr("importlib.metadata.version", lambda _name: (_ for _ in ()).throw(LookupError()))
    monkeypatch.setattr(m, "_source_tree_revision", lambda _source: None)

    assert m.tool_revision() == f"nemotron {__version__}"


def test_runtime_dependencies_include_an_exact_vcs_commit(monkeypatch) -> None:
    class FakeDistribution:
        version = "0.10.0+a8425c9"

        @staticmethod
        def read_text(filename: str) -> str | None:
            if filename != "direct_url.json":
                return None
            return json.dumps(
                {
                    "url": "https://example.invalid/run.git",
                    "vcs_info": {"vcs": "git", "commit_id": "a8425c9f11a45412e6d5338cecb1c014b0ecc0c4"},
                }
            )

    monkeypatch.setattr("importlib.metadata.distribution", lambda _name: FakeDistribution())

    assert m.runtime_dependencies()["nemo-run"] == {
        "version": "0.10.0+a8425c9",
        "commit_id": "a8425c9f11a45412e6d5338cecb1c014b0ecc0c4",
        "source_url": "https://example.invalid/run.git",
    }


# -- write / read -------------------------------------------------------------


def test_write_then_read_round_trips(tmp_path) -> None:
    path = tmp_path / "nested" / "run_manifest.json"

    m.write_manifest(path, _valid())

    assert m.read_manifest(path) == _valid()
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == m.SCHEMA_VERSION


def test_writing_refuses_a_non_conformant_manifest(tmp_path) -> None:
    path = tmp_path / "run_manifest.json"
    document = _valid()
    document["declared"]["attribution"] = "invented"

    with pytest.raises(ValueError, match="non-conformant"):
        m.write_manifest(path, document)

    assert not path.exists(), "a rejected manifest must not leave a file behind"


def test_writing_leaves_no_temp_file(tmp_path) -> None:
    path = tmp_path / "run_manifest.json"

    m.write_manifest(path, _valid())

    assert [p.name for p in tmp_path.iterdir()] == ["run_manifest.json"]
