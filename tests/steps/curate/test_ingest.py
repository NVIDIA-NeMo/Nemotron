# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""curate/ingest: raw corpus in, curatable corpus out.

The step exists because "provide a config and your data" was not true without
it. Most of these tests are about identity — a minted id that does not survive
resharding invalidates every claim subset and decontamination make.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
import yaml

from nemotron.steps.curate.nemo_curator.runtime import ingest as ingest_module
from nemotron.steps.curate.nemo_curator.scripts import run_ingest

from .._step_helpers import assert_step_static, step_dir

STEP_DIR = step_dir(__file__, "curate", "nemo_curator", "ingest")


def jsonl(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def parquet(path: Path, rows: list[dict]) -> str:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pq.write_table(pa.Table.from_pylist(rows), path)
    return str(path)


def cfg(tmp_path: Path, source: str, **overrides) -> dict:
    base = {
        "input": source,
        "output_dir": str(tmp_path / "ingested"),
        "text_field": "text",
        "id_from": None,
        "id_fields": ["text"],
        "on_duplicate": "refuse",
    }
    base.update(overrides)
    return base


def written(config: dict) -> list[dict]:
    out = Path(config["output_dir"])
    rows: list[dict] = []
    for shard in sorted(out.glob("part_*.jsonl")):
        rows += [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines()]
    return rows


# -- static -------------------------------------------------------------------


def test_ingest_step_static() -> None:
    assert_step_static(
        STEP_DIR,
        expected_name="steps/curate/ingest",
        expected_launch="python",
        expected_default_config="default",
    )


def test_ingest_needs_no_gpu() -> None:
    """Preparing data must not require a cluster; that is where people start."""
    assert "gpus_per_node = 0" in (STEP_DIR / "step.py").read_text(encoding="utf-8")


def test_the_default_refuses_duplicates_rather_than_choosing_for_you() -> None:
    config = yaml.safe_load((STEP_DIR / "config" / "default.yaml").read_text(encoding="utf-8"))

    assert config["on_duplicate"] == "refuse"


# -- format -------------------------------------------------------------------


def test_jsonl_is_read(tmp_path) -> None:
    source = jsonl(tmp_path / "a.jsonl", [{"text": f"document {i}"} for i in range(5)])
    config = cfg(tmp_path, source)

    report = run_ingest.run(config)

    assert report["input"]["format"] == "jsonl"
    assert len(written(config)) == 5


def test_parquet_is_read_without_a_ray_cluster(tmp_path) -> None:
    """Curator's ParquetReader is a Ray stage; ingestion must not need one."""
    source = parquet(tmp_path / "a.parquet", [{"text": f"document {i}", "url": f"u{i}"} for i in range(5)])
    config = cfg(tmp_path, source)

    report = run_ingest.run(config)

    assert report["input"]["format"] == "parquet"
    assert len(written(config)) == 5


def test_a_mixed_glob_is_refused_rather_than_half_read(tmp_path) -> None:
    jsonl(tmp_path / "a.jsonl", [{"text": "x"}])
    parquet(tmp_path / "b.parquet", [{"text": "y"}])

    with pytest.raises(ingest_module.IngestError, match="cannot infer one format"):
        run_ingest.run(cfg(tmp_path, str(tmp_path / "*")))


def test_a_directory_is_accepted(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    jsonl(raw / "a.jsonl", [{"text": "one"}, {"text": "two"}])

    report = run_ingest.run(cfg(tmp_path, str(raw)))

    assert report["counts"]["written"] == 2


# -- identity -----------------------------------------------------------------


def test_a_minted_id_survives_resharding(tmp_path) -> None:
    """The property Curator's positional AddId does not have.

    Re-split the corpus and every positional id changes, so any subset or
    decontamination claim keyed on the old ones silently describes documents
    that no longer carry those names.
    """
    rows = [{"text": f"document number {i}"} for i in range(6)]

    one = cfg(tmp_path, jsonl(tmp_path / "whole.jsonl", rows), output_dir=str(tmp_path / "o1"))
    run_ingest.run(one)

    split_dir = tmp_path / "split"
    split_dir.mkdir()
    jsonl(split_dir / "a.jsonl", rows[3:])  # reversed order, different shards
    jsonl(split_dir / "b.jsonl", rows[:3])
    two = cfg(tmp_path, str(split_dir), output_dir=str(tmp_path / "o2"))
    run_ingest.run(two)

    assert {r["id"] for r in written(one)} == {r["id"] for r in written(two)}


def test_a_minted_id_keeps_the_full_sha256_digest(tmp_path) -> None:
    config = cfg(tmp_path, jsonl(tmp_path / "a.jsonl", [{"text": "one document"}]))

    run_ingest.run(config)

    assert len(written(config)[0]["id"]) == 64


def test_identity_field_boundaries_cannot_collide() -> None:
    left = {"url": "alpha\nbeta", "text": "gamma"}
    right = {"url": "alpha", "text": "beta\ngamma"}

    assert ingest_module.mint_id(left, ["url", "text"], "") != ingest_module.mint_id(right, ["url", "text"], "")


def test_the_recipe_is_recorded_so_ids_can_be_regenerated(tmp_path) -> None:
    source = jsonl(tmp_path / "a.jsonl", [{"text": "x", "url": "u"}])
    config = cfg(tmp_path, source, id_fields=["url", "text"], id_prefix="c4vi-")

    identity = run_ingest.run(config)["identity"]

    assert identity["source"] == "minted"
    assert identity["fields"] == ["url", "text"]
    assert identity["prefix"] == "c4vi-"
    assert written(config)[0]["id"].startswith("c4vi-")


def test_a_corpus_that_carries_an_id_keeps_it(tmp_path) -> None:
    source = jsonl(tmp_path / "a.jsonl", [{"doc_id": "abc", "text": "x"}])
    config = cfg(tmp_path, source, id_from="doc_id")

    report = run_ingest.run(config)

    assert report["identity"]["source"] == "corpus"
    assert written(config)[0]["id"] == "abc"


def test_different_id_fields_give_different_ids(tmp_path) -> None:
    """So re-ingesting under another recipe cannot be mistaken for the same corpus."""
    rows = [{"text": "same text", "url": "a"}, {"text": "same text", "url": "b"}]
    source = jsonl(tmp_path / "a.jsonl", rows)

    by_text = cfg(tmp_path, source, id_fields=["text"], on_duplicate="suffix", output_dir=str(tmp_path / "t"))
    by_both = cfg(tmp_path, source, id_fields=["url", "text"], output_dir=str(tmp_path / "b"))
    run_ingest.run(by_text)
    run_ingest.run(by_both)

    assert len({r["id"] for r in written(by_both)}) == 2, "distinct urls are distinct documents"
    assert {r["id"] for r in written(by_text)} != {r["id"] for r in written(by_both)}


def test_minted_identity_fields_must_include_document_text(tmp_path) -> None:
    source = jsonl(tmp_path / "a.jsonl", [{"url": "u", "text": "document"}])

    with pytest.raises(ingest_module.IngestError, match="must include text_field"):
        run_ingest.run(cfg(tmp_path, source, id_fields=["url"]))


# -- duplicates ---------------------------------------------------------------


DUPES = [{"text": "identical"}, {"text": "identical"}, {"text": "unique"}]


def test_byte_identical_documents_are_refused_by_default(tmp_path) -> None:
    """Not hypothetical: 328 of 20,000 in one real Hindi corpus."""
    source = jsonl(tmp_path / "a.jsonl", DUPES)

    with pytest.raises(ingest_module.IngestError) as excinfo:
        run_ingest.run(cfg(tmp_path, source))

    message = str(excinfo.value)
    assert "1 document(s) repeat" in message
    assert "on_duplicate" in message, "the error must name the way out"


def test_refuse_leaves_no_partial_shards_or_success_report(tmp_path) -> None:
    config = cfg(tmp_path, jsonl(tmp_path / "a.jsonl", DUPES))

    with pytest.raises(ingest_module.IngestError):
        run_ingest.run(config)

    output = Path(config["output_dir"])
    assert not list(output.glob("part_*.jsonl"))
    assert not (output / "ingest_report.json").exists()


def test_a_failed_rerun_removes_the_previous_success_artifacts(tmp_path) -> None:
    source = tmp_path / "a.jsonl"
    config = cfg(tmp_path, jsonl(source, [{"text": "unique"}]))
    run_ingest.run(config)
    jsonl(source, DUPES)

    with pytest.raises(ingest_module.IngestError):
        run_ingest.run(config)

    assert not list(Path(config["output_dir"]).glob("part_*.jsonl"))
    assert not (Path(config["output_dir"]) / "ingest_report.json").exists()


def test_drop_keeps_the_first_of_each_group(tmp_path) -> None:
    config = cfg(tmp_path, jsonl(tmp_path / "a.jsonl", DUPES), on_duplicate="drop")

    report = run_ingest.run(config)

    assert report["counts"]["written"] == 2
    assert report["counts"]["duplicate_ids"] == 1
    assert any("on_duplicate: drop" in w for w in report["warnings"])


def test_suffix_keeps_every_copy_under_a_distinguishable_id(tmp_path) -> None:
    config = cfg(tmp_path, jsonl(tmp_path / "a.jsonl", DUPES), on_duplicate="suffix")

    report = run_ingest.run(config)
    ids = [r["id"] for r in written(config)]

    assert report["counts"]["written"] == 3
    assert len(set(ids)) == 3, "suffixing must actually disambiguate"


def test_suffix_never_collides_with_a_corpus_id(tmp_path) -> None:
    rows = [
        {"doc_id": "same-1", "text": "reserved suffix"},
        {"doc_id": "same", "text": "first"},
        {"doc_id": "same", "text": "second"},
    ]
    config = cfg(
        tmp_path,
        jsonl(tmp_path / "a.jsonl", rows),
        id_from="doc_id",
        on_duplicate="suffix",
    )

    run_ingest.run(config)

    ids = [row["id"] for row in written(config)]
    assert len(ids) == len(set(ids)) == 3
    assert "same-2" in ids


def test_corpus_id_duplicates_are_not_described_as_identical_content(tmp_path) -> None:
    rows = [
        {"doc_id": "same", "text": "first"},
        {"doc_id": "same", "text": "different"},
    ]
    config = cfg(tmp_path, jsonl(tmp_path / "a.jsonl", rows), id_from="doc_id")

    with pytest.raises(ingest_module.IngestError) as excinfo:
        run_ingest.run(config)

    message = str(excinfo.value)
    assert "duplicate 'doc_id'" in message
    assert "byte-identical" not in message


def test_duplicate_examples_are_distinct_ids(tmp_path) -> None:
    """The largest real group held 293 copies, so 'the first three' says nothing."""
    rows = [{"text": "a"}] * 4 + [{"text": "b"}] * 4 + [{"text": "c"}] * 4
    source = jsonl(tmp_path / "a.jsonl", rows)

    with pytest.raises(ingest_module.IngestError) as excinfo:
        run_ingest.run(cfg(tmp_path, source))

    message = str(excinfo.value)
    assert "3 group(s)" in message
    quoted = [part for part in message.split() if len(part.strip(",.()")) == 64]
    assert len(set(quoted)) == len(quoted), "the same id must not be quoted twice"


# -- what it refuses to do ----------------------------------------------------


def test_text_is_never_rewritten(tmp_path) -> None:
    """Normalisation that changes content is a filtering decision, not a read."""
    original = "  Tiếng Việt  \n\n  với khoảng trắng lạ  "
    config = cfg(tmp_path, jsonl(tmp_path / "a.jsonl", [{"text": original}]))

    run_ingest.run(config)

    assert written(config)[0]["text"] == original


def test_unlisted_columns_are_dropped(tmp_path) -> None:
    """So a later step cannot come to depend on a column nobody asked for."""
    source = jsonl(tmp_path / "a.jsonl", [{"text": "x", "url": "u", "junk": "z"}])
    config = cfg(tmp_path, source, keep_fields=["url"])
    run_ingest.run(config)

    row = written(config)[0]

    assert "url" in row
    assert "junk" not in row


def test_available_columns_are_reported_even_when_dropped(tmp_path) -> None:
    """Dropping is fine; dropping invisibly is not."""
    source = jsonl(tmp_path / "a.jsonl", [{"text": "x", "url": "u", "junk": "z"}])

    report = run_ingest.run(cfg(tmp_path, source, keep_fields=[]))

    assert set(report["columns_available"]) == {"text", "url", "junk"}


def test_records_without_text_are_counted_not_hidden(tmp_path) -> None:
    source = jsonl(tmp_path / "a.jsonl", [{"text": "ok"}, {"url": "no text"}, {"text": ""}])

    report = run_ingest.run(cfg(tmp_path, source))

    assert report["counts"]["written"] == 1
    assert report["counts"]["skipped_missing_text"] == 2
    assert any("no 'text'" in w or "had no" in w for w in report["warnings"])


def test_unparsable_lines_are_counted(tmp_path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"text": "ok"}\n{not json\n', encoding="utf-8")

    report = run_ingest.run(cfg(tmp_path, str(path)))

    assert report["counts"]["unparsable_lines"] == 1


def test_valid_non_object_json_values_are_counted(tmp_path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"text": "ok"}\n"scalar"\n[1, 2]\nnull\n', encoding="utf-8")

    report = run_ingest.run(cfg(tmp_path, str(path)))

    assert report["counts"]["records_read"] == 4
    assert report["counts"]["skipped_non_mapping"] == 3
    assert any("non-object JSON" in warning for warning in report["warnings"])


def test_missing_corpus_ids_are_warned_not_only_counted(tmp_path) -> None:
    rows = [{"doc_id": "kept", "text": "one"}, {"text": "missing id"}]
    config = cfg(tmp_path, jsonl(tmp_path / "a.jsonl", rows), id_from="doc_id")

    report = run_ingest.run(config)

    assert report["counts"]["skipped_missing_id"] == 1
    assert any("no 'doc_id'" in warning for warning in report["warnings"])


def test_a_corpus_with_nothing_usable_is_an_error(tmp_path) -> None:
    source = jsonl(tmp_path / "a.jsonl", [{"url": "a"}, {"url": "b"}])

    with pytest.raises(ingest_module.IngestError, match="no usable documents"):
        run_ingest.run(cfg(tmp_path, source))


def test_no_usable_documents_leave_no_empty_shard(tmp_path) -> None:
    config = cfg(tmp_path, jsonl(tmp_path / "a.jsonl", [{"url": "a"}, {"url": "b"}]))

    with pytest.raises(ingest_module.IngestError):
        run_ingest.run(config)

    assert not list(Path(config["output_dir"]).glob("part_*.jsonl"))


def test_exact_shard_boundary_does_not_create_a_trailing_empty_shard(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_ingest, "SHARD_SIZE", 2)
    config = cfg(
        tmp_path,
        jsonl(tmp_path / "a.jsonl", [{"text": "one"}, {"text": "two"}]),
    )

    run_ingest.run(config)

    assert [path.name for path in Path(config["output_dir"]).glob("part_*.jsonl")] == ["part_0.jsonl"]


def test_an_unmatched_input_is_an_error(tmp_path) -> None:
    with pytest.raises(ingest_module.IngestError, match="matched no files"):
        run_ingest.run(cfg(tmp_path, str(tmp_path / "nothing" / "*.jsonl")))


def test_every_error_is_documented() -> None:
    with (STEP_DIR / "step.toml").open("rb") as fh:
        manifest = tomllib.load(fh)

    documented = {e["name"] for e in manifest.get("errors", [])}

    assert {
        "duplicate_documents",
        "mixed_formats",
        "no_usable_documents",
        "invalid_identity_fields",
    } <= documented


# -- a step must not read its own output --------------------------------------
#
# Regression: with output_dir nested under the input directory — the layout the
# README's own example produces — a second run resolved the FIRST run's shards
# and its ingest_report.json as input. The report's lines were counted as corrupt
# documents, and the shards are deleted before the reader reaches them, so the
# reader raised FileNotFoundError from inside the loop.


def _nested(tmp_path: Path) -> dict:
    raw = tmp_path / "raw"
    raw.mkdir()
    jsonl(raw / "corpus.jsonl", [{"text": f"document {i}", "url": f"u{i}"} for i in range(3)])
    return cfg(tmp_path, str(raw), output_dir=str(raw / "ingested"), id_fields=["url", "text"])


def test_running_twice_over_a_nested_output_dir_is_idempotent(tmp_path) -> None:
    config = _nested(tmp_path)

    first = run_ingest.run(dict(config))
    second = run_ingest.run(dict(config))

    assert second["counts"]["records_read"] == first["counts"]["records_read"] == 3
    assert second["counts"]["written"] == first["counts"]["written"] == 3


def test_the_previous_report_is_not_read_back_as_corrupt_documents(tmp_path) -> None:
    """ingest_report.json ends in .json, which a bare directory reference matches."""
    config = _nested(tmp_path)
    run_ingest.run(dict(config))

    second = run_ingest.run(dict(config))

    assert second["counts"]["unparsable_lines"] == 0


def test_skipped_own_output_is_reported_rather_than_silent(tmp_path) -> None:
    """ "1 file" where the user expected 2 must be traceable without re-deriving it."""
    config = _nested(tmp_path)
    run_ingest.run(dict(config))

    second = run_ingest.run(dict(config))

    assert second["input"]["files"] == 1
    assert second["input"]["skipped_own_output"] == 1


def test_an_input_that_is_only_the_previous_output_is_refused(tmp_path) -> None:
    """Excluding own output must not turn an empty corpus into a silent success."""
    config = _nested(tmp_path)
    run_ingest.run(dict(config))

    with pytest.raises(ingest_module.IngestError, match="matched no files"):
        run_ingest.run(cfg(tmp_path, str(Path(config["output_dir"])), output_dir=config["output_dir"]))
